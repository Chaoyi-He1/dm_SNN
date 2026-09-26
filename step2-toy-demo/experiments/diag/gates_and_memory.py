"""TASK gates-and-memory: quantify the gate mismatch (A0 gates see h, A1 gates see ternary x) and the memory write strength.

Usage (from step2-toy-demo):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/diag/gates_and_memory.py part1   # (a)(b)(c)(d)
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/diag/gates_and_memory.py part2   # (e) variants
Writes results/conversion-sweep/diag/gates-and-memory/{part1,part2}.json.
Does not modify toy_demo/, snn_spec/ or convert_lab.py: gate modules are swapped on the model instance at runtime.
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

STEP2 = "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo"
sys.path.insert(0, STEP2)
sys.path.insert(0, STEP2 + "/experiments")

import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402
import torch.nn.functional as F                          # noqa: E402

import convert_lab as cl                                 # noqa: E402  (chdirs to STEP2)
from toy_demo.train import evaluate                      # noqa: E402
from toy_demo.convert import calibrate_thresholds        # noqa: E402
from toy_demo.model import l2norm                        # noqa: E402
from toy_demo.data import split_inputs_targets, decode, BOS, EOS   # noqa: E402
from snn_spec.gdn import ExternalGates                   # noqa: E402

OUT = Path(STEP2) / "results/conversion-sweep/diag/gates-and-memory"
OUT.mkdir(parents=True, exist_ok=True)
FQ = 0.75
CHUNK = 256


# ----------------------------------------------------------------------------- gate modules swapped in at runtime

class GateOracle(nn.Module):
    """Returns pre-recorded (alpha, beta) for token t of the current batch; counter wraps at L so that repeated
    model(inputs) calls on the same batch (e.g. inside calibrate_thresholds) all replay the same gates."""

    def __init__(self):
        super().__init__()
        self.alpha = self.beta = None
        self.t = 0

    def set(self, alpha, beta):
        self.alpha, self.beta, self.t = alpha, beta, 0

    def forward(self, x, ext=None):
        a, b = self.alpha[:, self.t], self.beta[:, self.t]
        self.t = (self.t + 1) % self.alpha.shape[1]
        return a, b


class ConstGates(nn.Module):
    """Constant per-head gates (A0 means)."""

    def __init__(self, alpha_h, beta_h):
        super().__init__()
        self.register_buffer("alpha_h", alpha_h.clone())
        self.register_buffer("beta_h", beta_h.clone())

    def forward(self, x, ext=None):
        B = x.shape[0]
        return self.alpha_h.expand(B, -1), self.beta_h.expand(B, -1)


# ----------------------------------------------------------------------------- helpers

def q_stats(x):
    """mean/std/quantiles of a flat tensor."""
    x = x.flatten().float()
    qs = torch.quantile(x, torch.tensor([.05, .25, .5, .75, .95], device=x.device)).tolist()
    return dict(mean=round(x.mean().item(), 4), std=round(x.std().item(), 4),
                q05=round(qs[0], 4), q25=round(qs[1], 4), q50=round(qs[2], 4), q75=round(qs[3], 4), q95=round(qs[4], 4))


def corr(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum().item() / math.sqrt((a * a).sum().item() * (b * b).sum().item() + 1e-12)


@torch.no_grad()
def a0_pass(a0, tokens, dev):
    """A0 over the full slice in chunks: per-block alpha/beta [N, L-1, H], A0 ||k||^2 stats, h maxima, A0 metrics."""
    N = tokens.shape[0]
    L = len(a0.blocks)
    alphas = [[] for _ in range(L)]; betas = [[] for _ in range(L)]
    ksq_sum = [0.0] * L; ksq_n = [0] * L
    write_sum = [0.0] * L
    hmax = {f"block{i}": dict(h_in=0.0, h_mid=0.0, h_out=0.0) for i in range(L)}
    hmax["h_final"] = 0.0
    nll = 0.0; n_correct = 0; n_tok = 0
    for s in range(0, N, CHUNK):
        inputs, targets, mask = split_inputs_targets(tokens[s:s + CHUNK].to(dev))
        logits, recs = a0(inputs, record=True)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        nll += ce[mask].sum().item(); n_correct += (logits.argmax(-1) == targets)[mask].sum().item(); n_tok += mask.sum().item()
        for i in range(L):
            al = torch.stack([recs[t][i]["alpha"] for t in range(len(recs))], 1)      # [B, L-1, H]
            be = torch.stack([recs[t][i]["beta"] for t in range(len(recs))], 1)
            alphas[i].append(al); betas[i].append(be)
            zk = torch.stack([recs[t][i]["zk"] for t in range(len(recs))], 1)[mask]   # [n, H*dk]
            k = l2norm(zk.view(zk.shape[0], a0.blocks[i].attn.H, a0.blocks[i].attn.dk))
            ksq = (k * k).sum(-1)                                                       # [n, H]
            ksq_sum[i] += ksq.sum().item(); ksq_n[i] += ksq.numel()
            write_sum[i] += (be[mask] * ksq).sum().item()
            for key in ("h_in", "h_mid", "h_out"):
                v = torch.stack([recs[t][i][key] for t in range(len(recs))], 1)[mask].abs().max().item()
                hmax[f"block{i}"][key] = max(hmax[f"block{i}"][key], v)
        hf = torch.stack([recs[t][L]["h_final"] for t in range(len(recs))], 1)[mask].abs().max().item()
        hmax["h_final"] = max(hmax["h_final"], hf)
        del recs
    alphas = [torch.cat(a, 0) for a in alphas]; betas = [torch.cat(b, 0) for b in betas]
    return dict(alphas=alphas, betas=betas,
                k_sqnorm_mean=[ksq_sum[i] / ksq_n[i] for i in range(L)],
                write_mean=[write_sum[i] / ksq_n[i] for i in range(L)],
                hmax=hmax, metrics=dict(token_acc=n_correct / n_tok, ppl=math.exp(nll / n_tok), n_tokens=n_tok))


@torch.no_grad()
def snn_manual_pass(m, tokens, dev, gate_source, own_gates=None):
    """Token-by-token SpikingLM run with blk.attn.gates swapped to ExternalGates; gate_source(chunk_slice, t, i, x1)
    returns (alpha [B,H], beta [B,H]). own_gates[i] is the model's original SigmoidGates (for mixed variants)."""
    N = tokens.shape[0]
    L = len(m.blocks)
    nll = 0.0; n_correct = 0; n_tok = 0
    hmax = {f"block{i}": dict(h_in=0.0, h_mid=0.0, h_out=0.0) for i in range(L)}
    for s in range(0, N, CHUNK):
        inputs, targets, mask = split_inputs_targets(tokens[s:s + CHUNK].to(dev))
        B, T = inputs.shape
        st = m.init_state(B, dev)
        outs = []
        for t in range(T):
            h = m.emb(inputs[:, t])
            for i, blk in enumerate(m.blocks):
                x1 = blk.q_in1(h)
                gates = gate_source(slice(s, s + B), t, i, x1)
                hin = h.abs()[mask[:, t]].max().item() if mask[:, t].any() else 0.0
                hmax[f"block{i}"]["h_in"] = max(hmax[f"block{i}"]["h_in"], hin)
                h, st[i], _ = blk.step(h, st[i], ext_gates=gates)
                if mask[:, t].any():
                    hmax[f"block{i}"]["h_out"] = max(hmax[f"block{i}"]["h_out"], h.abs()[mask[:, t]].max().item())
            outs.append(m.head(m.q_out(h)))
        logits = torch.stack(outs, 1)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        nll += ce[mask].sum().item(); n_correct += (logits.argmax(-1) == targets)[mask].sum().item(); n_tok += mask.sum().item()
    return dict(token_acc=n_correct / n_tok, ppl=math.exp(nll / n_tok), n_tokens=n_tok, hmax=hmax)


@torch.no_grad()
def paired_completions(m, a0, sl, demo, max_new, dev):
    """Greedy completions where the SNN uses A0's gates computed on the same token stream (A0 run in lock-step)."""
    rows = []
    for item in demo:
        st_s = m.init_state(1, dev); st_f = a0.init_state(1, dev)

        def step(tok):
            nonlocal st_s, st_f
            tt = torch.tensor([tok], device=dev)
            _, st_f, rf = a0.step(tt, st_f, record=True)
            h = m.emb(tt)
            for i, blk in enumerate(m.blocks):
                h, st_s[i], _ = blk.step(h, st_s[i], ext_gates=(rf[i]["alpha"], rf[i]["beta"]))
            return m.head(m.q_out(h))

        logits = None
        for tok in [BOS] + list(item["prefix"]):
            logits = step(tok)
        out = []
        for _ in range(max_new):
            nxt = int(logits.argmax(-1).item())
            if nxt == EOS:
                break
            out.append(nxt)
            logits = step(nxt)
        rows.append(dict(prefix=decode(sl, item["prefix"]), target=decode(sl, item["target"]),
                         output=decode(sl, out), exact=(out == list(item["target"]))))
    return rows


@torch.no_grad()
def snn_gates_on(m, tokens, dev):
    """A1's own alpha/beta on tokens (model forward with record): list of ([N, L-1, H], [N, L-1, H]) per block."""
    L = len(m.blocks)
    alphas = [[] for _ in range(L)]; betas = [[] for _ in range(L)]
    for s in range(0, tokens.shape[0], CHUNK):
        inputs, _, mask = split_inputs_targets(tokens[s:s + CHUNK].to(dev))
        _, recs = m(inputs, record=True)
        for i in range(L):
            alphas[i].append(torch.stack([recs[t][i]["alpha"] for t in range(len(recs))], 1))
            betas[i].append(torch.stack([recs[t][i]["beta"] for t in range(len(recs))], 1))
        del recs
    return [torch.cat(a, 0) for a in alphas], [torch.cat(b, 0) for b in betas]


def gate_compare(al0, be0, al1, be1, mask):
    """Per-block comparison of A1 gates vs A0 gates on the same valid tokens."""
    a0v, a1v, b0v, b1v = al0[mask], al1[mask], be0[mask], be1[mask]
    return dict(alpha_a0=q_stats(a0v), alpha_a1=q_stats(a1v), beta_a0=q_stats(b0v), beta_a1=q_stats(b1v),
                alpha_mae=round((a0v - a1v).abs().mean().item(), 4), alpha_corr=round(corr(a0v, a1v), 4),
                beta_mae=round((b0v - b1v).abs().mean().item(), 4), beta_corr=round(corr(b0v, b1v), 4),
                alpha_a0_head_mean=[round(v, 4) for v in a0v.mean(0).tolist()],
                alpha_a1_head_mean=[round(v, 4) for v in a1v.mean(0).tolist()],
                beta_a0_head_mean=[round(v, 4) for v in b0v.mean(0).tolist()],
                beta_a1_head_mean=[round(v, 4) for v in b1v.mean(0).tolist()],
                alpha_a0_head_std=[round(v, 4) for v in a0v.std(0).tolist()],
                alpha_a1_head_std=[round(v, 4) for v in a1v.std(0).tolist()],
                beta_a0_head_std=[round(v, 4) for v in b0v.std(0).tolist()],
                beta_a1_head_std=[round(v, 4) for v in b1v.std(0).tolist()])


@torch.no_grad()
def tf_metrics_with_oracle(m, oracles, tokens, alphas, betas, dev):
    """Teacher-forced metrics using model.forward with GateOracle modules replaying A0 gates per chunk."""
    nll = 0.0; n_correct = 0; n_tok = 0
    for s in range(0, tokens.shape[0], CHUNK):
        inputs, targets, mask = split_inputs_targets(tokens[s:s + CHUNK].to(dev))
        for i, o in enumerate(oracles):
            o.set(alphas[i][s:s + CHUNK], betas[i][s:s + CHUNK])
        logits = m(inputs)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        nll += ce[mask].sum().item(); n_correct += (logits.argmax(-1) == targets)[mask].sum().item(); n_tok += mask.sum().item()
    return dict(token_acc=n_correct / n_tok, ppl=math.exp(nll / n_tok), n_tokens=n_tok)


def brief(ev):
    return {k: ev[k] for k in ("token_acc", "ppl", "n_exact") if k in ev}


# ----------------------------------------------------------------------------- parts

def setup():
    torch.set_num_threads(1)
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env(fire_quantile=FQ)
    a0 = cl.load_a0(cfg, sl, dev)
    m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, FQ)
    return dev, cfg, sl, demo, tokens, a0, m, thr


def part1():
    t0 = time.time()
    dev, cfg, sl, demo, tokens, a0, m, thr = setup()
    res = dict(part="part1", fq=FQ, thresholds=thr)
    L = cfg.n_blocks
    # (a) model_stats on 512 sentences
    sub = tokens[:512]
    res["a0_stats_512"] = cl.model_stats(a0, sub, dev, is_float=True)
    res["a1_init_stats_512"] = cl.model_stats(m, sub, dev, is_float=False)
    print("[a] model_stats done", round(time.time() - t0, 1), "s", flush=True)
    # A0 full-slice pass: gates, ||k||^2, h maxima
    A = a0_pass(a0, tokens, dev)
    res["a0_metrics_full"] = A["metrics"]
    res["a0_hmax_full"] = A["hmax"]
    _, _, mask_all = split_inputs_targets(tokens.to(dev))
    # A1_init own gates on the full slice and per-block comparison
    al1, be1 = snn_gates_on(m, tokens, dev)
    res["gate_compare_full"] = {f"block{i}": gate_compare(A["alphas"][i], A["betas"][i], al1[i], be1[i], mask_all) for i in range(L)}
    # gate logit inputs: what the gate linear layers see (h vs x) -- norms
    _, _, mask512 = split_inputs_targets(sub.to(dev))
    with torch.no_grad():
        _, r0 = a0(sub.to(dev)[:, :-1], record=True)
        _, r1 = m(sub.to(dev)[:, :-1], record=True)
        gi = {}
        for i in range(L):
            h = torch.stack([r0[t][i]["h_in"] for t in range(len(r0))], 1)[mask512]
            x = torch.stack([r1[t][i]["x_attn"] for t in range(len(r1))], 1)[mask512]
            g0, g1 = a0.blocks[i].attn.gates, m.blocks[i].attn.gates
            gi[f"block{i}"] = dict(h_norm_mean=round(h.norm(dim=-1).mean().item(), 4), x_norm_mean=round(x.norm(dim=-1).mean().item(), 4),
                                   x_nonzero_rate=round((x != 0).float().mean().item(), 4),
                                   Wa_logit_a0=q_stats(g0.Wa(h)), Wa_logit_a1=q_stats(g1.Wa(x)),
                                   Wb_logit_a0=q_stats(g0.Wb(h)), Wb_logit_a1=q_stats(g1.Wb(x)),
                                   Wa_logit_corr=round(corr(g0.Wa(h), g1.Wa(x)), 4), Wb_logit_corr=round(corr(g0.Wb(h), g1.Wb(x)), 4))
        del r0, r1
    res["gate_inputs_512"] = gi
    print("[a] gate compare done", round(time.time() - t0, 1), "s", flush=True)
    # (b) write strength
    ws = {}
    for i in range(L):
        s1 = res["a1_init_stats_512"][f"block{i}"]
        b0 = A["betas"][i][mask_all]; b1 = be1[i][mask_all]
        ws[f"block{i}"] = dict(a0_k_sqnorm_mean_full=round(A["k_sqnorm_mean"][i], 6), a0_beta_mean_full=round(b0.mean().item(), 4),
                               a0_write_mean_full=round(A["write_mean"][i], 4),
                               a1_k_sqnorm_mean_512=s1["k_sqnorm_mean"], a1_beta_mean_full=round(b1.mean().item(), 4),
                               a1_rate_k_512=s1["rate_k"])
    # A1 write strength on the full slice: beta * ||k||^2 per (token, head), needs k from record
    with torch.no_grad():
        for i in range(L):
            wsum = 0.0; ksum = 0.0; n = 0; zero_write = 0
            for s in range(0, tokens.shape[0], CHUNK):
                inputs, _, mask = split_inputs_targets(tokens[s:s + CHUNK].to(dev))
                _, recs = m(inputs, record=True)
                k = torch.stack([recs[t][i]["k"] for t in range(len(recs))], 1)[mask]       # [n, H, dk] already /sqrt(dk)
                be = torch.stack([recs[t][i]["beta"] for t in range(len(recs))], 1)[mask]   # [n, H]
                ksq = (k * k).sum(-1)
                wsum += (be * ksq).sum().item(); ksum += ksq.sum().item(); n += ksq.numel(); zero_write += (ksq == 0).sum().item()
                del recs
            ws[f"block{i}"].update(a1_k_sqnorm_mean_full=round(ksum / n, 4), a1_write_mean_full=round(wsum / n, 4),
                                   a1_zero_write_frac_full=round(zero_write / n, 4),
                                   write_ratio_a1_over_a0=round((wsum / n) / A["write_mean"][i], 4))
    res["write_strength"] = ws
    print("[b] write strength done", round(time.time() - t0, 1), "s", flush=True)
    # (c) accumulator range
    emb_max = a0.emb.weight.abs().max().item()
    used = torch.unique(tokens)
    emb_max_used = a0.emb.weight[used.to(dev)].abs().max().item()
    res["accumulator"] = dict(emb_absmax_all_vocab=round(emb_max, 4), emb_absmax_used_tokens=round(emb_max_used, 4),
                              plan_bound=round(emb_max + 2 * L, 4), a0_hmax_full=A["hmax"],
                              a0_hmax_512={f"block{i}": {k: res["a0_stats_512"][f"block{i}"][k + "_absmax"] for k in ("h_in", "h_mid", "h_out")} for i in range(L)},
                              a1_hmax_512={f"block{i}": {k: res["a1_init_stats_512"][f"block{i}"][k + "_absmax"] for k in ("h_in", "h_mid", "h_out")} for i in range(L)},
                              a0_h_final_512=res["a0_stats_512"]["h_final_absmax"], a1_h_final_512=res["a1_init_stats_512"]["h_final_absmax"])
    # (d) KEY: A1_init with A0 gates, manual token loop with ExternalGates
    own = [blk.attn.gates for blk in m.blocks]
    for blk in m.blocks:
        blk.attn.gates = ExternalGates()
    al0, be0 = A["alphas"], A["betas"]

    def src_a0(sl_, t, i, x1):
        return al0[i][sl_, t], be0[i][sl_, t]

    def src_own(sl_, t, i, x1):
        return own[i](x1)

    def src_a0alpha(sl_, t, i, x1):
        a_, b_ = own[i](x1); return al0[i][sl_, t], b_

    def src_a0beta(sl_, t, i, x1):
        a_, b_ = own[i](x1); return a_, be0[i][sl_, t]

    ev_norm = evaluate(_with_gates(m, own), sl, demo, cfg, tokens)      # normal A1_init via evaluate()
    for blk in m.blocks:
        blk.attn.gates = ExternalGates()
    res["a1_init_evaluate"] = brief(ev_norm)
    res["a1_init_completions"] = ev_norm["completions"]
    r_own = snn_manual_pass(m, tokens, dev, src_own)
    res["a1_init_manual_loop_owngates"] = r_own
    r_a0 = snn_manual_pass(m, tokens, dev, src_a0)
    res["a1_init_a0gates"] = r_a0
    print(f"[d] A1_init own-gates (manual loop) acc {r_own['token_acc']:.4f} ppl {r_own['ppl']:.2f}; "
          f"A0-gates acc {r_a0['token_acc']:.4f} ppl {r_a0['ppl']:.2f}", round(time.time() - t0, 1), "s", flush=True)
    res["a1_init_a0alpha_ownbeta"] = snn_manual_pass(m, tokens, dev, src_a0alpha)
    res["a1_init_ownalpha_a0beta"] = snn_manual_pass(m, tokens, dev, src_a0beta)
    rows = paired_completions(m, a0, sl, demo, cfg.max_new_tokens, dev)
    res["a1_init_a0gates_completions"] = rows
    res["a1_init_a0gates"]["n_exact"] = sum(r["exact"] for r in rows)
    print("[d] mixed variants + paired completions done", round(time.time() - t0, 1), "s", flush=True)
    # (d2) A0 gates + thresholds recalibrated under A0 gates (GateOracle inside calibrate_thresholds)
    m2 = copy.deepcopy(m)
    oracles = [GateOracle() for _ in range(L)]
    for blk, o in zip(m2.blocks, oracles):
        blk.attn.gates = o
    for i, o in enumerate(oracles):
        o.set(al0[i][:cfg.calib_sentences], be0[i][:cfg.calib_sentences])
    r_chk = tf_metrics_with_oracle(m2, oracles, tokens, al0, be0, dev)
    res["a1_init_a0gates_oracle_check"] = r_chk           # must equal a1_init_a0gates (acc/ppl)
    for i, o in enumerate(oracles):
        o.set(al0[i][:cfg.calib_sentences], be0[i][:cfg.calib_sentences])
    thr2 = calibrate_thresholds(m2, tokens[:cfg.calib_sentences].to(dev), FQ)
    res["a1_init_a0gates_recalib"] = tf_metrics_with_oracle(m2, oracles, tokens, al0, be0, dev)
    res["a1_init_a0gates_recalib"]["thresholds"] = thr2
    print(f"[d2] A0-gates recalib acc {res['a1_init_a0gates_recalib']['token_acc']:.4f} ppl {res['a1_init_a0gates_recalib']['ppl']:.2f}",
          round(time.time() - t0, 1), "s", flush=True)
    # restore
    for blk, g in zip(m.blocks, own):
        blk.attn.gates = g
    # A0 per-head gate means (full slice) for part2
    res["a0_gate_means_full"] = {f"block{i}": dict(alpha=[round(v, 5) for v in al0[i][mask_all].mean(0).tolist()],
                                                    beta=[round(v, 5) for v in be0[i][mask_all].mean(0).tolist()]) for i in range(L)}
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "part1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("part1 written", res["seconds"], "s")


def _with_gates(m, gates):
    for blk, g in zip(m.blocks, gates):
        blk.attn.gates = g
    return m


def part2():
    t0 = time.time()
    dev, cfg, sl, demo, tokens, a0, m, thr = setup()
    L = cfg.n_blocks
    p1 = json.loads((OUT / "part1.json").read_text(encoding="utf-8"))
    res = dict(part="part2", fq=FQ)
    _, _, mask_all = split_inputs_targets(tokens.to(dev))
    A = a0_pass(a0, tokens, dev)
    al0, be0 = A["alphas"], A["betas"]
    # (e1) gate_refit=True
    mg, thr_g = cl.build_a1(a0, cfg, sl, tokens, dev, FQ, gate_refit=True)
    ev = evaluate(mg, sl, demo, cfg, tokens)
    res["a1_gate_refit"] = dict(**brief(ev), thresholds=thr_g, completions=ev["completions"])
    al1, be1 = snn_gates_on(mg, tokens, dev)
    res["a1_gate_refit_gate_compare_full"] = {f"block{i}": gate_compare(al0[i], be0[i], al1[i], be1[i], mask_all) for i in range(L)}
    res["a1_gate_refit_stats_512"] = cl.model_stats(mg, tokens[:512], dev)
    print(f"[e1] gate_refit acc {ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']}", round(time.time() - t0, 1), "s", flush=True)
    # (e1b) gate_refit gates but A1_init (non-refit) thresholds: isolates the gate change from the recalibration
    mgb = copy.deepcopy(m)
    for i in range(L):
        mgb.blocks[i].attn.gates.load_state_dict(mg.blocks[i].attn.gates.state_dict())
    ev = evaluate(mgb, sl, demo, cfg, tokens)
    res["a1_init_with_refit_gates_only"] = brief(ev)
    # (e2) constant gates = A0 per-head means (full slice), A1_init thresholds
    means = p1["a0_gate_means_full"]
    mc = copy.deepcopy(m)
    for i in range(L):
        mc.blocks[i].attn.gates = ConstGates(torch.tensor(means[f"block{i}"]["alpha"], device=dev),
                                             torch.tensor(means[f"block{i}"]["beta"], device=dev))
    ev = evaluate(mc, sl, demo, cfg, tokens)
    res["a1_init_constgates"] = dict(**brief(ev), completions=ev["completions"])
    print(f"[e2] const gates acc {ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']}", round(time.time() - t0, 1), "s", flush=True)
    # (e2b) constant gates + thresholds recalibrated under constant gates
    thr_c = calibrate_thresholds(mc, tokens[:cfg.calib_sentences].to(dev), FQ)
    ev = evaluate(mc, sl, demo, cfg, tokens)
    res["a1_init_constgates_recalib"] = dict(**brief(ev), thresholds=thr_c)
    print(f"[e2b] const gates recalib acc {ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']}", round(time.time() - t0, 1), "s", flush=True)
    # (e3) reference: A0 itself with constant gates (how much does A0 rely on token-dependent gates?)
    a0c = copy.deepcopy(a0)
    for i in range(L):
        a0c.blocks[i].attn.gates = ConstGates(torch.tensor(means[f"block{i}"]["alpha"], device=dev),
                                              torch.tensor(means[f"block{i}"]["beta"], device=dev))
    ev = evaluate(a0c, sl, demo, cfg, tokens)
    res["a0_constgates"] = brief(ev)
    print(f"[e3] A0 const gates acc {ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']}", round(time.time() - t0, 1), "s", flush=True)
    # (e4) A1_init thresholds recalibrated with own gates but with the default A0 evaluation numbers for reference
    res["a0_evaluate"] = brief(evaluate(a0, sl, demo, cfg, tokens))
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "part2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("part2 written", res["seconds"], "s")


if __name__ == "__main__":
    {"part1": part1, "part2": part2}[sys.argv[1]]()
