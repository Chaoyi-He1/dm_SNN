"""Part 3 of gates-and-memory: is the weak delta-rule write (beta * ||k||^2 ~ 0.25 * A0) the issue rather than the gates?
Variants on A1_init (fq 0.75, leak 0), all evaluated with toy_demo.train.evaluate on the full slice:
  - beta x2 / x4 (clamped to 1) with own gates, thresholds as calibrated and recalibrated
  - k threshold quantile 0.5 / 0.0 (k fires on ~50% / ~100% of components -> ||k||^2 ~ 0.5 / 1.0), other points at 0.75, forward-order recalibration
  - A0 gates + beta x4 (GateOracle replay), teacher-forced metrics only
Writes results/conversion-sweep/diag/gates-and-memory/part3.json.
"""
import copy
import json
import sys
import time
from pathlib import Path

STEP2 = "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo"
sys.path.insert(0, STEP2)
sys.path.insert(0, STEP2 + "/experiments")
sys.path.insert(0, STEP2 + "/experiments/diag")

import torch                                             # noqa: E402
import torch.nn as nn                                    # noqa: E402

import convert_lab as cl                                 # noqa: E402
from toy_demo.train import evaluate                      # noqa: E402
from toy_demo.convert import _block_points, _set_thr, calibrate_thresholds   # noqa: E402
from toy_demo.data import split_inputs_targets           # noqa: E402
from gates_and_memory import GateOracle, a0_pass, tf_metrics_with_oracle, brief, setup, OUT, FQ   # noqa: E402


class BetaScaled(nn.Module):
    def __init__(self, inner, s):
        super().__init__()
        self.inner, self.s = inner, float(s)

    def forward(self, x, ext=None):
        a, b = self.inner(x, ext)
        return a, (b * self.s).clamp(max=1.0)


@torch.no_grad()
def calibrate_custom(m, tokens, fq_default, fq_by_name):
    """calibrate_thresholds with a per-point quantile override (forward order, same as toy_demo.convert)."""
    inputs, _, mask = split_inputs_targets(tokens)
    out = {}

    def qof(fn, bi, ternary, q):
        _, recs = m(inputs, record=True)
        z = torch.stack([fn(recs[t][bi]) for t in range(inputs.shape[1])], dim=1)[mask]
        z = z.abs() if ternary else z
        return max(torch.quantile(z.flatten().float(), q).item(), 1e-3)

    for i, blk in enumerate(m.blocks):
        for name, mod, fn, ternary in _block_points(blk):
            thr = qof(fn, i, ternary, fq_by_name.get(name, fq_default))
            _set_thr(mod, thr); out[f"blocks.{i}.{name}"] = thr
    thr = qof(lambda r: r["h_final"], len(m.blocks), True, fq_default)
    _set_thr(m.q_out, thr); out["q_out"] = thr
    return out


def main():
    t0 = time.time()
    dev, cfg, sl, demo, tokens, a0, m, thr = setup()
    L = cfg.n_blocks
    calib = tokens[:cfg.calib_sentences].to(dev)
    res = dict(part="part3", fq=FQ)
    # beta scaling, own gates
    for s in (2.0, 4.0):
        mb = copy.deepcopy(m)
        for blk in mb.blocks:
            blk.attn.gates = BetaScaled(blk.attn.gates, s)
        ev = evaluate(mb, sl, demo, cfg, tokens)
        res[f"beta_x{s:g}"] = brief(ev)
        st = cl.model_stats(mb, tokens[:512], dev)
        res[f"beta_x{s:g}"]["write_mean_512"] = [round(st[f"block{i}"]["beta_mean"] * st[f"block{i}"]["k_sqnorm_mean"], 4) for i in range(L)]
        res[f"beta_x{s:g}"]["beta_mean_512"] = [round(st[f"block{i}"]["beta_mean"], 4) for i in range(L)]
        thr_b = calibrate_thresholds(mb, calib, FQ)
        ev = evaluate(mb, sl, demo, cfg, tokens)
        res[f"beta_x{s:g}_recalib"] = dict(**brief(ev), thresholds=thr_b)
        print(f"[beta x{s:g}] acc {res[f'beta_x{s:g}']['token_acc']:.4f} ppl {res[f'beta_x{s:g}']['ppl']:.2f} | recalib acc "
              f"{ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']}", round(time.time() - t0, 1), "s", flush=True)
    # k threshold quantile
    for qk in (0.5, 0.0):
        mk = copy.deepcopy(m)
        thr_k = calibrate_custom(mk, calib, FQ, {"k": qk})
        ev = evaluate(mk, sl, demo, cfg, tokens)
        st = cl.model_stats(mk, tokens[:512], dev)
        res[f"k_fq{qk:g}"] = dict(**brief(ev), thresholds=thr_k,
                                 rate_k_512=[st[f"block{i}"]["rate_k"] for i in range(L)],
                                 k_sqnorm_512=[st[f"block{i}"]["k_sqnorm_mean"] for i in range(L)],
                                 write_mean_512=[round(st[f"block{i}"]["beta_mean"] * st[f"block{i}"]["k_sqnorm_mean"], 4) for i in range(L)])
        print(f"[k fq {qk:g}] acc {ev['token_acc']:.4f} ppl {ev['ppl']:.2f} exact {ev['n_exact']} k_sqnorm {res[f'k_fq{qk:g}']['k_sqnorm_512']}",
              round(time.time() - t0, 1), "s", flush=True)
    # k fq 0 + beta x2 (write ~ 2*beta*1.0, i.e. over A0) is not meaningful; instead A0 gates + beta x4 via oracle
    A = a0_pass(a0, tokens, dev)
    al0, be0 = A["alphas"], A["betas"]
    mo = copy.deepcopy(m)
    oracles = [GateOracle() for _ in range(L)]
    for blk, o in zip(mo.blocks, oracles):
        blk.attn.gates = BetaScaled(o, 4.0)
    res["a0gates_beta_x4"] = tf_metrics_with_oracle(mo, oracles, tokens, al0, be0, dev)
    print(f"[A0 gates beta x4] acc {res['a0gates_beta_x4']['token_acc']:.4f} ppl {res['a0gates_beta_x4']['ppl']:.2f}", round(time.time() - t0, 1), "s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (OUT / "part3.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("part3 written", res["seconds"], "s")


if __name__ == "__main__":
    main()
