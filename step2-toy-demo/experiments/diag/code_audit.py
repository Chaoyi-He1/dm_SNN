"""code-audit: adversarial review of the A0 -> A1 conversion (non-hyperparameter loss sources).

Sections (run all or a subset with --only):
  keys     (4) transfer_weights key coverage
  dists    (1) signed / abs distributions of every pre-activation at A1_init, firing rates, per-channel dispersion,
               all-zero q/k heads, scale mismatch A0 vs A1_init (lambda folding), k norms (2)
  head     (5) logit std A0 vs A1_init; scalar / per-row / full-LS head rescale and its effect on acc/ppl
  drift    (6) thresholds in results/ckpt/a1.pt vs A1_init calibrated values
  sg       (7) surrogate gradient of ternary_fire and per-point gradient coverage
  extras   (8) eval/train mode, padding, PAD-embedding usage, A0+q_out-only ablation, per-channel threshold ablation
Outputs: results/conversion-sweep/diag/code-audit/<section>.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import convert_lab as cl  # noqa: E402
from toy_demo.convert import _block_points, _set_thr, calibrate_thresholds, transfer_weights  # noqa: E402
from toy_demo.data import PAD, EOS, batches, split_inputs_targets  # noqa: E402
from toy_demo.model import FloatLM, SpikingLM, l2norm  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402
from snn_spec.neurons import SpikeFn, ternary_fire  # noqa: E402

OUT = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/code-audit'
os.makedirs(OUT, exist_ok=True)
POINT_OUT = dict(q_in1='x_attn', q='q', k='k', v='v', pre='o_spk', out='y_attn', q_in2='x_ffn', g='g', u='u', y2='y_ffn')


def r(x, n=4):
    return round(float(x), n)


def stack(recs, i, key, mask):
    return torch.stack([recs[t][i][key] for t in range(len(recs))], dim=1)[mask]


def ev3(m, sl, demo, cfg, tokens):
    e = evaluate(m, sl, demo, cfg, tokens)
    return dict(token_acc=r(e['token_acc']), ppl=r(e['ppl'], 3), n_exact=e['n_exact'])


def save(name, obj):
    p = os.path.join(OUT, name + '.json')
    with open(p, 'w') as f:
        json.dump(obj, f, indent=1)
    print(f'--- {name} -> {p}')
    print(json.dumps(obj, indent=1)[:6000])


# ----------------------------------------------------------------------------- (4) keys
def sec_keys(ctx):
    a0, cfg, sl, dev = ctx['a0'], ctx['cfg'], ctx['sl'], ctx['dev']
    m = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev)
    k0 = set(a0.state_dict().keys())
    k1 = set(m.state_dict().keys())
    thr_keys = {k for k in k1 if 'log_thr' in k}
    res = m.load_state_dict(a0.state_dict(), strict=False)
    # value-level check after transfer
    transfer_weights(a0, m)
    sd0, sd1 = a0.state_dict(), m.state_dict()
    maxdiff = max((sd0[k].float() - sd1[k].float()).abs().max().item() for k in k0)
    out = dict(n_a0_keys=len(k0), n_a1_keys=len(k1), n_log_thr_keys=len(thr_keys),
               a0_equals_a1_minus_thr=(k0 == (k1 - thr_keys)),
               missing_non_thr=sorted(set(res.missing_keys) - thr_keys), unexpected=sorted(res.unexpected_keys),
               max_abs_diff_after_transfer=maxdiff, thr_keys=sorted(thr_keys),
               leak_in_state_dict=any('leak' in k for k in k1),
               n_params_a0=sum(p.numel() for p in a0.parameters()), n_params_a1=sum(p.numel() for p in m.parameters()))
    save('keys', out)


# ----------------------------------------------------------------------------- (1)(2) dists
@torch.no_grad()
def sec_dists(ctx):
    a0, cfg, sl, dev, tokens, demo = ctx['a0'], ctx['cfg'], ctx['sl'], ctx['dev'], ctx['tokens'], ctx['demo']
    m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, 0.75)
    calib = tokens[:cfg.calib_sentences].to(dev)
    inputs, _, mask = split_inputs_targets(calib)
    _, r1 = m(inputs, record=True)
    _, r0 = a0(inputs, record=True)
    qs = torch.tensor([.05, .25, .5, .75, .95], device=dev)
    points = {}
    for i, blk in enumerate(m.blocks):
        for name, mod, fn, ternary in _block_points(blk):
            z = torch.stack([fn(r1[t][i]) for t in range(len(r1))], dim=1)[mask]   # [N, C] pre-activation (=U at leak 0)
            spk = stack(r1, i, POINT_OUT[name], mask).reshape(z.shape[0], -1)
            rate_c = (spk != 0).float().mean(0)
            sq = torch.quantile(z.flatten(), qs).tolist()
            aq = torch.quantile(z.abs().flatten(), qs).tolist()
            d = dict(ternary=ternary, thr=r(thr[f'blocks.{i}.{name}']), frac_neg=r((z < 0).float().mean()),
                     signed_q=[r(v) for v in sq], abs_q=[r(v) for v in aq], z_std=r(z.std()), z_mean=r(z.mean()),
                     rate=r((spk != 0).float().mean()), rate_pos=r((spk > 0).float().mean()), rate_neg=r((spk < 0).float().mean()),
                     chan_rate_min=r(rate_c.min()), chan_rate_med=r(rate_c.median()), chan_rate_max=r(rate_c.max()),
                     frac_chan_rate_lt05=r((rate_c < 0.05).float().mean()), frac_chan_rate_gt50=r((rate_c > 0.5).float().mean()),
                     chan_std_min=r(z.std(0).min()), chan_std_max=r(z.std(0).max()),
                     n_channels=int(z.shape[1]))
            if not ternary:
                d['thr_if_abs_quantile'] = r(aq[3])
                d['rate_if_abs_quantile'] = r((z >= aq[3]).float().mean())
                d['floor_hit'] = bool(thr[f'blocks.{i}.{name}'] <= 1.001e-3)
            points[f'blocks.{i}.{name}'] = d
        # per-head q / k zero counts
        q = stack(r1, i, 'q', mask)            # [N, H, dk] binary
        k = stack(r1, i, 'k', mask)            # scaled
        nq = (q != 0).sum(-1).float()          # [N, H]
        nk = (k != 0).sum(-1).float()
        points[f'blocks.{i}.q_head'] = dict(ones_per_head_mean=r(nq.mean()), ones_min=int(nq.min()), ones_max=int(nq.max()),
                                            frac_head_tokens_all_zero=r((nq == 0).float().mean()),
                                            frac_tokens_any_head_zero=r((nq == 0).any(-1).float().mean()),
                                            q_sqnorm_a1_mean=r(nq.mean()), q_sqnorm_a0=1.0)
        points[f'blocks.{i}.k_head'] = dict(nonzero_per_head_mean=r(nk.mean()), nz_min=int(nk.min()), nz_max=int(nk.max()),
                                            frac_head_tokens_all_zero=r((nk == 0).float().mean()),
                                            k_sqnorm_a1_mean=r((k ** 2).sum(-1).mean()), k_sqnorm_a1_max=r((k ** 2).sum(-1).max()),
                                            k_sqnorm_a0=1.0, c_k=r(1 / math.sqrt(cfg.dk), 5))
    hf = stack(r1, len(m.blocks), 'h_final', mask)
    aq = torch.quantile(hf.abs().flatten(), qs).tolist()
    xo = m.q_out(hf)
    rate_c = (xo != 0).float().mean(0)
    points['q_out'] = dict(ternary=True, thr=r(thr['q_out']), abs_q=[r(v) for v in aq], rate=r((xo != 0).float().mean()),
                           chan_rate_min=r(rate_c.min()), chan_rate_max=r(rate_c.max()),
                           frac_chan_rate_lt05=r((rate_c < 0.05).float().mean()), frac_chan_rate_gt50=r((rate_c > 0.5).float().mean()))
    # ---- scale mismatch A0 vs A1_init (lambda folding table) ----
    scale = {}
    emb = a0.emb.weight[3:]
    scale['emb'] = dict(rms=r(emb.pow(2).mean().sqrt()), absmax=r(emb.abs().max()))
    for i in range(len(m.blocks)):
        s = {}
        for a0key, a1key in (('zo', 'y_attn'), ('z2', 'y_ffn'), ('o', 'o'), ('o', 'o_spk')):
            x0 = stack(r0, i, a0key, mask); x1 = stack(r1, i, a1key, mask)
            s[f'{a0key}->{a1key}'] = dict(a0_rms=r(x0.pow(2).mean().sqrt()), a0_absmean=r(x0.abs().mean()), a0_absmax=r(x0.abs().max()),
                                          a1_rms=r(x1.pow(2).mean().sqrt()), a1_absmean=r(x1.abs().mean()), a1_absmax=r(x1.abs().max()))
        z1, z3 = stack(r0, i, 'z1', mask), stack(r0, i, 'z3', mask)
        p0 = F.silu(z1) * z3; p1 = stack(r1, i, 'p', mask)
        s['silu(z1)*z3->p'] = dict(a0_rms=r(p0.pow(2).mean().sqrt()), a0_absmean=r(p0.abs().mean()), a0_frac_abs_gt_0p01=r((p0.abs() > 1e-2).float().mean()),
                                   a1_rms=r(p1.pow(2).mean().sqrt()), a1_rate=r((p1 != 0).float().mean()))
        for key in ('h_in', 'h_mid', 'h_out'):
            x0 = stack(r0, i, key, mask); x1 = stack(r1, i, key, mask)
            s[key] = dict(a0_rms=r(x0.pow(2).mean().sqrt()), a0_absmax=r(x0.abs().max()), a1_rms=r(x1.pow(2).mean().sqrt()), a1_absmax=r(x1.abs().max()),
                          cos_a0_a1=r(F.cosine_similarity(x0, x1, dim=-1).mean()))
        # A0 q/k after l2norm vs A1 binary/ternary
        zq0 = stack(r0, i, 'zq', mask).view(-1, cfg.n_heads, cfg.dk); q0 = l2norm(zq0)
        s['q_a0_l2norm_absmean'] = r(q0.abs().mean()); s['q_a0_frac_pos'] = r((q0 > 0).float().mean())
        s['alpha_mean_a0'] = r(stack(r0, i, 'alpha', mask).mean()); s['alpha_mean_a1'] = r(stack(r1, i, 'alpha', mask).mean())
        s['beta_mean_a0'] = r(stack(r0, i, 'beta', mask).mean()); s['beta_mean_a1'] = r(stack(r1, i, 'beta', mask).mean())
        scale[f'block{i}'] = s
    hf0 = stack(r0, len(m.blocks), 'h_final', mask)
    scale['h_final'] = dict(a0_rms=r(hf0.pow(2).mean().sqrt()), a0_absmax=r(hf0.abs().max()), a1_rms=r(hf.pow(2).mean().sqrt()), a1_absmax=r(hf.abs().max()),
                            cos_a0_a1=r(F.cosine_similarity(hf0, hf, dim=-1).mean()))
    out = dict(points=points, scale=scale, a1_init_eval=ev3(m, sl, demo, cfg, tokens), thresholds=thr)
    save('dists', out)
    ctx['a1_init'] = m
    ctx['thr'] = thr


# ----------------------------------------------------------------------------- (5) head
@torch.no_grad()
def sec_head(ctx):
    a0, cfg, sl, dev, tokens, demo = ctx['a0'], ctx['cfg'], ctx['sl'], ctx['dev'], ctx['tokens'], ctx['demo']
    m = ctx.get('a1_init')
    if m is None:
        m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, 0.75)
        ctx['a1_init'], ctx['thr'] = m, thr
    calib = tokens[:cfg.calib_sentences].to(dev)
    inputs, targets, mask = split_inputs_targets(calib)
    L0, rec0 = a0(inputs, record=True); L0 = L0[mask]
    L1, rec1 = m(inputs, record=True); L1 = L1[mask]
    out = dict(n_tokens=int(L0.shape[0]), vocab=int(L0.shape[1]))
    for nm, L in (('a0', L0), ('a1_init', L1)):
        out[f'{nm}_logit_std'] = r(L.std()); out[f'{nm}_logit_absmax'] = r(L.abs().max())
        out[f'{nm}_per_token_std_mean'] = r(L.std(-1).mean()); out[f'{nm}_max_logit_mean'] = r(L.max(-1).values.mean())
        out[f'{nm}_ce_calib'] = r(F.cross_entropy(L, targets[mask]))
    # trained A1 for context
    a1 = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev)
    a1.load_state_dict(torch.load('results/ckpt/a1.pt', map_location=dev)); a1.eval()
    La = a1(inputs)[mask]
    out['a1_trained_logit_std'] = r(La.std()); out['a1_trained_per_token_std_mean'] = r(La.std(-1).mean())
    out['a1_trained_max_logit_mean'] = r(La.max(-1).values.mean())
    out['head_weight_rms'] = r(a0.head.weight.pow(2).mean().sqrt())
    out['base_eval_a1_init'] = ev3(m, sl, demo, cfg, tokens)
    W = m.head.weight.clone()
    # (a) scalar LS
    s = (L0 * L1).sum() / (L1 * L1).sum()
    out['scalar_fit'] = dict(s=r(s), rel_resid=r(((s * L1 - L0).pow(2).sum() / L0.pow(2).sum()).sqrt()))
    m.head.weight.copy_(W * s); out['eval_scalar'] = ev3(m, sl, demo, cfg, tokens)
    # (b) per-vocab-row scale
    sv = (L0 * L1).sum(0) / (L1 * L1).sum(0).clamp_min(1e-8)
    out['row_fit'] = dict(s_min=r(sv.min()), s_med=r(sv.median()), s_max=r(sv.max()),
                          rel_resid=r(((sv * L1 - L0).pow(2).sum() / L0.pow(2).sum()).sqrt()))
    m.head.weight.copy_(W * sv[:, None]); out['eval_row'] = ev3(m, sl, demo, cfg, tokens)
    # (c) full LS refit of head (upper bound of what the head alone can recover): min ||xq W^T - L0||
    hf1 = stack(rec1, len(m.blocks), 'h_final', mask); xq = m.q_out(hf1)
    Wls, _ = cl._lstsq(xq, L0, ridge=1e-3)
    out['ls_fit'] = dict(rel_resid=r(((xq @ Wls.T - L0).pow(2).sum() / L0.pow(2).sum()).sqrt()))
    m.head.weight.copy_(Wls); out['eval_head_ls'] = ev3(m, sl, demo, cfg, tokens)
    # (d) oracle: A0's head on A0's h_final vs on A1's h_final (continuous, no q_out) — how much is the trunk vs the quantizer
    m.head.weight.copy_(W)
    hf0 = stack(rec0, len(m.blocks), 'h_final', mask)
    Lc = a0.head(hf1)
    out['a1_trunk_a0head_no_qout_ce_calib'] = r(F.cross_entropy(Lc, targets[mask]))
    out['a1_trunk_a0head_no_qout_acc_calib'] = r((Lc.argmax(-1) == targets[mask]).float().mean())
    out['a1_init_acc_calib'] = r((L1.argmax(-1) == targets[mask]).float().mean())
    out['a0_acc_calib'] = r((L0.argmax(-1) == targets[mask]).float().mean())
    out['cos_hfinal_a0_a1'] = r(F.cosine_similarity(hf0, hf1, dim=-1).mean())
    # (e) temperature that minimizes CE for A1_init (is ppl 695 "over-confidence"?)
    best = None
    for T in [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]:
        ce = F.cross_entropy(L1 / T, targets[mask]).item()
        if best is None or ce < best[1]:
            best = (T, ce)
    out['best_temperature_a1_init'] = dict(T=best[0], ce=r(best[1]), ppl=r(math.exp(best[1]), 2))
    save('head', out)


# ----------------------------------------------------------------------------- (6) drift
def sec_drift(ctx):
    cfg, sl, dev, tokens, demo = ctx['cfg'], ctx['sl'], ctx['dev'], ctx['tokens'], ctx['demo']
    with open('results/toy-demo-default-run.json') as f:
        init = json.load(f)['stages']['A1_init']['thresholds']
    sd = torch.load('results/ckpt/a1.pt', map_location='cpu')
    sdc = torch.load('results/ckpt/c.pt', map_location='cpu')
    name_map = {'q_in1': 'q_in1', 'attn.sn_q': 'q', 'attn.sn_k': 'k', 'attn.sn_v': 'v', 'attn.sn_pre': 'pre', 'attn.sn_out': 'out',
                'q_in2': 'q_in2', 'ffn.sn_1': 'g', 'ffn.sn_3': 'u', 'ffn.sn_2': 'y2'}
    rows = {}
    for k, v in sd.items():
        if 'log_thr' not in k:
            continue
        if k == 'q_out.log_thr':
            nm = 'q_out'
        else:
            parts = k.split('.')
            i = parts[1]
            sub = '.'.join(parts[2:]).replace('._thr.log_thr', '').replace('.log_thr', '')
            nm = f'blocks.{i}.{name_map[sub]}'
        t_init = init[nm]; t_a1 = math.exp(v.item()); t_c = math.exp(sdc[k].item())
        rows[nm] = dict(init=r(t_init), a1=r(t_a1), c=r(t_c), ratio_a1_init=r(t_a1 / t_init, 3), ratio_c_a1=r(t_c / t_a1, 3),
                        log_drift_a1=r(math.log(t_a1 / t_init), 3), at_floor_a1=t_a1 <= 1.01e-3, at_floor_c=t_c <= 1.01e-3)
    ratios = [d['ratio_a1_init'] for d in rows.values()]
    out = dict(rows=rows, ratio_min=min(ratios), ratio_max=max(ratios), n_floor_a1=sum(d['at_floor_a1'] for d in rows.values()),
               n_floor_c=sum(d['at_floor_c'] for d in rows.values()), n_exploded_gt10=sum(x > 10 for x in ratios), n_collapsed_lt0p1=sum(x < 0.1 for x in ratios))
    # firing rates of trained a1 vs init
    a1 = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev); a1.load_state_dict(torch.load('results/ckpt/a1.pt', map_location=dev)); a1.eval()
    st = cl.model_stats(a1, tokens[:256], dev)
    out['a1_trained_rates'] = {b: {k: v for k, v in st[b].items() if k.startswith('rate') or k == 'k_sqnorm_mean' or 'absmax' in k} for b in ('block0', 'block1')}
    out['a1_trained_rate_q_out'] = st['rate_q_out']; out['a1_trained_h_final_absmax'] = r(st['h_final_absmax'])
    m0 = ctx.get('a1_init')
    if m0 is not None:
        st0 = cl.model_stats(m0, tokens[:256], dev)
        out['a1_init_rates'] = {b: {k: v for k, v in st0[b].items() if k.startswith('rate') or k == 'k_sqnorm_mean' or 'absmax' in k} for b in ('block0', 'block1')}
        out['a1_init_rate_q_out'] = st0['rate_q_out']
    save('drift', out)


# ----------------------------------------------------------------------------- (7) surrogate
def sec_sg(ctx):
    dev = ctx['dev']
    out = {}
    U = torch.linspace(-3, 3, 601, dtype=torch.float64, requires_grad=True)
    thr = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    y = ternary_fire(U, thr, 2.0)
    gU = torch.autograd.grad(y.sum(), U, retain_graph=True)[0]
    # d y_j / d thr for each j separately
    gT = []
    for j in [0, 100, 200, 250, 300, 350, 400, 500, 600]:
        thr.grad = None
        yj = ternary_fire(U[j:j + 1], thr, 2.0)
        gT.append((r(U[j].item(), 2), r(torch.autograd.grad(yj.sum(), thr)[0].item())))
    samples = {}
    for u in (-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3):
        j = int(round((u + 3) / 6 * 600))
        samples[str(u)] = dict(y=r(y[j].item()), dy_dU=r(gU[j].item()))
    out['thr1_samples'] = samples
    out['dy_dthr_at_U'] = gT
    out['analytic'] = 'dy/dU = sg(U-thr) + sg(U+thr), sg(x) = (w/2)/(1+(pi/2*w*x)^2), w=2 -> peak 1 at U=+-thr, half-max at |U-+thr| = 1/pi = 0.318 (absolute units, not scaled by thr)'
    out['dy_dU_at_0_thr1'] = r(gU[300].item()); out['dy_dU_at_thr'] = r(gU[400].item())
    # coverage per point at A1_init: fraction of pre-activations within half-max window of either threshold, mean gradient
    m, thr_d = ctx.get('a1_init'), ctx.get('thr')
    if m is not None:
        cfg, tokens = ctx['cfg'], ctx['tokens']
        calib = tokens[:cfg.calib_sentences].to(dev)
        inputs, _, mask = split_inputs_targets(calib)
        with torch.no_grad():
            _, r1 = m(inputs, record=True)
        cov = {}
        for i, blk in enumerate(m.blocks):
            for name, mod, fn, ternary in _block_points(blk):
                z = torch.stack([fn(r1[t][i]) for t in range(len(r1))], dim=1)[mask]
                th = thr_d[f'blocks.{i}.{name}']
                def sg(x): return 1.0 / (1 + (math.pi * x) ** 2)
                g = sg(z - th) + (sg(z + th) if ternary else 0)
                cov[f'blocks.{i}.{name}'] = dict(thr=r(th), halfwidth_over_thr=r(0.318 / th, 3), mean_grad=r(g.mean()),
                                                 frac_in_window=r((g > 0.5).float().mean()), frac_grad_lt_0p1=r((g < 0.1).float().mean()))
            hf = stack(r1, len(m.blocks), 'h_final', mask); th = thr_d['q_out']
            g = 1 / (1 + (math.pi * (hf - th)) ** 2) + 1 / (1 + (math.pi * (hf + th)) ** 2)
            cov['q_out'] = dict(thr=r(th), halfwidth_over_thr=r(0.318 / th, 3), mean_grad=r(g.mean()), frac_in_window=r((g > 0.5).float().mean()), frac_grad_lt_0p1=r((g < 0.1).float().mean()))
        out['coverage_a1_init'] = cov
    save('sg', out)


# ----------------------------------------------------------------------------- (8) extras
def per_channel_calibrate(m, tokens, q=0.75):
    """Same forward-order calibration as toy_demo.convert.calibrate_thresholds but one threshold per channel.
    Only swaps the shape of the existing log_thr parameter; snn_spec code untouched (broadcast [B,C]-[C])."""
    inputs, _, mask = split_inputs_targets(tokens)
    out = {}
    def setvec(mod, vec):
        target_holder = mod if hasattr(mod, 'log_thr') else mod._thr
        target_holder.log_thr = nn.Parameter(vec.clamp_min(1e-3).log().to(vec.device))
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            for name, mod, fn, ternary in _block_points(blk):
                _, recs = m(inputs, record=True)
                z = torch.stack([fn(recs[t][i]) for t in range(inputs.shape[1])], dim=1)[mask]
                z = z.abs() if ternary else z
                v = torch.quantile(z.float(), q, dim=0)
                setvec(mod, v); out[f'blocks.{i}.{name}'] = dict(min=r(v.min()), med=r(v.median()), max=r(v.max()), n_floor=int((v <= 1e-3).sum()))
        _, recs = m(inputs, record=True)
        hf = torch.stack([recs[t][len(m.blocks)]['h_final'] for t in range(inputs.shape[1])], dim=1)[mask]
        v = torch.quantile(hf.abs().float(), q, dim=0); setvec(m.q_out, v); out['q_out'] = dict(min=r(v.min()), med=r(v.median()), max=r(v.max()))
    return out


@torch.no_grad()
def sec_extras(ctx):
    a0, cfg, sl, dev, tokens, demo = ctx['a0'], ctx['cfg'], ctx['sl'], ctx['dev'], ctx['tokens'], ctx['demo']
    out = {}
    # eval()/train(): any mode-dependent modules?
    m = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev)
    types = sorted({type(x).__name__ for x in m.modules()})
    out['module_types_a1'] = types
    out['mode_dependent_modules'] = [t for t in types if t in ('Dropout', 'BatchNorm1d', 'BatchNorm2d', 'LayerNorm')]
    # padding: batches produce per-batch max length; PAD ids appear in inputs at masked positions only?
    g = torch.Generator().manual_seed(0)
    b = next(batches(sl, cfg.batch_size, g))
    inp, tgt, mask = split_inputs_targets(b)
    out['batch_shape'] = list(b.shape)
    out['pad_inputs_at_unmasked_positions'] = int(((inp == PAD) & mask).sum())
    out['eos_inputs_at_unmasked_positions'] = int(((inp == EOS) & mask).sum())
    out['frac_masked_positions'] = r(1 - mask.float().mean())
    full = tokens; inp, tgt, mask = split_inputs_targets(full)
    out['full_slice_shape'] = list(full.shape); out['full_frac_masked'] = r(1 - mask.float().mean())
    # A0's PAD embedding row norm vs others (never trained)
    out['emb_row_norm_pad'] = r(a0.emb.weight[PAD].norm()); out['emb_row_norm_median'] = r(a0.emb.weight[3:].norm(dim=-1).median())
    out['emb_row_norm_bos'] = r(a0.emb.weight[0].norm())
    # A0 + q_out only (head quantizer alone), threshold 0.75 quantile of |h_final| of A0 on calib set
    calib = tokens[:cfg.calib_sentences].to(dev)
    inputs, targets, mask = split_inputs_targets(calib)
    _, rec0 = a0(inputs, record=True)
    hf0 = stack(rec0, len(a0.blocks), 'h_final', mask)
    th = torch.quantile(hf0.abs().flatten(), 0.75).item()
    out['a0_eval'] = ev3(a0, sl, demo, cfg, tokens)
    a0.head_input = lambda h: ternary_fire(h, torch.tensor(th, device=dev))
    out['a0_plus_qout_only'] = dict(thr=r(th), **ev3(a0, sl, demo, cfg, tokens))
    # with per-row head scale fit
    L0 = a0.head(hf0); L1 = a0.head(ternary_fire(hf0, torch.tensor(th, device=dev)))
    sv = (L0 * L1).sum(0) / (L1 * L1).sum(0).clamp_min(1e-8)
    W = a0.head.weight.clone(); a0.head.weight.copy_(W * sv[:, None])
    out['a0_plus_qout_only_rowscale'] = ev3(a0, sl, demo, cfg, tokens)
    # full LS head refit on ternary input
    Wls, _ = cl._lstsq(ternary_fire(hf0, torch.tensor(th, device=dev)), L0, ridge=1e-3)
    a0.head.weight.copy_(Wls); out['a0_plus_qout_only_headLS'] = ev3(a0, sl, demo, cfg, tokens)
    a0.head.weight.copy_(W)
    # per-channel quantile for q_out on A0 (is per-channel better?)
    thv = torch.quantile(hf0.abs(), 0.75, dim=0)
    a0.head_input = lambda h: ternary_fire(h, thv)
    out['a0_plus_qout_perchannel'] = dict(thr_min=r(thv.min()), thr_max=r(thv.max()), **ev3(a0, sl, demo, cfg, tokens))
    del a0.head_input
    # per-channel threshold calibration at A1_init (same quantile, one thr per channel)
    m2 = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev); transfer_weights(a0, m2)
    pc = per_channel_calibrate(m2, calib, 0.75)
    out['a1_init_perchannel_thr_summary'] = pc
    out['a1_init_perchannel_eval'] = ev3(m2, sl, demo, cfg, tokens)
    st = cl.model_stats(m2, tokens[:256], dev)
    out['a1_init_perchannel_rates'] = {b: {k: v for k, v in st[b].items() if k.startswith('rate')} for b in ('block0', 'block1')}
    # scalar baseline for reference (fresh)
    m3, thr3 = cl.build_a1(a0, cfg, sl, tokens, dev, 0.75)
    out['a1_init_scalar_eval'] = ev3(m3, sl, demo, cfg, tokens)
    save('extras', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='keys,dists,head,drift,sg,extras')
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env()
    a0 = cl.load_a0(cfg, sl, dev)
    ctx = dict(cfg=cfg, sl=sl, demo=demo, tokens=tokens, dev=dev, a0=a0)
    secs = dict(keys=sec_keys, dists=sec_dists, head=sec_head, drift=sec_drift, sg=sec_sg, extras=sec_extras)
    for s in args.only.split(','):
        t0 = time.time()
        secs[s](ctx)
        print(f'[{s}] {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
