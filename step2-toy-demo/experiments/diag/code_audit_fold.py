"""code-audit follow-up: does the missing lambda folding (spec scale-folding table) explain A1_init loss?

Variants (all at leak 0, fq 0.75, calibrated sequentially like toy_demo.convert.calibrate_thresholds):
  base        plain transfer + calibrate (reference, = A1_init)
  inj         per-block residual injection scale Q_0 (lambda_y, lambda_y'), LS-fitted A0 y onto A1 ternary y
  fold_lin    per-channel lambda_o folded into W_O columns, lambda_g*lambda_u into W_2 rows (spec rows 5, 7, 8)
  inj+fold    both
  emb_global  no model change: scale emb by 1/Q (global) — the spec-neutral approximation of a global Q_0
Also: head-only q_out quantile sweep on A0 (per-row head scale fit) to see how much information the 25%-dense
ternary head code discards.
Output: results/conversion-sweep/diag/code-audit/fold.json
"""
import json
import math
import os
import sys

sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import convert_lab as cl  # noqa: E402
from toy_demo.convert import _block_points, _set_thr, transfer_weights  # noqa: E402
from toy_demo.data import split_inputs_targets  # noqa: E402
from toy_demo.model import SpikingLM  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402
from snn_spec.block import SpikingBlock  # noqa: E402
from snn_spec.neurons import ternary_fire  # noqa: E402

OUT = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/code-audit'


def r(x, n=4):
    return round(float(x), n)


def ev3(m, sl, demo, cfg, tokens):
    e = evaluate(m, sl, demo, cfg, tokens)
    return dict(token_acc=r(e['token_acc']), ppl=r(e['ppl'], 3), n_exact=e['n_exact'])


class ScaledInjBlock(SpikingBlock):
    """SpikingBlock with the spec's injection charges Q_0 (attn) and Q_0' (ffn) as explicit scalars (default 1 = current code)."""
    Q1 = 1.0
    Q2 = 1.0

    def step(self, h, st, ext_gates=None, record=False):
        x1 = self.q_in1(h)
        y1, st['attn'], r1 = self.attn.step(x1, st['attn'], ext_gates=ext_gates, record=record)
        h_mid = h + self.Q1 * y1
        x2 = self.q_in2(h_mid)
        y2, st['ffn'], r2 = self.ffn.step(x2, st['ffn'], record=record)
        h_out = h_mid + self.Q2 * y2
        rec = None
        if record:
            rec = dict(h_in=h, x_attn=x1, alpha=r1['alpha'], beta=r1['beta'], q=r1['q'], k=r1['k'], v=r1['v'], S=r1['S'],
                       o=r1['o'], o_spk=r1['o_spk'], y_attn=y1, h_mid=h_mid, x_ffn=x2, g=r2['g'], u=r2['u'], p=r2['p'],
                       y_ffn=y2, h_out=h_out)
        return h_out, st, rec


def gather(recs, i, key, mask):
    return torch.stack([recs[t][i][key] for t in range(len(recs))], dim=1)[mask]


def ls_scalar(y0, y1):
    return ((y0 * y1).sum() / (y1 * y1).sum().clamp_min(1e-8)).item()


def ls_perchannel(y0, y1):
    return (y0 * y1).sum(0) / (y1 * y1).sum(0).clamp_min(1e-8)


@torch.no_grad()
def convert(a0, cfg, sl, dev, calib, fq, inj=False, fold_lin=False, emb_scale=1.0):
    m = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev)
    transfer_weights(a0, m)
    for blk in m.blocks:
        blk.__class__ = ScaledInjBlock
        blk.Q1, blk.Q2 = 1.0, 1.0
    if emb_scale != 1.0:
        m.emb.weight.mul_(emb_scale)
    inputs, _, mask = split_inputs_targets(calib)
    _, ref = a0(inputs, record=True)
    info = {}

    def run():
        _, recs = m(inputs, record=True)
        return recs

    def calib_point(i, blk, idx):
        name, mod, fn, ternary = _block_points(blk)[idx]
        recs = run()
        z = torch.stack([fn(recs[t][i]) for t in range(len(recs))], dim=1)[mask]
        z = z.abs() if ternary else z
        thr = max(torch.quantile(z.flatten().float(), fq).item(), 1e-3)
        _set_thr(mod, thr)
        info[f'thr.blocks.{i}.{name}'] = r(thr)
        return recs

    for i, blk in enumerate(m.blocks):
        for idx in range(5):                       # q_in1, q, k, v, pre
            recs = calib_point(i, blk, idx)
        if fold_lin:                                # lambda_o (per channel) into W_O columns
            recs = run()
            lam = ls_perchannel(gather(ref, i, 'o', mask), gather(recs, i, 'o_spk', mask))
            blk.attn.Wo.weight.mul_(lam[None, :])
            info[f'lambda_o.block{i}'] = dict(min=r(lam.min()), med=r(lam.median()), max=r(lam.max()))
        recs = calib_point(i, blk, 5)               # out
        if inj:
            recs = run()
            q1 = ls_scalar(gather(ref, i, 'zo', mask), gather(recs, i, 'y_attn', mask))
            blk.Q1 = q1; info[f'Q1.block{i}'] = r(q1)
        for idx in (6, 7, 8):                       # q_in2, g, u
            recs = calib_point(i, blk, idx)
        if fold_lin:                                # lambda_g*lambda_u (per channel) into W_2 rows(=input columns)
            recs = run()
            z1, z3 = gather(ref, i, 'z1', mask), gather(ref, i, 'z3', mask)
            lam = ls_perchannel(F.silu(z1) * z3, gather(recs, i, 'p', mask))
            blk.ffn.W2.weight.mul_(lam[None, :])
            info[f'lambda_p.block{i}'] = dict(min=r(lam.min()), med=r(lam.median()), max=r(lam.max()))
        recs = calib_point(i, blk, 9)               # y2
        if inj:
            recs = run()
            q2 = ls_scalar(gather(ref, i, 'z2', mask), gather(recs, i, 'y_ffn', mask))
            blk.Q2 = q2; info[f'Q2.block{i}'] = r(q2)
    recs = run()
    L = len(m.blocks)
    hf = gather(recs, L, 'h_final', mask)
    thr = max(torch.quantile(hf.abs().flatten().float(), fq).item(), 1e-3)
    _set_thr(m.q_out, thr); info['thr.q_out'] = r(thr)
    hf0 = gather(ref, L, 'h_final', mask)
    info['h_final_rms_a1'] = r(hf.pow(2).mean().sqrt()); info['h_final_rms_a0'] = r(hf0.pow(2).mean().sqrt())
    info['cos_hfinal'] = r(F.cosine_similarity(hf0, hf, dim=-1).mean())
    return m, info


@torch.no_grad()
def head_row_scale(m, a0, calib):
    inputs, _, mask = split_inputs_targets(calib)
    L0 = a0(inputs)[mask]; L1 = m(inputs)[mask]
    sv = (L0 * L1).sum(0) / (L1 * L1).sum(0).clamp_min(1e-8)
    m.head.weight.mul_(sv[:, None])
    return dict(s_med=r(sv.median()))


@torch.no_grad()
def main():
    torch.manual_seed(0)
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env()
    a0 = cl.load_a0(cfg, sl, dev)
    calib = tokens[:cfg.calib_sentences].to(dev)
    out = {}
    variants = dict(base=dict(), inj=dict(inj=True), fold_lin=dict(fold_lin=True), inj_fold=dict(inj=True, fold_lin=True),
                    emb_x0p5=dict(emb_scale=0.5), emb_x0p33=dict(emb_scale=1 / 3), emb_x0p25=dict(emb_scale=0.25), emb_x2=dict(emb_scale=2.0))
    if 'sweep-only' in sys.argv:
        variants = {}
    for name, kw in variants.items():
        m, info = convert(a0, cfg, sl, dev, calib, 0.75, **kw)
        e = ev3(m, sl, demo, cfg, tokens)
        hs = head_row_scale(m, a0, calib)
        e2 = ev3(m, sl, demo, cfg, tokens)
        out[name] = dict(eval=e, eval_head_rowscale=e2, head_scale=hs, **info)
        print(name, e, e2, {k: v for k, v in info.items() if not k.startswith('thr.')}, flush=True)
    # head-only quantile sweep on A0
    inputs, _, mask = split_inputs_targets(calib)
    _, rec0 = a0(inputs, record=True)
    hf0 = gather(rec0, len(a0.blocks), 'h_final', mask)
    W = a0.head.weight.clone()
    sweep = {}
    for q in (0.9, 0.75, 0.5, 0.25, 0.1, 0.0):
        th = torch.quantile(hf0.abs().flatten(), q).item() if q > 0 else 1e-3
        tt = torch.tensor(th, device=dev)
        a0.head_input = lambda h, tt=tt: ternary_fire(h, tt)
        e = ev3(a0, sl, demo, cfg, tokens)
        L0 = W @ hf0.T; L1 = a0.head(ternary_fire(hf0, tt))
        sv = (L0.T * L1).sum(0) / (L1 * L1).sum(0).clamp_min(1e-8)
        a0.head.weight.copy_(W * sv[:, None]); e2 = ev3(a0, sl, demo, cfg, tokens); a0.head.weight.copy_(W)
        sweep[f'q{q}'] = dict(thr=r(th), rate=r((ternary_fire(hf0, tt) != 0).float().mean()), eval=e, eval_rowscale=e2)
        print('head sweep', q, sweep[f'q{q}'], flush=True)
    del a0.head_input
    out['a0_head_qout_sweep'] = sweep
    with open(os.path.join(OUT, 'fold.json' if 'sweep-only' not in sys.argv else 'fold_sweep.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', os.path.join(OUT, 'fold.json'))


if __name__ == '__main__':
    main()
