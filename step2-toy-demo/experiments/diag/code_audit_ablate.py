"""code-audit follow-up: where are the 74 accuracy points lost? Hybrid float/spiking model built on A0 weights where each
of the 21 spike points can be switched on individually (leak 0 == stateless, so this is exactly A1_init when all are on).
Thresholds are calibrated the same way as toy_demo.convert.calibrate_thresholds (0.75 quantile; signed for binary q/g,
|z| for ternary), sequentially in forward order over the active points.
Modes: alone (each point on by itself) | cumul (switch on in forward order) | loo (all on except one) | groups
Output: results/conversion-sweep/diag/code-audit/ablate_<mode>.json
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import convert_lab as cl  # noqa: E402
from toy_demo.data import BOS, EOS, split_inputs_targets  # noqa: E402
from toy_demo.model import l2norm  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402
from snn_spec.gdn import SpikingGDN  # noqa: E402

OUT = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/code-audit'
PTS = ['xin1', 'q', 'k', 'v', 'pre', 'out', 'xin2', 'g', 'u', 'y2']
BINARY = {'q', 'g'}


def tern(z, t):
    return (z >= t).float() - (z <= -t).float()


def binr(z, t):
    return (z >= t).float()


class Hybrid:
    """A0 weights; per-point flags; thresholds dict name -> float."""

    def __init__(self, a0, cfg):
        self.a0, self.cfg = a0, cfg
        self.on = set()
        self.thr = {}
        self.device = a0.device
        self.rec = None

    def init_state(self, B, device):
        return self.a0.init_state(B, device)

    def _pt(self, name, z, ternary):
        if name in self.on:
            return binr(z, self.thr[name]) if not ternary else tern(z, self.thr[name])
        return None

    def step(self, tok, st, record=False):
        cfg = self.cfg
        h = self.a0.emb(tok)
        B = h.shape[0]
        recs = []
        for i, blk in enumerate(self.a0.blocks):
            a, f = blk.attn, blk.ffn
            rec = {}
            rec['xin1'] = h
            x1 = tern(h, self.thr[f'b{i}.xin1']) if f'b{i}.xin1' in self.on else h
            alpha, beta = a.gates(x1)
            zq, zk, zv = a.Wq(x1), a.Wk(x1), a.Wv(x1)
            rec['q'], rec['k'], rec['v'] = zq, zk, zv
            q = binr(zq, self.thr[f'b{i}.q']).view(B, a.H, a.dk) if f'b{i}.q' in self.on else l2norm(zq.view(B, a.H, a.dk))
            k = tern(zk, self.thr[f'b{i}.k']).view(B, a.H, a.dk) / math.sqrt(a.dk) if f'b{i}.k' in self.on else l2norm(zk.view(B, a.H, a.dk))
            v = tern(zv, self.thr[f'b{i}.v']).view(B, a.H, a.dv) if f'b{i}.v' in self.on else zv.view(B, a.H, a.dv)
            o, st[i]['attn']['S'] = SpikingGDN.core(k, v, q, alpha, beta, st[i]['attn']['S'])
            o = o.reshape(B, a.H * a.dv)
            rec['pre'] = o
            o_in = tern(o, self.thr[f'b{i}.pre']) if f'b{i}.pre' in self.on else o
            y = a.Wo(o_in)
            rec['out'] = y
            y1 = tern(y, self.thr[f'b{i}.out']) if f'b{i}.out' in self.on else y
            h_mid = h + y1
            rec['xin2'] = h_mid
            x2 = tern(h_mid, self.thr[f'b{i}.xin2']) if f'b{i}.xin2' in self.on else h_mid
            z1, z3 = f.W1(x2), f.W3(x2)
            rec['g'], rec['u'] = z1, z3
            gg = binr(z1, self.thr[f'b{i}.g']) if f'b{i}.g' in self.on else F.silu(z1)
            uu = tern(z3, self.thr[f'b{i}.u']) if f'b{i}.u' in self.on else z3
            y2 = f.W2(gg * uu)
            rec['y2'] = y2
            y2 = tern(y2, self.thr[f'b{i}.y2']) if f'b{i}.y2' in self.on else y2
            h = h_mid + y2
            recs.append(rec)
        recs.append(dict(head=h))
        hi = tern(h, self.thr['head']) if 'head' in self.on else h
        logits = self.a0.head(hi)
        return logits, st, (recs if record else None)

    def __call__(self, tokens, record=False):
        B, L = tokens.shape
        st = self.init_state(B, tokens.device)
        outs, recs = [], []
        for t in range(L):
            lg, st, rec = self.step(tokens[:, t], st, record)
            outs.append(lg); recs.append(rec)
        lg = torch.stack(outs, 1)
        return (lg, recs) if record else lg

    def eval(self):
        return self

    def train(self):
        return self

    def generate(self, prefix, max_new):
        from toy_demo.model import greedy_generate
        return greedy_generate(self, prefix, max_new)

    @torch.no_grad()
    def calibrate(self, active, calib, fq=0.75):
        """activate the points in `active` (forward order) one at a time, calibrating each on the current model."""
        self.on, self.thr = set(), {}
        inputs, _, mask = split_inputs_targets(calib)
        order = [f'b{i}.{p}' for i in range(len(self.a0.blocks)) for p in PTS] + ['head']
        for name in order:
            if name not in active:
                continue
            _, recs = self(inputs, record=True)
            if name == 'head':
                z = torch.stack([recs[t][-1]['head'] for t in range(len(recs))], 1)[mask]
                ternary = True
            else:
                i, p = int(name[1]), name.split('.')[1]
                z = torch.stack([recs[t][i][p] for t in range(len(recs))], 1)[mask]
                ternary = p not in BINARY
            z = z.abs() if ternary else z
            self.thr[name] = max(torch.quantile(z.flatten().float(), fq).item(), 1e-3)
            self.on.add(name)
        return dict(self.thr)


def ev3(m, sl, demo, cfg, tokens):
    e = evaluate(m, sl, demo, cfg, tokens)
    return dict(token_acc=round(e['token_acc'], 4), ppl=round(e['ppl'], 3), n_exact=e['n_exact'])


@torch.no_grad()
def main():
    mode = sys.argv[1]
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env()
    a0 = cl.load_a0(cfg, sl, dev)
    calib = tokens[:cfg.calib_sentences].to(dev)
    hy = Hybrid(a0, cfg)
    order = [f'b{i}.{p}' for i in range(cfg.n_blocks) for p in PTS] + ['head']
    out = {}
    t0 = time.time()

    def run(label, active, fq=0.75):
        thr = hy.calibrate(set(active), calib, fq)
        e = ev3(hy, sl, demo, cfg, tokens)
        out[label] = dict(eval=e, n_on=len(active), thr={k: round(v, 4) for k, v in thr.items()})
        print(label, e, f'{time.time() - t0:.0f}s', flush=True)

    if mode == 'alone':
        run('none', [])
        for name in order:
            run(name, [name])
    elif mode == 'cumul':
        for j in range(1, len(order) + 1):
            run('+' + order[j - 1], order[:j])
    elif mode == 'loo':
        run('all', order)
        for name in order:
            run('-' + name, [x for x in order if x != name])
    elif mode == 'groups':
        run('none', [])
        run('all', order)
        run('xin_only', [x for x in order if 'xin' in x])
        run('attn_only', [x for x in order if x.split('.')[-1] in ('q', 'k', 'v', 'pre', 'out')])
        run('qkv_only', [x for x in order if x.split('.')[-1] in ('q', 'k', 'v')])
        run('qk_only', [x for x in order if x.split('.')[-1] in ('q', 'k')])
        run('q_only', [x for x in order if x.split('.')[-1] == 'q'])
        run('k_only', [x for x in order if x.split('.')[-1] == 'k'])
        run('ffn_only', [x for x in order if x.split('.')[-1] in ('g', 'u', 'y2')])
        run('gu_only', [x for x in order if x.split('.')[-1] in ('g', 'u')])
        run('residual_out_only', [x for x in order if x.split('.')[-1] in ('out', 'y2')])
        run('block0_all', [x for x in order if x.startswith('b0')])
        run('block1_all', [x for x in order if x.startswith('b1')])
        run('all_but_head', [x for x in order if x != 'head'])
        run('all_but_qk', [x for x in order if x.split('.')[-1] not in ('q', 'k')])
    elif mode == 'density':
        for fq in (0.75, 0.5, 0.25, 0.0):
            run(f'xin_only@fq{fq}', [x for x in order if 'xin' in x], fq)
            run(f'k_only@fq{fq}', [x for x in order if x.split('.')[-1] == 'k'], fq)
            run(f'head_only@fq{fq}', ['head'], fq)
            run(f'all@fq{fq}', order, fq)
    with open(os.path.join(OUT, f'ablate_{mode}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('saved', mode)


if __name__ == '__main__':
    main()
