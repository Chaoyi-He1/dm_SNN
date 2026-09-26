"""code-audit follow-up (7): surrogate half-width is absolute (1/pi in U units) while thresholds span 0.12..1.6.
Compare 200-step A1 fine-tunes (lr 3e-3 cosine, seed 0, fq 0.75) with
  default   sg_width = 2 for every point (current code)
  scaled    sg_width = 2 / thr_init per module (half-width and peak expressed in threshold units; identical to default at thr = 1)
Usage: ... code_audit_sgw.py default|scaled  -> results/conversion-sweep/diag/code-audit/sgw_<mode>.json
"""
import json
import os
import sys
import time

sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch  # noqa: E402

import convert_lab as cl  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

OUT = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/code-audit'


def main():
    mode = sys.argv[1]
    torch.set_num_threads(1)
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env(fire_quantile=0.75)
    a0 = cl.load_a0(cfg, sl, dev)
    m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, 0.75, seed=0)
    widths = {}
    if mode == 'scaled':
        mods = cl.all_threshold_modules(m)
        for mod in mods:
            t = (mod.log_thr if hasattr(mod, 'log_thr') else mod._thr.log_thr).exp().item()
            mod.sg_width = 2.0 / t
            widths[str(id(mod))] = round(mod.sg_width, 3)
    init = evaluate(m, sl, demo, cfg, tokens)
    t0 = time.time()
    res = cl.train(m, sl, demo, cfg, tokens, lr=3e-3, steps=200, seed=0, schedule='cosine', eval_every=50, stop=False,
                   hist_path=os.path.join(OUT, f'sgw_{mode}.history.jsonl'))
    res['seconds'] = round(time.time() - t0, 1)
    res['init'] = {k: init[k] for k in ('token_acc', 'ppl', 'n_exact')}
    res['final'] = {k: res['final'][k] for k in ('token_acc', 'ppl', 'n_exact')}
    res['mode'] = mode
    res['sg_widths'] = sorted(widths.values())
    with open(os.path.join(OUT, f'sgw_{mode}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'history'}))


if __name__ == '__main__':
    main()
