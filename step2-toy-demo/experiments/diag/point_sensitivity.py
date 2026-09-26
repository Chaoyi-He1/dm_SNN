"""Point sensitivity at A1_init (fq 0.75, plain build): perturb ONE threshold module at a time
(x0.5, x2.0, set to 1e-3), evaluate token_acc/ppl, restore; then joint tests (Q_in group, LIF group,
per-block). Results appended to results/conversion-sweep/diag/point-sensitivity/measurements.jsonl
(resumable) and summarised in summary.json.
Run:  CUDA_VISIBLE_DEVICES=1 <venv python> experiments/diag/point_sensitivity.py
"""
import sys, os, json, math, time
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch
import convert_lab as cl                      # chdirs to step2-toy-demo
from toy_demo.convert import _block_points, _set_thr
from toy_demo.train import evaluate

OUT = 'results/conversion-sweep/diag/point-sensitivity'
os.makedirs(OUT, exist_ok=True)
MEAS = os.path.join(OUT, 'measurements.jsonl')
FQ = 0.75

torch.set_num_threads(1)
dev = cl.get_device()
cfg, sl, demo, tokens = cl.load_env(fire_quantile=FQ)
a0 = cl.load_a0(cfg, sl, dev)
t0 = time.time()
m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, FQ)
print(f'built A1 in {time.time()-t0:.1f}s', flush=True)
base_sd = {k: v.detach().clone() for k, v in m.state_dict().items()}

# --- named modules in cl.all_threshold_modules order ---------------------------------
names, mods, kinds = [], [], []
for i, blk in enumerate(m.blocks):
    for name, mod, _, ternary in _block_points(blk):
        names.append(f'b{i}.{name}'); mods.append(mod)
        kinds.append('Qin' if name.startswith('q_in') else ('LIF-ternary' if ternary else 'LIF-binary'))
names.append('q_out'); mods.append(m.q_out); kinds.append('Qin')
assert [id(x) for x in mods] == [id(x) for x in cl.all_threshold_modules(m)], 'order mismatch'
assert len(mods) == 21

def log_thr_param(mod):
    return mod.log_thr if hasattr(mod, 'log_thr') else mod._thr.log_thr

base_log = [log_thr_param(mod).detach().clone() for mod in mods]

@torch.no_grad()
def restore():
    for mod, v in zip(mods, base_log):
        log_thr_param(mod).copy_(v)

def thr_of(mod):
    return float(log_thr_param(mod).exp().item())

done = {}
if os.path.exists(MEAS):
    for line in open(MEAS):
        r = json.loads(line); done[r['key']] = r

def measure(key, apply_fn, **meta):
    if key in done:
        print(f'skip {key}: acc {done[key]["token_acc"]:.4f}', flush=True); return done[key]
    restore()
    with torch.no_grad():
        apply_fn()
    t = time.time()
    ev = evaluate(m, sl, demo, cfg, tokens)
    rec = dict(key=key, token_acc=ev['token_acc'], ppl=ev['ppl'], n_exact=ev['n_exact'],
               seconds=round(time.time() - t, 1), **meta)
    restore()
    with open(MEAS, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    done[key] = rec
    print(f'{key}: acc {rec["token_acc"]:.4f} ppl {rec["ppl"]:.2f} exact {rec["n_exact"]} ({rec["seconds"]}s)', flush=True)
    return rec

# --- baseline ------------------------------------------------------------------------
base = measure('baseline', lambda: None, kind='baseline')
# sanity: thresholds as built
print('thresholds:', json.dumps({n: round(thr_of(mod), 4) for n, mod in zip(names, mods)}), flush=True)

# --- single-point perturbations -----------------------------------------------------
for n, mod, kind in zip(names, mods, kinds):
    for tag, fn in (('x0.5', lambda mod=mod: cl.scale_thresholds(m, 0.5, only=[mod])),
                    ('x2.0', lambda mod=mod: cl.scale_thresholds(m, 2.0, only=[mod])),
                    ('=1e-3', lambda mod=mod: _set_thr(mod, 1e-3))):
        measure(f'{n} {tag}', fn, kind=kind, point=n, perturb=tag, thr0=thr_of(mod))

# --- joint tests ---------------------------------------------------------------------
qin_mods = [mod for mod, k in zip(mods, kinds) if k == 'Qin']
lif_mods = [mod for mod, k in zip(mods, kinds) if k != 'Qin']
assert len(qin_mods) == 5 and len(lif_mods) == 16
groups = {
    'Qin(all 5)': qin_mods,
    'LIF(all 16)': lif_mods,
    'block0 LIF(8)': [mod for n, mod, k in zip(names, mods, kinds) if n.startswith('b0.') and k != 'Qin'],
    'block1 LIF(8)': [mod for n, mod, k in zip(names, mods, kinds) if n.startswith('b1.') and k != 'Qin'],
    'block0 Qin(2)': [mod for n, mod, k in zip(names, mods, kinds) if n.startswith('b0.') and k == 'Qin'],
    'block1 Qin(2)': [mod for n, mod, k in zip(names, mods, kinds) if n.startswith('b1.') and k == 'Qin'],
    'block0 all(10)': [mod for n, mod in zip(names, mods) if n.startswith('b0.')],
    'block1 all(10)': [mod for n, mod in zip(names, mods) if n.startswith('b1.')],
    'all 21': mods,
}
for gname, gm in groups.items():
    for factor in (0.5, 2.0):
        measure(f'{gname} x{factor}', lambda gm=gm, factor=factor: cl.scale_thresholds(m, factor, only=gm),
                kind='joint', group=gname, perturb=f'x{factor}', n_mods=len(gm))

# --- baseline firing rates for context (512 sentences) --------------------------------
restore()
stats = cl.model_stats(m, tokens[:512], dev)

# --- summary --------------------------------------------------------------------------
b_acc, b_ppl = base['token_acc'], base['ppl']
rows = []
for n in names:
    r = {'point': n, 'thr0': done[f'{n} x0.5']['thr0']}
    for tag in ('x0.5', 'x2.0', '=1e-3'):
        d = done[f'{n} {tag}']
        r[tag] = dict(acc=d['token_acc'], ppl=d['ppl'], d_acc=d['token_acc'] - b_acc)
    r['max_abs_dacc'] = max(abs(r[t]['d_acc']) for t in ('x0.5', 'x2.0', '=1e-3'))
    r['best_dacc'] = max(r[t]['d_acc'] for t in ('x0.5', 'x2.0', '=1e-3'))
    rows.append(r)
rank = sorted(rows, key=lambda r: -r['max_abs_dacc'])
summary = dict(baseline=dict(token_acc=b_acc, ppl=b_ppl, n_exact=base['n_exact']),
               thresholds={n: thr_of(mod) for n, mod in zip(names, mods)},
               single=rows, ranked_by_max_abs_dacc=[(r['point'], round(r['max_abs_dacc'], 4)) for r in rank],
               ranked_by_best_gain=[(r['point'], round(r['best_dacc'], 4)) for r in sorted(rows, key=lambda r: -r['best_dacc'])],
               joint={k: v for k, v in done.items() if v.get('kind') == 'joint'},
               a1_init_rates=stats, total_seconds=round(time.time() - t0, 1))
json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
print('RANK (max |d_acc|):', summary['ranked_by_max_abs_dacc'], flush=True)
print('RANK (best gain):', summary['ranked_by_best_gain'], flush=True)
print('DONE', round(time.time() - t0, 1), 's', flush=True)
