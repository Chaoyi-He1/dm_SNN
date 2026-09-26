"""Render markdown tables from point-sensitivity measurements.jsonl (no GPU needed)."""
import json, sys
MEAS = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/point-sensitivity/measurements.jsonl'
rows = [json.loads(l) for l in open(MEAS)]
done = {r['key']: r for r in rows}
b = done['baseline']
ba, bp = b['token_acc'], b['ppl']
print(f"baseline: acc {ba:.4f} ppl {bp:.2f} exact {b['n_exact']}\n")
names = []
for r in rows:
    if r.get('point') and r['point'] not in names:
        names.append(r['point'])
print('| point | kind | thr0 | x0.5 acc (d) / ppl | x2.0 acc (d) / ppl | =1e-3 acc (d) / ppl | max abs d_acc |')
print('|---|---|---|---|---|---|---|')
single = []
for n in names:
    cells, ds = [], []
    for tag in ('x0.5', 'x2.0', '=1e-3'):
        k = f'{n} {tag}'
        if k in done:
            d = done[k]; da = d['token_acc'] - ba; ds.append(abs(da))
            cells.append(f"{d['token_acc']:.4f} ({da:+.4f}) / {d['ppl']:.1f}")
        else:
            cells.append('n/a')
    mx = max(ds) if ds else float('nan')
    single.append((n, mx))
    print(f"| {n} | {done[f'{n} x0.5']['kind']} | {done[f'{n} x0.5']['thr0']:.4f} | " + ' | '.join(cells) + f' | {mx:.4f} |')
print('\nranked by max |d_acc|:')
for n, mx in sorted(single, key=lambda t: -t[1]):
    print(f'  {n}: {mx:.4f}')
print('\n| joint group | x0.5 acc (d) / ppl | x2.0 acc (d) / ppl |')
print('|---|---|---|')
groups = []
for r in rows:
    if r.get('kind') == 'joint' and r['group'] not in groups:
        groups.append(r['group'])
for g in groups:
    cells = []
    for f in ('0.5', '2.0'):
        k = f'{g} x{f}'
        if k in done:
            d = done[k]; cells.append(f"{d['token_acc']:.4f} ({d['token_acc']-ba:+.4f}) / {d['ppl']:.1f}")
        else:
            cells.append('n/a')
    print(f'| {g} | ' + ' | '.join(cells) + ' |')
