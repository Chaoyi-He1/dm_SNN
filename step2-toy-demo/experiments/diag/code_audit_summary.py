"""Compact printout of results/conversion-sweep/diag/code-audit/*.json."""
import json
import sys

D = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/code-audit/'
which = sys.argv[1:] or ['keys', 'dists', 'drift', 'sg', 'extras']

if 'keys' in which:
    d = json.load(open(D + 'keys.json'))
    print('== keys'); print({k: v for k, v in d.items() if k != 'thr_keys'})
if 'dists' in which:
    d = json.load(open(D + 'dists.json'))
    print('== dists a1_init_eval', d['a1_init_eval'])
    P = d['points']
    print('name | tern | thr | frac_neg | sq75 | aq75 | rate | rate+ | rate- | chan_rate min/med/max | chanstd min/max | thr_if_abs | rate_if_abs')
    for k, v in P.items():
        if 'head' in k:
            print(k, v); continue
        print(k, v.get('ternary'), v['thr'], v.get('frac_neg'), v.get('signed_q', [None] * 5)[3], v['abs_q'][3], v['rate'], v.get('rate_pos'), v.get('rate_neg'),
              v['chan_rate_min'], v.get('chan_rate_med'), v['chan_rate_max'], v.get('chan_std_min'), v.get('chan_std_max'), v.get('thr_if_abs_quantile'), v.get('rate_if_abs_quantile'))
    print('-- scale'); print(json.dumps(d['scale']))
if 'drift' in which:
    d = json.load(open(D + 'drift.json'))
    print('== drift'); print({k: v for k, v in d.items() if k not in ('rows', 'a1_trained_rates', 'a1_init_rates')})
    for k, v in d['rows'].items():
        print(k, v)
    print('a1_trained_rates', json.dumps(d['a1_trained_rates'])); print('a1_trained_rate_q_out', d['a1_trained_rate_q_out'])
    print('a1_init_rates', json.dumps(d.get('a1_init_rates'))); print('a1_init_rate_q_out', d.get('a1_init_rate_q_out'))
if 'sg' in which:
    d = json.load(open(D + 'sg.json'))
    print('== sg'); print({k: v for k, v in d.items() if k != 'coverage_a1_init'})
    for k, v in d.get('coverage_a1_init', {}).items():
        print(k, v)
if 'extras' in which:
    d = json.load(open(D + 'extras.json'))
    print('== extras')
    for k, v in d.items():
        if k == 'a1_init_perchannel_thr_summary':
            print(k); [print('   ', kk, vv) for kk, vv in v.items()]
        else:
            print(k, v)
