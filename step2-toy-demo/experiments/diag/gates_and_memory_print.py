"""Print the measurements from gates_and_memory.py (part1.json / part2.json)."""
import json
import sys
from pathlib import Path

OUT = Path("/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/results/conversion-sweep/diag/gates-and-memory")


def part1():
    d = json.loads((OUT / "part1.json").read_text(encoding="utf-8"))
    print("== (a) model_stats 512 ==")
    for name in ("a0_stats_512", "a1_init_stats_512"):
        s = d[name]
        for b in ("block0", "block1"):
            st = s[b]
            print(name, b, "alpha_mean %.4f q %s beta_mean %.4f q %s | h_in %.3f h_mid %.3f h_out %.3f" % (
                st['alpha_mean'], st['alpha_q'], st['beta_mean'], st['beta_q'], st['h_in_absmax'], st['h_mid_absmax'], st['h_out_absmax']))
            if 'rate_k' in st:
                print("   rates", {k: v for k, v in st.items() if k.startswith('rate_')}, "k_sqnorm", st['k_sqnorm_mean'])
        print("  h_final_absmax", s['h_final_absmax'], "rate_q_out", s.get('rate_q_out'))
    print("== gate_compare_full ==")
    for b, g in d['gate_compare_full'].items():
        print(b)
        for k in ('alpha_a0', 'alpha_a1', 'beta_a0', 'beta_a1'):
            print("  ", k, g[k])
        print("   alpha_mae", g['alpha_mae'], "alpha_corr", g['alpha_corr'], "beta_mae", g['beta_mae'], "beta_corr", g['beta_corr'])
        print("   alpha head mean a0", g['alpha_a0_head_mean'], "a1", g['alpha_a1_head_mean'])
        print("   alpha head std  a0", g['alpha_a0_head_std'], "a1", g['alpha_a1_head_std'])
        print("   beta  head mean a0", g['beta_a0_head_mean'], "a1", g['beta_a1_head_mean'])
        print("   beta  head std  a0", g['beta_a0_head_std'], "a1", g['beta_a1_head_std'])
    print("== gate_inputs_512 ==")
    for b, g in d['gate_inputs_512'].items():
        print(b, {k: v for k, v in g.items() if not isinstance(v, dict)})
        for k in ('Wa_logit_a0', 'Wa_logit_a1', 'Wb_logit_a0', 'Wb_logit_a1'):
            print("  ", k, g[k])
    print("== (b) write_strength ==")
    for b, w in d['write_strength'].items():
        print(b, w)
    print("== (c) accumulator ==")
    print(json.dumps(d['accumulator']))
    print("== (d) ==")
    for k in ('a0_metrics_full', 'a1_init_evaluate', 'a1_init_manual_loop_owngates', 'a1_init_a0gates', 'a1_init_a0alpha_ownbeta',
              'a1_init_ownalpha_a0beta', 'a1_init_a0gates_oracle_check', 'a1_init_a0gates_recalib'):
        v = dict(d[k]); v.pop('thresholds', None); hm = v.pop('hmax', None)
        print(k, v)
        if hm:
            print("   hmax", hm)
    print("A0 gate means", d['a0_gate_means_full'])
    print("A1_init thr   ", d['thresholds'])
    print("recalib thr   ", d['a1_init_a0gates_recalib']['thresholds'])
    print("a0gates completions:")
    for r in d['a1_init_a0gates_completions']:
        print("  ", r['exact'], "|", r['prefix'], "->", r['output'][:80])
    print("seconds", d['seconds'])


def part2():
    d = json.loads((OUT / "part2.json").read_text(encoding="utf-8"))
    for k in ('a1_gate_refit', 'a1_init_with_refit_gates_only', 'a1_init_constgates', 'a1_init_constgates_recalib', 'a0_constgates', 'a0_evaluate'):
        v = dict(d[k]); v.pop('thresholds', None); v.pop('completions', None)
        print(k, v)
    print("== gate_refit gate_compare_full ==")
    for b, g in d['a1_gate_refit_gate_compare_full'].items():
        print(b)
        for k in ('alpha_a0', 'alpha_a1', 'beta_a0', 'beta_a1'):
            print("  ", k, g[k])
        print("   alpha_mae", g['alpha_mae'], "alpha_corr", g['alpha_corr'], "beta_mae", g['beta_mae'], "beta_corr", g['beta_corr'])
        print("   alpha head mean a0", g['alpha_a0_head_mean'], "a1", g['alpha_a1_head_mean'])
        print("   beta  head mean a0", g['beta_a0_head_mean'], "a1", g['beta_a1_head_mean'])
    print("== gate_refit stats 512 ==")
    for b in ("block0", "block1"):
        st = d['a1_gate_refit_stats_512'][b]
        print(b, "alpha_mean %.4f beta_mean %.4f" % (st['alpha_mean'], st['beta_mean']), {k: v for k, v in st.items() if k.startswith('rate_')}, "k_sqnorm", st['k_sqnorm_mean'])
    print("gate_refit thr", d['a1_gate_refit']['thresholds'])
    print("constgates_recalib thr", d['a1_init_constgates_recalib']['thresholds'])
    print("seconds", d['seconds'])


if __name__ == "__main__":
    {"part1": part1, "part2": part2}[sys.argv[1]]()
