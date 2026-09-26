"""fq-scan: A1_init quality (token_acc, ppl, n_exact) and firing rates versus the threshold-calibration quantile fq.
Usage: CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/diag/fq-scan.py --fqs 0.3,0.4 [--refit] --out results/conversion-sweep/diag/fq-scan/scan_plain.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo')
sys.path.insert(0, '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments')
import torch  # noqa: E402
import convert_lab as cl  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

RATE_KEYS = ("x_attn", "q", "k", "v", "o_spk", "y_attn", "x_ffn", "g", "u", "p", "y_ffn")


def summarize(stats):
    """Aggregate the per-block rate_* fields: mean over the 11 rate keys x blocks (+ q_out)."""
    rates = []
    per = {}
    blocks = [b for b in stats if b.startswith("block")]
    for b in blocks:
        st = stats[b]
        for k in RATE_KEYS:
            per[f"{b}.{k}"] = st[f"rate_{k}"]
            rates.append(st[f"rate_{k}"])
    per["q_out"] = stats["rate_q_out"]
    rates.append(stats["rate_q_out"])
    return dict(rate_mean_all=round(sum(rates) / len(rates), 4), per_point=per,
                k_sqnorm=[stats[b]["k_sqnorm_mean"] for b in blocks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fqs", required=True)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = cl.get_device()
    rows = []
    if os.path.exists(args.out):
        rows = json.load(open(args.out))
    for fq in [float(x) for x in args.fqs.split(",")]:
        cfg, sl, demo, tokens = cl.load_env(fire_quantile=fq)
        a0 = cl.load_a0(cfg, sl, dev)
        t0 = time.time()
        m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, fq, refit=args.refit)
        t_build = time.time() - t0
        ev = evaluate(m, sl, demo, cfg, tokens)
        st = cl.model_stats(m, tokens[:args.n], dev, is_float=False)
        summ = summarize(st)
        row = dict(fq=fq, refit=args.refit, token_acc=ev["token_acc"], ppl=ev["ppl"], n_exact=ev["n_exact"],
                   rate_mean_all=summ["rate_mean_all"], k_sqnorm_mean=summ["k_sqnorm"], rates=summ["per_point"],
                   stats=st, thresholds=thr, build_seconds=round(t_build, 1),
                   completions=[c["output"] for c in ev["completions"]])
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("fq", "refit", "token_acc", "ppl", "n_exact", "rate_mean_all",
                                               "k_sqnorm_mean", "build_seconds")}), flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        del m, a0
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
