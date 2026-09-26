"""A1_init 质量 vs 按点类型的分位数(零训练)。在 step2-toy-demo 目录:python experiments/diag/kinds_scan.py"""
import json
import sys
import time

sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo")
sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments")
import convert_lab as cl  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

SPECS = {
    "all075": "",
    "all050": "ternary=0.5,q=0.5,g=0.5",
    "K1_judge": "qin=0.5,k=0.5,head=0.5",
    "K2_a1rates": "q_in1=0.45,q_in2=0.45,q=0.6,k=0.33,v=0.25,pre=0.4,out=0.4,g=0.72,u=0.37,y2=0.4,q_out=0.25",
    "K3_dense": "qin=0.4,k=0.4,v=0.5,head=0.4,u=0.5,y2=0.5,q=0.6,pre=0.6,out=0.6,g=0.75",
    "K4_judge_ffn": "qin=0.5,k=0.5,head=0.5,u=0.5,y2=0.5",
    "K5_sparse_g": "qin=0.25,k=0.25,head=0.5,g=0.75,q=0.5,v=0.5,pre=0.5,out=0.5,u=0.5,y2=0.5",
    "K6_qin_head": "qin=0.35,head=0.35",
}

dev = cl.get_device()
cfg, sl, demo, tokens = cl.load_env()
a0 = cl.load_a0(cfg, sl, dev)
rows = []
for name, spec in SPECS.items():
    t0 = time.time()
    m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, 0.75, fq_kinds=cl.parse_fq_kinds(spec))
    ev = evaluate(m, sl, demo, cfg, tokens)
    st = cl.model_stats(m, tokens[:256], dev)
    rates = [v for b in ("block0", "block1") for k, v in st[b].items() if k.startswith("rate_")] + [st["rate_q_out"]]
    row = dict(name=name, spec=spec, token_acc=round(ev["token_acc"], 4), ppl=round(ev["ppl"], 1), n_exact=ev["n_exact"],
               mean_rate=round(sum(rates) / len(rates), 3), seconds=round(time.time() - t0, 1))
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open("results/conversion-sweep/diag/kinds_scan.json", "w"), indent=1)
print("| spec | acc | ppl | exact | mean rate |")
for r in rows:
    print(f"| {r['name']} | {r['token_acc']} | {r['ppl']} | {r['n_exact']} | {r['mean_rate']} |")
