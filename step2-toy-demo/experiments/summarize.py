"""汇总 results/conversion-sweep/*/result.json 为一张表(markdown)。用法:python experiments/summarize.py [sweep_dir] [--json out.json]"""
import json
import sys
from pathlib import Path

sweep = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "results/conversion-sweep")
rows = []
for p in sorted(sweep.glob("*/result.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    a = r["args"]
    f = r["final"]
    knobs = []
    if a["stage"] == "a1":
        knobs.append(f"fq {a['fq']}" + (f" kinds[{a['fq_kinds']}]" if a.get("fq_kinds") else ""))
        if a.get("refit"): knobs.append("refit")
        if a.get("gate_refit"): knobs.append("gate-refit")
    else:
        knobs.append(f"init {Path(a['init']).parent.name}")
        if a.get("lif_scale", 1.0) != 1.0: knobs.append(f"lif x{a['lif_scale']}")
        if a.get("thr_scale", 1.0) != 1.0: knobs.append(f"thr x{a['thr_scale']}")
        if a.get("leak_anneal"): knobs.append(f"anneal {a['leak_anneal']}")
        if a.get("recalib") is not None: knobs.append(f"recalib {a['recalib']}")
    knobs.append(f"lr {a['lr']} {a['schedule']}" + (f" warmup {a['warmup']}" if a.get("warmup") else ""))
    if a.get("kd"): knobs.append(f"kd {a['kd']} T{a['kd_T']}")
    knobs.append(f"steps {a['steps']}")
    if a.get("seed"): knobs.append(f"seed {a['seed']}")
    hist = r.get("history", [])
    best = max(hist, key=lambda h: h["token_acc"]) if hist else None
    start = r.get("init") or r.get("b_lif_scale") or r.get("b_thr_scale") or r.get("b_recalib") or r.get("b")
    rows.append(dict(name=a["name"], stage=a["stage"], knobs="; ".join(knobs),
                     start_acc=round(start["token_acc"], 4) if start else None,
                     acc=round(f["token_acc"], 4), ppl=round(f["ppl"], 3), exact=f"{f['n_exact']}/{f['n_demo']}",
                     best_acc=round(best["token_acc"], 4) if best else None,
                     best_exact=max(h["n_exact"] for h in hist) if hist else None,
                     stopped=r.get("stopped"), steps=hist[-1]["step"] if hist else None, seconds=r.get("seconds")))
rows.sort(key=lambda x: (x["stage"], -x["acc"]))
print("| run | stage | knobs | start acc | final acc | ppl | exact | best acc (max exact) | stopped | steps | s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    print(f"| {x['name']} | {x['stage']} | {x['knobs']} | {x['start_acc']} | {x['acc']} | {x['ppl']} | {x['exact']} | "
          f"{x['best_acc']} ({x['best_exact']}) | {x['stopped']} | {x['steps']} | {x['seconds']} |")
if "--json" in sys.argv:
    Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
