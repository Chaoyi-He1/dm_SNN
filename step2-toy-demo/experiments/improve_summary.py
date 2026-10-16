"""汇总改进方向实验:results/improve/*/result.json + 基线(conversion-sweep 的 L1、CL2 与 A0),写 results/improve/summary.json 并打印表格。

每行:运行名、阶段、级数(每 token 最多几个脉冲)、尺寸、步数、起点、最终 acc/ppl/补全、训练中最好 acc 与补全、最后 500 步的提升、
每神经元每 token 平均脉冲数、用时。
"""
import json
import os
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)

BASELINES = {
    "A0 (ckpt)": None,
    "L1_fq50_12k": "results/conversion-sweep/L1_fq50_12k",
    "CL2_l1_lif225_lr3e3_12k": "results/conversion-sweep/CL2_l1_lif225_lr3e3_12k",
}


def load_hist(d):
    p = Path(d) / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def row(name, d):
    r = json.loads((Path(d) / "result.json").read_text(encoding="utf-8"))
    a = r["args"]
    h = load_hist(d)
    last = h[-1]["step"] if h else 0
    older = [x["token_acc"] for x in h if x["step"] <= last - 500]
    gain500 = (h[-1]["token_acc"] - max(older)) if older else None
    best = max(h, key=lambda x: x["token_acc"]) if h else None
    size = "x".join(str(a.get(k)) for k in ("d", "dk", "d_ff")) if a.get("d") else "128x32x256"
    out = dict(name=name, stage=a["stage"], levels=a.get("levels") or 1, level_points=a.get("level_points") or "all",
               size=size, steps=a.get("steps"), init=a.get("init"), lif_scale=a.get("lif_scale"), lr=a.get("lr"), fq=a.get("fq"),
               acc=round(r["final"]["token_acc"], 4), ppl=round(r["final"]["ppl"], 3), exact=r["final"]["n_exact"],
               best_acc=round(best["token_acc"], 4) if best else None, best_step=best["step"] if best else None,
               max_exact=max((x["n_exact"] for x in h), default=None), gain_last500=round(gain500, 4) if gain500 is not None else None,
               stopped=r.get("stopped"), seconds=r.get("seconds"), n_params=r.get("n_params"),
               spikes=(r.get("spikes") or {}).get("mean_spikes_per_neuron_token"))
    for k in ("init", "b", "b_lif_scale"):
        if k in r:
            out[f"{k}_acc"] = round(r[k]["token_acc"], 4)
    return out


def main():
    rows = []
    for name, d in BASELINES.items():
        if d and (Path(d) / "result.json").exists():
            rows.append(dict(row(name, d), group="baseline"))
    for d in sorted(Path("results/improve").iterdir()):
        if d.is_dir() and (d / "result.json").exists():
            rows.append(dict(row(d.name, d), group="improve"))
    Path("results/improve/summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ("name", "stage", "levels", "level_points", "size", "steps", "lif_scale", "init_acc", "b_lif_scale_acc", "acc", "ppl", "exact",
            "best_acc", "max_exact", "gain_last500", "spikes", "seconds")
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
