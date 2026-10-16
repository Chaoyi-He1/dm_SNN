"""改进方向实验的最终表格:合并 result.json / history.jsonl、model_profile 与 beam_eval 的输出,打印 markdown 并写 results/improve/tables.json。

PIPES 定义每条流水线的 A1 与 C 目录;画像与束搜索结果按模型名从 results/improve/profile_*.json、results/improve/beam/*.json 里找。
"""
import json
import os
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
R = Path("results/improve")
CS = Path("results/conversion-sweep")

# 名字: (方向, 宽度, 每 token 最多脉冲数, 多级点, A1 目录, C 目录, 画像/束搜索里 A1 的名字, C 的名字)
PIPES = [
    ("基线 T=1(已有)", "训练更久", 128, 1, "all", CS / "L1_fq50_12k", CS / "CL2_l1_lif225_lr3e3_12k", "A1", "C"),
    ("C 续训 12k", "训练更久", 128, 1, "all", CS / "L1_fq50_12k", R / "LT_cont12k_fromCL2", "A1", "LT_cont12k"),
    ("C 训 24k", "训练更久", 128, 1, "all", CS / "L1_fq50_12k", R / "LT_c24k_fromL1", "A1", "LT_c24k_fromL1"),
    ("A1 24k + C 24k", "训练更久", 128, 1, "all", R / "LT_a1_24k", R / "LT_c24k_from24k", "LT_a1_24k", "LT_c24k_from24k"),
    ("T=2", "多脉冲", 128, 2, "all", R / "T2_a1", R / "T2_c", "T2_a1", "T2_c"),
    ("T=4", "多脉冲", 128, 4, "all", R / "T4_a1", R / "T4_c", "T4_a1", "T4_c"),
    ("T=8", "多脉冲", 128, 8, "all", R / "T8_a1", R / "T8_c", "T8_a1", "T8_c"),
    ("T=4 只在 q/k/v", "多脉冲", 128, 4, "mem", R / "T4mem_a1", R / "T4mem_c", "T4mem_a1", "T4mem_c"),
    ("加宽 d=256", "加宽", 256, 1, "all", R / "W_a1", R / "W_c", "W_a1", "W_c"),
    ("加宽 d=256 + T=4", "加宽+多脉冲", 256, 4, "all", R / "WT4_a1", R / "WT4_c", "WT4_a1", "WT4_c"),
]


def load_result(d):
    p = Path(d) / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_hist(d):
    p = Path(d) / "history.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def collect(pattern):
    out = {}
    for f in sorted(Path(".").glob(pattern)):
        d = json.loads(f.read_text(encoding="utf-8"))
        for name, rec in (d["models"].items() if "models" in d else d.items()):
            for r in rec.get("runs", []) if isinstance(rec, dict) else []:
                # beam_eval rounds unique_exact to 4 decimals; recover count / n so a second rounding is exact
                r["unique_exact"] = round(r["unique_exact"] * r["unique_n"]) / r["unique_n"]
            out[name] = rec
    return out


def best_beam(rec):
    if not rec:
        return None, None
    runs = rec["runs"]
    g = next((r for r in runs if r["K"] == 1), None)
    b = max(runs, key=lambda r: r["unique_exact"])
    return g, b


def stage_row(d):
    r = load_result(d)
    if r is None:
        return None
    h = load_hist(d)
    last10 = [e["n_exact"] for e in h[-10:]]
    return dict(acc=round(r["final"]["token_acc"], 4), ppl=round(r["final"]["ppl"], 3), exact=r["final"]["n_exact"],
                steps_run=h[-1]["step"] if h else None, stopped=r.get("stopped"), seconds=r.get("seconds"),
                exact5_last10=sum(x == 5 for x in last10), lif_scale=r["args"].get("lif_scale"))


def main():
    prof = collect("results/improve/profile_*.json")
    beam = collect("results/improve/beam/*.json")
    rows = []
    for name, direction, width, T, pts, a1, c, a1n, cn in PIPES:
        ra, rc = stage_row(a1), stage_row(c)
        ga, ba = best_beam(beam.get(a1n))
        gc, bc = best_beam(beam.get(cn))
        pc = prof.get(cn) or prof.get({"C": "C_CL2"}.get(cn, ""), None)
        pa = prof.get(a1n) or prof.get({"A1": "A1_L1"}.get(a1n, ""), None)
        rows.append(dict(name=name, direction=direction, width=width, T=T, points=pts, a1=ra, c=rc,
                         a1_greedy=ga and ga["unique_exact"], a1_beam=ba and ba["unique_exact"], a1_beam_cfg=ba and (ba["K"], ba["alpha"]),
                         c_greedy=gc and gc["unique_exact"], c_beam=bc and bc["unique_exact"], c_beam_cfg=bc and (bc["K"], bc["alpha"]),
                         c_spikes=pc and pc.get("total_spikes_per_token"), c_w2=pc and pc.get("w2_events_per_token"),
                         c_params=pc and pc.get("n_params"), c_late=pc and pc["by_position"]["16-25"],
                         c_mean_p=pc and pc.get("mean_p_true"), a1_spikes=pa and pa.get("total_spikes_per_token")))
    seeds = []
    for T in (2, 4):
        for s in (0, 1, 2):
            suf = "" if s == 0 else f"_s{s}"
            a1d, cd = R / f"T{T}_a1{suf}", R / f"T{T}_c{suf}"
            ra, rc = stage_row(a1d), stage_row(cd)
            gc, bc = best_beam(beam.get(f"T{T}_c{suf}"))
            seeds.append(dict(T=T, seed=s, a1=ra, c=rc, c_greedy=gc and gc["unique_exact"], c_beam=bc and bc["unique_exact"]))
    Path("results/improve/tables.json").write_text(json.dumps(dict(main=rows, seeds=seeds), ensure_ascii=False, indent=1),
                                                    encoding="utf-8")

    lines = []

    def out(*a):
        text = " ".join(str(x) for x in a)
        print(text)
        lines.append(text)

    def f(x, k=3):
        return "—" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))

    out("| 运行 | 方向 | 宽度 | T | A1 acc | A1 步数 | C acc | C 步数 | C demo | 最后 10 个评估点 5/5 次数 | C 贪心整句 | C 束搜索整句 | 每 token 脉冲 | W2 事件 | 参数 |")
    out("|" + "---|" * 15)
    for r in rows:
        a, c = r["a1"] or {}, r["c"] or {}
        out(f"| {r['name']} | {r['direction']} | {r['width']} | {r['T']}{'(q/k/v)' if r['points'] == 'mem' else ''} | {f(a.get('acc'))} | {f(a.get('steps_run'))} | "
              f"{f(c.get('acc'))} | {f(c.get('steps_run'))} | {f(c.get('exact'))}/5 | {f(c.get('exact5_last10'))} | {f(r['c_greedy'])} | {f(r['c_beam'])} | "
              f"{f(r['c_spikes'], 0)} | {f(r['c_w2'], 0)} | {f(r['c_params'])} |")
    out()
    out("| T | 种子 | A1 acc | A1 步数 | C acc | C 步数 | C demo | 最后 10 个评估点 5/5 次数 | C 贪心整句 | C 束搜索整句 |")
    out("|" + "---|" * 10)
    for r in seeds:
        a, c = r["a1"] or {}, r["c"] or {}
        out(f"| {r['T']} | {r['seed']} | {f(a.get('acc'))} | {f(a.get('steps_run'))} | {f(c.get('acc'))} | {f(c.get('steps_run'))} | "
              f"{f(c.get('exact'))}/5 | {f(c.get('exact5_last10'))} | {f(r['c_greedy'])} | {f(r['c_beam'])} |")
    Path("results/improve/tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
