"""汇总电路验收、误差隔离、逐项去掉、容限、诊断与噪声种子的结果,写 results/improve/circuit_tables.md 与 circuit_tables.json。
各运行目录里的 circuit*.json 由 experiments/circuit_levels.py 生成。"""
import json
import os
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
R = Path("results/improve")
RUNS = [("T = 1,原来的 C", "circuit_T1_CL2", 1, 0), ("T = 2,种子 0", "T2_c", 2, 0), ("T = 2,种子 1", "T2_c_s1", 2, 1),
        ("T = 2,种子 2", "T2_c_s2", 2, 2), ("T = 4,种子 0", "T4_c", 4, 0), ("T = 4,种子 1", "T4_c_s1", 4, 1), ("T = 4,种子 2", "T4_c_s2", 4, 2)]
ISO = [("T = 1", "circuit_T1_CL2"), ("T = 2", "T2_c"), ("T = 4", "T4_c")]
PART_NAMES = {"gdn": "GDN 子块", "ffn": "FFN 子块", "state": "状态阵列", "head": "输出头"}


def load(d, name):
    p = R / d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def outputs(cmp):
    rows = []
    for i, b in enumerate(cmp["blocks"]):
        for key in ("attn", "ffn"):
            p = b[key]
            rows.append(dict(block=i, part=key, pr=min(p[k] for k in ("pos_precision", "pos_recall", "neg_precision", "neg_recall")),
                             sign=p["sign_error"]))
    return rows


def pct(x, k=1):
    return f"{100 * x:.{k}f}%"


def main():
    L, J = [], {}
    # 1. 验收
    L += ["### 验收", "", "| 模型 | 软件 C | 理想电路 | 标称电路 acc | 标称 5 句 | 5 个噪声种子:acc 均值 ± 标准差,5/5 次数 | argmax 一致率 | 状态误差最大 | 符号错误率(块 1 注意力 / 块 1 前馈 / 块 2 注意力 / 块 2 前馈) | P/R 最小 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    acc_rows = []
    for label, d, T, s in RUNS:
        a, ns = load(d, "circuit.json"), load(d, "circuit_seeds.json")
        sw, ci, cn = a["software"], a["circuit_ideal"], a["circuit_nominal"]
        o = outputs(cn["compare"])
        ideal_ok = ci["argmax_equal_to_software"] and ci["logits_max_abs_diff"] == 0.0 and ci["compare"]["pass"] and ci["token_acc"] == sw["token_acc"]
        se = max(b["state_err_max"] for b in cn["compare"]["blocks"])
        five = sum(x == 5 for x in ns["n_exact"])
        row = dict(label=label, T=T, seed=s, sw_acc=sw["token_acc"], sw_exact=sw["n_exact"], ideal_identical=ideal_ok,
                   ideal_logits_diff=ci["logits_max_abs_diff"], nominal_acc=cn["token_acc"], nominal_exact=cn["n_exact"],
                   noise_mean=ns["acc_mean"], noise_std=ns["acc_std"], noise_five=five, noise_n=len(ns["n_exact"]),
                   agree=cn["compare"]["argmax_agreement"], state_err=se, signs=[x["sign"] for x in o], pr_min=min(x["pr"] for x in o),
                   drop=sw["token_acc"] - cn["token_acc"], spikes=a["spikes"]["total_spikes_per_token"], w2=a["spikes"]["w2_events_per_token"])
        acc_rows.append(row)
        L.append(f"| {label} | {sw['token_acc']:.3f},{sw['n_exact']}/5 | {'与软件逐位相同' if ideal_ok else '不同'} | {cn['token_acc']:.3f} | {cn['n_exact']}/5 | "
                 f"{ns['acc_mean']:.3f} ± {ns['acc_std']:.3f},{five}/{len(ns['n_exact'])} | {cn['compare']['argmax_agreement']:.3f} | {se:.4f} | "
                 f"{' / '.join(pct(x['sign']) for x in o)} | {min(x['pr'] for x in o):.3f} |")
    J["accept"] = acc_rows
    # 2. 误差隔离
    L += ["", "### 误差隔离(只让一部分取标称值,其余理想)", "", "| 模型 | 标称部分 | argmax 一致率 | 最差符号错误率 | 最差 P/R | 自由运行 acc | 5 句 |", "|---|---|---|---|---|---|---|"]
    J["isolate"] = {}
    for label, d in ISO:
        iso = load(d, "circuit_isolate.json")["isolation"]
        J["isolate"][label] = iso_rows = []
        for r in iso:
            s = r["summary"]
            name = "全部" if len(r["parts"]) == 4 else PART_NAMES[r["parts"][0]]
            iso_rows.append(dict(parts=r["parts"], agree=s["argmax_agreement"], sign=s["worst_sign_error"], pr=s["worst_pr"], acc=r["token_acc"], exact=r["n_exact"]))
            L.append(f"| {label} | {name} | {s['argmax_agreement']:.3f} | {pct(s['worst_sign_error'])} | {s['worst_pr']:.3f} | {r['token_acc']:.3f} | {r['n_exact']}/5 |")
    # 3. 逐项去掉:按 T 平均(T = 2、4 各 3 个模型)
    groups = {"T = 1": ["circuit_T1_CL2"], "T = 2": ["T2_c", "T2_c_s1", "T2_c_s2"], "T = 4": ["T4_c", "T4_c_s1", "T4_c_s2"]}
    labels = [r["label"] for r in load("T2_c", "circuit_leaveout.json")["leaveout"]]
    L += ["", "### 逐项去掉(其余保持标称;T = 2、T = 4 为 3 个模型的平均,括号里是 5 句全对的模型数;噪声种子 0)", "",
          "| 配置 | " + " | ".join(f"{g}:acc" for g in groups) + " | " + " | ".join(f"{g}:argmax 一致率" for g in groups) + " |",
          "|---|" + "---|" * (2 * len(groups))]
    J["leaveout"] = {}
    for lab in labels:
        accs, agrees = [], []
        for g, ds in groups.items():
            rows = [next(x for x in load(d, "circuit_leaveout.json")["leaveout"] if x["label"] == lab) for d in ds]
            a = sum(x["token_acc"] for x in rows) / len(rows)
            five = sum(x["n_exact"] == 5 for x in rows)
            ag = sum(x["summary"]["argmax_agreement"] for x in rows) / len(rows)
            J["leaveout"].setdefault(lab, {})[g] = dict(acc=a, five=five, n=len(rows), agree=ag, accs=[x["token_acc"] for x in rows],
                                                       exact=[x["n_exact"] for x in rows])
            accs.append(f"{a:.3f}({five}/{len(rows)})")
            agrees.append(f"{ag:.3f}")
        L.append(f"| {lab} | " + " | ".join(accs) + " | " + " | ".join(agrees) + " |")
    # 4. 容限
    L += ["", "### 一次只开一种非理想性(其余理想)时的最差符号错误率与 P/R 最小值", ""]
    J["knobs"] = {}
    for label, d in ISO:
        kn = load(d, "circuit_knobs.json")
        J["knobs"][label] = kn["tolerance"]
        L += [f"{label}:", "", "| 参数 | 标称值 | 各档(值:符号错误率 / P/R 最小 / argmax 一致率) | 满足判据的最差档 |", "|---|---|---|---|"]
        for knob, t in kn["tolerance"].items():
            rows = [r for r in kn["knobs"] if r["knob"] == knob]
            per = ";".join(f"{r['value']}:{pct(r['summary']['worst_sign_error'])} / {r['summary']['worst_pr']:.2f} / {r['summary']['argmax_agreement']:.4f}" for r in rows)
            L.append(f"| {knob} | {t['nominal']} | {per} | {t['tolerance'] if t['tolerance'] is not None else '没有'} |")
        L.append("")
    # 5. 诊断
    L += ["### 单步与累积", "", "| 模型 | 单步(全标称):计数不一致(四个输出平均) | 单步:最差符号错误率 | 单步:最差 P/R | 单步:输出头 argmax 一致 | 累积(四个输出平均):位置 1–3 | 位置 4–6 | 位置 26–41 |",
          "|---|---|---|---|---|---|---|---|"]
    J["diagnose"] = {}
    for label, d in ISO:
        dg = load(d, "circuit_diagnose.json")
        one = dg["one_step"][0]
        assert len(one["parts"]) == 4
        pp = dg["position_profile_nominal"]
        J["diagnose"][label] = dict(one_step=dg["one_step"], profile=pp, tiny=dg["tiny_perturbation"])
        L.append(f"| {label} | {pct(one['mean_count_mismatch'])} | {pct(one['worst_sign_error'], 2)} | {one['worst_pr']:.3f} | {one['head_argmax_agreement']:.3f} | "
                 f"{pct(pp['1-3'])} | {pct(pp['4-6'])} | {pct(pp['26-41'])} |")
    L += ["", "| 模型 | 单步:只开一部分(计数不一致 平均 / 输出头 argmax 一致) |", "|---|---|"]
    for label, d in ISO:
        one = load(d, "circuit_diagnose.json")["one_step"]
        L.append(f"| {label} | " + ";".join(f"{PART_NAMES[r['parts'][0]] if len(r['parts']) == 1 else '全部'} {pct(r['mean_count_mismatch'])} / {r['head_argmax_agreement']:.3f}" for r in one) + " |")
    L += ["", "| 模型 | 极小扰动(其余理想) | 单步:计数不一致(四个输出平均) | 单步:P/R 最小 | 累积(teacher forcing):最差符号错误率 | P/R 最小 | argmax 一致率 |", "|---|---|---|---|---|---|---|"]
    for label, d in ISO:
        for r in load(d, "circuit_diagnose.json")["tiny_perturbation"]:
            L.append(f"| {label} | {r['knob']} = {r['value']} | {pct(r['one_step']['mean_count_mismatch'], 2)} | {r['one_step']['worst_pr']:.3f} | "
                     f"{pct(r['teacher_forced']['worst_sign_error'], 2)} | {r['teacher_forced']['worst_pr']:.3f} | {r['teacher_forced']['argmax_agreement']:.4f} |")
    (R / "circuit_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (R / "circuit_tables.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
