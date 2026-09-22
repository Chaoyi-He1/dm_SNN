"""把流水线结果写成 toy-demo.json 与 toy-demo.md(计划第 5、6 节的产物)。"""
import json
from pathlib import Path

STAGES = ["A0", "A1_init", "A1", "B", "C", "circuit_ideal", "circuit_nominal"]
LABEL = {"A0": "A0 浮点", "A1_init": "A1 阈值初始化(未微调)", "A1": "A1 可脉冲化", "B": "B 折叠转换(零训练)",
         "C": "C 微调", "circuit_ideal": "电路·理想", "circuit_nominal": "电路·标称"}


def _cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_report(results, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "toy-demo.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    L = ["# 玩具模型 demo 结果", "",
         f"设备 {results['device']},torch {results['torch']},词表 {results['vocab_size']},句数 {results['n_sentences']}。", "",
         "## 各阶段", "", "| 阶段 | token 准确率 | 困惑度 | 补全正确 | 停止 | 种子 | 用时 s |", "|---|---|---|---|---|---|---|"]
    for s in STAGES:
        r = results["stages"].get(s)
        if not r:
            continue
        f = r["final"]
        L.append(f"| {LABEL[s]} | {f['token_acc']:.4f} | {f['ppl']:.3f} | {f['n_exact']}/{f['n_demo']} | "
                 f"{r.get('stopped', '')} | {r.get('seed', '')} | {r.get('seconds', '')} |")
    L += ["", "## 补全", ""]
    for s in STAGES:
        r = results["stages"].get(s)
        if not r:
            continue
        L += [f"### {LABEL[s]}", "", "| 前缀 | 目标 | 输出 | 一致 |", "|---|---|---|---|"]
        L += [f"| {_cell(c['prefix'])} | {_cell(c['target'])} | {_cell(c['output'])} | {'是' if c['exact'] else '否'} |"
              for c in r["final"]["completions"]]
        L.append("")
    for s in ("circuit_ideal", "circuit_nominal"):
        cmp = results["stages"].get(s, {}).get("final", {}).get("compare")
        if not cmp:
            continue
        L += [f"## teacher forcing 对照:{LABEL[s]}", "",
              f"输出头 argmax 一致率 {cmp['argmax_agreement']:.4f},{cmp['n_positions']} 个位置,判定 {'通过' if cmp['pass'] else '不通过'}。", "",
              "| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |", "|---|---|---|---|---|---|---|---|---|"]
        for i, b in enumerate(cmp["blocks"]):
            for name in ("attn", "ffn"):
                p = b[name]
                L.append(f"| {i} | {name} | {p['pos_precision']:.3f} | {p['pos_recall']:.3f} | {p['neg_precision']:.3f} | "
                         f"{p['neg_recall']:.3f} | {p['sign_error']:.4f} | {b['state_err_max']:.4f} | {b['state_err_mean']:.4f} |")
        L.append("")
    if results.get("sigma_g_sweep"):
        L += ["## 编程误差扫描", "", "| σ_G | token 准确率 | 困惑度 | 补全正确 |", "|---|---|---|---|"]
        L += [f"| {r['sigma_g']} | {r['token_acc']:.4f} | {r['ppl']:.3f} | {r['n_exact']} |" for r in results["sigma_g_sweep"]]
        L.append("")
    (out_dir / "toy-demo.md").write_text("\n".join(L), encoding="utf-8")
