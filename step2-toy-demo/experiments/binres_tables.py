"""二值(0/1)残差流实验的汇总表:把二值残差的各条链与三值残差、实值残差的对照并排,回答"离浮点 A0 还差多少、花了多少"。

用法(在 step2-toy-demo 目录执行;任意 cwd 也可,main 会切换):
    .venv/bin/python experiments/binres_tables.py \
        --runs binres_sr_T1,binres_sr_T2,binres_sr_T4,binres_sr_T8,binres_xor_T1,binres_xor_T2 \
        --ternary ternres_sat_T1,ternres_sat_T2,ternres_sat_T4 \
        --baselines T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k,T2=results/improve/T2_c,T4=results/improve/T4_c,T8=results/improve/T8_c \
        --beam results/improve/beam/binres.json,results/improve/beam/ternres.json,results/improve/beam/base.json \
        [--a0-acc 0.842] [--save results/improve/binres_tables.json]

只读 JSON / JSONL,不导入 torch;链还在跑时也能运行(缺的写 "—")。每条链的读取沿用 ternres_tables.read_run / read_baseline,
这里补上:残差流类型与宽度(取自 result.json 的 args)、残差流的电平数、与 A0 的差、整句背对率(取自 --beam 的各个
beam_eval 结果文件,按检查点路径对应)、是否达到"和 A0 差不多"的判据。

判据(设计文档 docs/superpowers/specs/2026-11-14-binary-residual-design.md):
    C 的 token 准确率 >= A0 - 0.01,最终 5 道 demo 题全对,且最后 10 次评估里至少 5 次全对。
整句背对率只报告,不进判据。
"""
import argparse
import json
import os
import sys
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP2))

from experiments.ternres_tables import DASH, _f, dig, dig_num, load_json, parse_baselines, read_baseline, read_run  # noqa: E402

A0_ACC = 0.842           # 浮点 A0 的切片 token 准确率(results/ckpt/a0.pt)
ACC_MARGIN = 0.01        # 判据:与 A0 相差不超过 1 个百分点
MIN_LAST10 = 5           # 判据:最后 10 次评估至少 5 次 5/5
KIND = {"binary": "二值", "ternary": "三值", "real": "实值"}

HEADERS = ["残差流", "合并", "T", "电平数", "宽度 d", "A1 初始 acc", "A1 acc(步数)", "C acc", "比 A0 低(百分点)",
           "C 的 demo", "最后 10 次 5/5", "整句贪心", "整句 K=4", "每 token 脉冲", "W2 事件", "达到判据"]


def stream_levels(residual, T):
    """残差流每个通道能取几个值:二值(无符号)T+1,三值 2T+1,实值不是离散的(返回 None)。"""
    if T is None or residual == "real":
        return None
    return int(T) + 1 if residual == "binary" else 2 * int(T) + 1


def meets_criterion(c_acc, n_exact, last10, a0_acc=A0_ACC):
    """三项都有数才判;缺任何一项返回 None(链没跑完)。"""
    if c_acc is None or n_exact is None or last10 is None:
        return None
    return bool(c_acc >= a0_acc - ACC_MARGIN - 1e-9 and n_exact == 5 and last10 >= MIN_LAST10)


def beam_index(paths):
    """读若干 beam_eval 结果文件 -> {检查点路径: {(K, alpha): unique_exact}}。读不到的文件跳过;同一检查点后读到的覆盖先读到的。"""
    out = {}
    for p in paths:
        j = load_json(p)
        for m in (dig(j, "models") or {}).values():
            path, runs = dig(m, "path"), dig(m, "runs")
            if not isinstance(path, str) or not isinstance(runs, list):
                continue
            rec = out.setdefault(os.path.normpath(path), {})
            for r in runs:
                if isinstance(r, dict) and "K" in r and "unique_exact" in r:
                    rec[(int(r["K"]), float(r.get("alpha", 0.0)))] = r["unique_exact"]
    return out


def enrich(rec, c_dir, beams, a0_acc=A0_ACC):
    """在 ternres_tables 的记录上补:residual、d、电平数、n_exact、与 A0 的差、整句背对率、判据。c_dir 是 C 运行目录。"""
    c = load_json(Path(c_dir) / "result.json")
    if rec.get("residual") == "real":              # read_baseline 的记录:实值残差,result.json 里没有 residual 键
        residual = "real"
    else:                                          # 链:C 的 args 优先;还没到 C 时看 A1 的 args,再不行按名字
        a1 = load_json(Path(str(c_dir)[:-2] + "_a1") / "result.json") if str(c_dir).endswith("_c") else None
        residual = (dig(c, "args", "residual") or dig(a1, "args", "residual")
                    or ("binary" if Path(c_dir).name.startswith("binres_") else "ternary"))
    rec = dict(rec, residual=residual)
    rec["d"] = dig_num(c, "args", "d") or 128
    rec["stream_levels"] = stream_levels(residual, rec.get("levels"))
    rec["c_n_exact"] = dig_num(c, "final", "n_exact")
    rec["gap_to_a0_pt"] = None if rec.get("c_acc") is None else round((a0_acc - rec["c_acc"]) * 100, 2)
    b = beams.get(os.path.normpath(str(Path(c_dir) / "model.pt")), {})
    rec["recall_greedy"], rec["recall_k4"] = b.get((1, 0.0)), b.get((4, 0.0))
    rec["meets"] = meets_criterion(rec.get("c_acc"), rec["c_n_exact"], rec.get("c_last10_exact"), a0_acc)
    return rec


def fmt_row(rec):
    a1 = _f(rec["a1_acc"], ".3f")
    if rec["a1_acc"] is not None and rec["a1_steps"] is not None:
        a1 += f" ({rec['a1_steps']})"
    demo = DASH if rec["c_n_exact"] is None else f"{rec['c_n_exact']}/5"
    meets = DASH if rec["meets"] is None else ("是" if rec["meets"] else "否")
    merge = rec.get("merge") or DASH
    name = rec["name"]
    if "_s" in name.rsplit("T", 1)[-1]:            # 种子后缀写进合并列,免得几行看起来一样
        merge += f"(种子 {name.rsplit('_s', 1)[-1]})"
    return [KIND.get(rec["residual"], str(rec["residual"])), merge, _f(rec["levels"], "g"),
            DASH if rec["stream_levels"] is None else str(rec["stream_levels"]), _f(rec["d"], "g"),
            _f(rec["a1_init_acc"], ".3f"), a1, _f(rec["c_acc"], ".3f"), _f(rec["gap_to_a0_pt"], ".1f"), demo,
            _f(rec["c_last10_exact"], "d"), _f(rec["recall_greedy"], ".3f"), _f(rec["recall_k4"], ".3f"),
            _f(rec["spikes_per_token"], ".0f"), _f(rec["w2_events"], ".0f"), meets]


def render_table(recs):
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
    lines += ["| " + " | ".join(fmt_row(r)) + " |" for r in recs]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="", help="二值残差的链名,逗号分隔")
    ap.add_argument("--ternary", default="", help="三值残差的链名(对照),逗号分隔")
    ap.add_argument("--baselines", default="", help="实值残差:label=C 运行目录,逗号分隔")
    ap.add_argument("--beam", default="", help="beam_eval 结果文件,逗号分隔")
    ap.add_argument("--out", default="results/improve", help="链的产物所在目录")
    ap.add_argument("--a0-acc", type=float, default=A0_ACC)
    ap.add_argument("--save", default="results/improve/binres_tables.json", help="JSON 输出路径;空串则不写")
    args = ap.parse_args(argv)
    try:
        baselines = parse_baselines(args.baselines)
    except ValueError as e:
        ap.error(str(e))
    os.chdir(STEP2)
    beams = beam_index([x.strip() for x in args.beam.split(",") if x.strip()])
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    recs = []
    for name in split(args.runs) + split(args.ternary):
        recs.append(enrich(read_run(args.out, name), Path(args.out) / f"{name}_c", beams, args.a0_acc))
    for label, path in baselines:
        recs.append(enrich(read_baseline(label, path), path, beams, args.a0_acc))
    print(render_table(recs))
    print("\n状态: " + "; ".join(f"{r['name']} {r['status']}" for r in recs))
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(a0_acc=args.a0_acc, criterion=dict(acc_margin=ACC_MARGIN, min_last10=MIN_LAST10),
                                        rows=recs), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写 {path}")


if __name__ == "__main__":
    sys.exit(main())
