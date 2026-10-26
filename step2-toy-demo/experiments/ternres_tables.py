"""三值残差流实验的汇总表:把 ternary_residual.py chain 的产物与实值残差基线的 C 运行并排,打印 markdown 表并写 JSON。

用法(在 step2-toy-demo 目录执行;任意 cwd 也可,main 会切换):
    .venv/bin/python experiments/ternres_tables.py \
        --runs ternres_sat_T1,ternres_and_T1,ternres_sat_T2,ternres_and_T2 \
        --baselines T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k,T2=results/improve/T2_c,T2_s1=results/improve/T2_c_s1 \
        [--out results/improve] [--save results/improve/ternres_tables.json]

只读 JSON / JSONL,不导入 torch,不启动训练;chain 还在跑时也能运行:读不到(或正写到一半)的文件按缺失处理,表里写 "—",JSON 里写 null。
每条 --runs 读 --out 下的 <name>_chain.json、<name>_iscan.json、<name>_a1/result.json、<name>_c/result.json,
status 由哪些文件在决定:missing(什么都没有)、a1 running(有 chain/iscan/a1 目录,a1 的 result.json 还没有)、
c running(a1 已完成,c 的 result.json 还没有;包含 bscan 阶段)、done(c 的 result.json 在)。
每条 --baselines 是 label=C 运行目录;A1 取该 C result.json 里 args.init 所在目录的 result.json,没有就写 "—"。
"""
import argparse
import json
import os
import sys
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]

DASH = "—"
N_DEMO = 5          # demo 共 5 句;history 里的 n_exact == 5 即 5/5
LAST_K = 10         # "最后 10 次"评估
BASELINE_RULE = "实值残差"

HEADERS = ["规则", "T", "选中的 fq", "A1 初始 acc", "A1 acc(步数)", "B acc", "选中的 LIF 倍数", "C acc", "C ppl",
           "demo 最后 10 次 5/5 次数", "C 收敛步数", "每 token 脉冲总数", "W2 事件", "用时(A1 + C, 分钟)"]


# ----------------------------------------------------------------------------- 读取

def load_json(path):
    """读 JSON 对象;文件不存在、正被写到一半(解析失败)或不是对象都返回 None。"""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):                  # JSONDecodeError、UnicodeDecodeError 都是 ValueError
        return None
    return obj if isinstance(obj, dict) else None


def dig(obj, *keys):
    """沿键路径取值;任何一层缺失或不是 dict 都返回 None。"""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def dig_num(obj, *keys):
    """dig,但只接受数(排除 bool),其余当缺失。"""
    v = dig(obj, *keys)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def first(*vals):
    return next((v for v in vals if v is not None), None)


def last_k_exact(history, k=LAST_K):
    """history 最后 k 条里 n_exact == 5 的条数;history 缺失或为空返回 None。"""
    if not isinstance(history, list) or not history:
        return None
    return sum(1 for h in history[-k:] if isinstance(h, dict) and h.get("n_exact") == N_DEMO)


def last_step(history):
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        s = history[-1].get("step")
        return s if isinstance(s, (int, float)) and not isinstance(s, bool) else None
    return None


def stage_fields(a1, c):
    """从 A1 与 C 的 result.json(可为 None)里取出表里要的数,键的含义见 fmt_row。缺什么就是 None。"""
    sec_a1, sec_c = dig_num(a1, "seconds"), dig_num(c, "seconds")
    secs = [s for s in (sec_a1, sec_c) if s is not None]
    return dict(
        a1_init_acc=dig_num(a1, "init", "token_acc"),
        a1_acc=dig_num(a1, "final", "token_acc"),
        a1_steps=last_step(dig(a1, "history")),
        b_acc=first(dig_num(c, "b_lif_scale", "token_acc"), dig_num(c, "b", "token_acc")),
        c_acc=dig_num(c, "final", "token_acc"),
        c_ppl=dig_num(c, "final", "ppl"),
        c_last10_exact=last_k_exact(dig(c, "history")),
        c_steps=last_step(dig(c, "history")),
        c_stopped=dig(c, "stopped"),
        spikes_per_token=dig_num(c, "spikes", "total_spikes_per_token"),
        w2_events=dig_num(c, "spikes", "w2_events_per_token"),
        seconds_a1=sec_a1,
        seconds_c=sec_c,
        minutes=sum(secs) / 60 if secs else None,     # 只有一段时是部分和,原始秒数在 seconds_a1 / seconds_c
    )


def run_status(chain, iscan, a1, c, a1_dir, c_dir):
    if c is not None:
        return "done"
    if a1 is not None or Path(c_dir).exists():
        return "c running"
    if chain is not None or iscan is not None or Path(a1_dir).exists():
        return "a1 running"
    return "missing"


def read_run(out_dir, name):
    """读 <out_dir>/<name>_{chain.json,iscan.json,a1/,c/},返回一条记录(dict,缺失项为 None)。"""
    out = Path(out_dir)
    a1_dir, c_dir = out / f"{name}_a1", out / f"{name}_c"
    chain, iscan = load_json(out / f"{name}_chain.json"), load_json(out / f"{name}_iscan.json")
    a1, c = load_json(a1_dir / "result.json"), load_json(c_dir / "result.json")
    rec = dict(name=name, status=run_status(chain, iscan, a1, c, a1_dir, c_dir), residual="ternary",
               merge=first(dig(chain, "merge"), dig(a1, "args", "merge"), dig(c, "args", "merge")),
               levels=first(dig(chain, "levels"), dig(c, "args", "levels"), dig(a1, "args", "levels")),
               fq=first(dig_num(chain, "fq"), dig_num(a1, "args", "fq"), dig_num(c, "args", "fq")),
               lif_scale=first(dig_num(chain, "lif_scale"), dig_num(c, "args", "lif_scale")),
               **stage_fields(a1, c))
    if rec["a1_init_acc"] is None:                   # A1 还在训练时 result.json 没有 init;iscan 对所选 fq 算的是同一个量
        rec["a1_init_acc"] = dig_num(chain, "iscan", "best", "token_acc")
    return rec


def read_baseline(label, path):
    """读基线 C 运行目录 path(其 result.json 的 args.init 指向 A1 的 model.pt,同目录的 result.json 是 A1 的数字)。"""
    c = load_json(Path(path) / "result.json")
    init = dig(c, "args", "init")
    a1 = load_json(Path(init).parent / "result.json") if isinstance(init, str) else None
    return dict(name=label, status="done" if c is not None else "missing", residual="real", merge=None,
                levels=first(dig(c, "args", "levels"), dig(a1, "args", "levels"), 1 if c is not None else None),
                fq=first(dig_num(a1, "args", "fq"), dig_num(c, "args", "fq")),
                lif_scale=dig_num(c, "args", "lif_scale"),
                **stage_fields(a1, c))


# ----------------------------------------------------------------------------- 格式化

def _f(v, spec):
    return DASH if v is None else format(v, spec)


def fmt_rule(rec):
    """规则列:新模型写 merge,基线写"实值残差";名字不是常规的(带种子后缀等)时在括号里补上,免得几行一样。"""
    if rec["residual"] == "real":
        plain = rec["levels"] is not None and rec["name"] == f"T{rec['levels']}"
        return BASELINE_RULE if plain else f"{BASELINE_RULE}({rec['name']})"
    if rec["merge"] is None:
        return rec["name"]                           # 什么都还没有:至少让这一行认得出是谁
    plain = rec["levels"] is not None and rec["name"] == f"ternres_{rec['merge']}_T{rec['levels']}"
    return rec["merge"] if plain else f"{rec['merge']}({rec['name']})"


def fmt_row(rec):
    """一条记录 -> 14 个单元格字符串(顺序同 HEADERS);缺失一律 "—"。"""
    a1 = _f(rec["a1_acc"], ".3f")
    if rec["a1_acc"] is not None and rec["a1_steps"] is not None:
        a1 += f" ({rec['a1_steps']})"
    steps = DASH
    if rec["c_steps"] is not None:
        steps = f"{rec['c_steps']}" + ("(跑满)" if rec["c_stopped"] == "max_steps" else "")
    return [fmt_rule(rec), _f(rec["levels"], "g"), _f(rec["fq"], "g"), _f(rec["a1_init_acc"], ".3f"), a1,
            _f(rec["b_acc"], ".3f"), _f(rec["lif_scale"], "g"), _f(rec["c_acc"], ".3f"), _f(rec["c_ppl"], ".2f"),
            _f(rec["c_last10_exact"], "d"), steps, _f(rec["spikes_per_token"], ".0f"), _f(rec["w2_events"], ".0f"),
            _f(rec["minutes"], ".1f")]


def render_table(recs):
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "|".join("---" for _ in HEADERS) + "|"]
    lines += ["| " + " | ".join(fmt_row(r)) + " |" for r in recs]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- 命令行

def parse_baselines(spec):
    """"T1=dir1,T2=dir2" -> [(label, dir), ...]"""
    out = []
    for item in [x for x in spec.split(",") if x.strip()]:
        label, sep, path = item.partition("=")
        if not sep or not label.strip() or not path.strip():
            raise ValueError(f"--baselines 项应为 label=目录,得到 {item!r}")
        out.append((label.strip(), path.strip()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default="", help="逗号分隔的 chain 名字,如 ternres_sat_T1,ternres_and_T1")
    ap.add_argument("--baselines", default="", help="label=C 运行目录,逗号分隔,如 T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k")
    ap.add_argument("--out", default="results/improve", help="chain 产物所在目录")
    ap.add_argument("--save", default="results/improve/ternres_tables.json", help="JSON 输出路径;空串则不写")
    args = ap.parse_args(argv)
    try:
        baselines = parse_baselines(args.baselines)
    except ValueError as e:
        ap.error(str(e))
    os.chdir(STEP2)
    names = [x.strip() for x in args.runs.split(",") if x.strip()]
    runs = {n: read_run(args.out, n) for n in names}
    bases = {label: read_baseline(label, p) for label, p in baselines}

    print(render_table(list(runs.values()) + list(bases.values())))
    if runs:
        print("\n状态: " + "; ".join(f"{n} {r['status']}" for n, r in runs.items()))
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(runs=runs, baselines=bases), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写 {path}")


if __name__ == "__main__":
    sys.exit(main())
