"""三值残差流模型 TernaryResidualLM 的转换实验:沿用 convert_lab 的训练循环与存取,走 A0 -> A1 -> B -> C。

在 step2-toy-demo 目录执行(任意 cwd 也可,脚本自行切换);结果写到 --out/<name>/(默认 results/improve):
    .venv/bin/python experiments/ternary_residual.py a1 --name ternres_sat_T2_a1 --merge sat --levels 2 --fq 0.75
    .venv/bin/python experiments/ternary_residual.py c --name ternres_sat_T2_c --init results/improve/ternres_sat_T2_a1/model.pt --lif-scale 2.25
    .venv/bin/python experiments/ternary_residual.py iscan --merge and --levels 2 --fq-list 0.5,0.75,0.9 --save results/improve/ternres_and_T2_iscan.json
    .venv/bin/python experiments/ternary_residual.py bscan --init results/improve/ternres_sat_T2_a1/model.pt --scale-list 1,2,3 --save results/improve/ternres_sat_T2_bscan.json
    .venv/bin/python experiments/ternary_residual.py chain --name ternres_sat_T2 --merge sat --levels 2    # iscan -> a1 -> bscan -> c,进程内串联

阶段:
    a1     A0 权重转移 + 阈值标定(--fq),leak 0 下微调 --steps 步
    c      装载 A1 检查点(merge、levels、fq、尺寸从 --init 同目录的 result.json 继承),leak = cfg.leak;
           训练前只把 LIF 阈值乘以 --lif-scale;最终以 leak = cfg.leak 评估
    iscan  对 --fq-list 里每个分位数做 A1 初始化,只算 teacher forcing 准确率、困惑度与平均脉冲数
    bscan  B 起点扫描:对 --scale-list 里每个倍数缩放 LIF 阈值,算 leak = cfg.leak 下的 teacher forcing 准确率
    chain  iscan(取 token_acc 最高的 fq)-> a1(<name>_a1)-> bscan(取 token_acc 最高的倍数)-> c(<name>_c);
           选择写进 <name>_iscan.json、<name>_bscan.json 与 <name>_chain.json
每次 _save 的 result.json args 含 stage、residual="ternary"、merge、levels、fq 与全部尺寸键,beam_eval 据此装载。
不改动 toy_demo 包与 convert_lab;不调用 apply_levels(levels 由模型构造参数给出)。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
sys.path.insert(0, str(STEP2))

from experiments.beam_eval import SIZE_KEYS                                              # noqa: E402
from experiments.convert_lab import _save, get_device, load_a0, load_env, scale_thresholds, size_overrides, train  # noqa: E402
from toy_demo.convert import set_leak, transfer_weights                                  # noqa: E402
from toy_demo.ternary_residual import TernaryResidualLM, calibrate_thresholds, spike_counts  # noqa: E402
from toy_demo.train import evaluate, teacher_forced_metrics                              # noqa: E402


# ----------------------------------------------------------------------------- 构造与装载

def make_model(cfg, sl, dev, merge, levels, leak):
    return TernaryResidualLM(cfg, sl.vocab_size, leak=leak, merge=merge, levels=levels).to(dev)


def build_a1(a0, cfg, sl, tokens, dev, fq, merge, levels, seed=0, leak=0.0):
    """A0 -> A1:权重转移 + 阈值标定。返回 (模型, 阈值字典)。"""
    torch.manual_seed(seed)
    m = make_model(cfg, sl, dev, merge, levels, leak)
    transfer_weights(a0, m)
    thr = calibrate_thresholds(m, tokens[:cfg.calib_sentences].to(dev), fq)
    return m, thr


def describe_init(init_path):
    """读 --init 同目录 result.json 的 args -> dict(merge, levels, fq, sizes);sizes 的键是 ToyConfig 字段名。"""
    a = json.loads((Path(init_path).parent / "result.json").read_text(encoding="utf-8"))["args"]
    return dict(merge=a["merge"], levels=int(a["levels"]), fq=a.get("fq"),
                sizes={key: a[k] for k, key in SIZE_KEYS if a.get(k) is not None})


def inherit_from_init(args):
    """c/bscan:把 merge、levels、fq 写进 args(进 result.json),返回传给 load_env 的覆盖项(命令行尺寸优先于继承值)。"""
    info = describe_init(args.init)
    args.merge, args.levels, args.fq = info["merge"], info["levels"], info["fq"]
    overrides = {**info["sizes"], **size_overrides(args)}
    if info["fq"] is not None:
        overrides["fire_quantile"] = info["fq"]
    return overrides


def safe_spike_counts(m, tokens, dev, n=256):
    try:
        return spike_counts(m, tokens[:n], dev)
    except Exception as e:                         # noqa: BLE001  统计是附带的,不让它毁掉一次长训练
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return dict(error=repr(e))


def save_stage(out_dir, m, res, args, cfg, extra):
    """补全 beam_eval.describe_checkpoint 需要的键(residual、尺寸取 cfg 的值)后存模型与 result.json。"""
    args.residual = "ternary"
    for k, key in SIZE_KEYS:
        setattr(args, k, getattr(cfg, key))
    _save(out_dir, m, res, args, extra)


def _brief(final):
    return {k: final[k] for k in ("token_acc", "ppl", "n_exact", "n_demo")}


def _write_json(path, obj):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- 命令

def cmd_a1(args):
    torch.set_num_threads(1)
    dev = get_device()
    cfg, sl, demo, tokens = load_env(fire_quantile=args.fq, **size_overrides(args))
    a0 = load_a0(cfg, sl, dev, args.a0)
    t0 = time.time()
    m, thr = build_a1(a0, cfg, sl, tokens, dev, args.fq, args.merge, args.levels, seed=args.seed)
    init = evaluate(m, sl, demo, cfg, tokens)
    print(f"[a1_init:{args.name}] merge {args.merge} levels {args.levels} acc {init['token_acc']:.4f} ppl {init['ppl']:.3f} "
          f"exact {init['n_exact']}/{init['n_demo']}")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = train(m, sl, demo, cfg, tokens, lr=args.lr, steps=args.steps, seed=args.seed, eval_every=args.eval_every,
                stop=not args.no_stop, hist_path=out_dir / "history.jsonl")
    res["seconds"] = round(time.time() - t0, 1)
    extra = dict(init=init, thresholds=thr)
    save_stage(out_dir, m, res, args, cfg, extra)           # 先存模型与结果,脉冲统计失败也不丢检查点
    extra["spikes"] = safe_spike_counts(m, tokens, dev)
    _save(out_dir, m, res, args, extra)
    return dict(dir=str(out_dir), final=res["final"], seconds=res["seconds"])


def cmd_c(args):
    torch.set_num_threads(1)
    dev = get_device()
    cfg, sl, demo, tokens = load_env(**inherit_from_init(args))
    t0 = time.time()
    m = make_model(cfg, sl, dev, args.merge, args.levels, cfg.leak)
    m.load_state_dict(torch.load(args.init, map_location=dev))
    b = evaluate(m, sl, demo, cfg, tokens)
    print(f"[b:{args.name}] merge {args.merge} levels {args.levels} acc {b['token_acc']:.4f} ppl {b['ppl']:.3f} "
          f"exact {b['n_exact']}/{b['n_demo']}")
    extra = dict(b=b)
    if args.lif_scale != 1.0:
        scale_thresholds(m, args.lif_scale, only=list(m.lif_modules()))
        extra["b_lif_scale"] = evaluate(m, sl, demo, cfg, tokens)
        print(f"[b_lif_scale:{args.name}] x{args.lif_scale} acc {extra['b_lif_scale']['token_acc']:.4f} "
              f"ppl {extra['b_lif_scale']['ppl']:.3f}")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = train(m, sl, demo, cfg, tokens, lr=args.lr, steps=args.steps, seed=args.seed, eval_every=args.eval_every,
                stop=not args.no_stop, hist_path=out_dir / "history.jsonl")
    set_leak(m, cfg.leak)
    res["final"] = evaluate(m, sl, demo, cfg, tokens)      # 以 leak=cfg.leak 的最终评估为准
    res["seconds"] = round(time.time() - t0, 1)
    save_stage(out_dir, m, res, args, cfg, extra)           # 先存模型与结果,脉冲统计失败也不丢检查点
    extra["spikes"] = safe_spike_counts(m, tokens, dev)
    _save(out_dir, m, res, args, extra)
    return dict(dir=str(out_dir), final=res["final"], seconds=res["seconds"])


def cmd_iscan(args):
    """A1 初始化扫描:对 --fq-list 里的每个分位数做权重转移 + 标定,只算切片 teacher forcing 准确率与困惑度。"""
    dev = get_device()
    cfg, sl, demo, tokens = load_env(**size_overrides(args))
    a0 = load_a0(cfg, sl, dev, args.a0)
    rows = []
    for fq in [float(x) for x in args.fq_list.split(",")]:
        m, _ = build_a1(a0, cfg, sl, tokens, dev, fq, args.merge, args.levels)
        r = teacher_forced_metrics(m, tokens)
        rows.append(dict(fq=fq, token_acc=round(r["token_acc"], 4), ppl=round(r["ppl"], 3),
                         spikes=spike_counts(m, tokens[:256], dev)["mean_spikes_per_neuron_token"]))
        print(json.dumps(rows[-1]), flush=True)
    out = dict(merge=args.merge, levels=args.levels, sizes=size_overrides(args), a0=args.a0, rows=rows)
    _write_json(args.save, out)
    return out


def cmd_bscan(args):
    """B 起点扫描:装载 A1 检查点、leak 设为 cfg.leak,对 --scale-list 里的每个倍数只缩放 LIF 阈值,算 teacher forcing 准确率。"""
    dev = get_device()
    cfg, sl, demo, tokens = load_env(**inherit_from_init(args))
    state = torch.load(args.init, map_location=dev)
    rows = []
    for sc in [float(x) for x in args.scale_list.split(",")]:
        m = make_model(cfg, sl, dev, args.merge, args.levels, cfg.leak)
        m.load_state_dict(state)
        if sc != 1.0:
            scale_thresholds(m, sc, only=list(m.lif_modules()))
        r = teacher_forced_metrics(m, tokens)
        rows.append(dict(lif_scale=sc, token_acc=round(r["token_acc"], 4), ppl=round(r["ppl"], 3)))
        print(json.dumps(rows[-1]), flush=True)
    out = dict(init=args.init, merge=args.merge, levels=args.levels, rows=rows)
    _write_json(args.save, out)
    return out


def cmd_chain(args):
    """进程内串联 iscan -> a1 -> bscan -> c;每步的选择打印出来并写进 <name>_chain.json(并列时取第一个)。"""
    out = Path(args.out)
    sizes = {a: getattr(args, a) for a, _ in SIZE_KEYS}
    common = dict(out=args.out, a0=args.a0, seed=args.seed, eval_every=args.eval_every, no_stop=args.no_stop,
                  lr=args.lr, steps=args.steps, **sizes)
    summary = dict(name=args.name, merge=args.merge, levels=args.levels, lr=args.lr, steps=args.steps, seed=args.seed)
    chain_path = out / f"{args.name}_chain.json"

    def stage(label, fn, **kw):
        t0 = time.time()
        r = fn(argparse.Namespace(stage=label, **common, **kw))
        return r, round(time.time() - t0, 1)

    iscan_path = out / f"{args.name}_iscan.json"
    r, sec = stage("iscan", cmd_iscan, name=f"{args.name}_iscan", merge=args.merge, levels=args.levels,
                   fq_list=args.fq_list, save=str(iscan_path))
    best = max(r["rows"], key=lambda row: row["token_acc"])
    summary.update(fq=best["fq"], iscan=dict(best=best, seconds=sec, file=str(iscan_path)))
    print(f"[chain:{args.name}] iscan picks fq {best['fq']} (token_acc {best['token_acc']})", flush=True)
    _write_json(chain_path, summary)

    r, sec = stage("a1", cmd_a1, name=f"{args.name}_a1", merge=args.merge, levels=args.levels, fq=best["fq"])
    summary["a1"] = dict(final=_brief(r["final"]), seconds=sec, dir=r["dir"])
    _write_json(chain_path, summary)

    bscan_path = out / f"{args.name}_bscan.json"
    init = str(Path(r["dir"]) / "model.pt")
    r, sec = stage("bscan", cmd_bscan, name=f"{args.name}_bscan", init=init, scale_list=args.scale_list, save=str(bscan_path))
    best_b = max(r["rows"], key=lambda row: row["token_acc"])
    summary.update(lif_scale=best_b["lif_scale"], bscan=dict(best=best_b, seconds=sec, file=str(bscan_path)))
    print(f"[chain:{args.name}] bscan picks lif_scale {best_b['lif_scale']} (token_acc {best_b['token_acc']})", flush=True)
    _write_json(chain_path, summary)

    r, sec = stage("c", cmd_c, name=f"{args.name}_c", init=init, lif_scale=best_b["lif_scale"])
    summary["c"] = dict(final=_brief(r["final"]), seconds=sec, dir=r["dir"])
    _write_json(chain_path, summary)
    print(f"[chain:{args.name}] done: a1 acc {summary['a1']['final']['token_acc']:.4f}, c acc {r['final']['token_acc']:.4f}", flush=True)


# ----------------------------------------------------------------------------- 命令行

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", default="run")
    common.add_argument("--out", default="results/improve")
    common.add_argument("--a0", default="results/ckpt/a0.pt")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--eval-every", type=int, default=100)
    common.add_argument("--no-stop", action="store_true", help="忽略停止条件,跑满步数")
    common.add_argument("--lr", type=float, default=3e-3)
    common.add_argument("--steps", type=int, default=12000)
    for a, _ in SIZE_KEYS:
        common.add_argument("--" + a.replace("_", "-"), dest=a, type=int, default=None,
                            help="模型尺寸(默认 ToyConfig;c/bscan 从 --init 继承)")

    def model_args(p):
        p.add_argument("--merge", default="sat", choices=["sat", "and", "learn"], help="残差合并方式")
        p.add_argument("--levels", type=int, default=1, help="每 token 最多几个脉冲")

    p1 = sub.add_parser("a1", parents=[common])
    model_args(p1)
    p1.add_argument("--fq", type=float, default=0.75, help="阈值标定分位数")
    p1.set_defaults(fn=cmd_a1)
    pc = sub.add_parser("c", parents=[common])
    pc.add_argument("--init", required=True, help="A1 的 model.pt(merge、levels、fq、尺寸从同目录 result.json 继承)")
    pc.add_argument("--lif-scale", type=float, default=1.0, help="训练前只把 LIF 阈值乘以该系数")
    pc.set_defaults(fn=cmd_c)
    pi = sub.add_parser("iscan", parents=[common])
    model_args(pi)
    pi.add_argument("--fq-list", default="0.5,0.75,0.9")
    pi.add_argument("--save", default=None)
    pi.set_defaults(fn=cmd_iscan)
    pb = sub.add_parser("bscan", parents=[common])
    pb.add_argument("--init", required=True)
    pb.add_argument("--scale-list", default="1,1.5,2,2.25,2.5,3,4")
    pb.add_argument("--save", default=None)
    pb.set_defaults(fn=cmd_bscan)
    pn = sub.add_parser("chain", parents=[common])
    model_args(pn)
    pn.add_argument("--fq-list", default="0.5,0.75,0.9")
    pn.add_argument("--scale-list", default="1,1.5,2,2.25,2.5,3,4")
    pn.set_defaults(fn=cmd_chain)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
