"""转换实验台:在 toy_demo 的 A0 -> A1 -> B -> C 路径上增加可记录的旋钮(计划第 5 节:三个种子都失败后才调整超参数并记录)。

在 step2-toy-demo 目录执行(任意 cwd 也可,脚本自行切换):
    .venv/bin/python experiments/convert_lab.py a1 --name fq50 --fq 0.5 --lr 3e-3 --steps 6000
    .venv/bin/python experiments/convert_lab.py c  --name fq50_c --init results/conversion-sweep/fq50/model.pt --steps 4000 --leak-anneal 1000
    .venv/bin/python experiments/convert_lab.py stats --fq 0.75          # A0/A1_init 的门、发放率、h 量程、||k||^2

旋钮(均写入 result.json 的 args,与 toy-demo.json 的 config 同为记录):
    --fq            阈值标定分位数(计划默认 0.75)
    --lr/--steps    学习率与步数上限;--schedule cosine|const|cosine_floor;--warmup N
    --kd W --kd-T T 蒸馏:loss = (1-W)*CE + W*T^2*KL(A0 || 模型)
    --refit         A1 初始化时按前向顺序用最小二乘把每个线性层重拟合到 A0 的预激活(输入换成脉冲后),再标定阈值
    --gate-refit    只重拟合门的线性层 Wa/Wb
    --leak-anneal N C 阶段:leak 从 0 线性升到 cfg.leak,用 N 步
    --recalib Q     C 阶段训练前在 leak=cfg.leak 下按分位数 Q 重新标定阈值
不改动 toy_demo 包本身;所有结果写到 --out/<name>/。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
sys.path.insert(0, str(STEP2))

from toy_demo.config import ToyConfig                                   # noqa: E402
from toy_demo.convert import _block_points, _set_thr, calibrate_thresholds, set_leak, transfer_weights  # noqa: E402
from toy_demo.data import all_sequences, batches, decode, load_slice, select_demo, split_inputs_targets  # noqa: E402
from toy_demo.model import FloatLM, SpikingLM                            # noqa: E402
from toy_demo.train import _should_stop, evaluate                         # noqa: E402


# ----------------------------------------------------------------------------- 环境

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_env(**overrides):
    cfg = ToyConfig(**overrides)
    sl = load_slice(cfg.slice_dir)
    demo = select_demo(sl, cfg)
    tokens = all_sequences(sl)
    return cfg, sl, demo, tokens


def load_a0(cfg, sl, dev, path="results/ckpt/a0.pt"):
    m = FloatLM(cfg, sl.vocab_size).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()


def build_a1(a0, cfg, sl, tokens, dev, fq, seed=0, leak=0.0, refit=False, gate_refit=False):
    """A0 -> A1:权重转移 + (可选)最小二乘重拟合 + 阈值标定。返回 (模型, 阈值字典)。"""
    torch.manual_seed(seed)
    m = SpikingLM(cfg, sl.vocab_size, leak=leak).to(dev)
    transfer_weights(a0, m)
    calib = tokens[:cfg.calib_sentences].to(dev)
    if refit or gate_refit:
        thr = refit_and_calibrate(a0, m, calib, fq, full=refit)
    else:
        thr = calibrate_thresholds(m, calib, fq)
    return m, thr


# ----------------------------------------------------------------------------- 最小二乘重拟合

def _lstsq(X, Z, ridge=1e-3, bias=False):
    """min ||X W^T + b - Z||^2 (+ ridge)。X [N, in], Z [N, out] -> (W [out, in], b [out] 或 None)。"""
    X = X.double()
    Z = Z.double()
    if bias:
        X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
    A = X.T @ X + ridge * X.shape[0] * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    W = torch.linalg.solve(A, X.T @ Z).T                 # [out, in(+1)]
    if bias:
        return W[:, :-1].float(), W[:, -1].float()
    return W.float(), None


def _gather(recs, block_index, key, mask):
    """recs[t][block_index][key] -> 在有效位置上拼成 [N_valid, C]。"""
    return torch.stack([recs[t][block_index][key] for t in range(len(recs))], dim=1)[mask]


@torch.no_grad()
def refit_and_calibrate(a0, m, tokens, fq, full=True, ridge=1e-3):
    """按前向顺序:对每个线性层,用 A1 当前的脉冲输入 X 与 A0 的预激活 Z 做最小二乘重拟合,然后标定紧随其后的阈值。
    full=False 时只重拟合门(Wa、Wb)。A0 的记录在同样的 token 上取一次。"""
    inputs, _, mask = split_inputs_targets(tokens)
    _, ref = a0(inputs, record=True)
    out = {}

    def run():
        _, recs = m(inputs, record=True)
        return recs

    def set_linear(lin, W, b=None):
        lin.weight.copy_(W)
        if b is not None and lin.bias is not None:
            lin.bias.copy_(b)

    for i, blk in enumerate(m.blocks):
        a, f = blk.attn, blk.ffn
        a0b = a0.blocks[i]
        # q_in1
        recs = run()
        thr = _q(_gather(recs, i, "h_in", mask).abs(), fq)
        _set_thr(blk.q_in1, thr); out[f"blocks.{i}.q_in1"] = thr
        recs = run()
        x1 = _gather(recs, i, "x_attn", mask)
        if full:
            for name, lin in (("Wq", a.Wq), ("Wk", a.Wk), ("Wv", a.Wv)):
                W, _ = _lstsq(x1, _gather(ref, i, "z" + name[1].lower(), mask), ridge)
                set_linear(lin, W)
        # 门:目标为 A0 的 sigmoid 前逻辑值
        for gname in ("Wa", "Wb"):
            lin0, lin = getattr(a0b.attn.gates, gname), getattr(a.gates, gname)
            h0 = _gather(ref, i, "h_in", mask)
            W, b = _lstsq(x1, lin0(h0), ridge, bias=True)
            set_linear(lin, W, b)
        # 神经元阈值 q, k, v(依次)
        for name, mod, fn, ternary in _block_points(blk)[1:4]:
            recs = run()
            z = torch.stack([fn(recs[t][i]) for t in range(len(recs))], dim=1)[mask]
            thr = _q(z.abs() if ternary else z, fq)
            _set_thr(mod, thr); out[f"blocks.{i}.{name}"] = thr
        # pre
        recs = run()
        o = _gather(recs, i, "o", mask).reshape(mask.sum().item(), -1)
        thr = _q(o.abs(), fq); _set_thr(a.sn_pre, thr); out[f"blocks.{i}.pre"] = thr
        # Wo
        recs = run()
        if full:
            o_spk = _gather(recs, i, "o_spk", mask)
            W, _ = _lstsq(o_spk, _gather(ref, i, "zo", mask), ridge)
            set_linear(a.Wo, W)
            recs = run()
        y = torch.stack([a.Wo(recs[t][i]["o_spk"]) for t in range(len(recs))], dim=1)[mask]
        thr = _q(y.abs(), fq); _set_thr(a.sn_out, thr); out[f"blocks.{i}.out"] = thr
        # q_in2
        recs = run()
        thr = _q(_gather(recs, i, "h_mid", mask).abs(), fq); _set_thr(blk.q_in2, thr); out[f"blocks.{i}.q_in2"] = thr
        recs = run()
        x2 = _gather(recs, i, "x_ffn", mask)
        if full:
            for name, lin in (("z1", f.W1), ("z3", f.W3)):
                W, _ = _lstsq(x2, _gather(ref, i, name, mask), ridge)
                set_linear(lin, W)
        for name, mod, fn, ternary in _block_points(blk)[7:9]:
            recs = run()
            z = torch.stack([fn(recs[t][i]) for t in range(len(recs))], dim=1)[mask]
            thr = _q(z.abs() if ternary else z, fq); _set_thr(mod, thr); out[f"blocks.{i}.{name}"] = thr
        recs = run()
        if full:
            p = _gather(recs, i, "p", mask)
            W, _ = _lstsq(p, _gather(ref, i, "z2", mask), ridge)
            set_linear(f.W2, W)
            recs = run()
        y2 = torch.stack([f.W2(recs[t][i]["p"]) for t in range(len(recs))], dim=1)[mask]
        thr = _q(y2.abs(), fq); _set_thr(f.sn_2, thr); out[f"blocks.{i}.y2"] = thr
    # 输出头
    L = len(m.blocks)
    recs = run()
    hf = _gather(recs, L, "h_final", mask)
    thr = _q(hf.abs(), fq); _set_thr(m.q_out, thr); out["q_out"] = thr
    if full:
        recs = run()
        xq = torch.stack([m.q_out(recs[t][L]["h_final"]) for t in range(len(recs))], dim=1)[mask]
        target = a0.head(_gather(ref, L, "h_final", mask))
        W, _ = _lstsq(xq, target, ridge)
        set_linear(m.head, W)
    return out


def _q(z, quantile):
    return max(torch.quantile(z.flatten().float(), quantile).item(), 1e-3)


@torch.no_grad()
def all_threshold_modules(m):
    """全部阈值模块(21 个):每块 10 个脉冲点 + 输出头前的 q_out。"""
    mods = []
    for blk in m.blocks:
        mods += [mod for _, mod, _, _ in _block_points(blk)]
    return mods + [m.q_out]


@torch.no_grad()
def scale_thresholds(m, factor, only=None):
    """把阈值乘以 factor;only 为模块子集时只缩放这些模块。"""
    for mod in (only or all_threshold_modules(m)):
        target = mod.log_thr if hasattr(mod, "log_thr") else mod._thr.log_thr
        target.add_(math.log(factor))


# ----------------------------------------------------------------------------- 训练

def lr_schedule(kind, base, steps, warmup):
    def f(step):
        if warmup and step < warmup:
            return base * (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        if kind == "const":
            return base
        if kind == "cosine":
            return base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
        if kind == "cosine_floor":
            return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))
        raise ValueError(kind)
    return f


def train(model, sl, demo, cfg, tokens, *, lr, steps, seed, schedule="cosine", warmup=0, kd=None, kd_w=0.0, kd_T=2.0,
          leak_anneal=0, leak_from=0.0, leak_to=None, eval_every=100, stop=True, log=print, hist_path=None):
    """Adam + 可选调度 + 梯度裁剪;每 eval_every 步评估;满足计划的停止条件(5 句全对且平台)或到 steps 停止。"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = model.device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_at = lr_schedule(schedule, lr, steps, warmup)
    leak_to = cfg.leak if leak_to is None else leak_to
    history, step, t0 = [], 0, time.time()
    model.train()
    hist_file = open(hist_path, "a", encoding="utf-8") if hist_path else None
    while step < steps:
        for batch in batches(sl, cfg.batch_size, g):
            leak = None
            if leak_anneal:
                leak = leak_from + (leak_to - leak_from) * min(1.0, step / leak_anneal)
                set_leak(model, leak)
            cur_lr = lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            inputs, targets, mask = split_inputs_targets(batch.to(dev))
            logits = model(inputs)
            ce = F.cross_entropy(logits[mask], targets[mask])
            loss = ce
            if kd is not None and kd_w > 0:
                with torch.no_grad():
                    tl = kd(inputs)[mask]
                kl = F.kl_div(F.log_softmax(logits[mask] / kd_T, -1), F.softmax(tl / kd_T, -1),
                              reduction="batchmean") * kd_T ** 2
                loss = (1 - kd_w) * ce + kd_w * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                ev = evaluate(model, sl, demo, cfg, tokens)
                entry = dict(step=step, loss=round(loss.item(), 4), ce=round(ce.item(), 4), token_acc=ev["token_acc"],
                             ppl=ev["ppl"], n_exact=ev["n_exact"], seconds=round(time.time() - t0, 1),
                             lr=cur_lr, leak=leak)
                history.append(entry)
                log(json.dumps(entry))
                if hist_file:
                    hist_file.write(json.dumps(entry) + "\n"); hist_file.flush()
                if stop and _should_stop(history, cfg):
                    model.eval()
                    return dict(history=history, final=ev, stopped="converged", seed=seed)
            if step >= steps:
                break
    model.eval()
    return dict(history=history, final=evaluate(model, sl, demo, cfg, tokens), stopped="max_steps", seed=seed)


# ----------------------------------------------------------------------------- 统计

@torch.no_grad()
def model_stats(model, tokens, dev, is_float=False):
    """门分布、每个脉冲点的发放率、h 量程、||k||^2。tokens 含 BOS/EOS/PAD。"""
    inputs, _, mask = split_inputs_targets(tokens.to(dev))
    _, recs = model(inputs, record=True)
    out = {}
    L = len(model.blocks)
    for i in range(L):
        r = {k: torch.stack([recs[t][i][k] for t in range(len(recs))], dim=1)[mask] for k in recs[0][i] if k != "S"}
        st = dict(alpha_mean=r["alpha"].mean().item(), alpha_q=[round(v, 3) for v in torch.quantile(r["alpha"].flatten(), torch.tensor([.05, .5, .95], device=dev)).tolist()],
                  beta_mean=r["beta"].mean().item(), beta_q=[round(v, 3) for v in torch.quantile(r["beta"].flatten(), torch.tensor([.05, .5, .95], device=dev)).tolist()],
                  h_in_absmax=r["h_in"].abs().max().item(), h_mid_absmax=r["h_mid"].abs().max().item(), h_out_absmax=r["h_out"].abs().max().item())
        if not is_float:
            for k in ("x_attn", "q", "k", "v", "o_spk", "y_attn", "x_ffn", "g", "u", "p", "y_ffn"):
                st[f"rate_{k}"] = round((r[k] != 0).float().mean().item(), 4)
            st["k_sqnorm_mean"] = round((r["k"] ** 2).sum(-1).mean().item(), 4)      # 已含 1/dk 缩放
        out[f"block{i}"] = st
    hf = torch.stack([recs[t][L]["h_final"] for t in range(len(recs))], dim=1)[mask]
    out["h_final_absmax"] = hf.abs().max().item()
    if not is_float:
        out["rate_q_out"] = round((model.q_out(hf) != 0).float().mean().item(), 4)
    return out


# ----------------------------------------------------------------------------- 命令

def _save(out_dir, model, result, args, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    payload = dict(args={k: v for k, v in vars(args).items() if k != "fn"}, torch=torch.__version__,
                   device=str(model.device), **(extra or {}), **result)
    (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    f = result["final"]
    print(f"[{args.stage}:{args.name}] acc {f['token_acc']:.4f} ppl {f['ppl']:.3f} exact {f['n_exact']}/{f['n_demo']} "
          f"stopped {result.get('stopped')} ({result.get('seconds', '?')} s)")


def cmd_a1(args):
    torch.set_num_threads(1)
    dev = get_device()
    cfg, sl, demo, tokens = load_env(fire_quantile=args.fq)
    a0 = load_a0(cfg, sl, dev, args.a0)
    t0 = time.time()
    m, thr = build_a1(a0, cfg, sl, tokens, dev, args.fq, seed=args.seed, refit=args.refit, gate_refit=args.gate_refit)
    init = evaluate(m, sl, demo, cfg, tokens)
    print(f"[a1_init:{args.name}] acc {init['token_acc']:.4f} ppl {init['ppl']:.3f} exact {init['n_exact']}/{init['n_demo']}")
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = train(m, sl, demo, cfg, tokens, lr=args.lr, steps=args.steps, seed=args.seed, schedule=args.schedule,
                warmup=args.warmup, kd=a0 if args.kd > 0 else None, kd_w=args.kd, kd_T=args.kd_T,
                eval_every=args.eval_every, stop=not args.no_stop, hist_path=out_dir / "history.jsonl")
    res["seconds"] = round(time.time() - t0, 1)
    _save(out_dir, m, res, args, dict(init=init, thresholds=thr))


def cmd_c(args):
    torch.set_num_threads(1)
    dev = get_device()
    cfg, sl, demo, tokens = load_env(fire_quantile=args.fq)
    a0 = load_a0(cfg, sl, dev, args.a0)
    t0 = time.time()
    m = SpikingLM(cfg, sl.vocab_size, leak=cfg.leak).to(dev)
    m.load_state_dict(torch.load(args.init, map_location=dev))
    b = evaluate(m, sl, demo, cfg, tokens)
    print(f"[b:{args.name}] acc {b['token_acc']:.4f} ppl {b['ppl']:.3f} exact {b['n_exact']}/{b['n_demo']}")
    extra = dict(b=b)
    if args.recalib is not None:
        extra["thresholds_recalib"] = calibrate_thresholds(m, tokens[:cfg.calib_sentences].to(dev), args.recalib)
        extra["b_recalib"] = evaluate(m, sl, demo, cfg, tokens)
        print(f"[b_recalib:{args.name}] acc {extra['b_recalib']['token_acc']:.4f} ppl {extra['b_recalib']['ppl']:.3f}")
    if args.thr_scale != 1.0:
        scale_thresholds(m, args.thr_scale)
        extra["b_thr_scale"] = evaluate(m, sl, demo, cfg, tokens)
        print(f"[b_thr_scale:{args.name}] x{args.thr_scale} acc {extra['b_thr_scale']['token_acc']:.4f} ppl {extra['b_thr_scale']['ppl']:.3f}")
    if args.leak_anneal:
        set_leak(m, 0.0)
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = train(m, sl, demo, cfg, tokens, lr=args.lr, steps=args.steps, seed=args.seed, schedule=args.schedule,
                warmup=args.warmup, kd=a0 if args.kd > 0 else None, kd_w=args.kd, kd_T=args.kd_T,
                leak_anneal=args.leak_anneal, leak_from=0.0, leak_to=cfg.leak,
                eval_every=args.eval_every, stop=not args.no_stop, hist_path=out_dir / "history.jsonl")
    set_leak(m, cfg.leak)
    res["final"] = evaluate(m, sl, demo, cfg, tokens)      # 以 leak=cfg.leak 的最终评估为准
    res["seconds"] = round(time.time() - t0, 1)
    _save(out_dir, m, res, args, extra)


def cmd_stats(args):
    dev = get_device()
    cfg, sl, demo, tokens = load_env(fire_quantile=args.fq)
    a0 = load_a0(cfg, sl, dev, args.a0)
    sub = tokens[:args.n]
    out = dict(a0=model_stats(a0, sub, dev, is_float=True))
    m, thr = build_a1(a0, cfg, sl, tokens, dev, args.fq, refit=args.refit, gate_refit=args.gate_refit)
    out["a1_init"] = model_stats(m, sub, dev)
    out["a1_init_eval"] = {k: evaluate(m, sl, demo, cfg, tokens)[k] for k in ("token_acc", "ppl", "n_exact")}
    out["thresholds"] = thr
    if args.init:
        m2 = SpikingLM(cfg, sl.vocab_size, leak=args.leak).to(dev)
        m2.load_state_dict(torch.load(args.init, map_location=dev))
        out["loaded"] = model_stats(m2, sub, dev)
        out["loaded_eval"] = {k: evaluate(m2, sl, demo, cfg, tokens)[k] for k in ("token_acc", "ppl", "n_exact")}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", required=False, default="run")
    common.add_argument("--out", default="results/conversion-sweep")
    common.add_argument("--a0", default="results/ckpt/a0.pt")
    common.add_argument("--fq", type=float, default=0.75)
    common.add_argument("--lr", type=float, default=None, help="默认 a1 3e-3、c 1e-3(计划第 5 节)")
    common.add_argument("--steps", type=int, default=2000)
    common.add_argument("--schedule", default="cosine", choices=["cosine", "const", "cosine_floor"])
    common.add_argument("--warmup", type=int, default=0)
    common.add_argument("--kd", type=float, default=0.0)
    common.add_argument("--kd-T", type=float, default=2.0)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--eval-every", type=int, default=100)
    common.add_argument("--no-stop", action="store_true", help="忽略停止条件,跑满步数")
    p1 = sub.add_parser("a1", parents=[common])
    p1.add_argument("--refit", action="store_true")
    p1.add_argument("--gate-refit", action="store_true")
    p1.set_defaults(fn=cmd_a1)
    pc = sub.add_parser("c", parents=[common])
    pc.add_argument("--init", required=True, help="A1 的 model.pt(或 results/ckpt/a1.pt)")
    pc.add_argument("--leak-anneal", type=int, default=0)
    pc.add_argument("--recalib", type=float, default=None)
    pc.add_argument("--thr-scale", type=float, default=1.0, help="训练前把全部阈值乘以该系数(LIF 与 Q_in 都乘)")
    pc.set_defaults(fn=cmd_c)
    ps = sub.add_parser("stats", parents=[common])
    ps.add_argument("--n", type=int, default=512, help="统计用的句子数")
    ps.add_argument("--refit", action="store_true")
    ps.add_argument("--gate-refit", action="store_true")
    ps.add_argument("--init", default=None, help="额外统计一个已保存的脉冲模型")
    ps.add_argument("--leak", type=float, default=0.0, help="--init 模型的 leak")
    ps.set_defaults(fn=cmd_stats)
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-3 if args.stage == "c" else 3e-3
    args.fn(args)


if __name__ == "__main__":
    main()
