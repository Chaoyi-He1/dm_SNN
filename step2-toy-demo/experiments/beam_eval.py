"""束搜索解码与补全评估(改进方向 3:换解码方式)。

束搜索:批量执行,N 个前缀 × K 条束放在同一个 batch 里,状态(GDN 的 S、所有膜电位)按父束下标重排。
已经输出 EOS 的束冻结(分数不变,留在束里与未完成的束比较);排名分数 = 累计 log 概率 / 长度^alpha
(alpha = 0 即原始联合概率)。K = 1 时与 toy_demo 的贪心解码逐 token 相同(测试见 --selftest)。

评估集:
  demo        计划第 3 节的 5 句 demo(前缀 5 个 token)
  unique      前 5 个 token 的 5-gram 在整个切片里只出现一次、且长度 >= 6 的全部句子:目标唯一,看补全是否逐 token 等于原句
  any         全部长度 >= 6 的句子按前缀去重:补全后的整句只要是语料里任意一句(同前缀)就算对,衡量"有没有说出一句背过的完整句子"

用法(在 step2-toy-demo 下):
  .venv/bin/python experiments/beam_eval.py --models A0=results/ckpt/a0.pt,C=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt \
      --beams 1,2,4,8 --alphas 0,1 --save results/improve/beam/base.json
模型类型由检查点同目录的 result.json(args.stage、levels、尺寸)决定;results/ckpt/{a0,a1,c}.pt 按文件名。
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
sys.path.insert(0, str(STEP2 / "experiments"))

from toy_demo.config import ToyConfig                                         # noqa: E402
from toy_demo.data import BOS, EOS, PAD, decode, load_slice, select_demo     # noqa: E402
from toy_demo.model import FloatLM, SpikingLM, greedy_generate               # noqa: E402
from multilevel import apply_levels                                          # noqa: E402

SIZE_KEYS = (("d", "d"), ("dk", "dk"), ("dv", "dv"), ("d_ff", "d_ff"), ("heads", "n_heads"))


# ----------------------------------------------------------------------------- 模型装载

def describe_checkpoint(path):
    """-> dict(kind=float|spiking, leak, levels, level_points, sizes)。"""
    path = Path(path)
    meta = path.parent / "result.json"
    if meta.exists():
        a = json.loads(meta.read_text(encoding="utf-8"))["args"]
        stage = a["stage"]
        sizes = {key: a[k] for k, key in SIZE_KEYS if a.get(k) is not None}
        if stage == "a0":
            return dict(kind="float", sizes=sizes)
        return dict(kind="spiking", leak=0.0 if stage == "a1" else None, levels=int(a.get("levels") or 1),
                    level_points=a.get("level_points") or "all", sizes=sizes)
    name = path.stem
    if name == "a0":
        return dict(kind="float", sizes={})
    if name in ("a1", "c"):
        return dict(kind="spiking", leak=0.0 if name == "a1" else None, levels=1, level_points="all", sizes={})
    raise ValueError(f"cannot tell the model type of {path}")


def load_model(path, sl, dev):
    info = describe_checkpoint(path)
    cfg = ToyConfig(**info["sizes"])
    if info["kind"] == "float":
        m = FloatLM(cfg, sl.vocab_size).to(dev)
    else:
        leak = cfg.leak if info["leak"] is None else info["leak"]
        m = SpikingLM(cfg, sl.vocab_size, leak=leak).to(dev)
        apply_levels(m, info["levels"], info["level_points"])
        info["leak"] = leak
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval(), cfg, info


# ----------------------------------------------------------------------------- 束搜索

def reindex(st, idx):
    """把状态(嵌套的 list/dict,叶子是 batch 维在前的张量)按 idx 重排。"""
    if torch.is_tensor(st):
        return st.index_select(0, idx)
    if isinstance(st, dict):
        return {k: reindex(v, idx) for k, v in st.items()}
    if isinstance(st, (list, tuple)):
        return type(st)(reindex(v, idx) for v in st)
    return st


@torch.no_grad()
def beam_search(model, prefixes, K, max_new, alpha=0.0):
    """prefixes: N 个等长的 id 列表(不含 BOS)。返回 N 个生成的 id 列表(不含 EOS)。"""
    dev = model.device
    N = len(prefixes)
    st = model.init_state(N, dev)
    toks = torch.tensor([[BOS] + list(p) for p in prefixes], device=dev)
    logits = None
    for t in range(toks.shape[1]):
        logits, st, _ = model.step(toks[:, t], st)
    V = logits.shape[-1]
    K = min(K, V)
    rows = torch.arange(N, device=dev).repeat_interleave(K)
    st = reindex(st, rows)
    logits = logits.index_select(0, rows)
    score = torch.full((N, K), -math.inf, device=dev)
    score[:, 0] = 0.0
    length = torch.zeros(N, K, device=dev)
    finished = torch.zeros(N, K, dtype=torch.bool, device=dev)
    seqs = torch.full((N, K, max_new), PAD, dtype=torch.long, device=dev)
    frozen = torch.full((V,), -math.inf, device=dev)
    frozen[EOS] = 0.0
    base = (torch.arange(N, device=dev) * K).unsqueeze(1)
    for s in range(max_new):
        logp = F.log_softmax(logits.float(), -1).view(N, K, V)
        logp = torch.where(finished.unsqueeze(-1), frozen.view(1, 1, V), logp)
        cand = score.unsqueeze(-1) + logp                                     # [N, K, V]
        cand_len = length.unsqueeze(-1) + (~finished).float().unsqueeze(-1)   # 冻结的束长度不变
        rank = cand if alpha == 0 else cand / cand_len.clamp(min=1.0) ** alpha
        top = rank.view(N, K * V).topk(K, dim=-1).indices                      # [N, K]
        parent, tok = top // V, top % V
        score = cand.view(N, K * V).gather(1, top)
        length = cand_len.expand(N, K, V).reshape(N, K * V).gather(1, top)
        was_finished = finished.gather(1, parent)
        seqs = seqs.gather(1, parent.unsqueeze(-1).expand(N, K, max_new)).clone()
        seqs[:, :, s] = torch.where(was_finished, torch.full_like(tok, PAD), tok)
        finished = was_finished | (tok == EOS)
        if bool(finished.all()):
            break
        flat = (base + parent).flatten()
        st = reindex(st, flat)
        feed = torch.where(finished, torch.full_like(tok, EOS), tok).flatten()
        logits, st, _ = model.step(feed, st)
    final = score if alpha == 0 else score / length.clamp(min=1.0) ** alpha
    best = final.argmax(-1)
    out = []
    for n in range(N):
        row = seqs[n, best[n]].tolist()
        gen = []
        for t in row:
            if t in (EOS, PAD):
                break
            gen.append(t)
        out.append(gen)
    return out


def beam_by_length(model, prefixes, K, max_new, alpha, chunk):
    """按前缀长度分组、分块调用 beam_search,返回与 prefixes 同序的结果。"""
    out = [None] * len(prefixes)
    groups = {}
    for i, p in enumerate(prefixes):
        groups.setdefault(len(p), []).append(i)
    for _, idx in groups.items():
        for j in range(0, len(idx), chunk):
            part = idx[j:j + chunk]
            res = beam_search(model, [prefixes[i] for i in part], K, max_new, alpha)
            for i, r in zip(part, res):
                out[i] = r
    return out


# ----------------------------------------------------------------------------- 评估集

def eval_sets(sl, cfg):
    n = cfg.demo_prefix_len
    counts = {}
    for row in sl.ids:
        for i in range(len(row) - n + 1):
            g = tuple(row[i:i + n])
            counts[g] = counts.get(g, 0) + 1
    unique = [i for i, row in enumerate(sl.ids) if len(row) > n and counts.get(tuple(row[:n]), 0) == 1]
    by_prefix = {}
    for row in sl.ids:
        if len(row) > n:
            by_prefix.setdefault(tuple(row[:n]), set()).add(tuple(row[n:]))
    return dict(unique=unique, any=by_prefix)


def run_eval(model, sl, cfg, demo, sets, K, alpha, chunk):
    n = cfg.demo_prefix_len
    mx = cfg.max_new_tokens
    t0 = time.time()
    d_out = beam_by_length(model, [d["prefix"] for d in demo], K, mx, alpha, chunk)
    demo_rows = [dict(prefix=decode(sl, d["prefix"]), target=decode(sl, d["target"]), output=decode(sl, o),
                      exact=(o == list(d["target"]))) for d, o in zip(demo, d_out)]
    u_idx = sets["unique"]
    u_out = beam_by_length(model, [sl.ids[i][:n] for i in u_idx], K, mx, alpha, chunk)
    u_exact = [o == sl.ids[i][n:] for i, o in zip(u_idx, u_out)]
    # 补全中与目标逐 token 一致的前缀长度占目标长度的比例
    def matched(o, tgt):
        k = 0
        while k < min(len(o), len(tgt)) and o[k] == tgt[k]:
            k += 1
        return k / len(tgt)
    u_frac = [matched(o, sl.ids[i][n:]) for i, o in zip(u_idx, u_out)]
    a_prefixes = list(sets["any"].keys())
    a_out = beam_by_length(model, [list(p) for p in a_prefixes], K, mx, alpha, chunk)
    a_ok = [tuple(o) in sets["any"][p] for p, o in zip(a_prefixes, a_out)]
    return dict(K=K, alpha=alpha, demo_exact=sum(r["exact"] for r in demo_rows), demo=demo_rows,
                unique_n=len(u_idx), unique_exact=round(sum(u_exact) / len(u_idx), 4),
                unique_matched_frac=round(sum(u_frac) / len(u_frac), 4),
                any_n=len(a_prefixes), any_corpus_sentence=round(sum(a_ok) / len(a_ok), 4),
                seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------------- 自检

@torch.no_grad()
def selftest(model, sl, cfg, demo):
    """K=1 与贪心逐 token 相同;重排后的状态与单独运行一致。"""
    prefixes = [sl.ids[i][:5] for i in range(0, 200, 7) if len(sl.ids[i]) > 5]
    g = [greedy_generate(model, p, cfg.max_new_tokens) for p in prefixes]
    b = beam_search(model, prefixes, 1, cfg.max_new_tokens)
    mism = sum(x != y for x, y in zip(g, b))
    # 宽束不应比贪心的联合 log 概率更差(K=4、alpha=0)
    return dict(n=len(prefixes), k1_vs_greedy_mismatch=mism)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="name=path[,name=path...]")
    ap.add_argument("--beams", default="1,2,4,8")
    ap.add_argument("--alphas", default="0,1")
    ap.add_argument("--chunk", type=int, default=512, help="每次束搜索的前缀数(行数 = chunk × K)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = ToyConfig()
    sl = load_slice(base_cfg.slice_dir)
    demo = select_demo(sl, base_cfg)
    sets = eval_sets(sl, base_cfg)
    print(f"eval sets: unique {len(sets['unique'])} sentences, any {len(sets['any'])} prefixes", flush=True)
    out = dict(sets=dict(unique=len(sets["unique"]), any=len(sets["any"])), models={})
    for item in args.models.split(","):
        name, path = item.split("=")
        model, cfg, info = load_model(path, sl, dev)
        rec = dict(path=path, info=info, runs=[])
        if args.selftest:
            rec["selftest"] = selftest(model, sl, cfg, demo)
            print(name, "selftest", rec["selftest"], flush=True)
        for K in [int(x) for x in args.beams.split(",")]:
            for alpha in [float(x) for x in args.alphas.split(",")]:
                if K == 1 and alpha != 0:
                    continue          # K=1 时长度归一化不改变选择
                r = run_eval(model, sl, cfg, demo, sets, K, alpha, max(1, args.chunk // K * 2))
                rec["runs"].append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "demo"} | dict(model=name)), flush=True)
        out["models"][name] = rec
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
