"""demo 命令(计划第 6 节)。在 step2-toy-demo 目录执行:
    .venv/bin/python -m toy_demo.demo                          # 5 个前缀,各阶段补全 + 指标表
    .venv/bin/python -m toy_demo.demo --prompt "The little dog"  # 新前缀(需要 requirements-data.txt)
"""
import argparse
import json
from pathlib import Path

import torch

from .circuit import CircuitLM
from .config import ToyConfig
from .convert import set_leak
from .data import decode, load_slice, select_demo
from .model import FloatLM, SpikingLM


def load_models(cfg, sl, results_dir):
    ck = Path(results_dir) / "ckpt"
    V = sl.vocab_size

    def load(m, name):
        m.load_state_dict(torch.load(ck / f"{name}.pt", map_location="cpu"))
        return m.eval()

    a0 = load(FloatLM(cfg, V), "a0")
    a1 = load(SpikingLM(cfg, V, leak=0.0), "a1")
    b = SpikingLM(cfg, V, leak=cfg.leak)
    b.load_state_dict(a1.state_dict())
    b.eval()
    c = load(SpikingLM(cfg, V, leak=cfg.leak), "c")
    return dict(A0=a0, A1=a1, B=b, C=c, circuit=CircuitLM(c, cfg, ideal=False))


def show_demo(models, sl, demo, cfg):
    for d in demo:
        print(f"前缀: {decode(sl, d['prefix'])!r}")
        print(f"目标: {decode(sl, d['target'])!r}")
        for name, m in models.items():
            out = m.generate(d["prefix"], cfg.max_new_tokens)
            mark = "一致" if out == list(d["target"]) else "不一致"
            print(f"  {name:8s} {mark:4s} {decode(sl, out)!r}")
        print()


def show_metrics(results_dir):
    p = Path(results_dir) / "toy-demo.json"
    if not p.exists():
        print("(没有 toy-demo.json;先运行 python -m toy_demo.run)")
        return
    r = json.loads(p.read_text(encoding="utf-8"))
    print("| 阶段 | token 准确率 | 困惑度 | 补全正确 |")
    print("|---|---|---|---|")
    for name, s in r["stages"].items():
        f = s["final"]
        print(f"| {name} | {f['token_acc']:.4f} | {f['ppl']:.3f} | {f['n_exact']}/{f['n_demo']} |")


def _load_tokenizer(meta):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(meta["tokenizer_repo"], "tokenizer.json", revision=meta["tokenizer_revision"])
    return Tokenizer.from_file(path)


def encode_prompt(text, sl):
    """用 Qwen 分词器处理新前缀并映射到紧凑索引;有词表外 token 时退出并列出。"""
    tok = _load_tokenizer(sl.meta)
    ids = tok.encode(text, add_special_tokens=False).ids
    lookup = {q: i for i, q in enumerate(sl.compact_to_qwen) if q >= 0}
    unknown = [tok.decode([q]) for q in ids if q not in lookup]
    if unknown:
        raise SystemExit(f"这些 token 不在紧凑词表里,模型无法处理:{unknown}")
    return [lookup[q] for q in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ToyConfig.results_dir)
    ap.add_argument("--prompt", default=None)
    a = ap.parse_args()
    cfg = ToyConfig()
    sl = load_slice(cfg.slice_dir)
    models = load_models(cfg, sl, a.results)
    if a.prompt:
        prefix = encode_prompt(a.prompt, sl)
        for name in ("C", "circuit"):
            print(f"{name:8s} {decode(sl, models[name].generate(prefix, cfg.max_new_tokens))!r}")
        return
    show_demo(models, sl, select_demo(sl, cfg), cfg)
    show_metrics(a.results)


if __name__ == "__main__":
    main()
