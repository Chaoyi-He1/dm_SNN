"""一次性构建 TinyStories 切片(需要网络;依赖 requirements-data.txt)。
在 step2-toy-demo 目录执行:.venv/bin/python -m toy_demo.build_slice [--target 4000] [--out data/tinystories-slice]
"""
import argparse
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

from .config import ToyConfig
from .data import make_slice, save_slice, sha256_of

REPO = "Qwen/Qwen3.5-0.8B"
DATASET = "roneneldan/TinyStories"
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def build(target, out_dir, revision="main", cfg=None):
    cfg = cfg or ToyConfig()
    sha = HfApi().model_info(REPO, revision=revision).sha
    tok_path = hf_hub_download(REPO, "tokenizer.json", revision=sha)
    tok = Tokenizer.from_file(tok_path)
    sentences, ids, n_stories, n_dropped = [], [], 0, 0
    for story in load_dataset(DATASET, split="train", streaming=True):
        n_stories += 1
        for s in SENT_SPLIT.split(story["text"].strip()):
            s = " ".join(s.split())
            if not s:
                continue
            enc = tok.encode(s, add_special_tokens=False).ids
            if cfg.min_len <= len(enc) <= cfg.max_len:
                sentences.append(s)
                ids.append(enc)
            else:
                n_dropped += 1
        if len(sentences) >= target:
            break
    meta = dict(source=f"https://huggingface.co/datasets/{DATASET}", split="train", license="CDLA-Sharing-1.0",
                n_stories=n_stories, n_sentences=len(sentences), n_dropped=n_dropped,
                min_len=cfg.min_len, max_len=cfg.max_len, tokenizer_repo=REPO, tokenizer_revision=sha,
                tokenizer_sha256=sha256_of(tok_path), sentence_split=SENT_SPLIT.pattern)
    sl = make_slice(sentences, ids, lambda q: tok.decode([q]), meta)
    save_slice(sl, out_dir)
    (Path(out_dir) / "README.md").write_text(readme(meta, target), encoding="utf-8")
    return sl


def readme(m, target):
    return (f"# TinyStories 切片\n\n"
            f"来源:{m['source']},{m['split']} split,按流式读取开头 {m['n_stories']} 篇故事。许可 {m['license']}。\n\n"
            f"分句正则 `{m['sentence_split']}`;保留 {m['min_len']}–{m['max_len']} 个 token 的句子,"
            f"共 {m['n_sentences']} 句,丢弃 {m['n_dropped']} 句。\n\n"
            f"分词器:`{m['tokenizer_repo']}` 修订 `{m['tokenizer_revision']}`,`tokenizer.json` SHA-256 `{m['tokenizer_sha256']}`。\n\n"
            f"文件:`slice.json`(句子、紧凑 id、紧凑 id 到 Qwen id 的映射、每个 token 的文本、元数据),`SHA256SUMS`。\n\n"
            f"重建:`.venv/bin/python -m toy_demo.build_slice --target {target}`\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=ToyConfig.target_sentences)
    ap.add_argument("--out", default=ToyConfig.slice_dir)
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    sl = build(a.target, a.out, a.revision)
    print(f"{len(sl.ids)} sentences, vocab {sl.vocab_size}, stories {sl.meta['n_stories']}")


if __name__ == "__main__":
    main()
