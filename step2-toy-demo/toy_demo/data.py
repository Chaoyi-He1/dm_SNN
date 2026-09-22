"""切片的读写、紧凑词表、批处理与 demo 句子选择。切片文件格式见 data/tinystories-slice/README.md。

紧凑索引:0/1/2 是 BOS/EOS/PAD(不对应任何 Qwen id,解码时忽略),其后按首次出现顺序排列切片里用到的 Qwen token。
"""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

BOS, EOS, PAD = 0, 1, 2
N_SPECIAL = 3
SPECIAL_TEXT = ["<bos>", "<eos>", "<pad>"]
SLICE_FILE = "slice.json"
SUMS_FILE = "SHA256SUMS"


@dataclass
class Slice:
    sentences: list                 # 原始句子文本
    ids: list                       # 每句的紧凑 id 列表(不含特殊 token)
    compact_to_qwen: list           # 紧凑 id -> Qwen id;特殊 token 为 -1
    token_text: list                # 紧凑 id -> 文本片段;特殊 token 为 "<bos>" 等
    meta: dict = field(default_factory=dict)

    @property
    def vocab_size(self):
        return len(self.compact_to_qwen)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_slice(sentences, qwen_ids, qwen_to_text, meta):
    """从句子文本、每句的 Qwen id 列表和 Qwen id -> 文本片段 的函数建立紧凑索引。"""
    compact_to_qwen = [-1] * N_SPECIAL
    token_text = list(SPECIAL_TEXT)
    lookup, ids = {}, []
    for seq in qwen_ids:
        row = []
        for q in seq:
            if q not in lookup:
                lookup[q] = len(compact_to_qwen)
                compact_to_qwen.append(int(q))
                token_text.append(qwen_to_text(q))
            row.append(lookup[q])
        ids.append(row)
    return Slice(list(sentences), ids, compact_to_qwen, token_text, dict(meta))


def save_slice(sl, slice_dir):
    d = Path(slice_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = dict(sentences=sl.sentences, ids=sl.ids, compact_to_qwen=sl.compact_to_qwen,
                   token_text=sl.token_text, meta=sl.meta)
    (d / SLICE_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (d / SUMS_FILE).write_text(f"{sha256_of(d / SLICE_FILE)}  {SLICE_FILE}\n", encoding="utf-8")


def load_slice(slice_dir):
    """读切片并核对 SHA-256;不一致时抛 ValueError。"""
    d = Path(slice_dir)
    expected = (d / SUMS_FILE).read_text(encoding="utf-8").split()[0]
    actual = sha256_of(d / SLICE_FILE)
    if actual != expected:
        raise ValueError(f"{d / SLICE_FILE}: SHA-256 {actual} != {expected}")
    sl = Slice(**json.loads((d / SLICE_FILE).read_text(encoding="utf-8")))
    assert sl.compact_to_qwen[:N_SPECIAL] == [-1] * N_SPECIAL
    return sl


def encode_sequence(ids):
    return [BOS] + list(ids) + [EOS]


def decode(sl, ids):
    return "".join(sl.token_text[i] for i in ids if i >= N_SPECIAL)


def pad_sequences(seqs, pad=PAD):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def all_sequences(sl):
    """整个切片:tokens [N, L],含 BOS/EOS,PAD 补齐。"""
    return pad_sequences([encode_sequence(s) for s in sl.ids])


def batches(sl, batch_size, generator):
    """随机打乱后按 batch 产出 tokens [B, L];每个 epoch 调一次。"""
    order = torch.randperm(len(sl.ids), generator=generator).tolist()
    for i in range(0, len(order), batch_size):
        yield pad_sequences([encode_sequence(sl.ids[j]) for j in order[i:i + batch_size]])


def split_inputs_targets(tokens):
    """tokens [B, L] -> (inputs [B, L-1], targets [B, L-1], mask [B, L-1]:目标不是 PAD 的位置)。"""
    return tokens[:, :-1], tokens[:, 1:], tokens[:, 1:] != PAD


def select_demo(sl, cfg):
    """计划第 3 节:长度 demo_len_min–demo_len_max,前 demo_prefix_len 个 token 的 n-gram 在整个切片里只出现一次;
    固定种子抽 n_demo 句。返回 [dict(index, prefix, target, text)]。"""
    n = cfg.demo_prefix_len
    counts = {}
    for row in sl.ids:
        for i in range(len(row) - n + 1):
            g = tuple(row[i:i + n])
            counts[g] = counts.get(g, 0) + 1
    cands = [i for i, row in enumerate(sl.ids)
             if cfg.demo_len_min <= len(row) <= cfg.demo_len_max and counts.get(tuple(row[:n]), 0) == 1]
    gen = torch.Generator().manual_seed(cfg.demo_seed)
    order = torch.randperm(len(cands), generator=gen).tolist()
    out = []
    for j in order[:cfg.n_demo]:
        row = sl.ids[cands[j]]
        out.append(dict(index=cands[j], prefix=row[:n], target=row[n:], text=sl.sentences[cands[j]]))
    if len(out) < cfg.n_demo:
        raise ValueError(f"only {len(out)} demo candidates, need {cfg.n_demo}")
    return out


def synthetic_slice(n_sentences=64, vocab=20, seed=0, min_len=6, max_len=16):
    """测试用:随机 token 序列的合成切片;token 文本为 ' w<i>',Qwen id 伪造为 1000+i。"""
    g = torch.Generator().manual_seed(seed)
    sentences, qwen_ids = [], []
    for _ in range(n_sentences):
        L = int(torch.randint(min_len, max_len + 1, (1,), generator=g))
        row = torch.randint(0, vocab, (L,), generator=g).tolist()
        qwen_ids.append([1000 + t for t in row])
        sentences.append("".join(f" w{t}" for t in row))
    return make_slice(sentences, qwen_ids, lambda q: f" w{q - 1000}", dict(source="synthetic", seed=seed))
