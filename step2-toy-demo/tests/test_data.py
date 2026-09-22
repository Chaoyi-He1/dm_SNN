import pytest
import torch

from toy_demo.data import (BOS, EOS, PAD, make_slice, save_slice, load_slice, encode_sequence, decode,
                           all_sequences, batches, split_inputs_targets, select_demo, synthetic_slice)


def test_make_slice_assigns_compact_ids_in_first_seen_order():
    sl = make_slice(["a b", "b c"], [[500, 600], [600, 700]], lambda q: f"<{q}>", {})
    assert sl.compact_to_qwen == [-1, -1, -1, 500, 600, 700]
    assert sl.ids == [[3, 4], [4, 5]]
    assert sl.token_text[:3] == ["<bos>", "<eos>", "<pad>"] and sl.token_text[3] == "<500>"
    assert sl.vocab_size == 6


def test_save_load_roundtrip_and_hash_check(tmp_path):
    sl = synthetic_slice(8)
    save_slice(sl, tmp_path)
    back = load_slice(tmp_path)
    assert back.ids == sl.ids and back.compact_to_qwen == sl.compact_to_qwen and back.token_text == sl.token_text
    p = tmp_path / "slice.json"
    p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        load_slice(tmp_path)


def test_encode_decode():
    sl = synthetic_slice(4)
    seq = encode_sequence(sl.ids[0])
    assert seq[0] == BOS and seq[-1] == EOS
    assert decode(sl, seq) == sl.sentences[0]


def test_all_sequences_padding_and_mask():
    sl = synthetic_slice(6)
    tok = all_sequences(sl)
    assert tok.shape == (6, max(len(s) for s in sl.ids) + 2)
    inp, tgt, mask = split_inputs_targets(tok)
    assert inp.shape == tgt.shape == mask.shape == (6, tok.shape[1] - 1)
    assert mask.sum().item() == sum(len(s) + 1 for s in sl.ids)      # 每句 len+1 个目标(含 EOS)
    assert (tgt[~mask] == PAD).all()


def test_batches_cover_every_sentence_once():
    sl = synthetic_slice(10)
    g = torch.Generator().manual_seed(0)
    assert sum(b.shape[0] for b in batches(sl, 4, g)) == 10


def test_select_demo_prefixes_are_unique_5grams(cfg):
    sl = synthetic_slice(64)
    demo = select_demo(sl, cfg)
    assert len(demo) == cfg.n_demo
    n = cfg.demo_prefix_len
    for d in demo:
        assert len(d["prefix"]) == n
        assert cfg.demo_len_min <= n + len(d["target"]) <= cfg.demo_len_max
        assert sl.ids[d["index"]] == d["prefix"] + d["target"]
        hits = sum(1 for row in sl.ids for i in range(len(row) - n + 1) if row[i:i + n] == d["prefix"])
        assert hits == 1
    assert select_demo(sl, cfg) == demo          # 固定种子可复现
