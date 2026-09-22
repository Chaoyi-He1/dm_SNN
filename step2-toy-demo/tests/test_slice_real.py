from pathlib import Path

import pytest

from toy_demo.config import ToyConfig
from toy_demo.data import load_slice, select_demo

SLICE = Path(__file__).resolve().parents[1] / ToyConfig.slice_dir


@pytest.mark.skipif(not SLICE.exists(), reason="切片未构建:python -m toy_demo.build_slice")
def test_real_slice_loads_and_meets_plan():
    cfg = ToyConfig()
    sl = load_slice(SLICE)
    assert len(sl.ids) >= cfg.target_sentences
    assert all(cfg.min_len <= len(r) <= cfg.max_len for r in sl.ids)
    assert 1000 <= sl.vocab_size <= 6000
    assert sl.meta["license"] == "CDLA-Sharing-1.0" and len(sl.meta["tokenizer_revision"]) == 40
    demo = select_demo(sl, cfg)
    assert len(demo) == cfg.n_demo
    assert all(t.strip() for d in demo for t in [d["text"]])
