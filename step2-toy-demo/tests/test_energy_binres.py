"""能耗事件计数对二值残差模型的处理:类型识别、输出头输入、比较器判决数;三值对照 satc 仍按三值残差计。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import numpy as np
import pytest
import torch

from experiments.energy_events import count_events, model_kind, model_matrices
from toy_demo.binary_residual import BinaryResidualLM, TernaryClampResidualLM
from toy_demo.data import all_sequences, split_inputs_targets
from toy_demo.model import SpikingLM


def _lower_thresholds(m):
    with torch.no_grad():
        m.q_emb.log_thr.fill_(-1.0)
        for lif in m.lif_modules():
            lif._thr.log_thr.fill_(-1.0)


@torch.no_grad()
@pytest.mark.parametrize("merge,levels", [("sr", 1), ("src", 2), ("xor", 2)])
def test_binres_events_count_unsigned_pulses_and_only_the_embedding_comparators(cfg, sl, merge, levels):
    torch.manual_seed(0)
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.9, merge=merge, levels=levels)
    _lower_thresholds(m)
    tok = all_sequences(sl)[:8]
    ev = count_events(m, tok, torch.device("cpu"))
    assert model_kind(m) == "binres" and ev["levels"] == levels
    assert ev["cmp_decisions"] == cfg.d * levels                      # 只有 q_emb,无符号,每通道 T 个比较器
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    L = inputs.shape[1]
    hf = torch.stack([recs[t][-1]["h_final"] for t in range(L)], 1)[mask]
    assert np.allclose(ev["rows"]["head"], (hf * levels).round().double().mean(0).numpy())
    x0 = torch.stack([recs[t][0]["x_attn"] for t in range(L)], 1)[mask]
    assert np.allclose(ev["rows"]["blocks.0.qkv"], (x0 * levels).round().double().mean(0).numpy())
    assert ev["rows"]["head"].sum() > 0 and ev["spikes_lif"] > 0
    mats = model_matrices(m)
    assert mats["head"].shape == (sl.vocab_size, cfg.d) and mats["embed"].shape == (cfg.d, sl.vocab_size)


def test_model_kind_separates_the_three_families(cfg, sl):
    assert model_kind(SpikingLM(cfg, sl.vocab_size, leak=0.9)) == "spiking"
    ctl = TernaryClampResidualLM(cfg, sl.vocab_size, leak=0.9, levels=2)
    _lower_thresholds(ctl)
    assert model_kind(ctl) == "ternres"
    ev = count_events(ctl, all_sequences(sl)[:4], torch.device("cpu"))
    assert ev["cmp_decisions"] == (cfg.d + cfg.n_blocks * 2 * cfg.d) * 2 * 2
