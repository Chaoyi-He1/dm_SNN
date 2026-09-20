"""验收指标的定义:按正负分的 precision/recall、符号错误率、两种归一化的状态误差、在环替换用的一致率。"""
import math
import torch

from snn_spec.metrics import spike_prf_by_sign, sign_error_rate, state_error, spike_agreement


def test_prf_perfect_match_is_one():
    ref = torch.tensor([[1.0, 0.0, -1.0, 0.0, 1.0]])
    m = spike_prf_by_sign(ref, ref)
    assert m["pos_precision"] == 1.0 and m["pos_recall"] == 1.0
    assert m["neg_precision"] == 1.0 and m["neg_recall"] == 1.0


def test_prf_counts_events_not_zeros():
    ref = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    ckt = torch.tensor([[1.0, 1.0, 0.0, 0.0]])   # 一个假正,一个漏负
    m = spike_prf_by_sign(ckt, ref)
    assert math.isclose(m["pos_precision"], 0.5) and m["pos_recall"] == 1.0
    assert m["neg_recall"] == 0.0


def test_all_zero_circuit_is_not_rewarded():
    ref = torch.tensor([[1.0, 0.0, -1.0, 1.0]])
    m = spike_prf_by_sign(torch.zeros_like(ref), ref)
    assert m["pos_recall"] == 0.0 and m["neg_recall"] == 0.0


def test_sign_error_rate_flipped_spikes():
    ref = torch.tensor([[1.0, 0.0, -1.0, 1.0]])
    assert sign_error_rate(-ref, ref) == 1.0
    assert sign_error_rate(ref, ref) == 0.0
    assert sign_error_rate(torch.tensor([[1.0, 0.0, 1.0, 1.0]]), ref) == 1.0 / 3.0


def test_state_error_two_normalizations():
    S_sw = torch.zeros(4, 4); S_sw[0, 0] = 2.0
    S_ckt = S_sw.clone(); S_ckt[0, 0] = 2.5
    e = state_error(S_ckt, S_sw, bound=10.0, ref_scale=2.0)
    assert math.isclose(e["by_bound"], 0.5 / 10.0)
    assert math.isclose(e["by_scale"], 0.5 / 2.0)


def test_spike_agreement_rate():
    ref = torch.tensor([[1.0, 0.0, -1.0, 0.0]])
    ckt = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert math.isclose(spike_agreement(ckt, ref), 0.75)
