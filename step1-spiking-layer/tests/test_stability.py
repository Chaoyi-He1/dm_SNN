"""S0.4 的稳定性命题与压力测试 1–3(直接向递推核心注入 k,v,q 和门值,绕过编码器)。"""
import math
import torch

from snn_spec.gdn import SpikingGDN
from snn_spec.stress import core_stress_tests, state_bound


def test_state_bound_formula():
    assert abs(state_bound(dv=16, alpha_max=0.99) - math.sqrt(16) / (1 - 0.99)) < 1e-9


def test_stress_1_zero_key_follows_alpha_product():
    res = core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, seed=0)
    assert res["zero_key_follows_alpha_product"]


def test_stress_2_all_one_key_bounded():
    res = core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, seed=0)
    assert res["all_one_key_bounded"]


def test_stress_3_dense_random_key_bounded():
    res = core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, seed=0)
    assert res["dense_key_bounded"]


def test_stress_results_report_norm_ratios_below_bound():
    res = core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, seed=0)
    assert res["max_norm_over_bound"] <= 1.0 + 1e-6


def test_unscaled_binary_key_diverges_but_fixed_scaling_does_not():
    """反例:不缩放的全一 key 使 beta*||k||^2 = 8 > 2,状态沿 k 方向放大;固定缩放后有界。"""
    B, H, dk, dv = 1, 1, 16, 16
    a = torch.full((B, H), 0.99); b = torch.full((B, H), 0.5)
    q = torch.ones(B, H, dk)
    v = torch.ones(B, H, dv)
    S_raw = torch.zeros(B, H, dk, dv); S_fix = torch.zeros(B, H, dk, dv)
    k_raw = torch.ones(B, H, dk)
    k_fix = k_raw / math.sqrt(dk)
    norms_raw = []
    for _ in range(40):
        _, S_raw = SpikingGDN.core(k_raw, v, q, a, b, S_raw)
        _, S_fix = SpikingGDN.core(k_fix, v, q, a, b, S_fix)
        norms_raw.append(S_raw.norm().item())
    assert norms_raw[-1] > 1e6 and norms_raw[-1] > norms_raw[10]     # 发散
    assert S_fix.norm().item() <= math.sqrt(dv) / (1 - 0.99) * 1.05    # 命题的界


def test_fast_decay_is_not_mistaken_for_instability():
    res = core_stress_tests(alpha=0.5, beta=0.99)
    assert res["zero_key_follows_alpha_product"]
