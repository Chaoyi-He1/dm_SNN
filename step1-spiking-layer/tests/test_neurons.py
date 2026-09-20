"""脉冲点(编码器)的单元测试:与稳定性测试分开,只检查给定输入下的输出脉冲。"""
import math
import torch
import pytest

from snn_spec.neurons import SpikeFn, ternary_fire, BinaryLIF, TernaryLIF, QuantizerIn

SG = 2.0  # 代理梯度宽度(BiSpikCLM 的 alpha=2;这里改名以免与 GDN 的门 alpha 混淆)


def test_spikefn_forward_is_heaviside_with_fire_at_zero():
    u = torch.tensor([-1.0, -1e-6, 0.0, 0.5])
    assert torch.equal(SpikeFn.apply(u, SG), torch.tensor([0.0, 0.0, 1.0, 1.0]))


def test_spikefn_backward_is_arctan_surrogate():
    u = torch.tensor([-1.0, 0.0, 0.3, 2.0], requires_grad=True)
    SpikeFn.apply(u, SG).sum().backward()
    expected = (SG / 2) / (1 + ((math.pi / 2) * SG * u.detach()) ** 2)
    assert torch.allclose(u.grad, expected)


def test_ternary_fire_maps_to_minus_one_zero_plus_one():
    U = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    out = ternary_fire(U, torch.tensor(1.0), SG)
    assert torch.equal(out, torch.tensor([-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]))


def test_binary_lif_leaks_integrates_and_soft_resets():
    n = BinaryLIF(thr=1.0, leak=0.9)
    U = torch.zeros(1)
    S, U = n(torch.tensor([0.6]), U)
    assert S.item() == 0.0 and abs(U.item() - 0.6) < 1e-6
    S, U = n(torch.tensor([0.6]), U)          # 0.9*0.6 + 0.6 = 1.14 >= 1 → 发放,软复位减阈值
    assert S.item() == 1.0 and abs(U.item() - 0.14) < 1e-6


def test_binary_lif_never_emits_negative_spikes():
    n = BinaryLIF(thr=1.0, leak=0.9)
    S, U = n(torch.tensor([-5.0]), torch.zeros(1))
    assert S.item() == 0.0 and abs(U.item() + 5.0) < 1e-6


def test_ternary_lif_negative_branch_fires_and_resets_toward_zero():
    n = TernaryLIF(thr=1.0, leak=0.9)
    S, U = n(torch.tensor([-1.5]), torch.zeros(1))
    assert S.item() == -1.0 and abs(U.item() + 0.5) < 1e-6   # -1.5 - (1.0 * -1) = -0.5


def test_ternary_lif_positive_branch():
    n = TernaryLIF(thr=1.0, leak=0.9)
    S, U = n(torch.tensor([1.5]), torch.zeros(1))
    assert S.item() == 1.0 and abs(U.item() - 0.5) < 1e-6


def test_lif_threshold_fixed_by_default_and_learnable_on_request():
    fixed = BinaryLIF(thr=0.7, leak=0.9)
    assert len(list(fixed.parameters())) == 0
    learn = BinaryLIF(thr=0.7, leak=0.9, learn_thr=True)
    assert len(list(learn.parameters())) == 1
    assert abs(learn.thr.item() - 0.7) < 1e-6
    I = torch.tensor([0.6, 0.8, 1.2])
    S, _ = learn(I, torch.zeros(3))
    S.sum().backward()
    assert next(learn.parameters()).grad is not None


def test_quantizer_in_is_stateless_ternary_threshold():
    q = QuantizerIn(thr0=1.0)
    h = torch.tensor([-3.0, -0.2, 0.0, 0.2, 3.0])
    assert torch.equal(q(h), torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(q(h), q(h))  # 无状态:同一输入两次结果相同


def test_quantizer_in_threshold_is_learnable_and_stays_positive():
    q = QuantizerIn(thr0=1.0)
    assert abs(q.thr.item() - 1.0) < 1e-6
    h = torch.tensor([-3.0, 0.9, 1.1, 3.0])   # 非对称输入,避免正负支路的梯度恰好抵消
    q(h).sum().backward()
    (p,) = list(q.parameters())
    assert p.grad is not None and p.grad.abs().sum() > 0
    with torch.no_grad():
        p.fill_(-50.0)  # 参数任意取值,阈值仍为正
    assert q.thr.item() > 0
