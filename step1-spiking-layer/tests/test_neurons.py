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


# ----------------------------------------------------------------------------- 每 token 多个脉冲(levels=T)

def _count_ref(U, thr, T):
    """参考定义:T 个子步,每次 U >= thr/T 发一个并减 thr/T;返回 c/T。"""
    out = torch.zeros_like(U)
    for i in range(U.numel()):
        u, c = U.flatten()[i].item(), 0
        for _ in range(T):
            if u >= thr / T - 1e-12:
                c += 1; u -= thr / T
        out.view(-1)[i] = c / T
    return out


def test_levels_binary_counts_match_subslot_process():
    from snn_spec.neurons import level_binary_fire, level_ternary_fire
    torch.manual_seed(0)
    U = torch.randn(400) * 2
    for T in (2, 4, 8):
        s = level_binary_fire(U, torch.tensor(1.0), T)
        assert torch.allclose(s, _count_ref(U.clamp(min=0), 1.0, T))
        st = level_ternary_fire(U, torch.tensor(1.0), T)
        assert torch.allclose(st, _count_ref(U.clamp(min=0), 1.0, T) - _count_ref((-U).clamp(min=0), 1.0, T))
        assert set((s * T).round().unique().tolist()) <= set(range(T + 1))


def test_levels_one_is_bit_identical_to_original():
    from snn_spec.neurons import level_ternary_fire
    torch.manual_seed(1)
    U = torch.randn(200)
    assert torch.equal(level_ternary_fire(U, torch.tensor(0.7), 1), ternary_fire(U, torch.tensor(0.7)))
    n1, n2 = TernaryLIF(0.5, 0.9), TernaryLIF(0.5, 0.9, levels=1)
    s1, u1 = n1(U, torch.zeros_like(U)); s2, u2 = n2(U, torch.zeros_like(U))
    assert torch.equal(s1, s2) and torch.equal(u1, u2)


def test_levels_lif_soft_reset_subtracts_c_times_thr_over_T():
    n = TernaryLIF(1.0, 0.9, levels=4)
    U0 = torch.tensor([0.8, -0.3, 1.9, 0.24])
    s, U = n(U0, torch.zeros(4))
    assert torch.equal(s, torch.tensor([0.75, -0.25, 1.0, 0.0]))
    assert torch.allclose(U, U0 - s)                    # 减 c 个 thr/T
    q = QuantizerIn(1.0, levels=2)
    assert torch.equal(q(torch.tensor([0.49, 0.5, 1.0, -0.6])), torch.tensor([0.0, 0.5, 1.0, -0.5]))


def test_levels_surrogate_gradient_integral_is_independent_of_T():
    from snn_spec.neurons import level_binary_fire
    # arctan 代理梯度是 Lorentzian(尺度 1/π),尾巴很长:区间取 [-60, 60],截掉的尾巴每侧约 1/(60·π²) ≈ 0.0017
    U = torch.linspace(-60, 60, 240001, requires_grad=True)
    du = 120 / 240000
    integrals = []
    for T in (1, 2, 4):
        s = level_binary_fire(U, torch.tensor(1.0), T)
        (g,) = torch.autograd.grad(s.sum(), U)
        integrals.append(g.sum().item() * du)
    assert all(abs(v - integrals[0]) < 1e-3 for v in integrals)      # 与 T 无关
    assert all(abs(v - 1.0) < 0.01 for v in integrals)                # 等于台阶的总高度 1
