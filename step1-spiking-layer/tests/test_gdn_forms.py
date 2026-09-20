"""递推规范:形式 A(先衰减,再检索,再写入,再读出)。含单步判别、手算例子、与 HF 参考实现的等价性。"""
import math
import torch
import pytest

from snn_spec.gdn import (
    gdn_step, transition_matrix, hf_reference_recurrence,
    SpikingGDN, QwenNativeGates, SigmoidGates, ExternalGates,
)


def _unit(i, n):
    e = torch.zeros(n)
    e[i] = 1.0
    return e


def test_single_step_discrimination_A_B_C_give_1p25_1p00_0p75():
    """S1.4 的单步判别:S0 只在 (0,0) 为 1;k=e0,v0=2,q=e0,alpha=beta=0.5。"""
    n = 16
    S0 = torch.zeros(n, n); S0[0, 0] = 1.0
    k = _unit(0, n); q = _unit(0, n)
    v = torch.zeros(n); v[0] = 2.0
    SA, _ = gdn_step(S0, k, v, q, alpha=0.5, beta=0.5, form="A")
    SB, _ = gdn_step(S0, k, v, q, alpha=0.5, beta=0.5, form="B")
    SC, _ = gdn_step(S0, k, v, q, alpha=0.5, beta=0.5, form="C")
    assert abs(SA[0, 0].item() - 1.25) < 1e-6
    assert abs(SB[0, 0].item() - 1.00) < 1e-6
    assert abs(SC[0, 0].item() - 0.75) < 1e-6


def test_form_A_matches_closed_form_alpha_I_minus_beta_kk_S_plus_beta_kv():
    torch.manual_seed(0)
    dk, dv = 8, 6
    S = torch.randn(dk, dv); k = torch.randn(dk); v = torch.randn(dv); q = torch.randn(dk)
    a, b = 0.8, 0.4
    S1, o = gdn_step(S, k, v, q, alpha=a, beta=b, form="A")
    A = transition_matrix(k, alpha=a, beta=b, form="A")
    assert torch.allclose(A, a * (torch.eye(dk) - b * torch.outer(k, k)))
    assert torch.allclose(S1, A @ S + b * torch.outer(k, v), atol=1e-6)
    assert torch.allclose(o, S1.T @ q, atol=1e-6)  # 读出读的是更新后的 S


def test_forms_B_and_C_transition_matrices():
    torch.manual_seed(1)
    dk = 5
    k = torch.randn(dk); a, b = 0.7, 0.5
    I = torch.eye(dk)
    assert torch.allclose(transition_matrix(k, a, b, "B"), a * I - b * torch.outer(k, k))
    assert torch.allclose(transition_matrix(k, a, b, "C"), a * (I - b * torch.outer(k, k)))
    # C 与 A 的转移矩阵相同,差别只在写入项多乘一个 alpha
    S = torch.randn(dk, 3); v = torch.randn(3); q = torch.randn(dk)
    SA, _ = gdn_step(S, k, v, q, a, b, "A")
    SC, _ = gdn_step(S, k, v, q, a, b, "C")
    assert torch.allclose(SC - SA, (a - 1) * b * torch.outer(k, v), atol=1e-6)


def test_hand_example_two_orthogonal_keys_correct_order_reads_0_3():
    """对话里的手算例子:正确顺序用 ka 读出 (0,3);先检索再衰减(形式 B)读出 (-0.5, 2.5)。"""
    ka = torch.tensor([1.0, 0.0]); kb = torch.tensor([0.0, 1.0])
    S = torch.zeros(2, 2)
    S, _ = gdn_step(S, ka, torch.tensor([1.0, 1.0]), ka, alpha=1.0, beta=1.0, form="A")
    S, _ = gdn_step(S, kb, torch.tensor([2.0, 0.0]), kb, alpha=1.0, beta=1.0, form="A")
    assert torch.allclose(S, torch.tensor([[1.0, 1.0], [2.0, 0.0]]))
    S3, o = gdn_step(S, ka, torch.tensor([0.0, 3.0]), ka, alpha=0.5, beta=1.0, form="A")
    assert torch.allclose(o, torch.tensor([0.0, 3.0]))
    _, o_wrong = gdn_step(S, ka, torch.tensor([0.0, 3.0]), ka, alpha=0.5, beta=1.0, form="B")
    assert torch.allclose(o_wrong, torch.tensor([-0.5, 2.5]))


def test_core_reproduces_hf_torch_recurrent_gated_delta_rule():
    """SpikingGDN.core 在连续输入上逐 token 复现 HF Qwen3-Next/3.5 的参考递推
    (l2norm(q,k)、q 乘 dk^-1/2、alpha=exp(g))。"""
    torch.manual_seed(2)
    B, H, L, dk, dv = 2, 3, 7, 8, 6
    q = torch.randn(B, H, L, dk); k = torch.randn(B, H, L, dk); v = torch.randn(B, H, L, dv)
    g = -torch.rand(B, H, L) * 2.0            # log alpha in (-2, 0)
    beta = torch.rand(B, H, L)
    o_ref, S_ref = hf_reference_recurrence(q, k, v, g, beta)

    def l2n(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    S = torch.zeros(B, H, dk, dv)
    outs = []
    for t in range(L):
        qt = l2n(q[:, :, t]) * dk ** -0.5
        kt = l2n(k[:, :, t])
        o, S = SpikingGDN.core(kt, v[:, :, t], qt, g[:, :, t].exp(), beta[:, :, t], S)
        outs.append(o)
    o_ours = torch.stack(outs, dim=2)
    assert torch.allclose(o_ours, o_ref, atol=1e-5)
    assert torch.allclose(S, S_ref, atol=1e-5)


def test_qwen_native_gates_functional_form_and_alpha_clamp():
    torch.manual_seed(3)
    d, Hv = 16, 4
    gates = QwenNativeGates(d, Hv, alpha_rng=(0.5, 0.99))
    x = torch.randint(-1, 2, (5, d)).float()
    a, b = gates(x)
    assert a.shape == (5, Hv) and b.shape == (5, Hv)
    assert torch.all(a >= 0.5) and torch.all(a <= 0.99)
    assert torch.all(b > 0) and torch.all(b < 1)
    # 函数形式:alpha = clamp(exp(-exp(A_log) * softplus(a_proj x + dt_bias))), beta = sigmoid(b_proj x)
    raw_a = torch.exp(-gates.A_log.exp() * torch.nn.functional.softplus(gates.a_proj(x) + gates.dt_bias))
    assert torch.allclose(a, raw_a.clamp(0.5, 0.99))
    assert torch.allclose(b, torch.sigmoid(gates.b_proj(x)))


def test_sigmoid_gates_range():
    gates = SigmoidGates(16, 4, alpha_rng=(0.5, 0.99))
    a, b = gates(torch.randn(3, 16) * 10)
    assert torch.all(a >= 0.5) and torch.all(a <= 0.99) and torch.all(b > 0) and torch.all(b < 1)


def test_external_gates_pass_through_given_values():
    gates = ExternalGates()
    a0 = torch.full((2, 3), 0.9); b0 = torch.full((2, 3), 0.3)
    a, b = gates(torch.zeros(2, 8), ext=(a0, b0))
    assert torch.equal(a, a0) and torch.equal(b, b0)


def test_spiking_gdn_step_shapes_codes_and_fixed_k_scaling():
    torch.manual_seed(4)
    d, Hk, Hv, dk, dv = 32, 2, 4, 8, 8
    m = SpikingGDN(d, n_k_heads=Hk, n_v_heads=Hv, dk=dk, dv=dv)
    B = 3
    st = m.init_state(B, torch.device("cpu"))
    x = torch.randint(-1, 2, (B, d)).float()
    y, st, rec = m.step(x, st, record=True)
    assert y.shape == (B, d) and set(y.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert st["S"].shape == (B, Hv, dk, dv)
    assert set(rec["q"].unique().tolist()) <= {0.0, 1.0}
    assert set(rec["v"].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert set(rec["o_spk"].unique().tolist()) <= {-1.0, 0.0, 1.0}
    k = rec["k"]                                  # 已缩放的 k: [B, Hv, dk]
    assert k.shape == (B, Hv, dk)
    assert torch.all((k * k).sum(-1) <= 1.0 + 1e-6)          # ||k||^2 = n/dk <= 1
    nz = (k != 0).float().sum(-1)
    assert torch.allclose((k * k).sum(-1), nz / dk, atol=1e-6)
    assert torch.allclose(k.abs()[k != 0], torch.full_like(k.abs()[k != 0], dk ** -0.5))


def test_spiking_gdn_repeats_kv_heads_for_grouped_query():
    torch.manual_seed(5)
    m = SpikingGDN(d=16, n_k_heads=1, n_v_heads=2, dk=4, dv=4)
    st = m.init_state(2, torch.device("cpu"))
    x = torch.randint(-1, 2, (2, 16)).float()
    _, _, rec = m.step(x, st, record=True)
    assert torch.equal(rec["q"][:, 0], rec["q"][:, 1])
    assert torch.equal(rec["k"][:, 0], rec["k"][:, 1])


def test_spiking_gdn_uses_external_gates_when_given():
    torch.manual_seed(6)
    m = SpikingGDN(d=16, n_k_heads=2, n_v_heads=2, dk=4, dv=4, gates=ExternalGates())
    st = m.init_state(1, torch.device("cpu"))
    x = torch.randint(-1, 2, (1, 16)).float()
    a0 = torch.full((1, 2), 0.8); b0 = torch.full((1, 2), 0.25)
    _, _, rec = m.step(x, st, ext_gates=(a0, b0), record=True)
    assert torch.equal(rec["alpha"], a0) and torch.equal(rec["beta"], b0)


def test_spiking_gdn_optional_fir_carries_history():
    torch.manual_seed(7)
    m = SpikingGDN(d=16, n_k_heads=2, n_v_heads=2, dk=4, dv=4, fir_taps=4)
    st = m.init_state(1, torch.device("cpu"))
    assert st["fir_q"].shape == (1, 3, 8)   # taps-1 个历史, Hk*dk 通道
    x = torch.randint(-1, 2, (1, 16)).float()
    _, st, _ = m.step(x, st, record=True)
    assert torch.allclose(st["fir_q"][:, -1], m.Wq(x))  # 最新一步的投影进了历史缓存
