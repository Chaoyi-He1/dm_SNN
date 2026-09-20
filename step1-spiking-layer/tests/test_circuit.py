"""行为级电路原语(S1.2 六参数神经元、S1.3 gain cell 相位模型、S2.3 的 crossbar/比较器/锁存)
与按 φ0–φ8 相位表执行的行为级仿真器。理想参数下必须与 S0.9 的软件层逐脉冲一致。"""
import math
import torch
import pytest

from snn_spec.block import SpikingBlock, run_sequence
from snn_spec.gdn import SpikingGDN
from snn_spec.metrics import spike_prf_by_sign
from snn_spec.circuit import (
    DMNeuronModel, IdealLIFModel, GainCellArray, CrossbarModel, ComparatorModel,
    BehavioralBlock, run_behavioral_sequence, fit_dm_neuron, alpha_from_tgap,
)


def _e(i, n):
    v = torch.zeros(1, 1, n); v[0, 0, i] = 1.0
    return v


# ----------------------------------------------------------------------------- gain cell 阵列与单步判别

def _single_step(**kw):
    arr = GainCellArray(B=1, H=1, dk=16, dv=16, **kw)
    S0 = torch.zeros(1, 1, 16, 16); S0[0, 0, 0, 0] = 1.0
    arr.reset(S0)
    a = torch.full((1, 1), 0.5); b = torch.full((1, 1), 0.5)
    _, S1 = arr.cycle(k=_e(0, 16), v=2 * _e(0, 16), q=_e(0, 16), alpha=a, beta=b)
    return S1[0, 0, 0, 0].item()


def test_gain_cell_leak_only_in_phi1_gives_form_A_1p25():
    assert abs(_single_step() - 1.25) < 1e-6


def test_gain_cell_leak_after_write_gives_form_C_0p75():
    assert abs(_single_step(leak_before_retrieve=0.0) - 0.75) < 1e-6


def test_gain_cell_retrieve_before_decay_gives_form_B_1p00():
    assert abs(_single_step(retrieve_before_decay=True) - 1.00) < 1e-6


def test_gain_cell_uniform_leak_lands_between_A_and_C():
    x = _single_step(leak_before_retrieve=0.5)
    assert 0.75 < x < 1.25


def test_gain_cell_cycle_equals_core_for_random_inputs():
    torch.manual_seed(0)
    B, H, dk, dv = 2, 3, 8, 6
    arr = GainCellArray(B, H, dk, dv)
    S0 = torch.randn(B, H, dk, dv); arr.reset(S0)
    k = torch.randint(-1, 2, (B, H, dk)).float() / math.sqrt(dk)
    v = torch.randint(-1, 2, (B, H, dv)).float()
    q = torch.randint(0, 2, (B, H, dk)).float()
    a = torch.rand(B, H) * 0.5 + 0.5; b = torch.rand(B, H)
    o_ref, S_ref = SpikingGDN.core(k, v, q, a, b, S0)
    o, S = arr.cycle(k, v, q, a, b)
    assert torch.allclose(o, o_ref, atol=1e-6) and torch.allclose(S, S_ref, atol=1e-6)


def test_gain_cell_write_and_read_errors_are_applied():
    torch.manual_seed(1)
    arr = GainCellArray(1, 1, 16, 16, write_err=0.05, read_err=0.05, seed=3)
    S0 = torch.zeros(1, 1, 16, 16); S0[0, 0, 0, 0] = 1.0
    arr.reset(S0)
    a = torch.full((1, 1), 0.5); b = torch.full((1, 1), 0.5)
    o, S1 = arr.cycle(k=_e(0, 16), v=2 * _e(0, 16), q=_e(0, 16), alpha=a, beta=b)
    assert abs(S1[0, 0, 0, 0].item() - 1.25) > 1e-4
    assert abs(S1[0, 0, 0, 0].item() - 1.25) < 0.2


def test_alpha_from_tgap():
    assert abs(alpha_from_tgap(t_gap=1.0, tau_leak=1.0) - math.exp(-1)) < 1e-9
    assert abs(alpha_from_tgap(t_gap=0.0, tau_leak=1.0) - 1.0) < 1e-9


# ----------------------------------------------------------------------------- 神经元模型

def test_ideal_lif_model_matches_spec_soft_reset():
    n = IdealLIFModel(thr=1.0, leak=0.9, ternary=False)
    st = n.init_state((1,))
    S, st = n.step(torch.tensor([0.6]), st); assert S.item() == 0.0
    S, st = n.step(torch.tensor([0.6]), st)
    assert S.item() == 1.0 and abs(st["U"].item() - 0.14) < 1e-6


def test_dm_neuron_hard_reset_to_hold_voltage():
    n = DMNeuronModel(beta_m=0.9, g_m=1.0, theta=1.0, U_hold=0.2, rho=0.0, n_ref=0)
    st = n.init_state((1,))
    S, st = n.step(torch.tensor([1.5]), st)
    assert S.item() == 1.0 and abs(st["U"].item() - 0.2) < 1e-6


def test_dm_neuron_partial_reset_rho():
    n = DMNeuronModel(beta_m=0.9, g_m=1.0, theta=1.0, U_hold=0.2, rho=0.5, n_ref=0)
    st = n.init_state((1,))
    S, st = n.step(torch.tensor([1.6]), st)         # U_pre = 1.6 → 0.2 + 0.5*(1.6-1.0) = 0.5
    assert S.item() == 1.0 and abs(st["U"].item() - 0.5) < 1e-6


def test_dm_neuron_refractory_blocks_firing():
    n = DMNeuronModel(beta_m=0.9, g_m=1.0, theta=1.0, U_hold=0.0, rho=0.0, n_ref=2)
    st = n.init_state((1,))
    fired = []
    for _ in range(5):
        S, st = n.step(torch.tensor([2.0]), st)
        fired.append(S.item())
    assert fired == [1.0, 0.0, 0.0, 1.0, 0.0]       # 发放后两个周期不应期


def test_dm_neuron_ternary_negative_branch():
    n = DMNeuronModel(beta_m=0.9, g_m=1.0, theta=1.0, U_hold=0.2, rho=0.0, n_ref=0, ternary=True)
    st = n.init_state((1,))
    S, st = n.step(torch.tensor([-1.5]), st)
    assert S.item() == -1.0 and abs(st["U"].item() + 0.2) < 1e-6


def test_fit_dm_neuron_recovers_parameters_from_synthetic_trace():
    """S1.2 的拟合流程:随机电流序列驱动、记录周期边界膜电压与发放事件、最小二乘拟合。
    真值由模型本身生成;拟合后在新序列上按正负分统计 precision/recall >= 0.95。"""
    torch.manual_seed(2)
    true = dict(beta_m=0.85, g_m=1.3, theta=0.8, U_hold=0.1, rho=0.3, n_ref=1)
    n = DMNeuronModel(**true, ternary=True)
    N, T = 64, 400
    I = torch.randn(T, N) * 0.6
    st = n.init_state((N,))
    U_hist, S_hist = [st["U"].clone()], []
    for t in range(T):
        S, st = n.step(I[t], st)
        U_hist.append(st["U"].clone()); S_hist.append(S)
    fit = fit_dm_neuron(I, torch.stack(U_hist), torch.stack(S_hist), ternary=True)
    assert abs(fit["beta_m"] - true["beta_m"]) < 0.02
    assert abs(fit["g_m"] - true["g_m"]) < 0.02
    assert abs(fit["theta"] - true["theta"]) < 0.05
    assert abs(fit["U_hold"] - true["U_hold"]) < 0.05
    assert abs(fit["rho"] - true["rho"]) < 0.05
    assert fit["n_ref"] == true["n_ref"]
    m = DMNeuronModel(**{k: fit[k] for k in true}, ternary=True)
    I2 = torch.randn(200, N) * 0.6
    st_a, st_b = n.init_state((N,)), m.init_state((N,))
    A, Bv = [], []
    for t in range(200):
        sa, st_a = n.step(I2[t], st_a); sb, st_b = m.step(I2[t], st_b)
        A.append(sa); Bv.append(sb)
    prf = spike_prf_by_sign(torch.stack(Bv), torch.stack(A))
    assert min(prf["pos_precision"], prf["pos_recall"], prf["neg_precision"], prf["neg_recall"]) >= 0.95


# ----------------------------------------------------------------------------- crossbar、比较器

def test_crossbar_ideal_equals_linear():
    torch.manual_seed(3)
    lin = torch.nn.Linear(16, 8, bias=False)
    xb = CrossbarModel(lin.weight.detach())
    x = torch.randint(-1, 2, (4, 16)).float()
    assert torch.allclose(xb(x), lin(x))


def test_crossbar_quantization_and_programming_error():
    torch.manual_seed(4)
    W = torch.randn(8, 16)
    xq = CrossbarModel(W, n_levels=4)
    assert len(torch.unique(xq.W_eff.abs())) <= 5          # 每个差分列 4 级电导 + 0
    xn = CrossbarModel(W, sigma_g=0.1, seed=1)
    assert not torch.allclose(xn.W_eff, W) and (xn.W_eff - W).abs().max() < 0.5 * W.abs().max()


def test_comparator_offset_shifts_threshold():
    c = ComparatorModel(thr=1.0)
    assert torch.equal(c(torch.tensor([-1.5, 0.5, 1.5])), torch.tensor([-1.0, 0.0, 1.0]))
    c2 = ComparatorModel(thr=1.0, offset=0.6)                # 有效阈值 1.6 / -0.4
    assert torch.equal(c2(torch.tensor([-1.5, 0.5, 1.5])), torch.tensor([-1.0, 0.0, 0.0]))


# ----------------------------------------------------------------------------- 行为级仿真器 = 软件层(理想参数)

def _block(seed):
    torch.manual_seed(seed)
    blk = SpikingBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4, thr=0.5)
    with torch.no_grad():
        for m in (blk.attn.Wq, blk.attn.Wk, blk.attn.Wv, blk.attn.Wo, blk.ffn.W1, blk.ffn.W3, blk.ffn.W2):
            m.weight.mul_(4.0)                                   # 让神经元真的发放
    return blk


def test_behavioral_block_with_ideal_primitives_reproduces_software_block():
    blk = _block(5)
    h_seq = torch.randn(2, 12, 16) * 2
    sw, sw_st = run_sequence(blk, h_seq)
    sim = BehavioralBlock(blk)
    hw, hw_st, recs = run_behavioral_sequence(sim, h_seq, record=True)
    assert torch.equal(hw, sw)
    assert torch.allclose(hw_st["attn"]["S"], sw_st["attn"]["S"], atol=1e-6)
    assert any(r["y_attn"].abs().sum() > 0 for r in recs) and any(r["y_ffn"].abs().sum() > 0 for r in recs)


def test_behavioral_block_always_on_leak_deviates_from_software():
    blk = _block(6)
    h_seq = torch.randn(1, 12, 16) * 2
    sw, sw_st = run_sequence(blk, h_seq)
    sim = BehavioralBlock(blk, gain_cell_kw=dict(leak_before_retrieve=0.0))
    hw, hw_st, _ = run_behavioral_sequence(sim, h_seq, record=True)
    assert not torch.allclose(hw_st["attn"]["S"], sw_st["attn"]["S"], atol=1e-6)


def test_behavioral_block_accepts_external_gates_and_dm_neurons():
    blk = _block(7)
    h_seq = torch.randn(1, 6, 16) * 2
    _, _, recs = run_sequence(blk, h_seq, record=True)
    alpha = torch.stack([r["alpha"] for r in recs], dim=1); beta = torch.stack([r["beta"] for r in recs], dim=1)
    sim = BehavioralBlock(blk, neuron_factory=lambda thr, leak, ternary:
                          DMNeuronModel(beta_m=leak, g_m=1.0, theta=thr, U_hold=0.0, rho=0.0, n_ref=0, ternary=ternary))
    hw, _, hrecs = run_behavioral_sequence(sim, h_seq, ext_gates_seq=(alpha, beta), record=True)
    assert hw.shape == h_seq.shape
    assert torch.equal(torch.stack([r["alpha"] for r in hrecs], dim=1), alpha)
