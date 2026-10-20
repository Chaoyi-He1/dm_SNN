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


# ----------------------------------------------------------------------------- 每 token 多个脉冲(levels=T)

def test_ideal_lif_levels_equals_sequential_subslots():
    torch.manual_seed(0)
    for T in (2, 4):
        n = IdealLIFModel(thr=1.0, leak=0.9, ternary=True, levels=T)
        st = n.init_state((64,))
        for _ in range(5):
            I = torch.randn(64) * 1.5
            U0 = 0.9 * st["U"] + I
            S, st = n.step(I, st)
            # 顺序子时隙:每次 U >= thr/T 发放并减 thr/T(负侧对称)
            u, c = U0.clone(), torch.zeros_like(U0)
            for _ in range(T):
                f = (u >= 1.0 / T).float() - (u <= -1.0 / T).float()
                c += f; u -= f / T
            assert torch.allclose(S, c / T) and torch.allclose(st["U"], u)


def test_comparator_levels_is_a_bank_of_2T_comparators():
    c = ComparatorModel(thr=1.0, levels=2)
    assert torch.equal(c(torch.tensor([-1.5, -0.5, 0.49, 0.5, 1.5])), torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]))
    c2 = ComparatorModel(thr=1.0, offset=0.1, levels=4)
    assert torch.equal(c2(torch.tensor([0.35, 0.6, 1.1, -0.15])), torch.tensor([0.25, 0.5, 1.0, -0.25]))
    with pytest.raises(ValueError):
        ComparatorModel(thr=1.0, hysteresis=0.1, levels=2)


def _set_levels(blk, T, points=None):
    mods = dict(q_in1=blk.q_in1, q=blk.attn.sn_q, k=blk.attn.sn_k, v=blk.attn.sn_v, pre=blk.attn.sn_pre,
                out=blk.attn.sn_out, q_in2=blk.q_in2, g=blk.ffn.sn_1, u=blk.ffn.sn_3, y2=blk.ffn.sn_2)
    for name, m in mods.items():
        if points is None or name in points:
            m.levels = T


@pytest.mark.parametrize("T", [2, 4])
def test_behavioral_block_reproduces_software_block_with_levels(T):
    blk = _block(8)
    _set_levels(blk, T)
    h_seq = torch.randn(2, 12, 16) * 2
    sw, sw_st, srecs = run_sequence(blk, h_seq, record=True)
    sim = BehavioralBlock(blk)
    assert sim.levels["q_in1"] == T and sim.levels["y2"] == T and sim.neu["k"].levels == T
    hw, hw_st, recs = run_behavioral_sequence(sim, h_seq, record=True)
    assert torch.equal(hw, sw)
    assert torch.allclose(hw_st["attn"]["S"], sw_st["attn"]["S"], atol=1e-6)
    for key in ("x_attn", "q", "k", "v", "o_spk", "y_attn", "g", "u", "p", "y_ffn"):
        assert all(torch.equal(a[key], b[key]) for a, b in zip(recs, srecs)), key
    g = torch.cat([r["g"] for r in recs]); u = torch.cat([r["u"] for r in recs]); p = torch.cat([r["p"] for r in recs])
    assert (g * T).round().max() > 1 or (u.abs() * T).round().max() > 1       # 真的用到了多级
    assert torch.allclose(p, g * u)                                              # W2 事件数 = c_g·c_u
    assert any(r["y_attn"].abs().sum() > 0 for r in recs) and any(r["y_ffn"].abs().sum() > 0 for r in recs)


def test_behavioral_block_levels_only_on_some_points():
    blk = _block(9)
    _set_levels(blk, 4, points=("q", "k", "v"))
    h_seq = torch.randn(1, 8, 16) * 2
    sw, _ = run_sequence(blk, h_seq)
    sim = BehavioralBlock(blk)
    assert sim.levels["k"] == 4 and sim.levels["g"] == 1 and sim.cmp1.levels == 1
    hw, _ = run_behavioral_sequence(sim, h_seq)
    assert torch.equal(hw, sw)


def test_old_three_argument_neuron_factory_still_works_for_levels_one():
    blk = _block(10)
    sim = BehavioralBlock(blk, neuron_factory=lambda thr, leak, ternary: IdealLIFModel(thr, leak, ternary))
    h_seq = torch.randn(1, 4, 16) * 2
    assert torch.equal(run_behavioral_sequence(sim, h_seq)[0], run_sequence(blk, h_seq)[0])


def test_spike_prf_by_sign_counts_for_levels():
    ref = torch.tensor([1.0, 0.5, -0.5, 0.0, -1.0])
    ckt = torch.tensor([0.5, 0.5, -1.0, 0.5, 0.0])
    r = spike_prf_by_sign(ckt, ref, levels=2)
    # 正:ref 计数 2+1=3,ckt 1+1+1=3,tp = min(1,2)+min(1,1)+min(1,0) = 2
    assert r["pos_tp"] == 2 and r["pos_pred"] == 3 and r["pos_ref"] == 3
    # 负:ref 1+2=3,ckt 2,tp = min(2,1)+min(0,2) = 1
    assert r["neg_tp"] == 1 and r["neg_pred"] == 2 and r["neg_ref"] == 3
    one = spike_prf_by_sign(torch.tensor([1.0, 0.0, -1.0]), torch.tensor([1.0, -1.0, -1.0]))
    assert one == spike_prf_by_sign(torch.tensor([1.0, 0.0, -1.0]), torch.tensor([1.0, -1.0, -1.0]), levels=1)


def test_behavioral_block_crossbar_kw_per_matrix():
    """crossbar_kw 按矩阵名给出时只有列出的矩阵带非理想性,其余与理想相同;共用字典的旧写法不变。"""
    blk = _block(7)
    noisy = dict(n_levels=8, sigma_g=0.05, seed=3)
    uni = BehavioralBlock(blk, crossbar_kw=noisy)
    per = BehavioralBlock(blk, crossbar_kw={"W1": noisy, "W2": noisy})
    ideal = BehavioralBlock(blk)
    for n in ("Wq", "Wk", "Wv", "Wo", "W3"):
        assert torch.equal(per.xb[n].W_eff, ideal.xb[n].W_eff)
    for n in ("W1", "W2"):
        assert torch.equal(per.xb[n].W_eff, uni.xb[n].W_eff) and not torch.equal(per.xb[n].W_eff, ideal.xb[n].W_eff)
    weights = dict(Wq=blk.attn.Wq, Wk=blk.attn.Wk, Wv=blk.attn.Wv, Wo=blk.attn.Wo, W1=blk.ffn.W1, W3=blk.ffn.W3, W2=blk.ffn.W2)
    for n, lin in weights.items():                                   # 共用字典送到全部 7 个矩阵
        assert torch.equal(uni.xb[n].W_eff, CrossbarModel(lin.weight, **noisy).W_eff)
    with pytest.raises(ValueError):
        BehavioralBlock(blk, crossbar_kw={"W1": noisy, "sigma_g": 0.1})
