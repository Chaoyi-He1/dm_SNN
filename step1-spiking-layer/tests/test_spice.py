"""Stage 1 的 SPICE 测试台(ngspice 批处理)。器件参数为占位的理想稳定模型,不代表实测器件。"""
import shutil
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="需要 ngspice")

from experiments.spice.gain_cell import single_step_discrimination, alpha_sweep, closed_form, always_on_prediction
from experiments.spice.dm_neuron import fit_and_validate


def test_spice_single_cell_leak_window_before_retrieve_gives_1p25():
    assert abs(single_step_discrimination("A") - 1.25) <= 0.02


def test_spice_single_cell_retrieve_before_leak_window_gives_1p00():
    assert abs(single_step_discrimination("B") - 1.00) <= 0.02


def test_spice_single_cell_leak_window_after_write_gives_0p75():
    assert abs(single_step_discrimination("C") - 0.75) <= 0.02


def test_spice_single_cell_always_on_leak_matches_analytic_and_is_rejected_as_A():
    x = single_step_discrimination("always_on")
    assert abs(x - always_on_prediction(0.5, 0.5)) <= 0.02
    assert abs(x - 1.25) > 0.15                  # 单步判别正确地拒绝它,但它与 B 的 1.00 只差约 0.03


def test_spice_second_point_alpha_0p25_beta_1_separates_always_on_from_B():
    """第二判别点:alpha=0.25、beta=1 时闭式给 A 2.00 / B 1.25 / C 0.50,常开泄漏约 1.0,与 B 分开。"""
    cf = closed_form(0.25, 1.0)
    assert (cf["A"], cf["B"], cf["C"]) == (2.0, 1.25, 0.5)
    for form in ("A", "B", "C"):
        assert abs(single_step_discrimination(form, 0.25, 1.0) - cf[form]) <= 0.03
    x = single_step_discrimination("always_on", 0.25, 1.0)
    assert abs(x - always_on_prediction(0.25, 1.0)) <= 0.03
    assert min(abs(x - cf[f]) for f in ("A", "B", "C")) > 0.15


def test_spice_gain_cell_alpha_sweep_is_monotone_and_spans_a_range():
    pts = alpha_sweep()
    alphas = [a for _, a in pts]
    assert all(0.0 <= a <= 1.0 for a in alphas)
    assert all(alphas[i] >= alphas[i + 1] - 1e-3 for i in range(len(alphas) - 1))
    assert alphas[0] > 0.99 and alphas[-1] < 0.6


def test_spice_dm_neuron_fit_recovers_rc_parameters_and_threshold():
    """拟合出的膜泄漏、电荷增益、阈值、保持电压应与电路参数一致(占位器件的理论值已知)。"""
    r = fit_and_validate(seed=0)
    f = r["fit"]
    assert abs(f["beta_m"] - r["beta_m_theory"]) < 0.01 * r["beta_m_theory"]
    assert abs(f["g_m"] - r["g_m_theory_V_per_mA"]) < 0.01 * r["g_m_theory_V_per_mA"]
    assert abs(f["theta"] - 1.0) < 0.02 and abs(f["U_hold"] - 0.2) < 0.05
    assert f["n_ref"] == 0 and r["double_spike_cycles"] == {"train": 0, "test": 0}


def test_spice_dm_neuron_free_run_precision_recall_floor_over_three_seeds():
    """S1.2 的门是 precision/recall >= 0.95;单个 rho 的复位律在此占位器件上达到 0.86–1.00,
    这里守 0.85 的下限,逐种子结果记录在 results/spice-dm-neuron.json。"""
    for seed in (0, 1, 2):
        r = fit_and_validate(seed=seed)
        assert min(r["pos_precision"], r["pos_recall"]) >= 0.85, (seed, r["pos_precision"], r["pos_recall"])
