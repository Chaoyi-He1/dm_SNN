"""Table 1 物理模型与 Zhao et al. 2025 正文实验数据的对照(experiments/adm_main_text.py 的可断言部分)。
输出脉冲 = Eq. S15 的 O = H(V_gs − V_th) 的上升沿,对应实验测到的输出电流尖峰;导通事件 = x 越过 1。"""
import pytest
import torch

from snn_spec.adm import ADMNeuronModel, TABLE1, pulse_train, onsets, first_fire_time

DT = 2e-6


def _fig2e(t_end, **over):
    return ADMNeuronModel(**dict(TABLE1, **over), dt=DT).run(pulse_train(0.8, 2e-3, 0.5, t_end, DT))


def _isi_ms(idx, t0, t1):
    t = idx.double() * DT
    t = t[(t >= t0) & (t <= t1)]
    return (t[1:] - t[:-1]) * 1e3


def test_constant_0p8V_first_turn_on_is_34p92ms_regression():
    """adm-neuron.md 记录的 Python 值 34.924 ms(SPICE 34.918 ms);锁定数值以便日后改模型时看得见变化。"""
    r = ADMNeuronModel(**TABLE1, dt=DT).run(torch.full((int(0.1 / DT),), 0.8, dtype=torch.float64))
    assert abs(first_fire_time(r["A"], DT) - 34.924e-3) < 0.05e-3


@pytest.mark.parametrize("Rc", (50e6, 100e6))
def test_small_control_resistance_gives_one_output_spike_per_pulse(Rc):
    """正文 Fig. 4d:R_c = 50 / 100 MΩ 时 ISI 均值 2.25 / 2.3 ms,即几乎每个 2 ms 脉冲一个尖峰。
    模型:R_c·C_g = 0.5 / 1 ms,栅电压在 1 ms 间隙内降回阈值以下,稳定后(0.6–0.9 s)输出脉冲 ISI 恰为 2 ms。"""
    r = _fig2e(0.9, Rc=Rc)
    isi = _isi_ms(onsets(r["O"]), 0.6, 0.9)
    assert len(isi) >= 140 and abs(isi.mean().item() - 2.0) < 0.1, (len(isi), isi.mean().item())


def test_table1_at_250M_latches_instead_of_spiking():
    """记录现状(与正文 Fig. 2c/2e/4d 不符,见 results/adm-neuron.md):R_c = 250 MΩ 时 Table 1 模型在首次发放后
    锁在 x ≈ 1、V_gs ≥ 0.3 V,导通事件每个脉冲多次(x 在 1 附近抖动),Eq. S15 的输出恒高、没有输出脉冲。"""
    r = _fig2e(0.9)
    late = r["V"][int(0.5 / DT):]
    assert late.min().item() > 0.3
    assert len(_isi_ms(onsets(r["O"]), 0.6, 0.9)) == 0
    t_on = onsets(r["A"]).double() * DT
    win = (t_on >= 0.6) & (t_on <= 0.9)
    assert win.sum().item() > 3 * len(torch.unique((t_on[win] // 2e-3).long()))    # 每个脉冲不止一次越过 1


@pytest.mark.xfail(strict=True, reason="正文 Fig. 4d:R_c = 250 MΩ 时 ISI 均值 ≈ 4 ms、输出电流在尖峰间回到基线;"
                                       "Table 1 模型给不出输出脉冲(锁死)。待 SI 核对 Eq. S1–S5 与 Table 1 后更新")
def test_fig4d_isi_at_250M_is_a_few_ms():
    r = _fig2e(0.9)
    isi = _isi_ms(onsets(r["O"]), 0.6, 0.9)
    assert len(isi) >= 10 and 2.0 < isi.mean().item() < 8.0
