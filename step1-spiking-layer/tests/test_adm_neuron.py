"""Zhao et al. 2025(Nature Electronics)补充材料 Note 4/6 的 1M1T1R 物理模型,确定性版本(σ₁=σ₂=0)。
测试取自补充材料里的三个实验:脉冲串发放(Fig. 17)、泄漏积分(Fig. 19)、内在可塑性(Fig. 20)。
发放事件 = 扩散忆阻器导通(x 越过 1)。器件时间尺度是毫秒到百毫秒,仿真时长按此设置。"""
import torch

from snn_spec.adm import ADMNeuronModel, TABLE1, pulse_train, first_fire_time, onsets

DT = 2e-6


def test_table1_pulse_train_0p8V_2ms_integrates_then_fires_repeatedly():
    m = ADMNeuronModel(**TABLE1, dt=DT)
    vin = pulse_train(amplitude=0.8, period=2e-3, duty=0.5, t_end=1.0, dt=DT)
    r = m.run(vin)
    ev = onsets(r["A"])
    assert len(ev) >= 5
    t_first = ev[0].item() * DT
    assert t_first > 20 * 2e-3                       # 先经过几十个脉冲的积分,再开始发放
    assert r["V"].max() < 0.8 and r["V"].max() > 0.2
    # 两次导通之间栅电压降不到 0.2 V(R_c·C_g = 2.5 ms 对 1 ms 间隙):Eq. S15 的 O 在此协议下会饱和
    assert r["O"][int(0.5 / DT):].min() > 0.5


def test_deterministic_when_sigma_is_zero():
    m = ADMNeuronModel(**TABLE1, dt=DT)
    vin = pulse_train(0.8, 2e-3, 0.5, 50e-3, DT)
    assert torch.equal(m.run(vin)["V"], m.run(vin)["V"])


def test_first_fire_time_decreases_with_input_voltage():
    times = {}
    for vin in (0.6, 0.8, 1.0):
        m = ADMNeuronModel(**TABLE1, dt=DT)
        r = m.run(torch.full((int(0.3 / DT),), vin, dtype=torch.float64))
        times[vin] = first_fire_time(r["A"], DT)
        assert times[vin] is not None
    assert times[1.0] < times[0.8] < times[0.6]
    assert 5e-3 < times[1.0] < 50e-3                   # 1 V 下几十毫秒量级(Fig. 10 用 25 ms 脉冲测积分时间)


def test_leaky_integration_more_pulses_before_firing_with_longer_intervals():
    """Fig. 19:0.8 V、1 ms 宽脉冲,间隔 0.2 ms 与 1.0 ms,首次发放前的脉冲数随间隔增加。"""
    counts = {}
    for gap in (0.2e-3, 1.0e-3):
        period = 1e-3 + gap
        m = ADMNeuronModel(**TABLE1, dt=DT)
        r = m.run(pulse_train(0.8, period, 1e-3 / period, 1.5, DT))
        t = first_fire_time(r["A"], DT)
        assert t is not None, f"gap={gap}: 1.5 s 内未发放"
        counts[gap] = int(t // period) + 1
    assert counts[0.2e-3] < counts[1.0e-3], counts


def test_intrinsic_plasticity_second_pulse_integrates_faster_and_gain_decays_with_interval():
    """Fig. 20:成对 1 V 脉冲,第二个脉冲的积分时间比第一个短;间隔越长(3.5 ms → 180 ms)缩短量越小。
    (Table 1 参数下 β=0 时 x 的不动点是 0.43,器件不发放,所以对照量是脉冲间隔而不是 β。)"""
    def gain(gap):
        m = ADMNeuronModel(**TABLE1, dt=DT)
        n_pulse, n_gap = int(60e-3 / DT), int(gap / DT)
        vin = torch.cat([torch.full((n_pulse,), 1.0), torch.zeros(n_gap), torch.full((n_pulse,), 1.0)]).double()
        r = m.run(vin)
        t1 = first_fire_time(r["A"][:n_pulse], DT)
        t2 = first_fire_time(r["A"][n_pulse + n_gap:], DT)
        assert t1 is not None and t2 is not None, (gap, t1, t2)
        return t1 - t2
    g_short, g_long = gain(3.5e-3), gain(180e-3)
    assert g_short > g_long > 0.0, (g_short, g_long)


def test_token_interface_returns_binary_spikes_and_carries_state():
    m = ADMNeuronModel(**TABLE1, dt=5e-6)
    st = m.init_state((3,))
    S, st = m.step(torch.tensor([0.0, 0.8, 1.0]), st, T_cyc=2e-3)
    assert S.shape == (3,) and set(S.tolist()) <= {0.0, 1.0}
    assert st["V"].shape == (3,) and st["F"].shape == (3,) and st["R"].shape == (3,)
    assert st["V"][0].item() == 0.0 and st["F"][2].item() > st["F"][1].item() > 0.0
