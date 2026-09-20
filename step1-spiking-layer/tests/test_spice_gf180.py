"""GF180MCU 真实晶体管上的泄漏路径测试(需要 ngspice 与 pdk/gf180mcu 的模型文件)。"""
import math
import shutil
import pytest

from experiments.spice.gf180 import PDK, idvg, alpha_leak, pick_vleak_for_half, spread

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None or not (PDK / "sm141064.ngspice").exists(),
                                reason="需要 ngspice 与 GF180MCU 模型")


def test_gf180_nmos_3p3_idvg_is_sane():
    pts = dict((round(v, 1), i) for v, i in idvg())
    assert pts[0.0] < 1e-10 and pts[3.3] > 1e-4          # 关态 pA 级,开态 > 100 uA
    assert all(pts[round(0.1 * k, 1)] <= pts[round(0.1 * (k + 1), 1)] * 1.0001 for k in range(33))


def test_gf180_ground_leak_alpha_depends_on_stored_value():
    vg, a = pick_vleak_for_half("ground", [0.30 + 0.05 * i for i in range(13)], 1.0)
    assert 0.3 < a < 0.7
    sp, vals = spread("ground", vg, (1.0, 0.5, 0.25))
    assert sp > 0.1, vals                                 # 恒流放电:alpha 随存储值变


def test_gf180_subthreshold_mid_leak_is_not_ohmic_at_100mV_swing():
    """亚阈区 I ∝ 1 - exp(-V_ds/V_t):摆幅 100 mV 时 alpha 随存储值明显变化,不能当欧姆泄漏用。"""
    vg, a = pick_vleak_for_half("subthreshold_mid", [0.20 + 0.025 * i for i in range(29)], 0.1)
    assert 0.25 < a < 0.75
    sp, vals = spread("subthreshold_mid", vg, (0.1, 0.05, -0.05, -0.1))
    assert sp > 0.05, vals


def test_gf180_resistor_leak_is_exponential_and_state_independent():
    for tg in (20e-9, 50e-9):
        sp, vals = spread("resistor", 0.0, (0.3, 0.1, -0.1, -0.3), cs=50e-15, t_gap=tg, r_leak=1e6)
        expect = math.exp(-tg / (1e6 * 50e-15))
        assert sp < 0.02, vals
        for a in vals.values():
            assert abs(a - expect) < 0.05 * expect + 0.01, (a, expect)
