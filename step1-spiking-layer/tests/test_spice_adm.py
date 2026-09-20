"""ngspice 里的 ADM 1M1T1R 与 Python 物理模型的交叉验证。"""
import shutil
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="需要 ngspice")

from snn_spec.adm import TABLE1
from experiments.spice.adm_spice import simulate, turn_on_times, compare_with_python


def test_spice_adm_constant_1V_fires_within_50ms_and_matches_python_model():
    r = compare_with_python(TABLE1, vin_const=1.0, t_end=60e-3)
    assert r["t_first_spice"] is not None and r["t_first_python"] is not None
    assert 5e-3 < r["t_first_spice"] < 50e-3
    assert abs(r["t_first_spice"] - r["t_first_python"]) < 1e-3   # 实测两者相同到 10 us
    assert r["rms_dV"] < 0.1                                        # 差异集中在导通瞬间的相位,随 dt 减小收敛(0.065/0.053/0.040)


def test_spice_adm_pulse_train_integrates_then_turns_on_repeatedly():
    sp = simulate(TABLE1, "PULSE(0 0.8 0 1u 1u 1m 2m)", 0.4, "paper")
    ev = turn_on_times(sp["t"], sp["x"])
    assert len(ev) >= 3 and ev[0] > 40e-3


def test_spice_adm_with_gf180_transistor_runs():
    sp = simulate(TABLE1, "DC 1.0", 40e-3, "gf180")
    assert len(turn_on_times(sp["t"], sp["x"])) >= 1
    assert sp["Vd"].min() < 1.0 - 1e-3            # 导通后晶体管拉低漏端
