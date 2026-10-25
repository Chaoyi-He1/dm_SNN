"""完整一层(GDN + SwiGLU + 两个累加器 + 两组块入口比较器)的 ngspice 网表,与软件 SpikingBlock 逐脉冲比对。

结构测试不需要 ngspice;仿真测试在 6 个 token 上同时跑 T = 1 与 T = 2(两个 ngspice 子进程并行,约 2.5 分钟)。"""
import math
import shutil
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from experiments.spice import block_layer as bl
from snn_spec.neurons import level_ternary_fire


@pytest.mark.parametrize("levels", [1, 2, 4])
def test_schedule_windows_are_ordered_and_fit_the_period(levels):
    PH, period, t_rec = bl.schedule(levels)
    # 每级 T 个子时隙,采样与作用交替
    for tag in "ABCDE":
        assert all(f"{tag}s{k}" in PH and f"{tag}a{k}" in PH for k in range(levels))
        assert f"{tag}s{levels}" not in PH
    seq = [k for k in PH if k not in ("clr", "load", "p1", "p3s")]
    starts = [PH[k][0] for k in seq]
    assert starts == sorted(starts)
    for a, b in zip(seq, seq[1:]):                      # 主序列里的窗口互不重叠
        assert PH[a][1] < PH[b][0], (a, b)
    assert PH["p1"][0] + 1.0 < PH["p3"][0]               # alpha = 0.5 的最长泄漏窗放得下(t_gap ≤ 1 µs)
    assert PH["p3"][0] < PH["p3s"][0] < PH["p3s"][1] == PH["p3"][1]
    assert PH["acc2"][1] < t_rec < period
    # T 每加 1,五个神经元级各多一对子时隙(0.4 µs)
    if levels > 1:
        assert math.isclose(period - bl.schedule(levels - 1)[1], 2.0, abs_tol=0.1)


def test_reference_sets_levels_and_thresholds_on_every_spike_point():
    ref = bl.make_reference(L=3, levels=2)
    for t in range(3):
        r = ref["recs"][t]
        for key in ("x_attn", "q", "v", "o_spk", "y_attn", "x_ffn", "g", "u", "y_ffn"):
            vals = (r[key].reshape(-1) * 2).tolist()
            assert all(abs(v - round(v)) < 1e-6 and abs(v) <= 2 + 1e-6 for v in vals), key   # 取值 c/2
        assert torch.allclose(r["p"], r["g"] * r["u"])
        assert torch.allclose(r["x_attn"], level_ternary_fire(r["h_in"], torch.tensor(0.7), 2))
    assert ref["thr"] == bl.DEFAULT_THR


def test_netlist_has_every_block_of_the_layer():
    for T in (1, 2):
        ref = bl.make_reference(L=2, levels=T)
        net = bl.netlist(ref).lower()
        lines = net.splitlines()
        count = lambda prefix: sum(1 for l in lines if l.startswith(prefix))  # noqa: E731
        d, dff = ref["d"], ref["d_ff"]
        assert count("cacc") == d and count("sld") == d and count("bacc") == d          # 累加器
        assert count("bcx") == d and count("bcw") == d                                   # 两组比较器
        bcx = next(l for l in lines if l.startswith("bcx0 "))
        assert bcx.count("u(") == 2 * T                                                  # 每通道 2T 个阈值
        assert count("band") == dff                                                      # 锁存 + AND
        assert count("cm") == d * d                                                      # gain cell 阵列
        for nm, n in (("q", d), ("k", d), ("v", d), ("p", d), ("y", d), ("g", dff), ("u", dff), ("z", d)):
            assert sum(1 for l in lines if l.startswith(f"sf{nm}0_")) == T, nm           # 每个神经元 T 个子时隙采样开关
            assert count(f"cs{nm}") == n, nm                                              # 脉冲锁存
        assert "method=gear" in net and f"tran {bl.TSTEP_US:g}u" in net


def test_threshold_margins_find_the_closest_decision():
    ref = bl.make_reference(L=16, levels=2)
    m = bl.threshold_margins(ref)
    assert set(m) == {"q", "k", "v", "pre", "out", "g", "u", "y2", "q_in1", "q_in2"}
    assert all(v["min"] >= 0 for v in m.values())
    # seed 0、T = 2 时 y2 的第 13 个神经元在第 8 个 token 的膜电位是 0.50004,离 θ/T = 0.5 只有 3.6e-5
    assert m["y2"]["token"] == 8 and m["y2"]["channel"] == 13 and m["y2"]["min"] < 1e-4


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="需要 ngspice")
def test_spice_full_layer_matches_software_spike_by_spike():
    refs = {T: bl.make_reference(L=6, levels=T) for T in (1, 2)}
    with ThreadPoolExecutor(max_workers=2) as ex:
        outs = dict(zip(refs, ex.map(lambda T: bl.simulate(refs[T]), refs)))
    for T, ref in refs.items():
        rep = bl.compare(ref, outs[T])
        for name, *_ in bl.SIGNALS:
            r = rep[name]
            assert min(r["pos_precision"], r["pos_recall"], r["neg_precision"], r["neg_recall"]) >= 0.9, (T, name, r)
            assert r["sign_error_rate"] <= 0.02, (T, name, r)
            assert r["max_level_dev"] < 1e-3, (T, name, r)               # 锁存电压落在脉冲级上
        # 所有判决离阈值都 > 1e-4 时,电路必须逐元素与软件相同
        if min(v["min"] for v in rep["margins"].values()) > 1e-4:
            assert rep["worst"]["agreement"] == 1.0, (T, rep["worst"])
        assert rep["h_mid"]["max_abs_err"] < 1e-3 and rep["h_out"]["max_abs_err"] < 1e-3, (T, rep["h_mid"], rep["h_out"])
        assert rep["S"]["err_over_bound"] <= 0.1 and rep["S"]["err_over_scale"] <= 1e-3, (T, rep["S"])
        assert rep["spikes_total_ref"] > 300 * T                        # 不是在一片沉默上通过的
