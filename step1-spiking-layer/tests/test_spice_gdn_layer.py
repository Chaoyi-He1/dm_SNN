"""单头 d=16 的 GDN 子块在 ngspice 里按相位表逐 token 执行,与软件模型(SpikingGDN,同权重、同输入、同门值)比对。"""
import shutil
import pytest
import torch

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="需要 ngspice")

from experiments.spice.gdn_layer import make_reference, simulate_layer, compare


def test_spice_gdn_subblock_matches_software_over_16_tokens():
    ref = make_reference(L=16, seed=0)
    out = simulate_layer(ref)
    rep = compare(ref, out)
    # S2.4 的判据:发放事件按正负分的 precision/recall >= 0.9,符号错误率 <= 2%,状态误差 <= 0.1·界
    for key in ("q", "k", "v", "o_spk", "y"):
        r = rep[key]
        assert min(r["pos_precision"], r["pos_recall"], r["neg_precision"], r["neg_recall"]) >= 0.9, (key, r)
        assert r["sign_error_rate"] <= 0.02, (key, r)
    assert rep["S"]["err_over_bound"] <= 0.1 and rep["S"]["err_over_scale"] <= 0.02, rep["S"]
    # 实测:q/k/v 逐 token 全对,o_spk、y 的分歧只来自临界神经元的翻转
    assert rep["q"]["agreement"] >= 0.99 and rep["k"]["agreement"] >= 0.99 and rep["v"]["agreement"] >= 0.99
    assert rep["spikes_total_ref"] > 100                     # 比对不是在一片沉默上通过的
