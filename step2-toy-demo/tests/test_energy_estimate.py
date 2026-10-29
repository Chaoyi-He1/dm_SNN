"""能耗估算:参数表、纯公式、事件计数、数字侧乘加数。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import numpy as np
import torch

from experiments import energy_model as em
from experiments.energy_params import COLUMNS, DEVICE, v
from experiments.spice import crossbar as xb
from toy_demo.config import ToyConfig


def test_every_parameter_has_a_source():
    for col in ("65nm", "advanced"):
        for name, p in COLUMNS[col].items():
            assert set(p) >= {"value", "unit", "source"} and p["source"], (col, name)
    assert v("65nm", "E_spike") == 1.39e-12 and v("65nm", "t_read") == 100e-9 and DEVICE["levels"]["value"] == 16


def test_mac_counts_match_analytic_shapes():
    cfg, V = ToyConfig(), 3226
    m = em.mac_counts(cfg, V)
    assert m["head"] == 128 * 3226 and m["proj_qkv"] == 2 * 3 * 128 * 128 and m["ffn_13"] == 2 * 2 * 128 * 256
    assert m["core"] == 2 * 4 * 4 * 32 * 32 and m["gates"] == 2 * 2 * 128 * 4
    assert m["params_embedding"] == 3226 * 128
    assert m["params_stream"] == 2 * (4 * 128 * 128 + 3 * 128 * 256 + 2 * 128 * 4 + 2 * 4) + 128 * 3226   # 不含嵌入表
    assert m["total_macs"] == sum(m[k] for k in ("proj_qkv", "proj_o", "ffn_13", "ffn_2", "gates", "core", "head"))


def test_row_conductance_sums_both_cells_of_each_weight():
    W = np.array([[1.0, -0.5], [0.0, 1.0]])                   # d_out=2, d_in=2;列 0:+1 与 0;列 1:−0.5 与 +1
    npos, nneg, _, _ = xb.quantize_weights(W, 16)
    Gp, Gn = xb.conductance_targets(npos, nneg, 16)
    g = em.row_conductance(W, 16, DEVICE["G_on"]["value"], DEVICE["G_off"]["value"])
    assert np.allclose(g, (Gp + Gn).sum(0)) and g.shape == (2,)
    assert g[1] > g[0]                                         # 行 1 有一个负权重与一个正权重,两个 cell 都导通


def test_crossbar_energy_formula_on_one_row():
    c = np.array([2.0, 0.0]); grow = np.array([1e-4, 1e-4])
    e = em.crossbar_energy(c, grow, d_out=4, col="65nm")
    V, t = v("65nm", "V_read"), v("65nm", "t_read")
    assert np.isclose(e["conduction"], V * V * t * 2.0 * 1e-4)
    assert np.isclose(e["gate"], 2.0 * 8 * v("65nm", "C_gate") * v("65nm", "V_sel") ** 2)
    assert np.isclose(e["wire"], 2.0 * 8 * v("65nm", "C_wire") * V * V)
    assert np.isclose(e["leak_off"], (16 - 16) * V * v("65nm", "I_off") * t) and e["leak_off"] >= 0
    assert np.isclose(e["integrator"], 4 * v("65nm", "I_int") * v("65nm", "V_dd") * t)
    assert np.isclose(e["total"], sum(e[k] for k in ("conduction", "gate", "wire", "leak_off", "integrator")))


def test_digital_dense_batch_amortises_only_weight_fetch():
    cfg, V = ToyConfig(), 3226
    e1, e64 = em.digital_dense_energy(cfg, V, "65nm", 1), em.digital_dense_energy(cfg, V, "65nm", 64)
    assert np.isclose(e64["weight"], e1["weight"] / 64) and e64["mac"] == e1["mac"] and e64["act"] == e1["act"]
    assert np.isclose(e1["weight"], em.mac_counts(cfg, V)["params_stream"] * 8 * v("65nm", "E_sram_far_bit"))
    assert e1["weight"] > 10 * e1["mac"]                       # batch 1 时权重取数支配


def _synthetic_events(cfg, V):
    d, dff, H = cfg.d, cfg.d_ff, cfg.n_heads
    rows = {}
    for i in range(cfg.n_blocks):
        rows[f"blocks.{i}.qkv"] = np.full(d, 0.5); rows[f"blocks.{i}.gates"] = np.full(d + 1, 0.5)
        rows[f"blocks.{i}.o"] = np.full(d, 0.25); rows[f"blocks.{i}.ffn13"] = np.full(d, 0.5); rows[f"blocks.{i}.w2"] = np.full(dff, 0.1)
    rows["head"] = np.full(d, 0.5)
    return dict(n_tokens=100, levels=1, rows=rows, embed_rows=np.full(V, 1.0 / V), spikes_lif=1700.0, n_lif=3200,
                cmp_decisions=5 * d * 2, gain=dict(k_rows=2 * H * 10.0, q_rows=2 * H * 6.0, alpha_mean=0.8, s_rms_norm=0.3))


def _synthetic_mats(cfg, V):
    rng = np.random.default_rng(0); d, dff, H = cfg.d, cfg.d_ff, cfg.n_heads
    mats = {}
    for i in range(cfg.n_blocks):
        mats[f"blocks.{i}.qkv"] = rng.normal(0, 0.5, (3 * d, d)); mats[f"blocks.{i}.gates"] = rng.normal(0, 0.5, (2 * H, d + 1))
        mats[f"blocks.{i}.o"] = rng.normal(0, 0.5, (d, d)); mats[f"blocks.{i}.ffn13"] = rng.normal(0, 0.5, (2 * dff, d)); mats[f"blocks.{i}.w2"] = rng.normal(0, 0.5, (d, dff))
    mats["head"] = rng.normal(0, 0.5, (V, d)); mats["embed"] = rng.normal(0, 0.5, (d, V))
    return mats


def test_memristor_energy_breakdown_adds_up_and_head_variants_differ():
    cfg, V = ToyConfig(), 3226
    ev, mats = _synthetic_events(cfg, V), _synthetic_mats(cfg, V)
    a = em.memristor_energy(ev, mats, cfg, V, "65nm", head="crossbar_adc")
    b = em.memristor_energy(ev, mats, cfg, V, "65nm", head="digital_event")
    for r in (a, b):
        assert np.isclose(r["total"], sum(r["by_component"].values())) and all(x >= 0 for x in r["by_component"].values())
    assert np.isclose(a["by_component"]["adc"], V * v("65nm", "E_adc")) and a["by_component"]["head_digital"] == 0
    assert b["by_component"]["adc"] == 0 and b["by_component"]["head_digital"] > 0 and b["by_component"]["crossbar_head"] == 0
    assert np.isclose(a["by_component"]["adm_spikes"], 1700.0 * v("65nm", "E_spike"))
    assert np.isclose(a["by_component"]["adm_leak"], 3200 * v("65nm", "V_gs") ** 2 / v("65nm", "R_c") * v("65nm", "t_token"))
    assert np.isclose(a["by_component"]["comparators"], ev["cmp_decisions"] * v("65nm", "E_cmp"))
    assert set(a["by_matrix"]) == set(mats)


def test_digital_event_energy_scales_with_driven_rows():
    cfg, V = ToyConfig(), 3226
    ev = _synthetic_events(cfg, V)
    e = em.digital_event_energy(ev, cfg, V, "65nm", batch=1)
    adds = sum(ev["rows"][k].sum() * {"qkv": 3 * cfg.d, "gates": 2 * cfg.n_heads, "o": cfg.d, "ffn13": 2 * cfg.d_ff, "w2": cfg.d}[k.split(".")[-1]]
               for k in ev["rows"] if k != "head") + ev["rows"]["head"].sum() * V
    assert np.isclose(e["add"], adds * v("65nm", "E_add16")) and np.isclose(e["weight"], adds * 8 * v("65nm", "E_sram_far_bit"))
    assert np.isclose(em.digital_event_energy(ev, cfg, V, "65nm", batch=16)["weight"], e["weight"] / 16)
