"""能耗估算:参数表、纯公式、事件计数、数字侧乘加数。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import numpy as np
import torch

from experiments import energy_model as em
from experiments.energy_events import count_events, model_kind, model_matrices
from experiments.energy_params import COLUMNS, DEVICE, v
from experiments.multilevel import apply_levels
from experiments.spice import crossbar as xb
from toy_demo.config import ToyConfig
from toy_demo.data import PAD, all_sequences, split_inputs_targets
from toy_demo.model import SpikingLM
from toy_demo.ternary_residual import TernaryResidualLM


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


def _lower_thresholds(m):
    with torch.no_grad():
        for lif in m.lif_modules():
            lif._thr.log_thr.fill_(-1.0)


@torch.no_grad()
def test_count_events_matches_manual_recount(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    tok = all_sequences(sl)[:12]
    ev = count_events(m, tok, torch.device("cpu"), batch=5)
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)[mask]
    assert ev["n_tokens"] == int(mask.sum()) and ev["levels"] == 1
    assert np.allclose(ev["rows"]["blocks.0.qkv"], x.abs().mean(0).numpy())
    assert np.allclose(ev["rows"]["blocks.0.gates"][:-1], x.abs().mean(0).numpy()) and ev["rows"]["blocks.0.gates"][-1] == 1.0
    g = torch.stack([recs[t][0]["g"] for t in range(inputs.shape[1])], 1)[mask]; u = torch.stack([recs[t][0]["u"] for t in range(inputs.shape[1])], 1)[mask]
    assert np.allclose(ev["rows"]["blocks.0.w2"], (g.abs() * u.abs()).mean(0).numpy())
    hf = torch.stack([recs[t][-1]["h_final"] for t in range(inputs.shape[1])], 1)[mask]
    assert np.allclose(ev["rows"]["head"], m.q_out(hf).abs().mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.n_blocks * 2 * cfg.d + cfg.d) * 2 and ev["n_lif"] == cfg.n_blocks * (6 * cfg.d + 2 * cfg.d_ff)
    assert np.isclose(ev["embed_rows"].sum(), 1.0) and ev["embed_rows"].shape == (sl.vocab_size,)
    assert 0 < ev["gain"]["alpha_mean"] < 1 and 0 <= ev["gain"]["s_rms_norm"] <= 1 and ev["gain"]["k_rows"] >= 0
    assert ev["spikes_lif"] > 0


@torch.no_grad()
def test_count_events_levels_two_counts_pulses_and_w2_products(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); apply_levels(m, 2); _lower_thresholds(m)
    tok = all_sequences(sl)[:8]
    ev = count_events(m, tok, torch.device("cpu"))
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)[mask]
    g = torch.stack([recs[t][0]["g"] for t in range(inputs.shape[1])], 1)[mask]; u = torch.stack([recs[t][0]["u"] for t in range(inputs.shape[1])], 1)[mask]
    assert ev["levels"] == 2
    assert np.allclose(ev["rows"]["blocks.0.qkv"], (x.abs() * 2).round().mean(0).numpy())
    assert np.allclose(ev["rows"]["blocks.0.w2"], ((g.abs() * 2).round() * (u.abs() * 2).round()).mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.n_blocks * 2 * cfg.d + cfg.d) * 4


@torch.no_grad()
def test_count_events_gain_and_spike_totals_match_recount(cfg, sl):
    for T in (1, 2):
        torch.manual_seed(0)
        m = SpikingLM(cfg, sl.vocab_size, leak=0.9)
        if T > 1:
            apply_levels(m, T)
        _lower_thresholds(m)
        tok = all_sequences(sl)[:8]
        ev = count_events(m, tok, torch.device("cpu"), batch=3)
        inputs, _, mask = split_inputs_targets(tok)
        _, recs = m(inputs, record=True)
        L, n = inputs.shape[1], int(mask.sum())
        rec = lambda i, key: torch.stack([recs[t][i][key] for t in range(L)], 1)[mask]      # 有效位置 [N, ...]
        pulses = lambda s: (s.abs() * T).round()
        k_rows = q_rows = spikes = 0.0
        alphas, states = [], []
        for i, blk in enumerate(m.blocks):
            k, q = rec(i, "k") / blk.attn.c_k, rec(i, "q")                                  # k 记录里已乘 c_k,先除掉
            for h in range(cfg.n_heads):                                                    # 每头 dk 个元素
                k_rows += float(pulses(k[:, h]).sum()); q_rows += float(pulses(q[:, h]).sum())
            spikes += sum(float(pulses(x).sum()) for x in (q, k, rec(i, "v"), rec(i, "o_spk"), rec(i, "y_attn"),
                                                          rec(i, "g"), rec(i, "u"), rec(i, "y_ffn")))
            alphas.append(rec(i, "alpha").reshape(-1))
            states.append(rec(i, "S").reshape(-1))
        S = torch.cat(states)
        assert k_rows > 0 and q_rows > 0 and spikes > k_rows + q_rows                       # 重算本身不是空的,去掉 /c_k 会被发现
        assert ev["levels"] == T
        assert np.isclose(ev["gain"]["k_rows"], k_rows / n) and np.isclose(ev["gain"]["q_rows"], q_rows / n)
        assert np.isclose(ev["spikes_lif"], spikes / n)
        assert np.isclose(ev["gain"]["alpha_mean"], float(torch.cat(alphas).mean()))
        assert np.isclose(ev["gain"]["s_rms_norm"], float((S * S).mean().sqrt() / S.abs().max()))
        assert ev["embed_rows"][PAD] == 0


@torch.no_grad()
def test_count_events_ignores_padding(cfg, sl):
    import pytest
    torch.manual_seed(1)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    tok = all_sequences(sl)[:6]
    inputs, _, mask = split_inputs_targets(tok)
    if mask.all():
        pytest.skip("切片里这 6 句等长,没有填充可测")
    ev = count_events(m, tok, torch.device("cpu"))
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)
    assert ev["n_tokens"] == int(mask.sum())
    assert np.allclose(ev["rows"]["blocks.0.qkv"], x[mask].abs().mean(0).numpy())                      # 只计有效位置
    assert not np.allclose(ev["rows"]["blocks.0.qkv"], x.reshape(-1, cfg.d).abs().mean(0).numpy())    # 含填充的平均不同


@torch.no_grad()
def test_ternres_events_use_h_final_as_head_input(cfg, sl):
    torch.manual_seed(0)
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.9, merge="sat", levels=1); _lower_thresholds(m)
    tok = all_sequences(sl)[:8]
    ev = count_events(m, tok, torch.device("cpu"))
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    hf = torch.stack([recs[t][-1]["h_final"] for t in range(inputs.shape[1])], 1)[mask]
    assert model_kind(m) == "ternres" and np.allclose(ev["rows"]["head"], hf.abs().mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.d + cfg.n_blocks * 2 * cfg.d) * 2


def test_model_matrices_shapes(cfg, sl):
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9)
    mats = model_matrices(m)
    d, dff, H, V = cfg.d, cfg.d_ff, cfg.n_heads, sl.vocab_size
    hq, hv = H * cfg.dk, H * cfg.dv                    # 真实配置里 hq = hv = d;tiny 配置里不等,按权重真实形状断言
    assert mats["blocks.0.qkv"].shape == (2 * hq + hv, d) and mats["blocks.0.gates"].shape == (2 * H, d + 1)
    assert mats["blocks.0.o"].shape == (d, hv) and mats["blocks.0.ffn13"].shape == (2 * dff, d) and mats["blocks.0.w2"].shape == (d, dff)
    assert mats["head"].shape == (V, d) and mats["embed"].shape == (d, V)
    assert np.allclose(mats["blocks.0.gates"][:H, -1], m.blocks[0].attn.gates.Wa.bias.detach().numpy())


from experiments.energy_gpu import joules_per_token, parse_power_lines


def test_parse_power_lines_and_joules_per_token():
    assert parse_power_lines(["20.08", " 25.9 ", "", "N/A"]) == [20.08, 25.9]
    r = joules_per_token(mean_w=60.0, idle_w=20.0, seconds=10.0, tokens=4000)
    assert np.isclose(r["total"], 600.0 / 4000) and np.isclose(r["incremental"], 400.0 / 4000) and np.isclose(r["ms_per_token"], 2.5)


from experiments.energy_estimate import embedding_quantization_check, sensitivity


def test_embedding_quantization_check_restores_weights(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    before = m.emb.weight.detach().clone()
    r = embedding_quantization_check(m, all_sequences(sl)[:16], levels=16)
    assert set(r) == {"acc_before", "acc_after", "delta"} and torch.equal(m.emb.weight.detach(), before)


def test_sensitivity_moves_with_t_read_and_adc():
    cfg, V = ToyConfig(), 3226
    s = sensitivity(_synthetic_events(cfg, V), _synthetic_mats(cfg, V), cfg, V)
    assert s["t_read"]["1e-06"] > s["t_read"]["1e-07"] > s["t_read"]["1e-08"]
    assert s["E_adc"]["1e-11"] > s["E_adc"]["1e-12"]
    assert s["G_scale"]["0.1"] < s["G_scale"]["1"]


def test_embedding_quantization_sweep_restores_and_orders(cfg, sl):
    from experiments.energy_estimate import embedding_quantization_sweep
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    before = m.emb.weight.detach().clone()
    r = embedding_quantization_sweep(m, all_sequences(sl)[:16], levels_list=(16, 64), scalings=("global", "per_column"))
    assert set(r) == {"global/16", "global/64", "per_column/16", "per_column/64", "best_within_0.5pt"}
    assert torch.equal(m.emb.weight.detach(), before)
    for k in ("global/16", "per_column/64"):
        assert set(r[k]) >= {"acc_before", "acc_after", "delta", "levels", "scaling", "w_max"}
    assert r["global/16"]["acc_before"] == r["per_column/64"]["acc_before"]


def test_unknown_override_key_raises():
    cfg, V = ToyConfig(), 3226
    import pytest
    with pytest.raises(ValueError):
        em.digital_dense_energy(cfg, V, "65nm", 1, overrides={"t_reed": 1e-7})


def test_dram_bit_is_not_node_scaled():
    assert np.isclose(v("65nm", "E_dram_bit"), 2e-9 / 64)
    assert np.isclose(v("65nm", "E_dram_bit"), 2e-9 / 64, rtol=1e-9, atol=0)      # 上一行的默认 atol=1e-8 比 3e-11 大,恒真
