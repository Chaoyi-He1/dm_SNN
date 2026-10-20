import torch

from toy_demo.circuit import CircuitLM, passes, primitive_kwargs, teacher_forced_compare
from toy_demo.convert import calibrate_thresholds
from toy_demo.data import all_sequences
from toy_demo.model import SpikingLM
from toy_demo.train import evaluate


def _model(cfg, sl, seed=0):
    torch.manual_seed(seed)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(3.0)                                   # 让神经元真的发放
    calibrate_thresholds(m, all_sequences(sl)[:16], 0.75)
    return m.eval()


def test_primitive_kwargs_ideal_and_nominal(cfg):
    xb, gc, frac = primitive_kwargs(cfg, ideal=True)
    assert xb == dict(seed=0) and gc == dict(seed=0) and frac == 0.0
    xb, gc, frac = primitive_kwargs(cfg, ideal=False, sigma_g=0.05, seed=3)
    assert xb == dict(n_levels=cfg.n_levels, sigma_g=0.05, read_noise=cfg.read_noise, seed=3)
    assert gc == dict(write_err=cfg.write_err, read_err=cfg.read_err, eps_hold=cfg.eps_hold, seed=3)
    assert frac == cfg.cmp_offset_frac


def test_ideal_circuit_matches_spiking_model(cfg, sl, demo):
    m = _model(cfg, sl)
    hw = CircuitLM(m, cfg, ideal=True)
    tok = all_sequences(sl)[:8]
    sw_logits, hw_logits = m(tok[:, :-1]), hw(tok[:, :-1])
    assert torch.allclose(sw_logits, hw_logits, atol=1e-5)
    assert torch.equal(sw_logits.argmax(-1), hw_logits.argmax(-1))
    cmp = teacher_forced_compare(m, hw, tok, cfg)
    for b in cmp["blocks"]:
        for part in (b["attn"], b["ffn"]):
            assert part["pos_precision"] == part["pos_recall"] == part["neg_precision"] == part["neg_recall"] == 1.0
            assert part["sign_error"] == 0.0
        assert b["state_err_max"] < 1e-6
    assert cmp["argmax_agreement"] == 1.0 and cmp["pass"] and passes(cmp)
    assert m.generate(demo[0]["prefix"], 8) == hw.generate(demo[0]["prefix"], 8)


def test_nominal_circuit_runs_and_reports(cfg, sl, demo):
    m = _model(cfg, sl)
    hw = CircuitLM(m, cfg, ideal=False)
    ev = evaluate(hw, sl, demo, cfg)
    assert ev["n_demo"] == cfg.n_demo and 0.0 <= ev["token_acc"] <= 1.0
    cmp = teacher_forced_compare(m, hw, all_sequences(sl)[:4], cfg)
    assert len(cmp["blocks"]) == cfg.n_blocks and isinstance(cmp["pass"], bool)
    assert cmp["n_positions"] == sum(len(sl.ids[i]) + 1 for i in range(4))


def test_nominal_parts_decompose_the_nominal_circuit(cfg, sl):
    """nominal_parts=() 与理想电路逐位相同;四部分全开与原来的全标称电路逐位相同;只开一部分时只有那一部分带误差。"""
    import pytest
    from toy_demo.circuit import PARTS
    m = _model(cfg, sl)
    tok = all_sequences(sl)[:6, :-1]
    ideal, nominal = CircuitLM(m, cfg, ideal=True), CircuitLM(m, cfg, ideal=False)
    assert torch.equal(CircuitLM(m, cfg, ideal=False, nominal_parts=()).forward(tok), ideal.forward(tok))
    assert torch.equal(CircuitLM(m, cfg, ideal=False, nominal_parts=PARTS).forward(tok), nominal.forward(tok))
    ffn = CircuitLM(m, cfg, ideal=False, nominal_parts=("ffn",))
    s_id, s_no = ideal.sims[0], nominal.sims[0]
    for n in ("Wq", "Wk", "Wv", "Wo"):
        assert torch.equal(ffn.sims[0].xb[n].W_eff, s_id.xb[n].W_eff)
    for n in ("W1", "W3", "W2"):
        assert torch.equal(ffn.sims[0].xb[n].W_eff, s_no.xb[n].W_eff)
    assert ffn.sims[0].cmp1.offset == 0.0 and ffn.sims[0].cmp2.offset == s_no.cmp2.offset != 0.0
    assert ffn.sims[0].gain_cell_kw == s_id.gain_cell_kw and torch.equal(ffn.head_xb.W_eff, ideal.head_xb.W_eff)
    assert ffn.nominal_parts == ("ffn",) and nominal.nominal_parts == PARTS and ideal.nominal_parts == ()
    gdn = CircuitLM(m, cfg, ideal=False, nominal_parts=("gdn",))
    for i, (sg, sn, si) in enumerate(zip(gdn.sims, nominal.sims, ideal.sims)):
        for n in ("Wq", "Wk", "Wv", "Wo"):                           # 四个都带误差,读噪声也带上
            assert torch.equal(sg.xb[n].W_eff, sn.xb[n].W_eff) and sg.xb[n].read_noise == sn.xb[n].read_noise == cfg.read_noise
        for n in ("W1", "W3", "W2"):
            assert torch.equal(sg.xb[n].W_eff, si.xb[n].W_eff) and sg.xb[n].read_noise == 0.0
        assert sg.cmp1.offset == sn.cmp1.offset != 0.0 and sg.cmp2.offset == 0.0
    # 只差输出头时,两块的全部记录(含读噪声的抽样)与全标称相同
    _, rec_a = CircuitLM(m, cfg, ideal=False, nominal_parts=("gdn", "ffn", "state")).forward(tok, record=True)
    _, rec_b = CircuitLM(m, cfg, ideal=False).forward(tok, record=True)
    for ra, rb in zip(rec_a, rec_b):
        for i in range(cfg.n_blocks):
            assert all(torch.equal(ra[i][k], rb[i][k]) for k in ("y_attn", "y_ffn", "S", "h_out"))
    with pytest.raises(ValueError):
        CircuitLM(m, cfg, ideal=True, nominal_parts=("ffn",))
    with pytest.raises(ValueError):
        CircuitLM(m, cfg, ideal=False, nominal_parts=("crossbar",))


def test_sigma_g_changes_head_weights(cfg, sl):
    m = _model(cfg, sl)
    a = CircuitLM(m, cfg, ideal=False, sigma_g=0.0).head_xb.W_eff
    b = CircuitLM(m, cfg, ideal=False, sigma_g=0.1).head_xb.W_eff
    assert not torch.equal(a, b)


def _ml_model(cfg, sl, T, points="all", seed=0):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from multilevel import apply_levels
    m = _model(cfg, sl, seed)
    apply_levels(m, T, points)
    return m.eval()


def test_ideal_circuit_matches_multilevel_model():
    import torch
    from toy_demo.config import tiny_config
    from toy_demo.data import synthetic_slice, all_sequences
    cfg, sl = tiny_config(), synthetic_slice()
    for T, points in ((2, "all"), (4, "mem")):
        m = _ml_model(cfg, sl, T, points)
        hw = CircuitLM(m, cfg, ideal=True)
        lv = hw.levels()
        assert lv["q_out"] == (T if points == "all" else 1) and lv["blocks.0.k"] == T and lv["blocks.1.g"] == (T if points == "all" else 1)
        tok = all_sequences(sl)[:8]
        sw_logits, hw_logits = m(tok[:, :-1]), hw(tok[:, :-1])
        assert torch.allclose(sw_logits, hw_logits, atol=1e-5)
        assert torch.equal(sw_logits.argmax(-1), hw_logits.argmax(-1))
        cmp = teacher_forced_compare(m, hw, tok, cfg)
        for b in cmp["blocks"]:
            for part in (b["attn"], b["ffn"]):
                assert part["pos_precision"] == part["pos_recall"] == part["neg_precision"] == part["neg_recall"] == 1.0
                assert part["sign_error"] == 0.0
            assert b["state_err_max"] < 1e-6
        assert cmp["pass"] and cmp["levels"] == lv
        # 多级确实出现:至少有一个 y 的计数超过 1
        _, recs = m(tok[:, :-1], record=True)
        ya = torch.stack([r[0]["y_attn"] for r in recs]); 
        assert (ya.abs() * T).round().max() > 1 or points != "all"


def test_nominal_circuit_runs_with_multilevel_model():
    from toy_demo.config import tiny_config
    from toy_demo.data import synthetic_slice, all_sequences, select_demo
    cfg, sl = tiny_config(), synthetic_slice()
    demo = select_demo(sl, cfg)
    m = _ml_model(cfg, sl, 2)
    hw = CircuitLM(m, cfg, ideal=False)
    ev = evaluate(hw, sl, demo, cfg)
    assert 0.0 <= ev["token_acc"] <= 1.0
    cmp = teacher_forced_compare(m, hw, all_sequences(sl)[:4], cfg)
    assert cmp["blocks"][0]["attn"]["levels"] == 2 and isinstance(cmp["pass"], bool)


def test_circuit_levels_script_tiny(tmp_path):
    """experiments/circuit_levels.py 在合成切片的 T = 2 小模型上跑验收与误差隔离。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import circuit_levels
    res = circuit_levels.main(["--tiny", "--out", str(tmp_path)])
    assert (tmp_path / "circuit.json").exists() and (tmp_path / "circuit.md").exists()
    assert res["circuit_ideal"]["compare"]["pass"] and res["circuit_ideal"]["argmax_equal_to_software"]
    assert len(res["sigma_g_sweep"]) == 4 and res["levels"]["q_out"] == 2
    iso = circuit_levels.main(["--tiny", "--mode", "isolate", "--out", str(tmp_path)])
    rows = {tuple(r["parts"]): r for r in iso["isolation"]}
    full = rows[("gdn", "ffn", "state", "head")]["summary"]
    nominal = circuit_levels.compact(res["circuit_nominal"]["compare"])
    assert full == nominal                                                 # 四部分全开 = 全标称,逐位相同
    assert rows[("gdn", "ffn", "state", "head")]["token_acc"] == res["circuit_nominal"]["token_acc"]
    assert rows[("head",)]["summary"]["worst_sign_error"] == 0.0           # 输出头的误差不影响块内脉冲
    assert "误差隔离" in (tmp_path / "circuit.md").read_text()
