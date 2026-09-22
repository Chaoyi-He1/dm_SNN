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


def test_sigma_g_changes_head_weights(cfg, sl):
    m = _model(cfg, sl)
    a = CircuitLM(m, cfg, ideal=False, sigma_g=0.0).head_xb.W_eff
    b = CircuitLM(m, cfg, ideal=False, sigma_g=0.1).head_xb.W_eff
    assert not torch.equal(a, b)
