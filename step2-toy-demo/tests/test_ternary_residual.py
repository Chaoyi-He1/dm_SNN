"""三值残差流语言模型:与 A0 的权重对接、前向与记录、脉冲点登记与标定、A1 → C 的装载、梯度。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import torch
import torch.nn.functional as F

from toy_demo.convert import set_leak, transfer_weights
from toy_demo.data import EOS, all_sequences
from toy_demo.model import FloatLM
from toy_demo.ternary_residual import TernaryResidualLM, calibrate_thresholds, named_points, spike_counts

POINTS = ("q", "k", "v", "pre", "out", "m1", "g", "u", "y2", "m2")
CALIB = ("q", "k", "v", "pre", "out", "g", "u", "y2")


def test_weights_transfer_from_float_and_extra_keys_are_thresholds(cfg):
    f, m = FloatLM(cfg, 20), TernaryResidualLM(cfg, 20, leak=0.0, merge="sat")
    transfer_weights(f, m)
    extra = set(m.state_dict()) - set(f.state_dict())
    assert all("log_thr" in k for k in extra)
    assert {"q_emb.log_thr", "blocks.0.m1.log_thr", "blocks.0.m2.log_thr"} <= extra
    assert not any(k.startswith(("q_out", "blocks.0.q_in")) for k in m.state_dict())


def test_forward_record_generate_and_residual_is_ternary(cfg, sl):
    tok = all_sequences(sl)[:4]
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.9)
    logits = m(tok[:, :-1])
    assert logits.shape == (4, tok.shape[1] - 1, sl.vocab_size)
    logits2, recs = m(tok[:, :-1], record=True)
    assert torch.equal(logits, logits2) and len(recs) == tok.shape[1] - 1
    assert len(recs[0]) == cfg.n_blocks + 1 and "h_final" in recs[0][-1]
    for rec in recs:
        for i in range(cfg.n_blocks):
            for key in ("h_in", "h_mid", "h_out"):
                assert set(rec[i][key].unique().tolist()) <= {-1.0, 0.0, 1.0}, key
        assert set(rec[-1]["h_final"].unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert torch.equal(rec[1]["h_in"], rec[0]["h_out"])          # 层间传的就是三值 r_out
    out = m.generate(sl.ids[0][:3], max_new=6)
    assert len(out) <= 6 and EOS not in out and all(0 <= t < sl.vocab_size for t in out)


def test_named_points_and_calibration_skip_the_merges(cfg, sl):
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="and")
    transfer_weights(FloatLM(cfg, sl.vocab_size), m)
    names = [n for n, _, _ in named_points(m)]
    assert names[0] == "q_emb" and names[1:11] == [f"blocks.0.{p}" for p in POINTS]
    assert len(names) == 1 + 10 * cfg.n_blocks
    thr = calibrate_thresholds(m, all_sequences(sl)[:16], 0.75)
    assert set(thr) == {"q_emb"} | {f"blocks.{i}.{p}" for i in range(cfg.n_blocks) for p in CALIB}
    assert all(v >= 1e-3 for v in thr.values())
    assert abs(m.q_emb.thr.item() - thr["q_emb"]) < 1e-5
    assert m.blocks[0].m1.thr.item() == 1.5 and m.blocks[0].m2.thr.item() == 1.5


def test_spike_counts_cover_every_point_including_merges(cfg, sl):
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, levels=2)
    sc = spike_counts(m, all_sequences(sl)[:8], torch.device("cpu"))
    assert set(sc["points"]) == {n for n, _, _ in named_points(m)}
    assert sc["points"]["blocks.0.m1"]["levels"] == 2 and sc["points"]["q_emb"]["neurons"] == cfg.d
    for key in ("mean_spikes_per_neuron_token", "total_spikes_per_token", "w2_events_per_token"):
        assert sc[key] >= 0


def test_a1_state_loads_into_leaky_c_model(cfg, sl):
    a = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sat", levels=2)
    c = TernaryResidualLM(cfg, sl.vocab_size, leak=cfg.leak, merge="sat", levels=2)
    c.load_state_dict(a.state_dict())
    set_leak(c, 0.5)
    assert all(lif.leak == 0.5 for lif in c.lif_modules()) and len(list(c.lif_modules())) == 8 * cfg.n_blocks


def test_gradient_flows_to_embedding_through_ternary_residual(cfg, sl):
    tok = all_sequences(sl)[:4]
    for merge in ("sat", "learn"):
        m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge=merge)
        with torch.no_grad():
            for lif in m.lif_modules():
                lif._thr.log_thr.fill_(-2.0)          # 阈值 0.135:默认 thr=1.0 时 k 几乎不发放,S 恒为 0,Wq 梯度恒为 0
        logits = m(tok[:, :-1])
        F.cross_entropy(logits.reshape(-1, sl.vocab_size), tok[:, 1:].reshape(-1)).backward()
        assert m.emb.weight.grad.abs().sum() > 0 and m.blocks[0].attn.Wq.weight.grad.abs().sum() > 0
        assert (m.blocks[0].m1.log_thr.grad is not None) == (merge == "learn")
