"""二值(0/1)残差流语言模型:与 A0 的权重对接、前向与记录、标定、A1 → C 的装载、梯度、检查点识别、驱动脚本的挂接。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import pytest
import torch
import torch.nn.functional as F

from snn_spec.binary_residual import BinaryMerge, BinaryQuantizer
from toy_demo.binary_residual import BinaryResidualLM, block_points, calibrate_thresholds, named_points, spike_counts
from toy_demo.convert import set_leak, transfer_weights
from toy_demo.data import EOS, all_sequences, split_inputs_targets
from toy_demo.model import FloatLM
from toy_demo.ternary_residual import TernaryResidualLM

POINTS = ("q", "k", "v", "pre", "out", "m1", "g", "u", "y2", "m2")
CALIB = ("q", "k", "v", "pre", "out", "g", "u", "y2")


def _lower_thresholds(m, log_thr=-2.0):
    """阈值降到 0.135:默认 thr=1.0 时小模型几乎不发放,残差流全是 0,测不出东西。"""
    with torch.no_grad():
        m.q_emb.log_thr.fill_(log_thr)
        for lif in m.lif_modules():
            lif._thr.log_thr.fill_(log_thr)


def test_weights_transfer_from_float_and_extra_keys_are_thresholds(cfg):
    f, m = FloatLM(cfg, 20), BinaryResidualLM(cfg, 20, leak=0.0, merge="sr")
    transfer_weights(f, m)
    extra = set(m.state_dict()) - set(f.state_dict())
    assert all("log_thr" in k for k in extra)
    assert {"q_emb.log_thr", "blocks.0.m1.log_thr", "blocks.0.m2.log_thr"} <= extra
    assert not any(k.startswith(("q_out", "blocks.0.q_in")) for k in m.state_dict())
    assert set(m.state_dict()) == set(TernaryResidualLM(cfg, 20, leak=0.0).state_dict())       # 键与三值残差模型相同
    assert isinstance(m.q_emb, BinaryQuantizer) and isinstance(m.blocks[0].m1, BinaryMerge)


@pytest.mark.parametrize("merge,levels", [("sr", 1), ("src", 1), ("xor", 1), ("sr", 2), ("src", 2), ("xor", 2)])
def test_forward_record_generate_and_residual_is_unsigned(cfg, sl, merge, levels):
    tok = all_sequences(sl)[:4]
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.9, merge=merge, levels=levels)
    _lower_thresholds(m)
    logits = m(tok[:, :-1])
    assert logits.shape == (4, tok.shape[1] - 1, sl.vocab_size)
    logits2, recs = m(tok[:, :-1], record=True)
    assert torch.equal(logits, logits2) and len(recs) == tok.shape[1] - 1
    grid = {j / levels for j in range(levels + 1)}
    seen = set()
    for rec in recs:
        for i in range(cfg.n_blocks):
            for key in ("h_in", "h_mid", "h_out"):
                vals = set(rec[i][key].unique().tolist())
                assert vals <= grid, (key, vals)
                seen |= vals
        assert set(rec[-1]["h_final"].unique().tolist()) <= grid
        assert torch.equal(rec[1]["h_in"], rec[0]["h_out"])          # 层间传的就是二值 r_out
        assert torch.equal(rec[-1]["h_final"], rec[cfg.n_blocks - 1]["h_out"])
    assert seen == grid                                              # 每个电平都出现过,不是全 0 的空测试
    out = m.generate(sl.ids[0][:3], max_new=6)
    assert len(out) <= 6 and EOS not in out and all(0 <= t < sl.vocab_size for t in out)


def test_calibration_uses_the_signed_quantile_for_the_embedding_and_skips_the_merges(cfg, sl):
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sr")
    transfer_weights(FloatLM(cfg, sl.vocab_size), m)
    names = [n for n, _, _ in named_points(m)]
    assert names[0] == "q_emb" and names[1:11] == [f"blocks.0.{p}" for p in POINTS]
    tokens = all_sequences(sl)[:16]
    thr = calibrate_thresholds(m, tokens, 0.75)
    assert set(thr) == {"q_emb"} | {f"blocks.{i}.{p}" for i in range(cfg.n_blocks) for p in CALIB}
    assert all(v >= 1e-3 for v in thr.values()) and abs(m.q_emb.thr.item() - thr["q_emb"]) < 1e-5
    assert m.blocks[0].m1.thr.item() == 1.0 and m.blocks[0].m2.thr.item() == 1.0
    inputs, _, mask = split_inputs_targets(tokens)
    e = m.emb(inputs)[mask]
    assert abs(torch.quantile(e.flatten(), 0.75).item() - thr["q_emb"]) < 1e-5            # 带符号的分位数,不是 |e| 的
    fired = m.q_emb(e)
    assert set(fired.unique().tolist()) <= {0.0, 1.0} and abs(fired.mean().item() - 0.25) < 0.02


def test_xor_calibrates_the_unsigned_subblock_outputs_on_the_positive_side(cfg, sl):
    sr = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sr")
    xor = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="xor")
    flags = lambda m: {name: ternary for name, _, _, ternary in block_points(m.blocks[0])}
    assert flags(sr) == dict(q=False, k=True, v=True, pre=True, out=True, g=False, u=True, y2=True)
    assert flags(BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="src")) == flags(sr)
    assert flags(xor) == dict(q=False, k=True, v=True, pre=True, out=False, g=False, u=True, y2=False)
    mods = {name: mod for name, mod, _, _ in block_points(xor.blocks[0])}
    assert mods["out"] is xor.blocks[0].attn.sn_out and mods["y2"] is xor.blocks[0].ffn.sn_2


def test_spike_counts_cover_every_point_including_merges(cfg, sl):
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sr", levels=2)
    _lower_thresholds(m)
    sc = spike_counts(m, all_sequences(sl)[:8], torch.device("cpu"))
    assert set(sc["points"]) == {n for n, _, _ in named_points(m)}
    assert sc["points"]["blocks.0.m1"]["levels"] == 2 and sc["points"]["q_emb"]["neurons"] == cfg.d
    assert sc["points"]["q_emb"]["spikes_per_token"] > 0 and sc["total_spikes_per_token"] > 0


def test_a1_state_loads_into_leaky_c_model(cfg, sl):
    for merge in ("sr", "src", "xor"):
        a = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge=merge, levels=2)
        c = BinaryResidualLM(cfg, sl.vocab_size, leak=cfg.leak, merge=merge, levels=2)
        c.load_state_dict(a.state_dict())
        set_leak(c, 0.5)
        assert all(lif.leak == 0.5 and lif.levels == 2 for lif in c.lif_modules())
        assert len(list(c.lif_modules())) == 8 * cfg.n_blocks


@pytest.mark.parametrize("merge", ["sr", "src", "xor"])
def test_gradient_flows_to_embedding_through_binary_residual(cfg, sl, merge):
    tok = all_sequences(sl)[:4]
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge=merge)
    _lower_thresholds(m)
    logits = m(tok[:, :-1])
    F.cross_entropy(logits.reshape(-1, sl.vocab_size), tok[:, 1:].reshape(-1)).backward()
    assert m.emb.weight.grad.abs().sum() > 0 and m.q_emb.log_thr.grad.abs().sum() > 0
    assert m.blocks[0].attn.Wq.weight.grad.abs().sum() > 0 and m.blocks[0].ffn.W2.weight.grad.abs().sum() > 0
    assert m.blocks[0].m1.log_thr.grad is None


def test_describe_and_load_checkpoint_recognise_binary_residual(cfg, sl, tmp_path):
    import json
    from experiments.beam_eval import SIZE_KEYS, describe_checkpoint, load_model
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="xor", levels=2)
    _lower_thresholds(m)
    d = tmp_path / "binres_xor_T2_a1"
    d.mkdir()
    torch.save(m.state_dict(), d / "model.pt")
    args = dict(stage="a1", residual="binary", merge="xor", levels=2)
    args.update({k: getattr(cfg, key) for k, key in SIZE_KEYS})
    (d / "result.json").write_text(json.dumps(dict(args=args)), encoding="utf-8")
    info = describe_checkpoint(d / "model.pt")
    assert info["kind"] == "binres" and info["merge"] == "xor" and info["levels"] == 2 and info["leak"] == 0.0
    mm, _, _ = load_model(d / "model.pt", sl, torch.device("cpu"))
    assert type(mm) is BinaryResidualLM and mm.merge == "xor" and mm.blocks[0].m1.levels == 2
    assert torch.equal(mm(all_sequences(sl)[:2, :-1]), m(all_sequences(sl)[:2, :-1]))
    args["residual"] = "ternary"                                     # 三值残差的识别不受影响
    args["merge"] = "sat"
    (d / "result.json").write_text(json.dumps(dict(args=args)), encoding="utf-8")
    assert describe_checkpoint(d / "model.pt")["kind"] == "ternres"


def test_driver_builds_binary_models_and_restores_the_ternary_driver(cfg, sl, demo):
    import experiments.ternary_residual as tr
    from experiments import binary_residual as br
    from experiments.convert_lab import train
    tokens = all_sequences(sl)
    originals = (tr.make_model, tr.calibrate_thresholds, tr.save_stage)
    m, thr = br.build_a1(FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, torch.device("cpu"), fq=0.75, merge="sr", levels=2, seed=0)
    assert type(m) is BinaryResidualLM and m.merge == "sr" and m.levels == 2 and "q_emb" in thr
    assert (tr.make_model, tr.calibrate_thresholds, tr.save_stage) == originals            # 退出后还原
    with br.binary_driver():
        assert tr.make_model is br.make_model and tr.save_stage is br.save_stage
    t, _ = tr.build_a1(FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, torch.device("cpu"), fq=0.75, merge="sat", levels=1, seed=0)
    assert type(t) is TernaryResidualLM
    res = train(m, sl, demo, cfg, tokens, lr=1e-3, steps=3, seed=0, eval_every=3, stop=False, log=lambda *_: None)
    assert res["history"][-1]["step"] == 3 and 0.0 <= res["final"]["token_acc"] <= 1.0


def test_save_stage_marks_the_run_as_binary(cfg, sl, tmp_path):
    import argparse
    import json
    from experiments import binary_residual as br
    m = BinaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sr", levels=2)
    args = argparse.Namespace(stage="a1", name="x", merge="sr", levels=2, fq=0.75)
    res = dict(final=dict(token_acc=0.5, ppl=2.0, n_exact=1, n_demo=5), stopped="max_steps", seconds=1.0)
    br.save_stage(tmp_path / "x", m, res, args, cfg, {})
    saved = json.loads((tmp_path / "x" / "result.json").read_text(encoding="utf-8"))["args"]
    assert saved["residual"] == "binary" and saved["merge"] == "sr" and saved["d"] == cfg.d
    from experiments.beam_eval import describe_checkpoint
    assert describe_checkpoint(tmp_path / "x" / "model.pt")["kind"] == "binres"
