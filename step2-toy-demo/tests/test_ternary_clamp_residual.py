"""三值对照模型 TernaryClampResidualLM(merge = satc):前向与 TernaryResidualLM 的 sat 相同、梯度不同;驱动与检查点装载的分派。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import json
import pytest
import torch
import torch.nn.functional as F

from snn_spec.binary_residual import SignedClampMerge
from toy_demo.binary_residual import BinaryResidualLM, TernaryClampResidualLM
from toy_demo.data import all_sequences
from toy_demo.model import FloatLM
from toy_demo.ternary_residual import TernaryResidualLM


def _lower_thresholds(m, log_thr=-2.0):
    with torch.no_grad():
        m.q_emb.log_thr.fill_(log_thr)
        for lif in m.lif_modules():
            lif._thr.log_thr.fill_(log_thr)


@pytest.mark.parametrize("levels", [1, 2])
def test_forward_is_bit_identical_to_the_sat_model_with_the_same_weights(cfg, sl, levels):
    tok = all_sequences(sl)[:6]
    sat = TernaryResidualLM(cfg, sl.vocab_size, leak=0.9, merge="sat", levels=levels)
    _lower_thresholds(sat)
    ctl = TernaryClampResidualLM(cfg, sl.vocab_size, leak=0.9, levels=levels)
    ctl.load_state_dict(sat.state_dict())                              # 键完全相同
    assert ctl.merge == "satc" and all(isinstance(m, SignedClampMerge) for b in ctl.blocks for m in (b.m1, b.m2))
    a, recs_a = sat(tok[:, :-1], record=True)
    b, recs_b = ctl(tok[:, :-1], record=True)
    assert torch.equal(a, b)
    assert all(torch.equal(recs_a[t][i]["h_out"], recs_b[t][i]["h_out"]) for t in range(len(recs_a)) for i in range(cfg.n_blocks))
    assert recs_a[0][0]["h_out"].abs().sum() > 0                        # 残差流不是全 0


def test_gradient_differs_from_the_sat_model_and_reaches_the_embedding(cfg, sl):
    tok = all_sequences(sl)[:6]
    grads = {}
    for name, cls, merge in (("sat", TernaryResidualLM, "sat"), ("satc", TernaryClampResidualLM, "satc")):
        torch.manual_seed(0)
        m = cls(cfg, sl.vocab_size, leak=0.0, merge=merge)
        _lower_thresholds(m)
        logits = m(tok[:, :-1])
        F.cross_entropy(logits.reshape(-1, sl.vocab_size), tok[:, 1:].reshape(-1)).backward()
        grads[name] = m.emb.weight.grad.clone()
    assert grads["satc"].abs().sum() > 0 and not torch.allclose(grads["sat"], grads["satc"])


def test_only_satc_is_accepted(cfg, sl):
    with pytest.raises(ValueError):
        TernaryClampResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sat")


def test_driver_dispatches_model_calibration_and_residual_label(cfg, sl, tmp_path):
    import argparse
    from experiments import binary_residual as br
    from experiments.beam_eval import describe_checkpoint, load_model
    tokens = all_sequences(sl)
    dev = torch.device("cpu")
    ctl, thr_c = br.build_a1(FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, dev, fq=0.75, merge="satc", levels=2, seed=0)
    ref, thr_r = __import__("experiments.ternary_residual", fromlist=["build_a1"]).build_a1(
        FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, dev, fq=0.75, merge="sat", levels=2, seed=0)
    assert type(ctl) is TernaryClampResidualLM and thr_c == thr_r               # 标定走三值残差的函数,阈值相同
    binm, _ = br.build_a1(FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, dev, fq=0.75, merge="src", levels=2, seed=0)
    assert type(binm) is BinaryResidualLM and binm.merge == "src"
    res = dict(final=dict(token_acc=0.5, ppl=2.0, n_exact=1, n_demo=5), stopped="max_steps", seconds=1.0)
    for name, model, merge, residual, kind in (("c", ctl, "satc", "ternary", "ternres"), ("b", binm, "src", "binary", "binres")):
        args = argparse.Namespace(stage="a1", name=name, merge=merge, levels=2, fq=0.75)
        br.save_stage(tmp_path / name, model, res, args, cfg, {})
        saved = json.loads((tmp_path / name / "result.json").read_text(encoding="utf-8"))["args"]
        assert saved["residual"] == residual and saved["merge"] == merge
        info = describe_checkpoint(tmp_path / name / "model.pt")
        assert info["kind"] == kind and info["merge"] == merge
        loaded, _, _ = load_model(tmp_path / name / "model.pt", sl, dev)
        assert type(loaded) is type(model)
        assert torch.equal(loaded(tokens[:2, :-1]), model.eval()(tokens[:2, :-1]))
