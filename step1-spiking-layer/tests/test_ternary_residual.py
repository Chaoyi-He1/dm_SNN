"""残差流全程三值的一层:两输入合并 M 的真值表、块的输出取值、梯度能否穿过。"""
import itertools
import torch

from snn_spec.neurons import QuantizerIn
from snn_spec.ternary_residual import ResidualMerge, TernaryResidualBlock

GRID = torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=2)))
A, B = GRID[:, 0], GRID[:, 1]


def test_sat_merge_is_saturating_add():
    assert torch.equal(ResidualMerge("sat")(A, B), torch.sign(A + B))


def test_and_merge_keeps_only_agreeing_signs():
    expected = torch.where((A + B).abs() == 2, torch.sign(A + B), torch.zeros_like(A))
    assert torch.equal(ResidualMerge("and")(A, B), expected)


def test_merge_levels_two_clamps_half_steps():
    grid = torch.tensor(list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0), repeat=2)))
    a, b = grid[:, 0], grid[:, 1]
    assert torch.equal(ResidualMerge("sat", levels=2)(a, b), torch.clamp(a + b, -1.0, 1.0))
    m = ResidualMerge("and", levels=2)                      # 阈值 0.75、1.5
    pairs = [((0.5, 0.0), 0.0), ((0.5, 0.5), 0.5), ((1.0, 0.5), 1.0), ((1.0, 1.0), 1.0), ((-1.0, -0.5), -1.0), ((1.0, -0.5), 0.0)]
    for (x, y), want in pairs:
        assert m(torch.tensor([x]), torch.tensor([y])).item() == want, (x, y)


def test_fixed_merges_have_frozen_threshold_and_learn_does_not():
    assert ResidualMerge("sat").thr.item() == 1.0 and not ResidualMerge("sat").log_thr.requires_grad
    assert ResidualMerge("and").thr.item() == 1.5 and not ResidualMerge("and").log_thr.requires_grad
    assert ResidualMerge("learn").thr.item() == 1.0 and ResidualMerge("learn").log_thr.requires_grad
    try:
        ResidualMerge("max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown merge must raise")


def _block(**kw):
    torch.manual_seed(0)
    return TernaryResidualBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4, **kw)


def test_block_residual_stream_is_ternary_and_merges_are_used():
    blk = _block(merge="sat")
    st = blk.init_state(3, torch.device("cpu"))
    r = torch.randint(-1, 2, (3, 16)).float()
    r_out, st, rec = blk.step(r, st, record=True)
    for key in ("x_attn", "h_mid", "x_ffn", "h_out", "y_attn", "y_ffn"):
        assert set(rec[key].unique().tolist()) <= {-1.0, 0.0, 1.0}, key
    assert torch.equal(rec["x_attn"], r) and torch.equal(rec["x_ffn"], rec["h_mid"]) and torch.equal(r_out, rec["h_out"])
    assert torch.equal(rec["h_mid"], blk.m1(rec["x_attn"], rec["y_attn"]))
    assert torch.equal(rec["h_out"], blk.m2(rec["h_mid"], rec["y_ffn"]))
    assert torch.equal(rec["s_mid"], r + rec["y_attn"]) and torch.equal(rec["s_out"], rec["h_mid"] + rec["y_ffn"])
    assert len(blk.lif_modules()) == 8


def test_block_levels_two_sets_every_spike_point_and_residual_to_half_steps():
    blk = _block(merge="sat", levels=2)
    assert all(m.levels == 2 for m in blk.lif_modules()) and blk.m1.levels == 2 and blk.m2.levels == 2
    r = torch.randint(-2, 3, (3, 16)).float() / 2
    _, _, rec = blk.step(r, blk.init_state(3, torch.device("cpu")), record=True)
    for key in ("h_mid", "h_out", "y_attn", "y_ffn", "q", "v"):
        v = rec[key]
        assert torch.equal((v * 2).round(), v * 2) and v.abs().max() <= 1.0, key


def test_silent_subblocks_pass_the_residual_through():
    blk = _block(merge="sat")
    with torch.no_grad():
        blk.attn.Wo.weight.zero_()
        blk.ffn.W2.weight.zero_()
    r = torch.randint(-1, 2, (3, 16)).float()
    r_out, _, rec = blk.step(r, blk.init_state(3, torch.device("cpu")), record=True)
    assert torch.all(rec["y_attn"] == 0) and torch.all(rec["y_ffn"] == 0)
    assert torch.equal(r_out, r)


def test_gradient_reaches_input_and_weights_through_the_ternary_residual():
    for merge in ("sat", "learn"):
        blk = _block(merge=merge, learn_thr=True, thr=0.5)   # 单 token、零初始状态:默认 thr=1.0 时 k 常常一个都不发放,梯度恒为 0
        e = (torch.randn(3, 16) * 2).requires_grad_()
        r = QuantizerIn(0.5)(e)
        r_out, _, _ = blk.step(r, blk.init_state(3, torch.device("cpu")))
        (r_out ** 2).sum().backward()
        assert e.grad is not None and e.grad.abs().sum() > 0
        assert blk.attn.Wq.weight.grad.abs().sum() > 0 and blk.ffn.W1.weight.grad.abs().sum() > 0
        assert (blk.m1.log_thr.grad is not None) == (merge == "learn")
