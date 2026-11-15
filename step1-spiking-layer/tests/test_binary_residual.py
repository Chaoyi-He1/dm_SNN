"""残差流全程二值(0/1)的一层:无符号量化器、两种合并规则的真值表与梯度、块的输出取值、恒等通路。"""
import itertools
import pytest
import torch

from snn_spec.binary_residual import BinaryMerge, BinaryQuantizer, BinaryResidualBlock
from snn_spec.neurons import BinaryLIF, TernaryLIF


def _grid(rs, ys):
    g = torch.tensor(list(itertools.product(rs, ys)))
    return g[:, 0], g[:, 1]


# ----------------------------------------------------------------------------- 量化器

def test_quantizer_is_unsigned_and_drops_the_negative_half():
    q = BinaryQuantizer(0.5)
    h = torch.tensor([-3.0, -0.5, 0.0, 0.49, 0.5, 2.0])
    assert torch.equal(q(h), torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
    assert q.log_thr.requires_grad and abs(q.thr.item() - 0.5) < 1e-6


def test_quantizer_levels_two_counts_half_thresholds():
    q = BinaryQuantizer(1.0, levels=2)
    h = torch.tensor([-1.0, 0.49, 0.5, 0.99, 1.0, 5.0])
    assert torch.equal(q(h), torch.tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0]))


# ----------------------------------------------------------------------------- 合并:sr

def test_sr_merge_is_a_set_reset_latch():
    r, y = _grid((0.0, 1.0), (-1.0, 0.0, 1.0))
    out = BinaryMerge("sr")(r, y)
    assert torch.equal(out, torch.clamp(r + y, 0.0, 1.0))
    latch = ((r > 0) | (y > 0)) & ~(y < 0)                 # r' = (r OR y+) AND NOT y-
    assert torch.equal(out, latch.float())


def test_sr_merge_levels_two_is_a_saturating_counter():
    r, y = _grid((0.0, 0.5, 1.0), (-1.0, -0.5, 0.0, 0.5, 1.0))
    assert torch.equal(BinaryMerge("sr", levels=2)(r, y), torch.clamp(r + y, 0.0, 1.0))
    r, y = _grid((0.0, 0.25, 0.5, 0.75, 1.0), tuple(j / 4 for j in range(-4, 5)))
    assert torch.equal(BinaryMerge("sr", levels=4)(r, y), torch.clamp(r + y, 0.0, 1.0))


def test_sr_threshold_is_one_and_frozen_and_unknown_merge_raises():
    m = BinaryMerge("sr")
    assert m.thr.item() == 1.0 and not m.log_thr.requires_grad and "log_thr" in m.state_dict()
    with pytest.raises(ValueError):
        BinaryMerge("sat")


# ----------------------------------------------------------------------------- 合并:xor

def test_xor_merge_toggles_and_generalises_to_absolute_difference():
    r, y = _grid((0.0, 1.0), (0.0, 1.0))
    assert torch.equal(BinaryMerge("xor")(r, y), (r != y).float())
    r, y = _grid((0.0, 0.5, 1.0), (0.0, 0.5, 1.0))
    assert torch.equal(BinaryMerge("xor", levels=2)(r, y), (r - y).abs())


def test_xor_gradient_uses_the_feasible_one_sided_slope_at_ties():
    cases = [((0.0, 0.0), (1.0, 1.0)),      # 都为 0:只能往上走,斜率 +1(恒等通路的梯度是 1)
             ((1.0, 0.0), (1.0, -1.0)),
             ((0.0, 1.0), (-1.0, 1.0)),
             ((1.0, 1.0), (-1.0, -1.0)),    # 都为 1:只能往下走
             ((0.5, 0.5), (0.0, 0.0)),      # 中间电平相等:尖点,取 0
             ((1.0, 0.5), (1.0, -1.0))]
    for (rv, yv), (gr, gy) in cases:
        r, y = torch.tensor([rv], requires_grad=True), torch.tensor([yv], requires_grad=True)
        BinaryMerge("xor", levels=2)(r, y).sum().backward()
        assert (r.grad.item(), y.grad.item()) == (gr, gy), (rv, yv)


# ----------------------------------------------------------------------------- 块

def _block(**kw):
    torch.manual_seed(0)
    return BinaryResidualBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4, **kw)


def _on_grid(v, T, signed):
    lo = -1.0 if signed else 0.0
    return torch.equal((v * T).round(), v * T) and v.min() >= lo and v.max() <= 1.0


@pytest.mark.parametrize("merge", ["sr", "xor"])
def test_block_residual_stream_is_binary_and_merges_are_used(merge):
    blk = _block(merge=merge, thr=0.3)
    r = torch.randint(0, 2, (5, 16)).float()
    r_out, _, rec = blk.step(r, blk.init_state(5, torch.device("cpu")), record=True)
    for key in ("x_attn", "h_mid", "x_ffn", "h_out"):
        assert set(rec[key].unique().tolist()) <= {0.0, 1.0}, key
    assert torch.equal(rec["x_attn"], r) and torch.equal(rec["x_ffn"], rec["h_mid"]) and torch.equal(r_out, rec["h_out"])
    assert torch.equal(rec["h_mid"], blk.m1(r, rec["y_attn"])) and torch.equal(rec["h_out"], blk.m2(rec["h_mid"], rec["y_ffn"]))
    assert isinstance(blk.m1, BinaryMerge) and isinstance(blk.m2, BinaryMerge) and blk.merge == merge
    assert len(blk.lif_modules()) == 8
    signed = merge == "sr"                                  # sr:子块输出仍是三值;xor:子块输出也无符号
    for key in ("y_attn", "y_ffn"):
        assert _on_grid(rec[key], 1, signed), key
        assert rec[key].abs().sum() > 0, key                # 测试确实让子块发放了
    want = TernaryLIF if signed else BinaryLIF
    assert type(blk.attn.sn_out) is want and type(blk.ffn.sn_2) is want
    assert type(blk.attn.sn_k) is TernaryLIF and type(blk.ffn.sn_3) is TernaryLIF      # 内部脉冲点不动


def test_sr_block_sets_and_resets_channels():
    blk = _block(merge="sr", thr=0.2)
    r = torch.randint(0, 2, (32, 16)).float()
    _, _, rec = blk.step(r, blk.init_state(32, torch.device("cpu")), record=True)
    for y, before, after in ((rec["y_attn"], r, rec["h_mid"]), (rec["y_ffn"], rec["h_mid"], rec["h_out"])):
        assert (y < 0).any() and (y > 0).any()
        assert (after[y < 0] == 0).all() and (after[y > 0] == 1).all()          # 负脉冲清 0,正脉冲置 1
        assert torch.equal(after[y == 0], before[y == 0])                        # 没有脉冲时保持


@pytest.mark.parametrize("merge", ["sr", "xor"])
def test_block_levels_two_sets_every_spike_point_and_keeps_the_stream_unsigned(merge):
    blk = _block(merge=merge, levels=2, thr=0.3)
    assert all(m.levels == 2 for m in blk.lif_modules()) and blk.m1.levels == 2 and blk.m2.levels == 2
    r = torch.randint(0, 3, (5, 16)).float() / 2
    _, _, rec = blk.step(r, blk.init_state(5, torch.device("cpu")), record=True)
    for key in ("h_mid", "h_out"):
        assert _on_grid(rec[key], 2, signed=False), key
    for key in ("y_attn", "y_ffn"):
        assert _on_grid(rec[key], 2, signed=(merge == "sr")), key


@pytest.mark.parametrize("merge", ["sr", "xor"])
def test_silent_subblocks_pass_the_residual_through(merge):
    blk = _block(merge=merge)
    with torch.no_grad():
        blk.attn.Wo.weight.zero_()
        blk.ffn.W2.weight.zero_()
    r = torch.randint(0, 2, (3, 16)).float()
    r_out, _, rec = blk.step(r, blk.init_state(3, torch.device("cpu")), record=True)
    assert torch.all(rec["y_attn"] == 0) and torch.all(rec["y_ffn"] == 0)
    assert torch.equal(r_out, r)


@pytest.mark.parametrize("merge", ["sr", "xor"])
def test_gradient_reaches_input_and_weights_through_the_binary_residual(merge):
    blk = _block(merge=merge, learn_thr=True, thr=0.2)       # 单 token、零初始状态:阈值高时 k、v 不发放,S 恒为 0,Wq 梯度恰为 0
    e = (torch.randn(8, 16) * 2).requires_grad_()
    r = BinaryQuantizer(0.5)(e)
    r_out, _, rec = blk.step(r, blk.init_state(8, torch.device("cpu")), record=True)
    assert rec["k"].abs().sum() > 0 and rec["v"].abs().sum() > 0
    (r_out ** 2).sum().backward()
    assert e.grad is not None and e.grad.abs().sum() > 0
    assert blk.attn.Wq.weight.grad.abs().sum() > 0 and blk.ffn.W1.weight.grad.abs().sum() > 0
    assert blk.attn.Wo.weight.grad.abs().sum() > 0 and blk.ffn.W2.weight.grad.abs().sum() > 0
    assert blk.m1.log_thr.grad is None
