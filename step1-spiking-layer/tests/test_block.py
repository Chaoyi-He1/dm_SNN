"""一层 = GDN + SwiGLU + 两个累加器残差;折叠 T=1,逐 token,状态跨 token/跨分段携带。"""
import torch

from snn_spec.block import SpikingSwiGLU, SpikingBlock, run_sequence


def test_swiglu_gating_is_and_with_sign_of_content():
    torch.manual_seed(0)
    m = SpikingSwiGLU(d=16, d_ff=32)
    st = m.init_state(4, torch.device("cpu"))
    x = torch.randint(-1, 2, (4, 16)).float()
    y, st, rec = m.step(x, st, record=True)
    g, u, p = rec["g"], rec["u"], rec["p"]
    assert set(g.unique().tolist()) <= {0.0, 1.0}
    assert set(u.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(p, g * u)                       # 幅度 AND,符号取 u
    assert torch.all(p[g == 0] == 0)
    assert set(y.unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_block_step_updates_real_accumulator_by_ternary_outputs():
    torch.manual_seed(1)
    blk = SpikingBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4)
    st = blk.init_state(3, torch.device("cpu"))
    h = torch.randn(3, 16) * 2
    h_out, st, rec = blk.step(h, st, record=True)
    assert h_out.shape == (3, 16)
    assert torch.allclose(h_out, h + rec["y_attn"] + rec["y_ffn"])
    assert set(rec["y_attn"].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert set(rec["y_ffn"].unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert set(rec["x_attn"].unique().tolist()) <= {-1.0, 0.0, 1.0}   # 块输入是 Q_in(h)
    assert set(rec["x_ffn"].unique().tolist()) <= {-1.0, 0.0, 1.0}


def test_run_sequence_in_two_segments_equals_one_pass():
    torch.manual_seed(2)
    blk = SpikingBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4)
    h_seq = torch.randn(2, 8, 16) * 2
    full, _ = run_sequence(blk, h_seq)
    part1, st = run_sequence(blk, h_seq[:, :4])
    part2, _ = run_sequence(blk, h_seq[:, 4:], st)
    assert torch.allclose(full, torch.cat([part1, part2], dim=1))


def test_run_sequence_fresh_state_differs_from_carried_state():
    # Controlled subthreshold drive: first token stores 0.6; the next reaches
    # 0.9 * 0.6 + 0.6 = 1.14. Random weights can produce identical silent outputs.
    blk = SpikingBlock(d=4, d_ff=4, n_k_heads=1, n_v_heads=1, dk=4, dv=4)
    with torch.no_grad():
        blk.attn.Wo.weight.zero_()
        blk.ffn.W1.weight.copy_(0.6 * torch.eye(4))
        blk.ffn.W3.weight.copy_(0.6 * torch.eye(4))
        blk.ffn.W2.weight.copy_(torch.eye(4))
    h_seq = torch.ones(1, 1, 4)
    first, st = run_sequence(blk, h_seq)
    carried, _ = run_sequence(blk, h_seq, st)
    fresh, _ = run_sequence(blk, h_seq)
    assert torch.equal(first, h_seq)
    assert torch.equal(fresh, h_seq)
    assert torch.equal(carried, h_seq + 1)


def test_backward_reaches_projection_weights_and_input_thresholds():
    torch.manual_seed(4)
    blk = SpikingBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4)
    h_seq = torch.randn(1, 6, 16) * 2
    out, _ = run_sequence(blk, h_seq)
    (out ** 2).sum().backward()
    assert blk.attn.Wq.weight.grad is not None and blk.attn.Wq.weight.grad.abs().sum() > 0
    assert blk.ffn.W2.weight.grad is not None
    assert blk.q_in1.log_thr.grad is not None


def test_record_contains_signals_needed_for_stage2_replay():
    torch.manual_seed(5)
    blk = SpikingBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4)
    st = blk.init_state(1, torch.device("cpu"))
    _, _, rec = blk.step(torch.randn(1, 16), st, record=True)
    for key in ["h_in", "x_attn", "alpha", "beta", "q", "k", "v", "S", "o", "o_spk",
                "y_attn", "h_mid", "x_ffn", "g", "u", "p", "y_ffn", "h_out"]:
        assert key in rec, key
