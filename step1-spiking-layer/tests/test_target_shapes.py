"""目标层(Qwen3.5-0.8B 的一个原生 GDN 层)按规范实例化后的形状与规模。
数值来自 HF `Qwen/Qwen3.5-0.8B` config.json 的 text_config(2026-09-14 读取)。"""
import torch

from snn_spec.block import SpikingBlock

D, D_FF = 1024, 3584                      # hidden_size, intermediate_size
HK, HV, DK, DV = 16, 16, 128, 128         # linear_num_key/value_heads, linear_key/value_head_dim
N_LAYERS, FULL_ATTN_IDX = 24, [3, 7, 11, 15, 19, 23]


def test_layer_types_give_18_gdn_and_6_full_attention_layers():
    layer_types = ["full_attention" if i in FULL_ATTN_IDX else "linear_attention" for i in range(N_LAYERS)]
    assert layer_types.count("linear_attention") == 18
    assert layer_types.count("full_attention") == 6
    assert all((i + 1) % 4 == 0 for i in FULL_ATTN_IDX)          # full_attention_interval = 4


def test_target_block_instantiates_and_steps_with_qwen35_0p8b_shapes():
    torch.manual_seed(0)
    blk = SpikingBlock(d=D, d_ff=D_FF, n_k_heads=HK, n_v_heads=HV, dk=DK, dv=DV)
    st = blk.init_state(1, torch.device("cpu"))
    assert st["attn"]["S"].shape == (1, HV, DK, DV)
    h_out, st, rec = blk.step(torch.randn(1, D), st, record=True)
    assert h_out.shape == (1, D)
    assert rec["q"].shape == (1, HV, DK) and rec["k"].shape == (1, HV, DK) and rec["v"].shape == (1, HV, DV)
    assert rec["o"].shape == (1, HV, DV) and rec["alpha"].shape == (1, HV)


def test_target_layer_component_counts():
    blk = SpikingBlock(d=D, d_ff=D_FF, n_k_heads=HK, n_v_heads=HV, dk=DK, dv=DV)
    attn_weights = sum(m.weight.numel() for m in (blk.attn.Wq, blk.attn.Wk, blk.attn.Wv, blk.attn.Wo))
    ffn_weights = sum(m.weight.numel() for m in (blk.ffn.W1, blk.ffn.W3, blk.ffn.W2))
    assert attn_weights == 8_388_608                 # 3 x 1024x2048 + 2048x1024
    assert ffn_weights == 11_010_048                 # 3 x 1024x3584
    st = blk.init_state(1, torch.device("cpu"))
    neurons = sum(v.numel() for grp in st.values() for k, v in grp.items() if k.startswith("U"))
    assert neurons == 17_408                         # q,k,v,pre 各 2048;out 1024;g,u 各 3584;ffn-out 1024
    assert st["attn"]["S"].numel() == 262_144        # 16 头 x 128 x 128 个 gain cell
