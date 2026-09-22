"""A0 -> A1 的权重转移与阈值初始化;A1 -> B 的 leak 切换(计划第 4 节)。

阈值标定按前向顺序逐点进行:一个脉冲点的预激活只依赖它之前的点(块内前馈;跨 token 只经由更早的块和状态),
所以先定前面的阈值再取后面的分位数是精确的。二值点取预激活的分位数,三值点取绝对值的分位数。
"""
import math

import torch

from .data import split_inputs_targets


def transfer_weights(float_lm, spiking_lm):
    res = spiking_lm.load_state_dict(float_lm.state_dict(), strict=False)
    bad = [k for k in res.missing_keys if "log_thr" not in k]
    if bad or res.unexpected_keys:
        raise ValueError(f"missing {bad}, unexpected {list(res.unexpected_keys)}")
    return spiking_lm


def set_leak(spiking_lm, leak):
    for m in spiking_lm.lif_modules():
        m.leak = float(leak)
    return spiking_lm


def _set_thr(module, value):
    """QuantizerIn 直接持有 log_thr;LIF 经 _thr.log_thr。阈值下限 1e-3。"""
    target = module.log_thr if hasattr(module, "log_thr") else module._thr.log_thr
    with torch.no_grad():
        target.fill_(math.log(max(float(value), 1e-3)))


def _block_points(blk):
    """块内脉冲点,按前向顺序:(名字, 模块, 由记录算预激活 [B, C], 是否三值)。"""
    a, f = blk.attn, blk.ffn
    return [
        ("q_in1", blk.q_in1, lambda r: r["h_in"], True),
        ("q", a.sn_q, lambda r: a.Wq(r["x_attn"]), False),
        ("k", a.sn_k, lambda r: a.Wk(r["x_attn"]), True),
        ("v", a.sn_v, lambda r: a.Wv(r["x_attn"]), True),
        ("pre", a.sn_pre, lambda r: r["o"].reshape(r["o"].shape[0], -1), True),
        ("out", a.sn_out, lambda r: a.Wo(r["o_spk"]), True),
        ("q_in2", blk.q_in2, lambda r: r["h_mid"], True),
        ("g", f.sn_1, lambda r: f.W1(r["x_ffn"]), False),
        ("u", f.sn_3, lambda r: f.W3(r["x_ffn"]), True),
        ("y2", f.sn_2, lambda r: f.W2(r["p"]), True),
    ]


@torch.no_grad()
def calibrate_thresholds(spiking_lm, tokens, quantile=0.75):
    """tokens [N, L] 含 BOS/EOS/PAD。每个点:运行模型,取有效位置上的预激活,阈值 = 分位数。返回 {点名: 阈值}。"""
    inputs, _, mask = split_inputs_targets(tokens)
    out = {}

    def quantile_of(fn, block_index, ternary):
        _, recs = spiking_lm(inputs, record=True)
        z = torch.stack([fn(recs[t][block_index]) for t in range(inputs.shape[1])], dim=1)[mask]
        z = z.abs() if ternary else z
        return max(torch.quantile(z.flatten().float(), quantile).item(), 1e-3)

    for i, blk in enumerate(spiking_lm.blocks):
        for name, mod, fn, ternary in _block_points(blk):
            thr = quantile_of(fn, i, ternary)
            _set_thr(mod, thr)
            out[f"blocks.{i}.{name}"] = thr
    thr = quantile_of(lambda r: r["h_final"], len(spiking_lm.blocks), True)
    _set_thr(spiking_lm.q_out, thr)
    out["q_out"] = thr
    return out
