"""残差流全程二值(0/1)的语言模型 BinaryResidualLM,以及它的阈值标定。

与 TernaryResidualLM 的差别:嵌入查表后过的是无符号量化器(嵌入值 >= 阈值得 1,否则 0,负的一半在这里丢掉),
各块是 snn_spec.binary_residual.BinaryResidualBlock(残差流 r ∈ {0, 1},levels=T 时 {0, 1/T, ..., 1};
合并 sr、src 或 xor),输出头直接读二值的 r。前向、记录字段与脉冲点的名字都沿用 TernaryResidualLM,
所以 named_points、spike_counts 直接复用(二值量取绝对值不变)。
权重名称与 FloatLM 一致,transfer_weights 可直接装载(多出的只有各处阈值 log_thr)。
"""
import torch
import torch.nn as nn

from snn_spec.binary_residual import BinaryQuantizer, BinaryResidualBlock, SignedClampMerge
from snn_spec.gdn import SigmoidGates

from .convert import _set_thr
from .data import split_inputs_targets
from .model import _LMBase
from .ternary_residual import TernaryResidualLM, named_points, spike_counts  # noqa: F401  (后两个原样复用并从这里导出)
from .ternary_residual import block_points as _ternary_block_points


class BinaryResidualLM(TernaryResidualLM):
    """step、lif_modules 沿用 TernaryResidualLM;只有构造不同(无符号的 q_emb 与二值残差块)。"""

    def __init__(self, cfg, vocab, leak, merge="sr", levels=1):
        _LMBase.__init__(self)
        self.cfg, self.vocab, self.merge, self.levels = cfg, vocab, merge, int(levels)
        rng = (cfg.alpha_min, cfg.alpha_max)
        self.emb = nn.Embedding(vocab, cfg.d)
        self.q_emb = BinaryQuantizer(1.0, levels=levels)
        self.blocks = nn.ModuleList([
            BinaryResidualBlock(cfg.d, cfg.d_ff, cfg.n_heads, cfg.n_heads, cfg.dk, cfg.dv,
                                gates=SigmoidGates(cfg.d, cfg.n_heads, rng), alpha_rng=rng,
                                leak=leak, thr=1.0, learn_thr=True, merge=merge, levels=levels)
            for _ in range(cfg.n_blocks)])
        self.head = nn.Linear(cfg.d, vocab, bias=False)


class TernaryClampResidualLM(TernaryResidualLM):
    """对照模型(merge = "satc"):三值残差流,前向与 TernaryResidualLM 的 sat 逐位相同,只是两处合并换成 SignedClampMerge,
    反向用 clamp 的精确梯度而不是代理梯度。二值的 src 用的是精确梯度,拿它和原来的 sat(代理梯度)比会混进梯度写法的差别,
    所以三值这边补一个同样写法的。标定、脉冲计数都沿用 toy_demo.ternary_residual 的函数;state_dict 的键与 TernaryResidualLM 相同。"""

    def __init__(self, cfg, vocab, leak, merge="satc", levels=1):
        if merge != "satc":
            raise ValueError(f"TernaryClampResidualLM only implements merge='satc', got {merge!r}")
        super().__init__(cfg, vocab, leak, merge="sat", levels=levels)
        self.merge = "satc"
        for blk in self.blocks:
            blk.merge = "satc"
            blk.m1, blk.m2 = SignedClampMerge(levels), SignedClampMerge(levels)


def block_points(blk):
    """块内按预激活分位数标定的 8 个 LIF 点:(名字, 模块, 由记录算预激活, 是否三值)。
    与三值残差相同,只是 xor 合并时子块输出 out、y2 是无符号的二值 LIF,按正的一侧取分位数。"""
    pts = _ternary_block_points(blk)
    if blk.merge == "xor":
        pts = [(name, mod, fn, ternary and name not in ("out", "y2")) for name, mod, fn, ternary in pts]
    return pts


@torch.no_grad()
def calibrate_thresholds(model, tokens, quantile=0.75):
    """tokens [N, L] 含 BOS/EOS/PAD。按前向顺序逐点标定(先 q_emb,再每块的 8 个 LIF),下限 1e-3;合并 m1/m2 不动。
    q_emb 与二值点的阈值 = 有效位置上预激活(带符号)的分位数,即约 1 - quantile 的通道达到满幅 1
    (levels=T > 1 时还有一部分通道只到 j/T);三值点取绝对值的分位数。
    返回 {点名: 阈值}。"""
    inputs, _, mask = split_inputs_targets(tokens)
    out = {}

    def to_thr(z, ternary):
        z = z.abs() if ternary else z
        return max(torch.quantile(z.flatten().float(), quantile).item(), 1e-3)

    def quantile_of(fn, block_index, ternary):
        _, recs = model(inputs, record=True)
        z = torch.stack([fn(recs[t][block_index]) for t in range(inputs.shape[1])], dim=1)[mask]
        return to_thr(z, ternary)

    thr = to_thr(model.emb(inputs)[mask], False)
    _set_thr(model.q_emb, thr)
    out["q_emb"] = thr
    for i, blk in enumerate(model.blocks):
        for name, mod, fn, ternary in block_points(blk):
            thr = quantile_of(fn, i, ternary)
            _set_thr(mod, thr)
            out[f"blocks.{i}.{name}"] = thr
    return out
