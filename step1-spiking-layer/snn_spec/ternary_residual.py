"""残差流全程三值的一层:r → GDN → h' = M1(r + y1) → SwiGLU → r_out = M2(h' + y2)。

与 SpikingBlock 的差别:SpikingBlock 的残差流 h 是实值累加器,块入口要过 Q_in 才变成三值;
这里的残差流本身就是三值(取值 {-1,0,+1},levels=T 时是 j/T 的整数倍),块入口直接把 r 送进线性层,
没有 Q_in,也没有实值的 h。两个"加法"换成两输入合并 M(两个三值相加,再过一对比较器)。

电路含义:层间连线和 GDN→FFN 之间的连线只传 ±1/0,所以累加器电容和 DAC 都不需要了;
每个通道只保留一次三值加法加一个比较器(levels=T 时是 2T 个比较器,阈值 ±j·thr/T)。

合并方式 merge:
  sat    阈值 1.0,饱和加法 clamp(a+b, -1, 1)(T=1 时等于 sign(a+b))。
  and    阈值 1.5,只在两路同号时才输出,异号或只有一路非零时输出 0(T=1;T>1 时阈值 ±j·1.5/T,单路满幅得 1/T)。
  learn  阈值初值 1.0,逐层可学(恒正,存成 log_thr)。
"""
import torch
import torch.nn as nn

from .neurons import level_ternary_fire
from .gdn import SpikingGDN
from .block import SpikingSwiGLU

_MERGE_THR0 = {"sat": 1.0, "and": 1.5, "learn": 1.0}


class ResidualMerge(nn.Module):
    """两输入合并 M(a, b) = 三值阈值(a + b)。无状态。
    阈值 thr = exp(log_thr);只有 merge == "learn" 时 log_thr 参与训练,其余两种固定。"""

    def __init__(self, merge="sat", levels=1, sg_width=2.0):
        super().__init__()
        if merge not in _MERGE_THR0:
            raise ValueError(f"merge must be one of {tuple(_MERGE_THR0)}, got {merge!r}")
        self.merge = merge
        self.levels = int(levels)
        self.sg_width = float(sg_width)
        self.log_thr = nn.Parameter(torch.log(torch.tensor(_MERGE_THR0[merge])),
                                    requires_grad=(merge == "learn"))

    @property
    def thr(self):
        return self.log_thr.exp()

    def forward(self, a, b):
        return level_ternary_fire(a + b, self.thr, self.levels, self.sg_width)


class TernaryResidualBlock(nn.Module):
    """一个 decoder block,残差流全程三值:GDN → M1 合并 → SwiGLU → M2 合并。"""

    def __init__(self, d, d_ff, n_k_heads, n_v_heads, dk, dv, gates=None, fir_taps=0,
                 alpha_rng=(0.5, 0.99), leak=0.9, thr=1.0, learn_thr=False, merge="sat", levels=1):
        super().__init__()
        self.merge = merge
        self.levels = int(levels)
        self.attn = SpikingGDN(d, n_k_heads, n_v_heads, dk, dv, gates=gates, fir_taps=fir_taps,
                               alpha_rng=alpha_rng, leak=leak, thr=thr, learn_thr=learn_thr)
        self.ffn = SpikingSwiGLU(d, d_ff, leak=leak, thr=thr, learn_thr=learn_thr)
        self.m1 = ResidualMerge(merge, levels)
        self.m2 = ResidualMerge(merge, levels)
        for m in self.lif_modules():
            m.levels = self.levels

    def lif_modules(self):
        a, f = self.attn, self.ffn
        return [a.sn_q, a.sn_k, a.sn_v, a.sn_pre, a.sn_out, f.sn_1, f.sn_3, f.sn_2]

    def init_state(self, B, device):
        return dict(attn=self.attn.init_state(B, device), ffn=self.ffn.init_state(B, device))

    def step(self, r, st, ext_gates=None, record=False):
        """r: [B, d] 三值残差(块入口)。返回 (r_out, st, rec)。"""
        y1, st["attn"], r1 = self.attn.step(r, st["attn"], ext_gates=ext_gates, record=record)
        h_mid = self.m1(r, y1)
        y2, st["ffn"], r2 = self.ffn.step(h_mid, st["ffn"], record=record)
        r_out = self.m2(h_mid, y2)
        rec = None
        if record:
            rec = dict(h_in=r, x_attn=r, alpha=r1["alpha"], beta=r1["beta"], q=r1["q"], k=r1["k"],
                       v=r1["v"], S=r1["S"], o=r1["o"], o_spk=r1["o_spk"], y_attn=y1, h_mid=h_mid,
                       x_ffn=h_mid, g=r2["g"], u=r2["u"], p=r2["p"], y_ffn=y2, h_out=r_out,
                       s_mid=r + y1, s_out=h_mid + y2)
        return r_out, st, rec
