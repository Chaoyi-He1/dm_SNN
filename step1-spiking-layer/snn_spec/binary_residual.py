"""残差流全程二值(0/1)的一层:r → GDN → h' = M1(r, y1) → SwiGLU → r_out = M2(h', y2)。

与 TernaryResidualBlock 的差别:残差流 r 只取 {0, 1}(levels=T 时 {0, 1/T, ..., 1},一根线上每 token 最多 T 个脉冲),
没有负脉冲。子块内部的脉冲点不变(q、g 二值,k、v、o_spk、u 三值)。两处合并 M 换成无符号的规则 merge:

  sr   置位 / 复位。子块输出 y 仍是三值,只在块内走一小段:y 的正脉冲把 r 置 1,负脉冲把 r 清 0,y = 0 时保持,
       r' = clamp(r + y, 0, 1)。T = 1 时就是 SR 锁存器 r' = (r OR y+) AND NOT y-;levels=T 时是每通道一个 0..T 的饱和计数。
       实现成阈值为 1.0 的无符号 T 级阶跃(与三值残差的 sat 同一种写法),阈值固定,不需要标定;反向走 arctan 代理梯度。
       代理梯度的中心在 r + y = 1:y = 0 时恒等通路的梯度在 r = 1 处是 1,在 r = 0 处只有 0.092(T = 1)。
       阶跃的阈值正好落在电平上,只对 T 为 2 的幂时精确(其余 T 有浮点误判),所以 sr 只接受这样的 T。
  src  前向与 sr 完全相同,r' = clamp(r + y, 0, 1);反向用 clamp 自己的梯度(0 <= r + y <= 1 时为 1,饱和时为 0),
       不经代理梯度。这个合并本来就是分段线性的,这是它的精确(次)梯度:y = 0 时恒等通路的梯度恒为 1,和实值残差一样。
       电路与 sr 相同,差别只在训练。
  xor  翻转。子块输出 y 也是无符号的(sn_out、sn_2 换成 BinaryLIF),r' = |r - y|。
       T = 1 时是异或(y = 1 把 r 翻转);levels=T 时等于把两路对齐的脉冲串(温度计码)逐子时隙异或,剩 |c_r - c_y| 个脉冲。
       没有比较器,也没有阈值。

三种规则在 y = 0 时都有 r' = r,残差的恒等通路保留。

电路含义:层间连线与 GDN→FFN 的连线每通道一根线、只传"有 / 无"脉冲;sr / src 每通道一个置位复位锁存器(T > 1 时是小计数器),
xor 每通道一个异或门。块入口没有 Q_in,也没有实值累加器 h。
"""
import torch
import torch.nn as nn

from .neurons import BinaryLIF, level_binary_fire
from .ternary_residual import TernaryResidualBlock

MERGES = ("sr", "src", "xor")


def clamp_exact(u, lo, hi):
    """clamp(u, lo, hi),梯度在 lo <= u <= hi(含端点)时为 1,饱和时为 0。
    torch.clamp 在端点处的梯度是 0,而 T = 1 时合并的所有取值都在端点上,所以梯度自己写(加的一项数值恒为 0)。"""
    ud = u.detach()
    inside = ((ud >= lo - 1e-6) & (ud <= hi + 1e-6)).to(u.dtype)
    return torch.clamp(ud, lo, hi) + inside * (u - ud)


class BinaryQuantizer(nn.Module):
    """无符号量化器 x = Q(h):h >= thr 得 1,否则 0;levels=T 时 x = (1/T) Σ_j 1[h >= j·thr/T] ∈ {0, 1/T, ..., 1}。
    无状态,阈值逐层可学且恒正。电路上是 1 个比较器(levels=T 时 T 个)。负的输入一律得 0。"""

    def __init__(self, thr0=1.0, sg_width=2.0, levels=1):
        super().__init__()
        self.log_thr = nn.Parameter(torch.log(torch.as_tensor(float(thr0))))
        self.sg_width = float(sg_width)
        self.levels = int(levels)

    @property
    def thr(self):
        return self.log_thr.exp()

    def forward(self, h):
        return level_binary_fire(h, self.thr, self.levels, self.sg_width)


class BinaryMerge(nn.Module):
    """两输入合并 M(r, y),r 是无符号的残差,y 是子块输出(sr、src:三值;xor:无符号)。无状态。
    log_thr 只是为了与 ResidualMerge 的接口一致(固定为 0,即阈值 1.0,不参与训练);只有 sr 用它。"""

    def __init__(self, merge="sr", levels=1, sg_width=2.0):
        super().__init__()
        if merge not in MERGES:
            raise ValueError(f"merge must be one of {MERGES}, got {merge!r}")
        if merge == "sr" and int(levels) & (int(levels) - 1):
            raise ValueError(f"sr 的阈值落在电平上,只对 2 的幂的 levels 精确,得到 {levels};其余 levels 用 src")
        self.merge = merge
        self.levels = int(levels)
        self.sg_width = float(sg_width)
        self.log_thr = nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def thr(self):
        return self.log_thr.exp()

    def forward(self, r, y):
        if self.merge == "sr":
            return level_binary_fire(r + y, self.thr, self.levels, self.sg_width)
        if self.merge == "src":
            return clamp_exact(r + y, 0.0, 1.0)     # 两路都在电平上,和也在电平上,clamp 之后仍在电平上
        d = r - y
        # |d| 在 d = 0 处的梯度是 0。相等时取可行方向上的单侧斜率:都为 0 只能往上走(+1,恒等通路的梯度就是 1),
        # 都为 1 只能往下走(-1),中间电平是尖点(0)。加的这一项数值恒为 0,只提供梯度。
        rd = r.detach()
        slope = torch.where(rd <= 0, torch.ones_like(rd), torch.where(rd >= 1, -torch.ones_like(rd), torch.zeros_like(rd)))
        tie = (d.detach() == 0).to(r.dtype) * slope
        return d.abs() + tie * ((r - rd) + (y - y.detach()))


class SignedClampMerge(nn.Module):
    """三值残差流的对照用合并 satc:M(r, y) = clamp(r + y, -1, 1)。两路都在电平上时前向与 ResidualMerge("sat") 逐位相同,
    反向用 clamp 自己的梯度(clamp_exact),不经代理梯度。用它是为了让"二值 src 对三值"的比较只差残差流的取值,不差梯度的写法。
    接口同 ResidualMerge / BinaryMerge(log_thr 固定为 0,不参与训练,也不被用到)。"""

    merge = "satc"

    def __init__(self, levels=1):
        super().__init__()
        self.levels = int(levels)
        self.log_thr = nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def thr(self):
        return self.log_thr.exp()

    def forward(self, r, y):
        return clamp_exact(r + y, -1.0, 1.0)


class BinaryResidualBlock(TernaryResidualBlock):
    """一个 decoder block,残差流全程二值:GDN → M1 合并 → SwiGLU → M2 合并。step 与记录字段沿用 TernaryResidualBlock。"""

    def __init__(self, d, d_ff, n_k_heads, n_v_heads, dk, dv, gates=None, fir_taps=0,
                 alpha_rng=(0.5, 0.99), leak=0.9, thr=1.0, learn_thr=False, merge="sr", levels=1):
        if merge not in MERGES:
            raise ValueError(f"merge must be one of {MERGES}, got {merge!r}")
        super().__init__(d, d_ff, n_k_heads, n_v_heads, dk, dv, gates=gates, fir_taps=fir_taps, alpha_rng=alpha_rng,
                         leak=leak, thr=thr, learn_thr=learn_thr, merge="sat", levels=levels)
        self.merge = merge
        self.m1 = BinaryMerge(merge, levels)
        self.m2 = BinaryMerge(merge, levels)
        if merge == "xor":                          # 子块输出也改成无符号
            self.attn.sn_out = BinaryLIF(thr, leak, learn_thr=learn_thr, levels=levels)
            self.ffn.sn_2 = BinaryLIF(thr, leak, learn_thr=learn_thr, levels=levels)
