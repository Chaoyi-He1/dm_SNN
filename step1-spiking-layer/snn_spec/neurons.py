"""脉冲点(编码器):阈值函数、有状态的 LIF 神经元、无状态的块输入量化器。

约定
----
- 所有神经元每次调用处理一个 token(折叠 T=1)。膜电位 U 显式传入传出,跨 token 携带。
- 二值脉冲取 {0,1};三值脉冲取 {-1,0,+1}。幅度恒为 1,任何尺度(QCFS 的 lambda、1/sqrt(dk))
  都折到别处,见设计文档"尺度折算表"。
- 复位是"减阈值"的软复位:U <- U - thr * S。三值时 S 带符号,负支路复位到零方向。
- 反向传播用 arctan 代理梯度;代理宽度参数命名为 sg_width,不叫 alpha,以免与 GDN 的门混淆。
- 扩展:每 token 最多 T 个脉冲(levels=T,默认 1)。token 内 T 个子步不泄漏,每次 U >= thr/T 就发一个脉冲
  并减 thr/T,输出 s = c/T ∈ {0, 1/T, ..., 1}(三值对称到负侧),软复位 U <- U - thr*s 正好减去 c 个 thr/T。
  前向写成 T 个阶跃的平均 (1/T) Σ_j H(U - j·thr/T),每个阶跃用同一代理梯度,总积分与 T=1 相同。
  T=1 时代码路径与原实现完全相同。
"""
import math
import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """前向:u >= 0 发放(Heaviside)。反向:arctan 代理梯度
    dS/du = (w/2) / (1 + (pi/2 * w * u)^2),w = sg_width。"""

    @staticmethod
    def forward(ctx, u, sg_width):
        ctx.save_for_backward(u)
        ctx.sg_width = sg_width
        return (u >= 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        w = ctx.sg_width
        x = (math.pi / 2) * w * u
        return g * (w / 2) / (1 + x * x), None


def ternary_fire(U, thr, sg_width=2.0):
    """无状态三值阈值:U >= thr 得 +1,U <= -thr 得 -1,其余 0。thr 可为需要梯度的张量。"""
    return SpikeFn.apply(U - thr, sg_width) - SpikeFn.apply(-U - thr, sg_width)


def level_steps(U, thr, levels, sg_width=2.0):
    """(1/T) Σ_{j=1..T} H(U - j·thr/T):T 级阶跃的平均,levels == 1 时就是 H(U - thr)。"""
    if levels == 1:
        return SpikeFn.apply(U - thr, sg_width)
    j = torch.arange(1, levels + 1, device=U.device, dtype=U.dtype) / levels
    return SpikeFn.apply(U.unsqueeze(-1) - thr * j, sg_width).mean(-1)


def level_binary_fire(U, thr, levels, sg_width=2.0):
    return level_steps(U, thr, levels, sg_width)


def level_ternary_fire(U, thr, levels, sg_width=2.0):
    if levels == 1:
        return ternary_fire(U, thr, sg_width)
    return level_steps(U, thr, levels, sg_width) - level_steps(-U, thr, levels, sg_width)


class PositiveScalar(nn.Module):
    """恒正的标量:thr = exp(log_thr)。learnable=False 时是 buffer。"""

    def __init__(self, init, learnable=False):
        super().__init__()
        log_init = torch.log(torch.as_tensor(float(init)))
        if learnable:
            self.log_thr = nn.Parameter(log_init.clone())
        else:
            self.register_buffer("log_thr", log_init.clone())

    def forward(self):
        return self.log_thr.exp()


class _LIFBase(nn.Module):
    def __init__(self, thr=1.0, leak=0.9, sg_width=2.0, learn_thr=False, levels=1):
        super().__init__()
        self._thr = PositiveScalar(thr, learnable=learn_thr)
        self.leak = float(leak)
        self.sg_width = float(sg_width)
        self.levels = int(levels)                   # 每 token 最多几个脉冲;普通属性,不进 state_dict

    @property
    def thr(self):
        return self._thr()

    def fire(self, U, thr):
        raise NotImplementedError

    def forward(self, I, U):
        """I: 输入电流(投影输出);U: 上一 token 结束时的膜电位。返回 (S, U_new)。"""
        thr = self.thr
        U = self.leak * U + I
        S = self.fire(U, thr)
        U = U - thr * S
        return S, U


class BinaryLIF(_LIFBase):
    """二值 LIF,发放 {0,1}(levels=T 时 {0, 1/T, ..., 1})。"""

    def fire(self, U, thr):
        if self.levels == 1:
            return SpikeFn.apply(U - thr, self.sg_width)
        return level_binary_fire(U, thr, self.levels, self.sg_width)


class TernaryLIF(_LIFBase):
    """三值 LIF,发放 {-1,0,+1}(电路上是互补的两条 DM 支路;levels=T 时 {-1, ..., -1/T, 0, 1/T, ..., 1})。"""

    def fire(self, U, thr):
        if self.levels == 1:
            return ternary_fire(U, thr, self.sg_width)
        return level_ternary_fire(U, thr, self.levels, self.sg_width)


class QuantizerIn(nn.Module):
    """块输入量化器 x = Q_in(h):无状态三值阈值,阈值逐层可学且恒正。电路上是两个比较器
    (levels=T 时是 2T 个比较器,阈值 ±j·thr/T)。"""

    def __init__(self, thr0=1.0, sg_width=2.0, levels=1):
        super().__init__()
        self.log_thr = nn.Parameter(torch.log(torch.as_tensor(float(thr0))))
        self.sg_width = float(sg_width)
        self.levels = int(levels)

    @property
    def thr(self):
        return self.log_thr.exp()

    def forward(self, h):
        if self.levels == 1:
            return ternary_fire(h, self.thr, self.sg_width)
        return level_ternary_fire(h, self.thr, self.levels, self.sg_width)
