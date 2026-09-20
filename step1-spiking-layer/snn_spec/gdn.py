"""Gated DeltaNet 递推(形式 A)与折叠 T=1 的脉冲 GDN 层。

递推约定(形式 A,与 GDN 原式、FLA 内核、HF Qwen3-Next/Qwen3.5 参考实现一致):
    S~_t = alpha_t * S_{t-1}                 衰减:读 S_{t-1}
    r_t  = S~_t^T k_t                        检索:读衰减后的 S~_t
    S_t  = S~_t + beta_t * k_t (v_t - r_t)^T 写入:产生 S_t
    o_t  = S_t^T q_t                         读出:读更新后的 S_t
状态 S 的形状是 [dk, dv](多头时 [B, H, dk, dv])。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import BinaryLIF, TernaryLIF


# ----------------------------------------------------------------------------- 参考递推(单头)

def transition_matrix(k, alpha, beta, form="A"):
    """状态转移矩阵 A,使 S_t = A S_{t-1} + (写入项)。A、C 同为 alpha (I - beta k k^T);B 为 alpha I - beta k k^T。"""
    I = torch.eye(k.shape[-1], dtype=k.dtype, device=k.device)
    kk = torch.outer(k, k)
    if form in ("A", "C"):
        return alpha * (I - beta * kk)
    if form == "B":
        return alpha * I - beta * kk
    raise ValueError(form)


def gdn_step(S, k, v, q, alpha, beta, form="A"):
    """单头单 token 的参考递推。S: [dk, dv],k, q: [dk],v: [dv]。返回 (S_new, o)。

    form="A":先衰减,再从衰减后的状态检索,再写入(规范)。
    form="B":从衰减前的状态检索,再衰减并写入(擦除项少乘一个 alpha)。
    form="C":先检索并写入,再对整个结果衰减(写入项多乘一个 alpha)。
    """
    if form == "A":
        S_t = alpha * S
        r = S_t.T @ k
        S_new = S_t + beta * torch.outer(k, v - r)
    elif form == "B":
        r = S.T @ k
        S_new = alpha * S + beta * torch.outer(k, v - r)
    elif form == "C":
        r = S.T @ k
        S_new = alpha * (S + beta * torch.outer(k, v - r))
    else:
        raise ValueError(form)
    o = S_new.T @ q
    return S_new, o


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def hf_reference_recurrence(query, key, value, g, beta, initial_state=None):
    """HF transformers `torch_recurrent_gated_delta_rule`(Qwen3-Next / Qwen3.5)的逐 token 转写。
    query, key: [B, H, L, dk];value: [B, H, L, dv];g: [B, H, L](log alpha);beta: [B, H, L]。
    内部对 q, k 做 L2 归一化,q 再乘 dk^-1/2(FLA 默认 scale)。返回 (o: [B, H, L, dv], S: [B, H, dk, dv])。"""
    query = _l2norm(query.float()); key = _l2norm(key.float()); value = value.float()
    B, H, L, dk = key.shape
    dv = value.shape[-1]
    query = query * dk ** -0.5
    out = torch.zeros(B, H, L, dv, dtype=value.dtype)
    S = torch.zeros(B, H, dk, dv, dtype=value.dtype) if initial_state is None else initial_state.clone()
    for i in range(L):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        S = S * g[:, :, i].exp()[..., None, None]                       # 衰减
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)                   # 检索(读衰减后的状态)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)                # 写入
        out[:, :, i] = (S * q_t.unsqueeze(-1)).sum(dim=-2)             # 读出
    return out, S


# ----------------------------------------------------------------------------- 门

class QwenNativeGates(nn.Module):
    """Qwen3-Next / Qwen3.5 原生门的函数形式:
        alpha = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias)),  beta = sigmoid(b_proj(x))
    alpha 钳到硬件可实现区间 [alpha_min, alpha_max]。装载原生层时把 a_proj/b_proj/dt_bias/A_log 原样搬进来。"""

    def __init__(self, d, n_v_heads, alpha_rng=(0.5, 0.99)):
        super().__init__()
        self.a_proj = nn.Linear(d, n_v_heads, bias=False)
        self.b_proj = nn.Linear(d, n_v_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(n_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(n_v_heads).uniform_(1.0, 16.0)))
        self.alpha_min, self.alpha_max = alpha_rng

    def forward(self, x, ext=None):
        alpha = torch.exp(-self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias))
        return alpha.clamp(self.alpha_min, self.alpha_max), torch.sigmoid(self.b_proj(x))


class SigmoidGates(nn.Module):
    """从零训练时的门:alpha = lo + (hi - lo) sigmoid(Wa x),beta = sigmoid(Wb x)。"""

    def __init__(self, d, n_v_heads, alpha_rng=(0.5, 0.99)):
        super().__init__()
        self.Wa = nn.Linear(d, n_v_heads)
        self.Wb = nn.Linear(d, n_v_heads)
        self.alpha_min, self.alpha_max = alpha_rng

    def forward(self, x, ext=None):
        a = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.Wa(x))
        return a, torch.sigmoid(self.Wb(x))


class ExternalGates(nn.Module):
    """首个闭环用:门值由外部(软件记录)给定,电路上对应直接施加泄漏管栅压与电流镜比例。"""

    def forward(self, x, ext=None):
        if ext is None:
            raise ValueError("ExternalGates 需要 ext=(alpha, beta)")
        return ext


# ----------------------------------------------------------------------------- 可选 FIR(Qwen 的 conv1d)

class CausalFIR(nn.Module):
    """逐通道因果 FIR(Qwen 的 depthwise causal conv1d,核长 taps,无偏置),放在神经元之前。
    权重形状 [C, taps],最后一个抽头乘当前输入(与 HF conv1d 权重 squeeze 后一致)。
    历史缓存 hist: [B, taps-1, C],最旧在前。默认初始化为恒等(只有当前抽头为 1)。"""

    def __init__(self, channels, taps):
        super().__init__()
        assert taps >= 2
        w = torch.zeros(channels, taps)
        w[:, -1] = 1.0
        self.weight = nn.Parameter(w)
        self.taps = taps

    def init_state(self, B, device):
        return torch.zeros(B, self.taps - 1, self.weight.shape[0], device=device)

    def forward(self, z, hist):
        full = torch.cat([hist, z.unsqueeze(1)], dim=1)          # [B, taps, C],最旧 → 当前
        y = (full * self.weight.T.unsqueeze(0)).sum(dim=1)
        return y, full[:, 1:]


# ----------------------------------------------------------------------------- 脉冲 GDN 层

class SpikingGDN(nn.Module):
    """折叠 T=1 的脉冲 Gated DeltaNet 层。

    x(块输入,三值)→ W_q/W_k/W_v →(可选 FIR)→ 神经元 → q 二值、k 三值/sqrt(dk)、v 三值
      → core(形式 A)→ o(连续)→ SN_pre → 三值 → W_O → SN_out → y 三值。
    S 保持连续(电路里是 gain cell 的电荷);alpha, beta 保持浮点标量(电路里是模拟控制量)。
    n_v_heads 可为 n_k_heads 的整数倍(q, k 按组重复),与 Qwen 的 linear_num_key/value_heads 对应。
    """

    def __init__(self, d, n_k_heads, n_v_heads, dk, dv, gates=None, fir_taps=0,
                 alpha_rng=(0.5, 0.99), leak=0.9, thr=1.0):
        super().__init__()
        assert n_v_heads % n_k_heads == 0
        self.d, self.Hk, self.Hv, self.dk, self.dv = d, n_k_heads, n_v_heads, dk, dv
        self.Wq = nn.Linear(d, n_k_heads * dk, bias=False)
        self.Wk = nn.Linear(d, n_k_heads * dk, bias=False)
        self.Wv = nn.Linear(d, n_v_heads * dv, bias=False)
        self.Wo = nn.Linear(n_v_heads * dv, d, bias=False)
        self.gates = gates if gates is not None else QwenNativeGates(d, n_v_heads, alpha_rng)
        self.sn_q = BinaryLIF(thr, leak)
        self.sn_k = TernaryLIF(thr, leak)
        self.sn_v = TernaryLIF(thr, leak)
        self.sn_pre = TernaryLIF(thr, leak)
        self.sn_out = TernaryLIF(thr, leak)
        self.c_k = 1.0 / math.sqrt(dk)                 # 固定缩放:||k||^2 = n/dk <= 1
        self.fir_taps = fir_taps
        if fir_taps > 1:
            self.fir_q = CausalFIR(n_k_heads * dk, fir_taps)
            self.fir_k = CausalFIR(n_k_heads * dk, fir_taps)
            self.fir_v = CausalFIR(n_v_heads * dv, fir_taps)

    def init_state(self, B, device):
        z = lambda *s: torch.zeros(*s, device=device)
        st = dict(S=z(B, self.Hv, self.dk, self.dv),
                  Uq=z(B, self.Hk * self.dk), Uk=z(B, self.Hk * self.dk),
                  Uv=z(B, self.Hv * self.dv), Upre=z(B, self.Hv * self.dv), Uout=z(B, self.d))
        if self.fir_taps > 1:
            st["fir_q"] = self.fir_q.init_state(B, device)
            st["fir_k"] = self.fir_k.init_state(B, device)
            st["fir_v"] = self.fir_v.init_state(B, device)
        return st

    @staticmethod
    def core(k, v, q, alpha, beta, S):
        """递推核心(形式 A)。k, q: [B, H, dk](k 已缩放);v: [B, H, dv];alpha, beta: [B, H];
        S: [B, H, dk, dv]。返回 (o: [B, H, dv], S_new)。独立于门的来源和编码器。"""
        S_t = alpha[..., None, None] * S                          # 衰减:读 S_{t-1}
        r = torch.einsum("bhij,bhi->bhj", S_t, k)                 # 检索:读 S~_t
        dv = beta[..., None] * (v - r)
        S_new = S_t + torch.einsum("bhi,bhj->bhij", k, dv)        # 写入:产生 S_t
        o = torch.einsum("bhij,bhi->bhj", S_new, q)               # 读出:读 S_t
        return o, S_new

    def step(self, x, st, ext_gates=None, record=False):
        """x: [B, d] 三值块输入脉冲。返回 (y: [B, d] 三值, st, rec)。rec 在 record=False 时为 None。"""
        B = x.shape[0]
        alpha, beta = self.gates(x, ext=ext_gates)
        zq, zk, zv = self.Wq(x), self.Wk(x), self.Wv(x)
        if self.fir_taps > 1:
            zq, st["fir_q"] = self.fir_q(zq, st["fir_q"])
            zk, st["fir_k"] = self.fir_k(zk, st["fir_k"])
            zv, st["fir_v"] = self.fir_v(zv, st["fir_v"])
        q, st["Uq"] = self.sn_q(zq, st["Uq"])
        k, st["Uk"] = self.sn_k(zk, st["Uk"])
        v, st["Uv"] = self.sn_v(zv, st["Uv"])
        rep = self.Hv // self.Hk
        q = q.view(B, self.Hk, self.dk).repeat_interleave(rep, dim=1)
        k = (k.view(B, self.Hk, self.dk) * self.c_k).repeat_interleave(rep, dim=1)
        v = v.view(B, self.Hv, self.dv)
        o, st["S"] = self.core(k, v, q, alpha, beta, st["S"])
        o_spk, st["Upre"] = self.sn_pre(o.reshape(B, self.Hv * self.dv), st["Upre"])
        y, st["Uout"] = self.sn_out(self.Wo(o_spk), st["Uout"])
        rec = None
        if record:
            rec = dict(alpha=alpha, beta=beta, q=q, k=k, v=v, S=st["S"], o=o, o_spk=o_spk, y=y)
        return y, st, rec
