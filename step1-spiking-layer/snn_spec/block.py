"""一层 = GDN + SwiGLU + 两个累加器残差。折叠 T=1,逐 token 执行,所有状态跨 token 携带。

残差约定:残差流 h 是实值累加器,永远不进任何线性层;每个块的输入是 x = Q_in(h)(无状态三值阈值)。
电路上 h 是每通道一个电容,块输出的每个三值分量向它注入 ±Q_0 或不注入,Q_in 是两个比较器。
"""
import torch
import torch.nn as nn

from .neurons import BinaryLIF, TernaryLIF, QuantizerIn
from .gdn import SpikingGDN


class SpikingSwiGLU(nn.Module):
    """W_2( g ⊙ u ),g = BinaryLIF(W_1 x) ∈ {0,1},u = TernaryLIF(W_3 x) ∈ {-1,0,+1}。
    折叠模式下逐元素乘就是"锁存 + AND",符号取 u(因 g >= 0)。"""

    def __init__(self, d, d_ff, leak=0.9, thr=1.0):
        super().__init__()
        self.d, self.d_ff = d, d_ff
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W3 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)
        self.sn_1 = BinaryLIF(thr, leak)
        self.sn_3 = TernaryLIF(thr, leak)
        self.sn_2 = TernaryLIF(thr, leak)

    def init_state(self, B, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return dict(U1=z(B, self.d_ff), U3=z(B, self.d_ff), U2=z(B, self.d))

    def step(self, x, st, record=False):
        g, st["U1"] = self.sn_1(self.W1(x), st["U1"])
        u, st["U3"] = self.sn_3(self.W3(x), st["U3"])
        p = g * u                                   # 锁存 + AND,符号取 u
        y, st["U2"] = self.sn_2(self.W2(p), st["U2"])
        rec = dict(g=g, u=u, p=p, y=y) if record else None
        return y, st, rec


class SpikingBlock(nn.Module):
    """一个完整的 decoder block:Q_in → GDN → 累加 → Q_in → SwiGLU → 累加。"""

    def __init__(self, d, d_ff, n_k_heads, n_v_heads, dk, dv, gates=None, fir_taps=0,
                 alpha_rng=(0.5, 0.99), leak=0.9, thr=1.0, thr_in=1.0):
        super().__init__()
        self.q_in1 = QuantizerIn(thr_in)
        self.attn = SpikingGDN(d, n_k_heads, n_v_heads, dk, dv, gates=gates, fir_taps=fir_taps,
                               alpha_rng=alpha_rng, leak=leak, thr=thr)
        self.q_in2 = QuantizerIn(thr_in)
        self.ffn = SpikingSwiGLU(d, d_ff, leak=leak, thr=thr)

    def init_state(self, B, device):
        return dict(attn=self.attn.init_state(B, device), ffn=self.ffn.init_state(B, device))

    def step(self, h, st, ext_gates=None, record=False):
        """h: [B, d] 实值累加器(块入口)。返回 (h_out, st, rec)。"""
        x1 = self.q_in1(h)
        y1, st["attn"], r1 = self.attn.step(x1, st["attn"], ext_gates=ext_gates, record=record)
        h_mid = h + y1
        x2 = self.q_in2(h_mid)
        y2, st["ffn"], r2 = self.ffn.step(x2, st["ffn"], record=record)
        h_out = h_mid + y2
        rec = None
        if record:
            rec = dict(h_in=h, x_attn=x1, alpha=r1["alpha"], beta=r1["beta"], q=r1["q"], k=r1["k"],
                       v=r1["v"], S=r1["S"], o=r1["o"], o_spk=r1["o_spk"], y_attn=y1, h_mid=h_mid,
                       x_ffn=x2, g=r2["g"], u=r2["u"], p=r2["p"], y_ffn=y2, h_out=h_out)
        return h_out, st, rec


def run_sequence(block, h_seq, st=None, ext_gates_seq=None, record=False):
    """逐 token 跑一段序列。h_seq: [B, L, d] 每个 token 的入口累加器值。
    st=None 表示新序列(全部状态清零);传入上一段的 st 表示接着上一段继续(长文本分段)。
    ext_gates_seq: 可选的 (alpha: [B, L, H], beta: [B, L, H]),用于首个闭环的外部门值。
    返回 (h_out: [B, L, d], st) 或 record=True 时 (h_out, st, recs)。"""
    B, L, _ = h_seq.shape
    if st is None:
        st = block.init_state(B, h_seq.device)
    outs, recs = [], []
    for t in range(L):
        ext = None if ext_gates_seq is None else (ext_gates_seq[0][:, t], ext_gates_seq[1][:, t])
        h_out, st, rec = block.step(h_seq[:, t], st, ext_gates=ext, record=record)
        outs.append(h_out)
        recs.append(rec)
    out = torch.stack(outs, dim=1)
    return (out, st, recs) if record else (out, st)
