"""行为级电路原语与按 φ0–φ8 相位表执行的行为级仿真器(S1.2、S1.3、S2.3)。

每个原语都有一组"理想参数",在理想参数下整个仿真器与 S0.9 的软件层逐脉冲一致;
Stage 1 用 SPICE 标定出的参数替换理想参数,得到带物理非理想性的仿真器。
"""
import math
import torch

from .gdn import ExternalGates


def alpha_from_tgap(t_gap, tau_leak):
    """gain cell 的衰减门:alpha = exp(-t_gap / tau_leak),tau_leak 由泄漏管栅压设定。"""
    return math.exp(-t_gap / tau_leak)


# ----------------------------------------------------------------------------- 神经元模型

def _fire(U, theta, ternary):
    if ternary:
        return (U >= theta).float() - (U <= -theta).float()
    return (U >= theta).float()


class IdealLIFModel:
    """S0.9 的 LIF(减阈值软复位、无不应期),作为行为级仿真器的理想神经元。接口与 DMNeuronModel 相同。"""

    def __init__(self, thr, leak, ternary=False):
        self.thr, self.leak, self.ternary = float(thr), float(leak), bool(ternary)

    def init_state(self, shape, device="cpu"):
        return dict(U=torch.zeros(shape, device=device))

    def step(self, I, st):
        U = self.leak * st["U"] + I
        S = _fire(U, self.thr, self.ternary)
        return S, dict(U=U - self.thr * S)


class DMNeuronModel:
    """S1.2 的六参数状态模型 (beta_m, g_m, theta, U_hold, rho, n_ref):
        U_t = beta_m U_{t-1} + g_m I_t;  S_t = 1[c_{t-1} = 0] · fire(U_t)
        发放时 U_t <- U_hold·sign(S_t) + rho (U_t - theta·sign(S_t)),c_t <- n_ref;否则 c_t <- max(c_{t-1}-1, 0)
    复位律的物理图像:器件导通把膜电容放电到保持电压 U_hold 后关断,越阈的剩余部分 (U_t - theta) 中有比例 rho
    在周期剩余时间里继续积分进来。rho = 0 为硬复位到 U_hold;rho = 1、U_hold = 0 即"减阈值"软复位。
    三值神经元的负支路对称。"""

    def __init__(self, beta_m, g_m, theta, U_hold, rho, n_ref, ternary=False):
        self.beta_m, self.g_m, self.theta = float(beta_m), float(g_m), float(theta)
        self.U_hold, self.rho, self.n_ref = float(U_hold), float(rho), int(n_ref)
        self.ternary = bool(ternary)

    def init_state(self, shape, device="cpu"):
        return dict(U=torch.zeros(shape, device=device), c=torch.zeros(shape, dtype=torch.long, device=device))

    def step(self, I, st):
        U = self.beta_m * st["U"] + self.g_m * I
        c = st["c"]
        S = _fire(U, self.theta, self.ternary) * (c == 0).float()
        fired = S != 0
        sgn = torch.sign(S)
        U = torch.where(fired, self.U_hold * sgn + self.rho * (U - self.theta * sgn), U)
        c = torch.where(fired, torch.full_like(c, self.n_ref), (c - 1).clamp(min=0))
        return S, dict(U=U, c=c)


def _robust_lstsq(A, b, k=5.0):
    """两遍最小二乘:第一遍拟合后剔除 |残差| > k·MAD 的样本再拟合一遍(边界采样落在放电中途的少数周期是离群点)。"""
    x = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    r = (A @ x - b).abs()
    mad = r.median().item()
    keep = r <= k * max(mad, 1e-9)
    if keep.sum() >= A.shape[1] + 1 and keep.sum() < len(b):
        x = torch.linalg.lstsq(A[keep], b[keep].unsqueeze(1)).solution.squeeze(1)
    return x.tolist()


def _best_threshold(mag, fired):
    """在候选阈值里选使 `fire ⇔ |U_pre| >= theta` 分类错误最少的那个,取相邻样本的中点。"""
    vals, order = torch.sort(mag)
    f = fired[order].double()
    n = len(vals)
    cum_f = torch.cat([torch.zeros(1, dtype=torch.double), torch.cumsum(f, 0)])          # 位置 j 之前发放的个数
    cum_nf = torch.cat([torch.zeros(1, dtype=torch.double), torch.cumsum(1 - f, 0)])
    err = cum_f[:n + 1] + (cum_nf[n] - cum_nf[:n + 1])                                   # 阈值放在位置 j
    j = int(torch.argmin(err).item())
    if j == 0:
        return vals[0].item() - 1e-6
    if j >= n:
        return vals[-1].item() + 1e-6
    return 0.5 * (vals[j - 1].item() + vals[j].item())


def fit_dm_neuron(I, U, S, ternary=False, n_ref_max=8):
    """S1.2 的拟合:I: [T, N] 每周期一个电流值;U: [T+1, N] 周期边界上的膜电压;S: [T, N] 发放事件。
    返回六参数字典。非发放步做线性最小二乘得 (beta_m, g_m);n_ref 取最小发放间隔减一;
    theta 在非不应期样本上按"fire ⇔ |U_pre| >= theta"分类错误最少来选;最小二乘均为剔除离群点的两遍拟合;
    发放步按 U_next·sgn = (U_hold - rho·theta) + rho·(U_pre·sgn) 做线性最小二乘得 (rho, U_hold)。"""
    I, U, S = I.double(), U.double(), S.double()
    U_prev, U_next = U[:-1], U[1:]
    T, N = S.shape
    fired = S != 0
    # 1) beta_m, g_m
    m = ~fired
    A = torch.stack([U_prev[m], I[m]], dim=1)
    beta_m, g_m = _robust_lstsq(A, U_next[m])
    U_pre = beta_m * U_prev + g_m * I
    # 2) n_ref:最小发放间隔 - 1
    n_ref = 0
    isis = []
    for j in range(N):
        t = torch.nonzero(fired[:, j]).squeeze(1)
        if len(t) > 1:
            isis.append((t[1:] - t[:-1]).min().item())
    if isis:
        n_ref = max(0, min(min(isis) - 1, n_ref_max))
    # 3) theta:排除不应期内的步
    refr = torch.zeros_like(fired)
    for d in range(1, n_ref + 1):
        refr[d:] |= fired[:-d]
    mag = U_pre.abs()
    theta = _best_threshold(mag[~refr], fired[~refr]) if fired.any() else mag.max().item() + 1e-6
    # 4) rho, U_hold(发放步:U_next·sgn = (U_hold - rho·theta) + rho·U_pre·sgn)
    sgn = torch.sign(S[fired])
    A2 = torch.stack([torch.ones_like(sgn), U_pre[fired] * sgn], dim=1)
    a, rho = _robust_lstsq(A2, U_next[fired] * sgn)
    U_hold = a + rho * theta
    return dict(beta_m=beta_m, g_m=g_m, theta=theta, U_hold=U_hold, rho=rho, n_ref=int(n_ref))


# ----------------------------------------------------------------------------- gain cell 阵列

class GainCellArray:
    """3T1C gain cell 阵列承载状态 S: [B, H, dk, dv]。按相位工作:
        phi1_decay:泄漏窗,S <- S·alpha^f_pre(f_pre = leak_before_retrieve,规范为 1)
        phi3_retrieve:r = S^T k;phi4_write:S <- S + k Δv^T;phi5_readout:o = S^T q
        end_of_cycle:S <- S·alpha^(1-f_pre)·eps_hold(写入后仍在泄漏的部分,规范为 0)
    retrieve_before_decay=True 表示检索发生在泄漏窗之前(形式 B 的时序错误)。
    write_err、read_err 为相对误差的标准差;eps_hold 为 φ2–φ8 期间的本征保持因子。"""

    def __init__(self, B, H, dk, dv, leak_before_retrieve=1.0, retrieve_before_decay=False,
                 write_err=0.0, read_err=0.0, eps_hold=1.0, seed=0, device="cpu"):
        self.shape = (B, H, dk, dv)
        self.f_pre = float(leak_before_retrieve)
        self.retrieve_before_decay = bool(retrieve_before_decay)
        self.write_err, self.read_err, self.eps_hold = float(write_err), float(read_err), float(eps_hold)
        self.gen = torch.Generator().manual_seed(seed)
        self.device = device
        self.reset()

    def reset(self, S0=None):
        self.S = torch.zeros(self.shape, device=self.device) if S0 is None else S0.detach().clone().to(self.device)

    def _noise(self, x, sigma):
        if sigma == 0.0:
            return x
        return x * (1.0 + sigma * torch.randn(x.shape, generator=self.gen).to(x.device))

    def phi1_decay(self, alpha):
        self.S = self.S * alpha[..., None, None] ** self.f_pre

    def phi3_retrieve(self, k):
        return self._noise(torch.einsum("bhij,bhi->bhj", self.S, k), self.read_err)

    def phi4_write(self, k, dv):
        self.S = self.S + self._noise(torch.einsum("bhi,bhj->bhij", k, dv), self.write_err)

    def phi5_readout(self, q):
        return self._noise(torch.einsum("bhij,bhi->bhj", self.S, q), self.read_err)

    def end_of_cycle(self, alpha):
        self.S = self.S * (alpha[..., None, None] ** (1.0 - self.f_pre)) * self.eps_hold

    def cycle(self, k, v, q, alpha, beta):
        """一个 token 周期里与状态阵列有关的全部相位。返回 (o, S)。"""
        if self.retrieve_before_decay:
            r = self.phi3_retrieve(k)
            self.phi1_decay(alpha)
        else:
            self.phi1_decay(alpha)
            r = self.phi3_retrieve(k)
        self.phi4_write(k, beta[..., None] * (v - r))
        o = self.phi5_readout(q)
        self.end_of_cycle(alpha)
        return o, self.S.clone()


# ----------------------------------------------------------------------------- crossbar、比较器、锁存

class CrossbarModel:
    """1T1R 差分 crossbar 承载一个权重矩阵 W: [out, in]。电导量化到 n_levels 级、编程误差 sigma_g(相对)、
    读噪声 read_noise(相对)。输入为脉冲(单位幅度),输出为 bitline 电流 x @ W_eff^T。"""

    def __init__(self, W, n_levels=None, sigma_g=0.0, read_noise=0.0, seed=0):
        W = W.detach().clone().float()
        self.gen = torch.Generator().manual_seed(seed)
        if n_levels:
            step = W.abs().max() / n_levels
            W = torch.round(W / step) * step
        if sigma_g:
            W = W * (1.0 + sigma_g * torch.randn(W.shape, generator=self.gen))
        self.W_eff, self.read_noise = W, float(read_noise)

    def __call__(self, x):
        y = x @ self.W_eff.T
        if self.read_noise:
            y = y * (1.0 + self.read_noise * torch.randn(y.shape, generator=self.gen))
        return y


class ComparatorModel:
    """块入口 Q_in 的两个比较器:输出 sign(h)·1[|h - offset| >= thr]。hysteresis > 0 时需传入上一输出 prev。"""

    def __init__(self, thr, offset=0.0, hysteresis=0.0):
        self.thr, self.offset, self.hys = float(thr), float(offset), float(hysteresis)

    def __call__(self, h, prev=None):
        z = h - self.offset
        if self.hys == 0.0 or prev is None:
            return torch.sign(z) * (z.abs() >= self.thr).float()
        turn_on = torch.sign(z) * (z.abs() >= self.thr + self.hys).float()
        stay = (z.abs() >= self.thr - self.hys).float() * prev
        return torch.where(prev != 0, stay, turn_on)


def latch_and(g, u):
    """锁存 + AND:幅度 AND,符号取 u(g >= 0)。"""
    return g * u


# ----------------------------------------------------------------------------- 行为级仿真器

class BehavioralBlock:
    """把 S0.9 的 SpikingBlock 逐原语替换成电路模型,按 S0.7 的相位表逐 token 执行。
    neuron_factory(thr, leak, ternary) 返回神经元模型;默认 IdealLIFModel。"""

    def __init__(self, block, neuron_factory=None, crossbar_kw=None, gain_cell_kw=None, comparator_kw=None):
        self.block = block
        nf = neuron_factory or (lambda thr, leak, ternary: IdealLIFModel(thr, leak, ternary))
        ck, gk = crossbar_kw or {}, comparator_kw or {}
        attn, ffn = block.attn, block.ffn
        self.gain_cell_kw = gain_cell_kw or {}
        with torch.no_grad():
            self.xb = {n: CrossbarModel(m.weight, **ck) for n, m in
                       [("Wq", attn.Wq), ("Wk", attn.Wk), ("Wv", attn.Wv), ("Wo", attn.Wo),
                        ("W1", ffn.W1), ("W3", ffn.W3), ("W2", ffn.W2)]}
            self.neu = {
                "q": nf(attn.sn_q.thr.item(), attn.sn_q.leak, False),
                "k": nf(attn.sn_k.thr.item(), attn.sn_k.leak, True),
                "v": nf(attn.sn_v.thr.item(), attn.sn_v.leak, True),
                "pre": nf(attn.sn_pre.thr.item(), attn.sn_pre.leak, True),
                "out": nf(attn.sn_out.thr.item(), attn.sn_out.leak, True),
                "g": nf(ffn.sn_1.thr.item(), ffn.sn_1.leak, False),
                "u": nf(ffn.sn_3.thr.item(), ffn.sn_3.leak, True),
                "y2": nf(ffn.sn_2.thr.item(), ffn.sn_2.leak, True),
            }
            self.cmp1 = ComparatorModel(block.q_in1.thr.item(), **gk)
            self.cmp2 = ComparatorModel(block.q_in2.thr.item(), **gk)

    def init_state(self, B, device="cpu"):
        a, f = self.block.attn, self.block.ffn
        arr = GainCellArray(B, a.Hv, a.dk, a.dv, device=device, **self.gain_cell_kw)
        st = dict(attn=dict(_arr=arr, S=arr.S,
                            nq=self.neu["q"].init_state((B, a.Hk * a.dk), device),
                            nk=self.neu["k"].init_state((B, a.Hk * a.dk), device),
                            nv=self.neu["v"].init_state((B, a.Hv * a.dv), device),
                            npre=self.neu["pre"].init_state((B, a.Hv * a.dv), device),
                            nout=self.neu["out"].init_state((B, a.d), device)),
                  ffn=dict(ng=self.neu["g"].init_state((B, f.d_ff), device),
                           nu=self.neu["u"].init_state((B, f.d_ff), device),
                           ny2=self.neu["y2"].init_state((B, f.d), device)))
        if a.fir_taps > 1:
            st["attn"]["fir_q"] = a.fir_q.init_state(B, device)
            st["attn"]["fir_k"] = a.fir_k.init_state(B, device)
            st["attn"]["fir_v"] = a.fir_v.init_state(B, device)
        return st

    @torch.no_grad()
    def step(self, h, st, ext_gates=None, record=False):
        a = self.block.attn
        B = h.shape[0]
        sa, sf = st["attn"], st["ffn"]
        # φ0:比较器读累加器;门值(外部给定或门电路/门函数)
        x = self.cmp1(h)
        if ext_gates is not None:
            alpha, beta = ext_gates
        elif isinstance(a.gates, ExternalGates):
            raise ValueError("该层的门为 ExternalGates,需提供 ext_gates")
        else:
            alpha, beta = a.gates(x)
        # φ2:投影 crossbar → (FIR) → 神经元,锁存 q, k, v
        zq, zk, zv = self.xb["Wq"](x), self.xb["Wk"](x), self.xb["Wv"](x)
        if a.fir_taps > 1:
            zq, sa["fir_q"] = a.fir_q(zq, sa["fir_q"])
            zk, sa["fir_k"] = a.fir_k(zk, sa["fir_k"])
            zv, sa["fir_v"] = a.fir_v(zv, sa["fir_v"])
        q, sa["nq"] = self.neu["q"].step(zq, sa["nq"])
        k, sa["nk"] = self.neu["k"].step(zk, sa["nk"])
        v, sa["nv"] = self.neu["v"].step(zv, sa["nv"])
        rep = a.Hv // a.Hk
        q = q.view(B, a.Hk, a.dk).repeat_interleave(rep, dim=1)
        k = (k.view(B, a.Hk, a.dk) * a.c_k).repeat_interleave(rep, dim=1)
        v = v.view(B, a.Hv, a.dv)
        # φ1/φ3/φ4/φ5:状态阵列的衰减、检索、写入、读出(时序由 GainCellArray 的配置决定)
        o, S = sa["_arr"].cycle(k, v, q, alpha, beta)
        sa["S"] = S
        o_spk, sa["npre"] = self.neu["pre"].step(o.reshape(B, a.Hv * a.dv), sa["npre"])
        # φ6:W_O、输出神经元、累加器、第二个比较器
        y, sa["nout"] = self.neu["out"].step(self.xb["Wo"](o_spk), sa["nout"])
        h_mid = h + y
        x2 = self.cmp2(h_mid)
        # φ7:W_1、W_3、两路神经元、锁存 AND
        g, sf["ng"] = self.neu["g"].step(self.xb["W1"](x2), sf["ng"])
        u, sf["nu"] = self.neu["u"].step(self.xb["W3"](x2), sf["nu"])
        p = latch_and(g, u)
        # φ8:W_2、输出神经元、累加器
        y2, sf["ny2"] = self.neu["y2"].step(self.xb["W2"](p), sf["ny2"])
        h_out = h_mid + y2
        rec = None
        if record:
            rec = dict(h_in=h, x_attn=x, alpha=alpha, beta=beta, q=q, k=k, v=v, S=S, o=o, o_spk=o_spk,
                       y_attn=y, h_mid=h_mid, x_ffn=x2, g=g, u=u, p=p, y_ffn=y2, h_out=h_out)
        return h_out, st, rec


def run_behavioral_sequence(sim, h_seq, st=None, ext_gates_seq=None, record=False):
    """与 block.run_sequence 同语义:st=None 为新序列;传入上一段 st 则接着跑。"""
    B, L, _ = h_seq.shape
    if st is None:
        st = sim.init_state(B, h_seq.device)
    outs, recs = [], []
    for t in range(L):
        ext = None if ext_gates_seq is None else (ext_gates_seq[0][:, t], ext_gates_seq[1][:, t])
        h_out, st, rec = sim.step(h_seq[:, t], st, ext_gates=ext, record=record)
        outs.append(h_out); recs.append(rec)
    out = torch.stack(outs, dim=1)
    return (out, st, recs) if record else (out, st)
