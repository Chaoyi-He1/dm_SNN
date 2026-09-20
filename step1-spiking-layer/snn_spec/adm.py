"""Zhao et al., "A spiking artificial neuron based on one diffusive memristor, one transistor and one resistor",
Nature Electronics 8, 1211–1221 (2025),补充材料 Note 4 与 Note 6 的 1M1T1R 物理模型。

状态变量:细丝长度 F(x₁)、残留长度 R(x₂)、栅电压 V(V_gs,即膜电位)。输入为施加在扩散忆阻器上的电压 V_in。
    dF = (−γ₁F − βF + μE) dt,  dR = (−γ₂R + βF) dt,  E = (V_in − V)/(α + |1 − x|)·1[V_in > V],  x = F + R
    R_d = R_on·x + R_off·(e^{(1−x)/λ} − 1)/(e^{1/λ} − 1)
    C_g dV/dt = (V_in − V)/R_d − V/R_c,  O = H(V − V_th)
随机项 N₁、N₂ 的方差系数 σ₁、σ₂ 默认置零,即"理想稳定"版本。x 在计算 R_d 时钳到 [0, 1]。

发放事件的定义:扩散忆阻器导通(x 自下而上越过 1,R_d 塌缩到 R_on)。论文 Eq. S15 的输出 O = H(V − V_th) 一并给出,
但在 50% 占空比的脉冲串下 V 在两次导通之间降不到 V_th 以下(R_c·C_g = 2.5 ms),O 会恒为 1,不能用它计脉冲。
"""
import math
import torch

# 补充材料 Table 1:与 Fig. 2e 实验匹配的参数(ms 时间尺度);单位:μ、β、γ 为 s⁻¹ 量级的速率,Ω、F、V。
TABLE1 = dict(mu=150.0, alpha=0.01, beta=20.0, gamma1=600.0, gamma2=5.0, lam=0.2,
              Ron=1e6, Roff=1e12, Cg=10e-12, Rc=250e6, Vth=0.2)
# Table 7:65 nm 节点能耗估计所用(R_c、R_l 提到 1 GΩ,μ 300)
TABLE7 = dict(TABLE1, mu=300.0, Rc=1e9)
# Table 8:3 nm FinFET 的快速投影(动力学快 1e5 倍,aF 级电容);此处仅保留忆阻器与电容、R_c
TABLE8 = dict(mu=3e7, alpha=0.01, beta=2e6, gamma1=6e7, gamma2=5e5, lam=0.2,
              Ron=1e8, Roff=4e18, Cg=4.14e-18, Rc=242e9, Vth=0.2)


class ADMNeuronModel:
    def __init__(self, mu, alpha, beta, gamma1, gamma2, lam, Ron, Roff, Cg, Rc, Vth, dt, sigma1=0.0, sigma2=0.0):
        self.mu, self.alpha, self.beta = float(mu), float(alpha), float(beta)
        self.gamma1, self.gamma2, self.lam = float(gamma1), float(gamma2), float(lam)
        self.Ron, self.Roff, self.Cg, self.Rc, self.Vth = float(Ron), float(Roff), float(Cg), float(Rc), float(Vth)
        self.dt, self.sigma1, self.sigma2 = float(dt), float(sigma1), float(sigma2)
        self._den = math.exp(1.0 / self.lam) - 1.0

    def init_state(self, shape, device="cpu"):
        z = lambda: torch.zeros(shape, device=device, dtype=torch.float64)
        return dict(F=z(), R=z(), V=z())

    def Rd(self, x):
        xc = x.clamp(0.0, 1.0)
        return (self.Ron * xc + self.Roff * (torch.exp((1.0 - xc) / self.lam) - 1.0) / self._den).clamp(min=self.Ron)

    def substep(self, vin, st):
        """一个 dt 的显式欧拉步(Eq. S12–S14)。vin: 张量,与状态同形。"""
        F, R, V = st["F"], st["R"], st["V"]
        x = F + R
        E = torch.where(vin > V, (vin - V) / (self.alpha + (1.0 - x).abs()), torch.zeros_like(V))
        dF = (-self.gamma1 * F - self.beta * F + self.mu * E) * self.dt
        dR = (-self.gamma2 * R + self.beta * F) * self.dt
        if self.sigma1 > 0:
            dF = dF + torch.randn_like(F) * torch.sqrt((self.sigma1 * dF).abs())
        if self.sigma2 > 0:
            dR = dR + torch.randn_like(R) * torch.sqrt((self.sigma2 * dR).abs())
        Rd = self.Rd(x)
        dV = self.dt / (Rd * self.Cg) * (vin - V) - self.dt / (self.Rc * self.Cg) * V
        F, R, V = F + dF, R + dR, V + dV
        return dict(F=F, R=R, V=V)

    def run(self, vin_seq, st=None):
        """按 dt 逐点跑一段输入电压序列 vin_seq: [T](单个神经元,确定性,纯标量循环以便跑秒级仿真)。
        返回 dict(V, O, F, R, A),各 [T];A = 1[x >= 1] 为忆阻器导通指示,其上升沿即发放事件。"""
        if self.sigma1 > 0 or self.sigma2 > 0:
            raise NotImplementedError("run() 只实现确定性版本;随机项用 substep()")
        F, R, V = (0.0, 0.0, 0.0) if st is None else (float(st["F"]), float(st["R"]), float(st["V"]))
        mu, alpha, beta, g1, g2, lam = self.mu, self.alpha, self.beta, self.gamma1, self.gamma2, self.lam
        Ron, Roff, Cg, Rc, Vth, dt, den = self.Ron, self.Roff, self.Cg, self.Rc, self.Vth, self.dt, self._den
        exp = math.exp
        Vs, Os, Fs, Rs, As = [], [], [], [], []
        for v in vin_seq.tolist():
            x = F + R
            E = (v - V) / (alpha + abs(1.0 - x)) if v > V else 0.0
            dF = (-g1 * F - beta * F + mu * E) * dt
            dR = (-g2 * R + beta * F) * dt
            xc = min(max(x, 0.0), 1.0)
            Rd = max(Ron * xc + Roff * (exp((1.0 - xc) / lam) - 1.0) / den, Ron)
            V = V + dt / (Rd * Cg) * (v - V) - dt / (Rc * Cg) * V
            F, R = F + dF, R + dR
            Vs.append(V); Os.append(1.0 if V >= Vth else 0.0); Fs.append(F); Rs.append(R)
            As.append(1.0 if F + R >= 1.0 else 0.0)
        t = lambda a: torch.tensor(a, dtype=torch.float64)
        return dict(V=t(Vs), O=t(Os), F=t(Fs), R=t(Rs), A=t(As))

    def step(self, vin, st, T_cyc):
        """token 接口:一个周期内输入电压恒为 vin(张量 [B]),积分 T_cyc/dt 个子步;
        周期内忆阻器只要发生一次导通(x 自下而上越过 1)即记一个脉冲。返回 (S ∈ {0,1}, st)。"""
        n = max(1, int(round(T_cyc / self.dt)))
        vin = vin.to(torch.float64)
        prev = (st["F"] + st["R"]) >= 1.0
        fired = torch.zeros_like(st["V"], dtype=torch.bool)
        for _ in range(n):
            st = self.substep(vin, st)
            now = (st["F"] + st["R"]) >= 1.0
            fired |= now & ~prev
            prev = now
        return fired.double(), st


def pulse_train(amplitude, period, duty, t_end, dt):
    t = torch.arange(0.0, t_end, dt, dtype=torch.float64)
    return torch.where((t % period) < duty * period, torch.full_like(t, amplitude), torch.zeros_like(t))


def onsets(A):
    """指示序列 A ∈ {0,1} 的上升沿位置(索引)。"""
    return torch.nonzero((A[1:] - A[:-1]) > 0.5).squeeze(1) + 1


def first_fire_time(A, dt):
    idx = onsets(A) if len(A) > 1 else torch.zeros(0)
    if len(idx) == 0 and len(A) > 0 and A[0] > 0.5:
        return 0.0
    return None if len(idx) == 0 else idx[0].item() * dt
