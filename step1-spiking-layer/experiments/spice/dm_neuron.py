"""S1.2:理想稳定 DM 神经元的 SPICE 瞬态 → 六参数状态模型拟合 → 在新序列上验证。

器件模型:膜电容 Cm 与泄漏电阻 Rm 做泄漏积分;扩散忆阻器用带迟滞的阈值开关代替
(V_th 导通、降到 V_hold 关断,导通电阻恒定),串一个电阻 Ron;这是"稳定、确定性"的占位模型,
参数待器件组标定后替换。每个周期注入一个恒定电流,周期边界采样膜电压,周期内开关是否导通记为发放。
"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from snn_spec.circuit import DMNeuronModel, fit_dm_neuron   # noqa: E402
from snn_spec.metrics import spike_prf_by_sign               # noqa: E402
from experiments.spice.runner import run_ngspice, parse_wrdata  # noqa: E402

C_M, R_M, T_CYC = 1e-9, 1e4, 1e-6          # tau_m = 10 us,beta_m 理论值 exp(-0.1) = 0.905
V_TH, V_HOLD, R_ON = 1.0, 0.2, 10.0         # 导通放电 tau = (1+R_on)·C_m = 11 ns,远短于周期
I_MEAN_MA, I_SIGMA_MA = 0.3, 0.6            # 每周期恒定电流的均值与标准差(mA)
# 单次发放量程:最坏情况是周期一开始就越阈复位,之后整个周期都在再充电;不再次越阈要求
# I < C_m (V_th - U_hold) / T_cyc = 0.8 mA。另外导通期间 I > V_hold / (1 + R_on) = 18 mA 会使开关锁住不关断。
I_MAX_MA = 0.75


def _pwl(I_A):
    pts = []
    for k, i in enumerate(I_A):
        pts.append(f"{k * T_CYC:.10g} {i:.6g}")
        pts.append(f"{(k + 1) * T_CYC - 1e-9:.10g} {i:.6g}")
    body = "\n+ ".join(" ".join(pts[j:j + 8]) for j in range(0, len(pts), 8))
    return "PWL(" + body + ")"


def _netlist(I_A):
    t_end = len(I_A) * T_CYC
    return f"""* ideal stable DM neuron: RC membrane + hysteretic threshold switch
.model TS sw vt={(V_TH + V_HOLD) / 2} vh={(V_TH - V_HOLD) / 2} ron=1 roff=1e9
Cm mem 0 {C_M:.6g} ic=0
Rm mem 0 {R_M:.6g}
Iin 0 mem {_pwl(I_A)}
Sdm mem d mem 0 TS
Ron d 0 {R_ON}
.control
set wr_singlescale
set wr_vecnames
tran 5n {t_end:.10g} uic
wrdata wave.txt v(mem) v(d)
.endc
.end
"""


def simulate(I_mA):
    """返回 (U: 周期边界膜电压 [T+1], fired: [T])。"""
    I_A = [i * 1e-3 for i in I_mA]
    log, files = run_ngspice(_netlist(I_A), files=("wave.txt",))
    if "wave.txt" not in files:
        raise RuntimeError("ngspice 未输出波形:\n" + log[-2000:])
    w = parse_wrdata(files["wave.txt"])
    t = torch.tensor(w["time"]); vm = torch.tensor(w["v(mem)"]); vd = torch.tensor(w["v(d)"])
    n = len(I_mA)
    U, fired = [], []
    for k in range(n + 1):
        idx = torch.argmin((t - k * T_CYC).abs())
        U.append(vm[idx].item())
    on = (vd > 0.1).float()
    onset = torch.zeros_like(on); onset[1:] = ((on[1:] - on[:-1]) > 0.5).float()   # 导通的起始边沿
    onsets = []
    for k in range(n):
        m = (t >= k * T_CYC) & (t < (k + 1) * T_CYC)
        onsets.append(int(onset[m].sum().item()))
        fired.append(1.0 if onsets[-1] > 0 else 0.0)
    simulate.last_onsets = onsets
    return torch.tensor(U), torch.tensor(fired)


def fit_and_validate(seed=0, n_train=400, n_test=200):
    g = torch.Generator().manual_seed(seed)
    I_tr = (I_MEAN_MA + torch.randn(n_train, generator=g) * I_SIGMA_MA).clamp(max=I_MAX_MA).tolist()
    U_tr, S_tr = simulate(I_tr)
    double_tr = sum(1 for c in simulate.last_onsets if c > 1)
    fit = fit_dm_neuron(torch.tensor(I_tr)[:, None], U_tr[:, None], S_tr[:, None], ternary=False)
    I_te = (I_MEAN_MA + torch.randn(n_test, generator=g) * I_SIGMA_MA).clamp(max=I_MAX_MA).tolist()
    _, S_te = simulate(I_te)
    double_te = sum(1 for c in simulate.last_onsets if c > 1)
    model = DMNeuronModel(**fit, ternary=False)
    st = model.init_state((1,))
    pred = []
    for i in I_te:
        s, st = model.step(torch.tensor([i]), st)
        pred.append(s.item())
    prf = spike_prf_by_sign(torch.tensor(pred), S_te)
    return dict(fit=fit, pos_precision=prf["pos_precision"], pos_recall=prf["pos_recall"],
                n_train_spikes=int(S_tr.sum().item()), n_test_spikes=int(S_te.sum().item()),
                double_spike_cycles=dict(train=double_tr, test=double_te), I_max_mA=I_MAX_MA,
                beta_m_theory=math.exp(-T_CYC / (R_M * C_M)), g_m_theory_V_per_mA=R_M * (1 - math.exp(-T_CYC / (R_M * C_M))) * 1e-3)


def main():
    rows = [fit_and_validate(seed) for seed in (0, 1, 2)]
    report = dict(device=dict(C_m=C_M, R_m=R_M, T_cycle=T_CYC, V_th=V_TH, V_hold=V_HOLD, R_on=R_ON, I_mean_mA=I_MEAN_MA, I_sigma_mA=I_SIGMA_MA),
                  note="RC 膜 + 带迟滞阈值开关的理想稳定 DM 占位模型;无不应期机制;参数待实测替换",
                  runs=rows)
    (ROOT / "results/spice-dm-neuron.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
