"""S1.3 / S1.4 的 gain cell 测试台。

single_step_discrimination:单个 gain cell(电容)配相位开关,用规范 S1.4 的注入量跑一个周期,
    泄漏窗的位置决定得到形式 A(1.25)、B(1.00)、C(0.75),或"泄漏一直开"的中间值。
alpha_sweep:泄漏管用通用 MOSFET 模型(占位器件),扫栅压得到 alpha = V(t_gap)/V(0) 的可调区间。
"""
import json
import math
from pathlib import Path

from .runner import run_ngspice, parse_meas

TAU = 1e-6                    # 泄漏窗内 R·C
T_GAP = math.log(2) * TAU     # alpha = 0.5 对应的泄漏窗长度
CS = 1e-9                     # 单 cell 电容(理想元件,便于观察)
V_IN, S0 = 2.0, 1.0


def _pulse(t_on, pw):
    return f"PULSE(0 1 {t_on:.10g} 1p 1p {pw:.10g} 1)"


def _netlist(t_leak_on, leak_pw, t_sample, sample_dur, t_write, write_dur, t_end, BETA=0.5):
    return f"""* single gain cell with phase-switched leak, sample-and-hold retrieval, gated write
.model SW sw vt=0.5 vh=0.01 ron=1 roff=1e12
Cs s 0 {CS:.6g} ic={S0}
Rleak s lk {TAU / CS:.6g}
Sleak lk 0 phi1 0 SW
Vphi1 phi1 0 {_pulse(t_leak_on, leak_pw)}
E1 buf 0 s 0 1
Ssh buf hold phi3 0 SW
Chold hold 0 1p ic=0
Vphi3 phi3 0 {_pulse(t_sample, sample_dur)}
Vphi4 phi4 0 {_pulse(t_write, write_dur)}
Bw 0 s I = u(V(phi4)-0.5)*{BETA}*({V_IN}-V(hold))*{CS / write_dur:.6g}
.control
tran 1n {t_end + 10e-9:.10g} uic
meas tran s_end find v(s) at={t_end:.10g}
meas tran r_hold find v(hold) at={t_end:.10g}
.endc
.end
"""


def schedules(alpha=0.5):
    """相位表:(泄漏窗起点, 泄漏窗长度, 采样起点, 采样长度, 写入起点, 写入长度, 周期末)。泄漏窗长度由 alpha 定。"""
    tg = -math.log(alpha) * TAU
    return {
        "A": (0.0, tg, tg + 0.10e-6, 50e-9, tg + 0.20e-6, 100e-9, tg + 0.5e-6),        # 规范:泄漏窗 → 检索 → 写入
        "B": (0.15e-6, tg, 0.05e-6, 50e-9, tg + 0.20e-6, 100e-9, tg + 0.5e-6),        # 检索在泄漏窗之前
        "C": (0.30e-6, tg, 0.05e-6, 50e-9, 0.15e-6, 100e-9, tg + 0.5e-6),             # 泄漏窗在写入之后
        "always_on": (0.0, tg, tg / 2 - 5e-9, 10e-9, tg / 2 + 15e-9, 10e-9, tg),       # 泄漏管整周期开着,周期中点检索并写入
    }


def closed_form(alpha, beta, s0=S0, v=V_IN):
    return dict(A=alpha * (1 - beta) * s0 + beta * v,
                B=(alpha - beta) * s0 + beta * v,
                C=alpha * (1 - beta) * s0 + alpha * beta * v)


def always_on_prediction(alpha, beta, s0=S0, v=V_IN):
    """泄漏常开时的解析值:S = alpha·s0 + beta·(v - alpha^{f_s} s0)·alpha^{1-f_w},f_s、f_w 为采样结束、写入结束在周期内的位置。"""
    _, tg, ts, sd, tw, wd, t_end = schedules(alpha)["always_on"]
    f_s, f_w = (ts + sd) / t_end, (tw + wd) / t_end
    return alpha * s0 + beta * (v - alpha ** f_s * s0) * alpha ** (1 - f_w)


def single_step_discrimination(schedule="A", alpha=0.5, beta=0.5):
    log, _ = run_ngspice(_netlist(*schedules(alpha)[schedule], BETA=beta))
    m = parse_meas(log)
    if "s_end" not in m:
        raise RuntimeError("ngspice 未给出 s_end:\n" + log[-2000:])
    return m["s_end"]


# ----------------------------------------------------------------------------- alpha 扫描

GENERIC_MODELS = {
    "bsim3_default": ".model nch nmos level=8 version=3.3.0",
    "level1": ".model nch nmos level=1 vto=0.4 kp=2e-5 lambda=0.02",
}


def _alpha_netlist(model_line, v_leak, t_gap, cs, v0):
    return f"""* 3T1C storage node with leak transistor (generic placeholder device)
{model_line}
Cs s 0 {cs:.6g} ic={v0}
M1 s g 0 0 nch W=1u L=0.5u
Vg g 0 {v_leak}
.control
tran {t_gap / 1000:.6g} {t_gap * 1.01:.6g} uic
meas tran v_end find v(s) at={t_gap:.6g}
.endc
.end
"""


def alpha_at(v_leak, t_gap=1e-6, cs=1e-12, v0=1.0, model="level1"):
    log, _ = run_ngspice(_alpha_netlist(GENERIC_MODELS[model], v_leak, t_gap, cs, v0))
    m = parse_meas(log)
    if "v_end" not in m:
        raise RuntimeError(f"ngspice 未给出 v_end(model={model}):\n" + log[-2000:])
    return min(1.0, max(0.0, m["v_end"] / v0))


V_LEAKS = tuple(round(0.0 + 0.05 * i, 3) for i in range(17))    # 0 … 0.8 V


def alpha_sweep(v_leaks=V_LEAKS, t_gap=1e-6, cs=1e-12, v0=1.0, model="level1"):
    """返回 [(V_leak, alpha)]。默认 level-1 通用 NMOS(ngspice 的 BSIM3 默认模型卡在此偏置下几乎不导通,不采用)。"""
    return [(v, alpha_at(v, t_gap, cs, v0, model)) for v in v_leaks]


def main():
    root = Path(__file__).resolve().parents[2]
    points = {}
    for alpha, beta in ((0.5, 0.5), (0.25, 1.0)):
        key = f"alpha={alpha},beta={beta}"
        points[key] = dict(spice={k: single_step_discrimination(k, alpha, beta) for k in schedules(alpha)},
                           closed_form=closed_form(alpha, beta), always_on_prediction=always_on_prediction(alpha, beta))
    sweep = alpha_sweep()
    v0_dep = {str(v0): alpha_at(0.55, v0=v0) for v0 in (1.0, 0.5, 0.25)}
    report = dict(single_step=points,
                  alpha_sweep=[dict(v_leak=v, alpha=a) for v, a in sweep],
                  alpha_v0_dependence_at_vleak_0p55=v0_dep,
                  device_note="泄漏管为 ngspice level-1 通用 NMOS(vto 0.4 V, kp 2e-5, W=1u, L=0.5u),存储电容 1 pF,t_gap 1 us;占位器件,非实测")
    (root / "results/spice-gain-cell.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
