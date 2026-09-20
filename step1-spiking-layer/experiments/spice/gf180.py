"""GF180MCU 真实晶体管上的 gain cell 泄漏路径对比(S1.3 第 5 项:泄漏是否欧姆型)。

四种泄漏路径,同一存储电容、同一泄漏窗:
  ground            存储节点经 NMOS 漏到地(饱和/亚阈区),栅压调 alpha —— 恒流放电,alpha 依赖存储值
  subthreshold_mid  NMOS 源接 V_mid,亚阈区,小摆幅;I ∝ (1 - exp(-V_ds/V_t)),只有摆幅 ≪ 26 mV 才近似欧姆
  triode_mid        长沟道 NMOS 源接 V_mid,强反型三极管区;I ∝ V_ov·V_ds - V_ds²/2,摆幅 ≪ V_ov 才近似欧姆
  resistor          开关(NMOS,栅 3.3 V)串电阻到 V_mid,alpha = exp(-t_gap/RC),由泄漏窗长度 t_gap 定 —— 欧姆型
"""
import json
import math
from pathlib import Path

from .runner import run_ngspice, parse_meas, parse_wrdata

ROOT = Path(__file__).resolve().parents[2]
PDK = ROOT / "pdk" / "gf180mcu"
LINKS = (PDK / "design.ngspice", PDK / "sm141064.ngspice")
HEADER = ".include design.ngspice\n.param sw_stat_global=0 sw_stat_mismatch=0\n.lib sm141064.ngspice typical\n"   # 关掉统计变差,确定性 typical
VMID = 1.0


def _run(netlist):
    log, files = run_ngspice(HEADER + netlist, files=("wave.txt",), links=LINKS)
    return log, files


def idvg(vd=1.0, w=1e-6, l=0.28e-6):
    """nmos_3p3 的 Id–Vg(0 … 3.3 V,步长 0.1 V)。返回 [(Vg, Id)]。"""
    net = f"""* nmos_3p3 Id-Vg
M1 d g 0 0 nmos_3p3 W={w:.3g} L={l:.3g}
Vd d 0 {vd}
Vg g 0 0
.control
set wr_singlescale
set wr_vecnames
dc Vg 0 3.3 0.1
let id = -i(Vd)
wrdata wave.txt id
.endc
.end
"""
    log, files = _run(net)
    if "wave.txt" not in files:
        raise RuntimeError(log[-1500:])
    w_ = parse_wrdata(files["wave.txt"])
    return list(zip(w_["v-sweep"], w_["id"]))


def alpha_leak(mode, v_leak=0.0, signal=0.1, cs=50e-15, t_gap=1e-6, w=0.22e-6, l=0.28e-6, r_leak=1e6, sw_w=2e-6):
    """一个泄漏窗后的 alpha = (V_end - V_mid) / signal(ground 模式 V_mid = 0)。v_leak 在 *_mid 模式下是 V_gs。"""
    tstop = t_gap * 1.02
    if mode == "ground":
        body = f"""M1 s g 0 0 nmos_3p3 W={w:.3g} L={l:.3g}
Vg g 0 {v_leak}
Cs s 0 {cs:.4g} ic={signal}
"""
        vmid = 0.0
    elif mode in ("subthreshold_mid", "triode_mid"):
        body = f"""Vmid mid 0 {VMID}
M1 s g mid 0 nmos_3p3 W={w:.3g} L={l:.3g}
Vg g 0 {VMID + v_leak}
Cs s 0 {cs:.4g} ic={VMID + signal}
"""
        vmid = VMID
    elif mode == "resistor":
        body = f"""Vmid mid 0 {VMID}
Rl s x {r_leak:.4g}
M1 x g mid 0 nmos_3p3 W={sw_w:.3g} L=0.28u
Vg g 0 PULSE(0 3.3 0 1p 1p {t_gap:.6g} 1)
Cs s 0 {cs:.4g} ic={VMID + signal}
"""
        vmid = VMID
        tstop = t_gap + 20e-9
    else:
        raise ValueError(mode)
    net = body + f""".control
tran {t_gap / 500:.4g} {tstop:.6g} uic
meas tran v_end find v(s) at={t_gap:.6g}
.endc
.end
"""
    log, _ = _run(net)
    m = parse_meas(log)
    if "v_end" not in m:
        raise RuntimeError(f"ngspice 未给出 v_end({mode}):\n" + log[-1500:])
    return (m["v_end"] - vmid) / signal


def pick_vleak_for_half(mode, v_leaks, signal, **kw):
    """在 v_leaks 里找 alpha(signal) 最接近 0.5 的栅压。返回 (v_leak, alpha)。"""
    best = None
    for v in v_leaks:
        a = alpha_leak(mode, v_leak=v, signal=signal, **kw)
        if best is None or abs(a - 0.5) < abs(best[1] - 0.5):
            best = (v, a)
    return best


def spread(mode, v_leak, signals, **kw):
    a = [alpha_leak(mode, v_leak=v_leak, signal=s, **kw) for s in signals]
    return max(a) - min(a), dict(zip([str(s) for s in signals], a))


def main():
    out = dict(pdk="GF180MCU typical, nmos_3p3", vmid=VMID)
    out["idvg_W1u_L0p28u_Vd1"] = [dict(vg=round(v, 2), id=i) for v, i in idvg()]
    # ground:C=50 fF,t_gap 1 us
    vg, a = pick_vleak_for_half("ground", [0.30 + 0.025 * i for i in range(25)], 1.0)
    sp, vals = spread("ground", vg, (1.0, 0.5, 0.25))
    out["ground"] = dict(cs=50e-15, t_gap=1e-6, v_leak=vg, alpha_at_1V=a, alpha_by_signal=vals, spread=sp)
    # subthreshold_mid:C=50 fF,t_gap 1 us,摆幅 ±0.1 / ±0.02 / ±0.005
    vg, a = pick_vleak_for_half("subthreshold_mid", [0.20 + 0.025 * i for i in range(29)], 0.02)
    res = {}
    for sw in (0.1, 0.02, 0.005):
        sp, vals = spread("subthreshold_mid", vg, (sw, sw / 2, -sw / 2, -sw))
        res[str(sw)] = dict(spread=sp, alpha_by_signal=vals)
    out["subthreshold_mid"] = dict(cs=50e-15, t_gap=1e-6, v_gs=vg, alpha_ref=a, by_swing=res)
    # triode_mid:长沟道 W/L = 0.22u/10u,C = 1 pF,t_gap 100 ns,V_gs 1.5 … 3.3
    kw = dict(cs=1e-12, t_gap=100e-9, l=10e-6)
    vg, a = pick_vleak_for_half("triode_mid", [1.5 + 0.1 * i for i in range(19)], 0.05, **kw)
    res = {}
    for sw in (0.2, 0.1, 0.05, 0.02):
        sp, vals = spread("triode_mid", vg, (sw, sw / 2, -sw / 2, -sw), **kw)
        res[str(sw)] = dict(spread=sp, alpha_by_signal=vals)
    out["triode_mid"] = dict(**kw, v_gs=vg, alpha_ref=a, by_swing=res)
    # resistor:R = 1 MΩ(理想),C = 50 fF,tau = 50 ns,t_gap 扫描
    res = {}
    for tg in (10e-9, 20e-9, 35e-9, 50e-9, 100e-9):
        sp, vals = spread("resistor", 0.0, (0.3, 0.1, -0.1, -0.3), cs=50e-15, t_gap=tg, r_leak=1e6)
        res[str(tg)] = dict(spread=sp, alpha_by_signal=vals, alpha_exp=math.exp(-tg / (1e6 * 50e-15)))
    out["resistor"] = dict(cs=50e-15, r_leak=1e6, switch="nmos_3p3 W=2u L=0.28u, gate 3.3 V during t_gap", by_tgap=res)
    (ROOT / "results" / "spice-gf180-leak.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "idvg_W1u_L0p28u_Vd1"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
