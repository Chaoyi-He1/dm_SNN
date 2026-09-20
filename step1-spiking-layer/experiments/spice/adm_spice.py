"""ngspice 里的 1M1T1R 神经元:扩散忆阻器用 Zhao et al. 2025 补充材料 Note 4 的物理模型(B 源 + 积分电容实现
状态方程),晶体管可选论文的等效参数(level 1,K = 3.75e-4 A/V²、V_th = 0.22 V)或 GF180MCU 的 nmos_3p3。"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from .runner import run_ngspice, parse_wrdata
from .gf180 import HEADER as GF_HEADER, LINKS as GF_LINKS

ROOT = Path(__file__).resolve().parents[2]
SCALE = 1e-3   # 状态变量积分电容取 1 mF、电流乘 1e-3,数值上等价于 1 F 与原电流


def netlist(P, vin_spec, t_end, transistor="paper", Rl=47e3, Vdd=1.0, tstep=2e-6, tmax=2e-6):
    den = math.exp(1.0 / P["lam"]) - 1.0
    if transistor == "paper":
        header, mos = "", ".model nch nmos level=1 kp=7.5e-4 vto=0.22\nM1 d g 0 0 nch W=1u L=1u"
    elif transistor == "gf180":
        header, mos = GF_HEADER, "M1 d g 0 0 nmos_3p3 W=1u L=0.28u"
    else:
        raise ValueError(transistor)
    return f"""* 1M1T1R neuron, ADM physics model (Zhao 2025 SI Note 4), deterministic
{header}Vin in 0 {vin_spec}
Cx1 x1 0 {SCALE} ic=0
Cx2 x2 0 {SCALE} ic=0
BE e 0 V = u(V(in)-V(g))*(V(in)-V(g))/({P['alpha']}+abs(1-V(x1)-V(x2)))
Bx1 0 x1 I = {SCALE}*(-{P['gamma1'] + P['beta']}*V(x1) + {P['mu']}*V(e))
Bx2 0 x2 I = {SCALE}*(-{P['gamma2']}*V(x2) + {P['beta']}*V(x1))
Bxc xc 0 V = min(max(V(x1)+V(x2),0),1)
Bgd gd 0 V = 1/max({P['Ron']}*V(xc) + {P['Roff']}*(exp((1-V(xc))/{P['lam']})-1)/{den}, {P['Ron']})
Bd in g I = (V(in)-V(g))*V(gd)
Cg g 0 {P['Cg']} ic=0
Rc g 0 {P['Rc']}
{mos}
Rl vdd d {Rl}
Vdd vdd 0 {Vdd}
.control
set wr_singlescale
set wr_vecnames
tran {tstep} {t_end} 0 {tmax} uic
wrdata wave.txt v(g) v(d) v(x1) v(x2)
.endc
.end
"""


def simulate(P, vin_spec, t_end, transistor="paper", **kw):
    """返回 dict(t, Vg, Vd, x)(numpy 数组,ngspice 内部时间点)。"""
    log, files = run_ngspice(netlist(P, vin_spec, t_end, transistor, **kw), files=("wave.txt",),
                             links=GF_LINKS if transistor == "gf180" else ())
    if "wave.txt" not in files:
        raise RuntimeError("ngspice 未输出波形:\n" + log[-2500:])
    w = parse_wrdata(files["wave.txt"])
    t = np.array(w["time"]); Vg = np.array(w["v(g)"]); Vd = np.array(w["v(d)"])
    x = np.array(w["v(x1)"]) + np.array(w["v(x2)"])
    return dict(t=t, Vg=Vg, Vd=Vd, x=x)


def turn_on_times(t, x):
    on = (x >= 1.0).astype(float)
    idx = np.nonzero(np.diff(on) > 0.5)[0] + 1
    return t[idx]


def compare_with_python(P, vin_const=1.0, t_end=60e-3, dt=2e-6, transistor="paper"):
    """同一恒定输入下,SPICE 与 Python 模型(snn_spec.adm)的栅电压与首次导通时刻对照。"""
    from snn_spec.adm import ADMNeuronModel, first_fire_time
    sp = simulate(P, f"DC {vin_const}", t_end, transistor, tstep=dt, tmax=dt)
    m = ADMNeuronModel(**P, dt=dt)
    r = m.run(torch.full((int(t_end / dt),), vin_const, dtype=torch.float64))
    tpy = np.arange(1, len(r["V"]) + 1) * dt
    Vg_sp = np.interp(tpy, sp["t"], sp["Vg"])
    Vg_py = r["V"].numpy()
    on_sp = turn_on_times(sp["t"], sp["x"])
    t_py = first_fire_time(r["A"], dt)
    return dict(vin=vin_const, t_first_spice=float(on_sp[0]) if len(on_sp) else None, t_first_python=t_py,
                n_on_spice=int(len(on_sp)), rms_dV=float(np.sqrt(np.mean((Vg_sp - Vg_py) ** 2))),
                max_dV=float(np.max(np.abs(Vg_sp - Vg_py))))


def main():
    from snn_spec.adm import TABLE1
    out = dict(model="Zhao 2025 SI Note 4, Table 1, sigma=0", transistor_paper="level1 kp=7.5e-4 vto=0.22, Rl=47k, Vdd=1V")
    out["cross_check"] = [compare_with_python(TABLE1, v) for v in (0.6, 0.8, 1.0)]
    sp = simulate(TABLE1, "PULSE(0 0.8 0 1u 1u 1m 2m)", 0.4, "paper")
    ev = turn_on_times(sp["t"], sp["x"])
    out["pulse_train_paper_transistor"] = dict(t_end=0.4, n_turn_on=int(len(ev)), first=float(ev[0]) if len(ev) else None,
                                              Vg_max=float(sp["Vg"].max()), Vd_min=float(sp["Vd"].min()))
    sp2 = simulate(TABLE1, "PULSE(0 0.8 0 1u 1u 1m 2m)", 0.4, "gf180")
    ev2 = turn_on_times(sp2["t"], sp2["x"])
    out["pulse_train_gf180_nmos_3p3"] = dict(t_end=0.4, n_turn_on=int(len(ev2)), first=float(ev2[0]) if len(ev2) else None,
                                            Vg_max=float(sp2["Vg"].max()), Vd_min=float(sp2["Vd"].min()),
                                            note="Rl=47k, Vdd=1 V 沿用论文;GF180 nmos_3p3 阈值约 0.7 V,输出摆幅小,Rl/Vdd 需按 3.3 V 工艺重设")
    (ROOT / "results" / "spice-adm-neuron.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
