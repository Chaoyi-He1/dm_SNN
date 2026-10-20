"""1T1R 差分 crossbar 的列求和(S1.5 第一项):忆阻电阻 + GF180MCU nmos_3p3 选通管 + 行线/位线线电阻,
位线由电流传输器钳在 0 V 并把电流灌进膜电容。

每个权重 W_ji 用一对忆阻器 (G⁺, G⁻) 表示:行 i 有两条驱动线,电压 +V_read·x_i 与 −V_read·x_i;列 j 一条位线,
电流 I_j = V_read Σ_i (G⁺_ji − G⁻_ji) x_i。电导量化到 LEVELS 级;编程目标按 1T1R 总电阻(写-校验),即忆阻电阻
取 1/G_target − R_on,R_on 为选通管在工作点的导通电阻。

三种模式:
  ideal  B 源直接做 Σ_i W_ji x_i(gdn_layer 的首个闭环)
  R      忆阻电阻 + 选通管用导通电阻为 R_on 的理想开关代替(线性)+ 线电阻
  1T1R   忆阻电阻 + 真实 nmos_3p3 选通管(栅 3.3 V)+ 线电阻
"""
import json
import math
from pathlib import Path

import numpy as np

from .runner import run_ngspice, parse_meas, parse_wrdata
from .gf180 import HEADER, LINKS

ROOT = Path(__file__).resolve().parents[2]

G_ON, G_OFF = 100e-6, 1e-6          # 忆阻器导电窗:LRS 10 kΩ,HRS 1 MΩ
LEVELS = 16                         # 每个忆阻器的电导级数(4 bit)
V_READ = 0.2                        # 行线读电压(V),低于忆阻器的开关阈值
R_LINE = 1.0                        # 行线、位线每个 cell 一段的线电阻(Ω)
SEL_W, SEL_L = 2e-6, 0.28e-6        # 选通管尺寸
V_SEL = 3.3                         # 选通栅压
T_INT = 1.0e-6                      # 积分窗(s):与 gdn_layer 的 φ2、φ6 等宽
_R_ON_CACHE = {}


def r_on_sel(vds=0.01, w=SEL_W, l=SEL_L):
    """选通管在栅 3.3 V、源接 0 V(位线虚地)、漏 vds 时的导通电阻(Ω)。vds < 0 时源漏互换。"""
    key = (vds, w, l)
    if key not in _R_ON_CACHE:
        net = HEADER + f"""* nmos_3p3 R_on
M1 d g 0 0 nmos_3p3 W={w:.3g} L={l:.3g}
Vg g 0 {V_SEL}
Vd d 0 {vds}
.control
op
let r_on = abs({vds}) / abs(i(Vd))
print r_on
.endc
.end
"""
        log, _ = run_ngspice(net, links=LINKS)
        m = parse_meas(log)
        if "r_on" not in m:
            raise RuntimeError(log[-1500:])
        _R_ON_CACHE[key] = m["r_on"]
    return _R_ON_CACHE[key]


def quantize_weights(W, levels=LEVELS, w_max=None):
    """W → 量化整数对 (n⁺, n⁻)(0 … levels−1)与量化后的权重 W_q = (n⁺ − n⁻)·w_max/(levels−1)。"""
    W = np.asarray(W, dtype=float)
    w_max = float(np.abs(W).max()) if w_max is None else float(w_max)
    n = np.rint(np.abs(W) / w_max * (levels - 1)).astype(int)
    npos = np.where(W > 0, n, 0)
    nneg = np.where(W < 0, n, 0)
    Wq = (npos - nneg) * w_max / (levels - 1)
    return npos, nneg, Wq, w_max


def unit_scale(w_max, levels=LEVELS):
    """软件 1 个单位对应的位线电流(A):s = V_read·(G_on − G_off)/w_max。膜电容取 s·T_int 使 1 单位 = 1 V。"""
    return V_READ * (G_ON - G_OFF) / w_max


def conductance_targets(npos, nneg, levels=LEVELS):
    dG = (G_ON - G_OFF) / (levels - 1)
    return G_OFF + npos * dG, G_OFF + nneg * dG


def xbar_lines(tag, npos, nneg, rowp, rown, sel, mode, r_line=R_LINE, r_on=None, levels=LEVELS):
    """生成一个 crossbar 的网表行。npos/nneg: [n_out, n_in] 的电导级;rowp[i]/rown[i]:行 i 的 ± 驱动节点;
    sel:选通栅节点(1T1R)。返回 (网表行列表, 每列位线末端节点列表)。"""
    n_out, n_in = npos.shape
    Gp, Gn = conductance_targets(npos, nneg, levels)
    r_on = r_on_sel() if r_on is None else r_on
    L = [f".model SWSEL{tag} sw vt={V_SEL / 2:.4g} vh=0.05 ron={r_on:.6g} roff=1e12"]   # R 模式:理想开关,导通电阻 = 选通管 R_on
    ends = []
    for j in range(n_out):
        for i in range(n_in):
            # 位线段:bl{j}_{i} 自上而下,末端 bl{j}_{n_in}
            L.append(f"Rbl{tag}{j}_{i} bl{tag}{j}_{i} bl{tag}{j}_{i + 1} {r_line:.4g}")
        ends.append(f"bl{tag}{j}_{n_in}")
    for i in range(n_in):
        for pol, rows, G in (("p", rowp, Gp), ("n", rown, Gn)):
            # 行线段:row 节点自左向右 r{pol}{i}_{j}
            prev = rows[i]
            for j in range(n_out):
                node = f"r{tag}{pol}{i}_{j}"
                L.append(f"Rrl{tag}{pol}{i}_{j} {prev} {node} {r_line:.4g}")
                rm = 1.0 / G[j, i] - r_on
                mid = f"m{tag}{pol}{i}_{j}"
                L.append(f"Rm{tag}{pol}{i}_{j} {node} {mid} {rm:.6g}")
                if mode == "1T1R":
                    L.append(f"M{tag}{pol}{i}_{j} {mid} {sel} bl{tag}{j}_{i} 0 nmos_3p3 W={SEL_W:.3g} L={SEL_L:.3g}")
                elif mode == "R":
                    L.append(f"S{tag}{pol}{i}_{j} {mid} bl{tag}{j}_{i} {sel} 0 SWSEL{tag}")
                else:
                    raise ValueError(mode)
                prev = node
    return L, ends


def bench_netlist(npos, nneg, xs, mode, r_line=R_LINE, t_win=2e-6, t_sel=T_INT):
    """测试台:行线按窗口依次加载输入向量 xs(每窗 t_win,选通在窗中段开 t_sel),每列位线经电流传输器
    灌进积分电容(每窗起点清零)。测每窗末的积分电压与选通中点的位线电流。"""
    n_out, n_in = npos.shape
    cm = 10e-12
    L = [f"* 1T1R crossbar bench, {n_out}x{n_in}, mode {mode}"]
    L.append(".options method=gear")               # 梯形法在选通沿后给出不衰减的电流振铃(电荷仍守恒),改用 Gear
    L.append(".model SW sw vt=0.5 vh=0.01 ron=1 roff=1e12")
    n_win = len(xs)
    # 行驱动:PWL,每窗常数
    for i in range(n_in):
        pts = ["0 0"]
        for w, x in enumerate(xs):
            a, b = w * t_win, (w + 1) * t_win
            pts += [f"{a + 1e-9:.4g} {V_READ * x[i]:.6g}", f"{b - 1e-9:.4g} {V_READ * x[i]:.6g}", f"{b:.4g} 0"]
        pwl = "PWL(" + "\n+ ".join(" ".join(pts[k:k + 6]) for k in range(0, len(pts), 6)) + ")"
        L.append(f"Vrp{i} rp{i} 0 {pwl}")
        L.append(f"Ern{i} rn{i} 0 rp{i} 0 -1")
    t0 = (t_win - t_sel) / 2
    L.append(f"Vsel sel 0 PULSE(0 {V_SEL} {t0:.4g} 1n 1n {t_sel:.4g} {t_win:.4g})")
    L.append(f"Vclr clr 0 PULSE(0 1 0 1p 1p 50n {t_win:.4g})")
    lines, ends = xbar_lines("x", npos, nneg, [f"rp{i}" for i in range(n_in)], [f"rn{i}" for i in range(n_in)], "sel", mode, r_line)
    L += lines
    for j in range(n_out):
        L.append(f"Vs{j} {ends[j]} 0 0")
        L.append(f"F{j} 0 u{j} Vs{j} 1")
        L.append(f"Cm{j} u{j} 0 {cm:.4g} ic=0")
        L.append(f"Sclr{j} u{j} 0 clr 0 SW")
    L.append(".control")
    L.append("set wr_singlescale\nset wr_vecnames")
    L.append(f"tran 2n {n_win * t_win:.6g} 0 10n uic")
    for w in range(n_win):
        tm = w * t_win + t_win / 2
        te = (w + 1) * t_win - 20e-9
        for j in range(n_out):
            L.append(f"meas tran i{w}_{j} find i(Vs{j}) at={tm:.6g}")
            L.append(f"meas tran u{w}_{j} find v(u{j}) at={te:.6g}")
    L.append(".endc\n.end")
    return "\n".join(L) + "\n", cm


def run_bench(W, xs, mode, r_line=R_LINE, levels=LEVELS):
    """返回每窗每列的 (位线电流, 积分电压) 与理想值(量化权重)。"""
    npos, nneg, Wq, w_max = quantize_weights(W, levels)
    net, cm = bench_netlist(npos, nneg, xs, mode, r_line)
    log, _ = run_ngspice(HEADER + net, links=LINKS, timeout=1800)
    m = parse_meas(log)
    n_out, n_in = npos.shape
    s = unit_scale(w_max, levels)
    I = np.zeros((len(xs), n_out)); U = np.zeros((len(xs), n_out))
    for w in range(len(xs)):
        for j in range(n_out):
            if f"i{w}_{j}" not in m:
                raise RuntimeError(f"缺 i{w}_{j}:\n" + log[-2000:])
            I[w, j] = m[f"i{w}_{j}"]
            U[w, j] = m[f"u{w}_{j}"]
    ideal_units = np.asarray(xs) @ Wq.T                      # 软件单位
    I_ideal = ideal_units * s                                # A
    U_ideal = I_ideal * T_INT / cm                           # V
    return dict(I=I, U=U, I_ideal=I_ideal, U_ideal=U_ideal, units=ideal_units, s=s, cm=cm, Wq=Wq, w_max=w_max)


def main():
    rng = np.random.default_rng(0)
    d = 16
    W = rng.normal(0, 0.5, (d, d))
    npos, nneg, Wq, w_max = quantize_weights(W)
    out = dict(pdk="GF180MCU typical, nmos_3p3 W=2u L=0.28u", g_on=G_ON, g_off=G_OFF, levels=LEVELS, v_read=V_READ,
               r_line=R_LINE, t_int=T_INT, d=d)
    # 选通管导通电阻与非线性
    out["r_on"] = {str(v): r_on_sel(v) for v in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, -0.01, -0.05)}
    # 量化本身的权重误差
    out["quantization"] = dict(w_max=w_max, max_abs_err=float(np.abs(W - Wq).max()),
                               rms_err_over_rms_w=float(np.sqrt(((W - Wq) ** 2).mean()) / np.sqrt((W ** 2).mean())))
    # 输入:16 个单位向量(提取有效权重)+ 16 个随机三值向量 + 全 1(最坏 IR drop)
    xs = [list((np.arange(d) == i).astype(float)) for i in range(d)]
    xs += [list(rng.integers(-1, 2, d).astype(float)) for _ in range(16)]
    xs += [[1.0] * d]
    res = {}
    for mode in ("R", "1T1R"):
        r = run_bench(W, xs, mode)
        W_eff = (r["U"][:d] / 1.0).T                              # 单位向量窗:U = W_eff·1 V
        err_w = np.abs(W_eff - Wq)
        rel_u = np.abs(r["U"][d:] - r["U_ideal"][d:]).max() / np.abs(r["U_ideal"][d:]).max()
        res[mode] = dict(w_eff_max_abs_err=float(err_w.max()), w_eff_max_rel_err=float(err_w.max() / w_max),
                         u_random_max_abs_err=float(np.abs(r["U"][d:d + 16] - r["U_ideal"][d:d + 16]).max()),
                         u_random_max_rel_err=float(rel_u),
                         u_allones_max_abs_err=float(np.abs(r["U"][-1] - r["U_ideal"][-1]).max()),
                         i_random_max_rel_err=float(np.abs(r["I"][d:d + 16] - r["I_ideal"][d:d + 16]).max() / np.abs(r["I_ideal"][d:d + 16]).max()),
                         u_ideal_max=float(np.abs(r["U_ideal"]).max()), s_A_per_unit=r["s"], cm=r["cm"])
    out["modes"] = res
    # 线电阻扫描(R 模式,线性,快)
    sweep = {}
    for rl in (0.0, 1.0, 5.0, 20.0, 50.0):
        r = run_bench(W, xs[d:], "R", r_line=rl)
        sweep[str(rl)] = dict(u_max_abs_err=float(np.abs(r["U"] - r["U_ideal"]).max()),
                              u_max_rel_err=float(np.abs(r["U"] - r["U_ideal"]).max() / np.abs(r["U_ideal"]).max()))
    out["r_line_sweep_R_mode"] = sweep
    (ROOT / "results" / "spice-crossbar.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
