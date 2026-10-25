"""把 experiments/spice/ 里各个 SPICE 网表画成原理图(schemdraw),输出到 schematics/。
每张图对应一个网表生成函数;元件值与节点名与网表一致。"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import schemdraw
import schemdraw.elements as elm
from schemdraw import flow

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "schematics"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from experiments.spice.gain_cell import schedules, always_on_prediction  # noqa: E402

_names = {f.name for f in fm.fontManager.ttflist}
CJK_FONT = next((n for n in ("Arial Unicode MS", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "Noto Sans CJK SC",
                                "Noto Sans CJK JP", "Microsoft YaHei")   # Linux 的 NotoSansCJK .ttc 在 matplotlib 里登记为 JP,字形覆盖全部 CJK
                 if n in _names), "DejaVu Sans")
VDOTS = "︙" if CJK_FONT.startswith("Noto") else "⋮"      # schemdraw 只用一种字体;Noto CJK 没有 U+22EE,用 CJK 竖排省略号
plt.rcParams["font.family"] = [CJK_FONT, "DejaVu Sans"]
schemdraw.config(font=CJK_FONT)


def save(d, name):
    d.save(str(OUT / f"{name}.svg"))
    d.save(str(OUT / f"{name}.png"), dpi=200)


def lbl(d, xy, text, halign="left", fontsize=None):
    """在显式坐标放文字(避开 schemdraw 对竖直元件 left/right 标签位置的约定)。"""
    kw = dict(halign=halign, valign="center")
    if fontsize:
        kw["fontsize"] = fontsize
    d += elm.Label().at(xy).label(text, **kw)


# ----------------------------------------------------------------------------- 1. 单 gain cell 测试电路(gain_cell._netlist)
def gain_cell_single_step():
    d = schemdraw.Drawing(unit=3.5)
    d.config(fontsize=11)
    # 写入电流源 B_w(φ4),从地注入 s
    d += (Bw := elm.SourceControlledI().at((-5, 0)).up())
    d += elm.Ground().at((-5, 0))
    lbl(d, (-5.6, 1.75), "B_w", halign="right")
    lbl(d, (-5, -1.4), "I = u(φ4)·β·(v − V(hold))·C_s/t_w\nβ = 0.5, v = 2 V, t_w = 100 ns", halign="center", fontsize=9)
    d += elm.Line().at(Bw.end).right().tox(0)
    # 存储电容 C_s
    d += (Cs := elm.Capacitor().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (0.5, 1.75), "C_s 1 nF\nic = S₀ = 1 V")
    d += elm.Dot().at(Cs.end).label("s", loc="top", ofst=0.15)
    # 泄漏支路 R_leak + S_leak(φ1)
    d += (Rl := elm.Resistor().at(Cs.end).right().label("R_leak 1 kΩ", loc="top").label("R·C = τ = 1 µs", loc="bottom"))
    d += (Sl := elm.Switch().at(Rl.end).down())
    d += elm.Ground().at(Sl.end)
    lbl(d, (Rl.end[0] + 0.6, 1.75), "S_leak\nφ1 泄漏窗")
    # 采样保持:E1(VCVS ×1,向上画使正极在上)→ S_sh(φ3)→ C_hold
    y_rail = 3.5 + 2.0
    d += elm.Line().at(Cs.end).up().length(2.0)
    d += elm.Line().right().tox(8)
    d += elm.Dot().at((8, y_rail)).label("buf", loc="top", ofst=0.15)
    d += elm.SourceControlledV().at((8, y_rail - 3.5)).up()
    d += elm.Ground().at((8, y_rail - 3.5))
    lbl(d, (7.4, y_rail - 1.75), "E1 = 1·V(s)\n(电压跟随)", halign="right")
    d += (Ssh := elm.Switch().at((8, y_rail)).right().label("S_sh(φ3 采样)", loc="top", ofst=0.3))
    d += elm.Dot().at(Ssh.end).label("hold = r", loc="top", ofst=0.15)
    d += (Ch := elm.Capacitor().at(Ssh.end).down())
    d += elm.Ground().at(Ch.end)
    lbl(d, (Ssh.end[0] + 0.5, y_rail - 1.75), "C_hold 1 pF")
    lbl(d, (3.5, -2.4),
        "一个周期:φ1 闭合 S_leak 一段 t_gap,S₀ → e^(−t_gap/RC)·S₀ = α·S₀;φ3 闭合 S_sh 采样 r = V(s);\n"
        "φ4 注入 ΔV = β·(v − r);周期末测 V(s) = S₁。泄漏窗相对检索/写入的位置决定 A(1.25)/B(1.00)/C(0.75)。",
        halign="center", fontsize=10)
    save(d, "01_gain_cell_single_step")


# ----------------------------------------------------------------------------- 2. 四种时序的相位图(gain_cell.schedules)
def phase_timing():
    res = json.load(open(ROOT / "results" / "spice-gain-cell.json"))["single_step"]["alpha=0.5,beta=0.5"]["spice"]
    sch = schedules(0.5)
    names = {"A": "A  泄漏窗 → 检索 → 写入(规范)", "B": "B  检索在泄漏窗之前", "C": "C  泄漏窗在写入之后",
             "always_on": "泄漏管整周期常开,周期中点检索并写入"}
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5))
    for ax, key in zip(axes, ["A", "B", "C", "always_on"]):
        t_leak, leak_pw, t_s, s_dur, t_w, w_dur, t_end = sch[key]
        for i, (lab, t0, dur) in enumerate([("φ1 leak", t_leak, leak_pw), ("φ3 sample", t_s, s_dur), ("φ4 write", t_w, w_dur)]):
            y = 2 - i
            ax.plot([0, t0 * 1e6, t0 * 1e6, (t0 + dur) * 1e6, (t0 + dur) * 1e6, t_end * 1e6],
                    [y, y, y + 0.7, y + 0.7, y, y], lw=2)
            ax.text(-0.02 * t_end * 1e6, y + 0.3, lab, ha="right", va="center", fontsize=10)
        ax.set_xlim(-0.3 * t_end * 1e6, t_end * 1e6 * 1.02); ax.set_ylim(-0.3, 3.1); ax.set_yticks([])
        ax.set_title(f"{names[key]}   →   S₁ = {res[key]:.3f}", fontsize=10, loc="left")
        ax.set_xlabel("t (µs)" if key == "always_on" else "")
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("单 cell 时序判别:α = 0.5, β = 0.5, S₀ = 1, v = 2(闭式 A 1.25 / B 1.00 / C 0.75;常开解析 %.3f)"
                 % always_on_prediction(0.5, 0.5), fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02_phase_timing_A_B_C.png", dpi=200); fig.savefig(OUT / "02_phase_timing_A_B_C.svg")
    plt.close(fig)


# ----------------------------------------------------------------------------- 3. GF180 四种泄漏路径(gf180.alpha_leak)
def _panel(title, kind, caption):
    d = schemdraw.Drawing(unit=3.0); d.config(fontsize=10)
    d += (Cs := elm.Capacitor().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (-0.5, 1.5), "C_s", halign="right")
    d += elm.Dot().at(Cs.end).label("s", loc="top", ofst=0.15)
    if kind == "resistor":
        d += (R := elm.Resistor().at(Cs.end).right().label("R 1 MΩ", loc="top").label("RC = 50 ns", loc="bottom"))
        d += elm.Dot().at(R.end).label("x", loc="top", ofst=0.15)
        d += elm.Line().right().length(1.0)
    else:
        d += elm.Line().at(Cs.end).right().length(3.0)
    d += (M := elm.NFet().anchor("drain"))
    lbl(d, (M.drain[0] + 1.2, M.gate[1]), title)
    d += elm.Line().at(M.gate).left().length(1.2)
    d += elm.Dot(open=True).label({"ground": "V_leak", "resistor": "φ_leak\n3.3 V 持续 t_gap"}.get(kind, "V_mid + V_gs"),
                                 loc="left", ofst=0.15)
    if kind == "ground":
        d += elm.Ground().at(M.source)
        y_cap = -1.2
    else:
        sx, sy = M.source
        d += elm.SourceV().at((sx, sy - 3.0)).up()
        d += elm.Ground().at((sx, sy - 3.0))
        lbl(d, (sx + 0.7, sy - 1.5), "V_mid 1.0 V")
        y_cap = sy - 4.2
    lbl(d, (2.5, y_cap), caption, halign="center", fontsize=9)
    return d


def leak_paths():
    panels = [
        ("03a_leak_ground", "nmos_3p3\nW/L = 0.22/0.28 µm", "ground",
         "接地泄漏:C_s 50 fF, t_gap 1 µs。饱和/亚阈区恒流放电,\nα 随存储值 1.0/0.5/0.25 V 为 0.42/0.22/0.016"),
        ("03b_leak_subthreshold_mid", "nmos_3p3\nW/L = 0.22/0.28 µm\nV_gs ≈ 0.5 V", "mid",
         "亚阈区到 V_mid:C_s 50 fF, t_gap 1 µs, 摆幅 ±0.1 V。\nI ∝ 1 − e^(−V_ds/V_t),α 离散 0.48,非欧姆"),
        ("03c_leak_triode_mid", "nmos_3p3\nW/L = 0.22/10 µm\nV_gs = 3.3 V", "mid",
         "三极管区到 V_mid:C_s 1 pF, t_gap 100 ns, 摆幅 ±0.2 V。\nI ∝ V_ov·V_ds − V_ds²/2,正负不对称,离散 0.13"),
        ("03d_leak_switch_resistor", "nmos_3p3 W = 2 µm\n开关", "resistor",
         "开关 + 电阻:C_s 50 fF。α = e^(−t_gap/RC),与 e^(−t/RC) 差 ≤ 0.3%,\n四种存储值离散 ≤ 1.4e-4(欧姆型,采用)"),
    ]
    for name, title, kind, cap in panels:
        save(_panel(title, kind, cap), name)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (name, *_rest) in zip(axes.ravel(), panels):
        ax.imshow(plt.imread(OUT / f"{name}.png")); ax.axis("off")
    fig.suptitle("GF180MCU nmos_3p3 上的四种 gain cell 泄漏路径(gf180.alpha_leak;衬底接地)", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "03_leak_paths_gf180.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- 4. 理想阈值开关神经元占位(dm_neuron._netlist)
def ideal_switch_neuron():
    d = schemdraw.Drawing(unit=3.5); d.config(fontsize=11)
    d += (Iin := elm.SourceI().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (-0.6, 1.75), "I_in", halign="right")
    lbl(d, (0, -1.4), "PWL:每周期 1 µs 恒定电流\n取 N(0.3, 0.6²) mA,截断 ≤ 0.75 mA", halign="center", fontsize=9)
    d += elm.Dot().at(Iin.end).label("mem", loc="top", ofst=0.15)
    d += elm.Line().right().length(3.0)
    d += (Cm := elm.Capacitor().down())
    d += elm.Ground().at(Cm.end)
    lbl(d, (Cm.start[0] + 0.5, 1.75), "C_m 1 nF")
    d += elm.Line().at(Cm.start).right().length(3.0)
    d += (Rm := elm.Resistor().down())
    d += elm.Ground().at(Rm.end)
    lbl(d, (Rm.start[0] + 0.5, 1.75), "R_m 10 kΩ\nτ_m = 10 µs")
    d += elm.Line().at(Rm.start).right().length(3.0)
    d += (S := elm.Switch().right().label("S_dm:带迟滞的阈值开关,r_on 1 Ω\n导通 V(mem) ≥ 1.0 V,关断 V(mem) ≤ 0.2 V", loc="bottom", ofst=0.4))
    d += elm.Dot().at(S.end).label("d(输出)", loc="top", ofst=0.15)
    d += (Ron := elm.Resistor().down())
    d += elm.Ground().at(Ron.end)
    lbl(d, (S.end[0] + 0.5, 1.75), "R_on 10 Ω")
    lbl(d, (6.5, -2.4),
        "理想稳定 DM 的占位模型:RC 泄漏积分 + 迟滞阈值开关放电到保持电压。\n"
        "用于 S1.2 六参数拟合流程的验证;真实器件见 05_adm_1m1t1r。", halign="center", fontsize=10)
    save(d, "04_ideal_switch_neuron_placeholder")


# ----------------------------------------------------------------------------- 5. ADM 1M1T1R(adm_spice.netlist)
def adm_1m1t1r():
    d = schemdraw.Drawing(unit=3.5); d.config(fontsize=11)
    d += (Vin := elm.SourceSquare().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (-0.6, 1.75), "V_in", halign="right")
    lbl(d, (0, -1.4), "脉冲 0.8 V / 2 ms / 50%\n或恒压 0.6–1.0 V", halign="center", fontsize=9)
    d += elm.Dot().at(Vin.end).label("in", loc="top", ofst=0.15)
    d += (ADM := elm.Memristor().at(Vin.end).right().label("ADM 扩散忆阻器 R_d(x)\n(状态方程见 05b)", loc="top", ofst=0.7))
    d += elm.Dot().at(ADM.end).label("g = V_gs", loc="top", ofst=0.15)
    d += (Cg := elm.Capacitor().at(ADM.end).down())
    d += elm.Ground().at(Cg.end)
    lbl(d, (Cg.start[0] - 0.5, 1.75), "C_g 10 pF", halign="right")
    d += elm.Line().at(ADM.end).right().length(3.0)
    d += (Rc := elm.Resistor().down())
    d += elm.Ground().at(Rc.end)
    lbl(d, (Rc.start[0] + 0.5, 1.75), "R_c 250 MΩ\n(不应期旋钮)")
    d += elm.Line().at(Rc.start).right().length(4.0)
    d += (M := elm.NFet().anchor("gate"))
    lbl(d, (M.drain[0] + 1.2, M.gate[1] - 0.2),
        "M1\n论文等效:level-1, K = 3.75e-4 A/V², V_T = 0.22 V\n或 GF180MCU nmos_3p3(W/L = 1/0.28 µm)")
    d += elm.Ground().at(M.source)
    d += elm.Dot().at(M.drain).label("d(输出 V_d, I_ds)", loc="left", ofst=0.2)
    d += (Rl := elm.Resistor().at(M.drain).up())
    lbl(d, (M.drain[0] + 0.5, M.drain[1] + 1.75), "R_l 47 kΩ")
    d += elm.Vdd().at(Rl.end).label("V_dd 1 V")
    save(d, "05_adm_1m1t1r")

    d2 = schemdraw.Drawing(unit=2.5); d2.config(fontsize=9.5)
    d2 += (b1 := flow.Box(w=7.6, h=1.5).label("E = u(V_in − V_g)·(V_in − V_g) / (α + |1 − x|)"))
    d2 += flow.Arrow().right().length(1.0)
    d2 += (b2 := flow.Box(w=6.6, h=1.5).label("∫  dx₁/dt = −(γ₁+β)·x₁ + μ·E\n(B 源电流 → 1 mF 积分电容)"))
    d2 += flow.Arrow().right().length(1.0)
    d2 += (b3 := flow.Box(w=5.6, h=1.5).label("∫  dx₂/dt = −γ₂·x₂ + β·x₁\n(残留 = 内在可塑性记忆)"))
    d2 += flow.Arrow().at(b3.S).down().length(1.0)
    d2 += (b4 := flow.Box(w=10.4, h=1.7).anchor("N").label("x = clamp(x₁ + x₂, 0, 1)\nG_d = 1 / max( R_on·x + R_off·(e^((1−x)/λ) − 1)/(e^(1/λ) − 1), R_on )"))
    d2 += flow.Arrow().at(b4.W).left().length(1.0)
    d2 += (b5 := flow.Box(w=6.6, h=1.5).anchor("E").label("B_d:I(in → g) = (V_in − V_g)·G_d\n灌入 C_g ∥ R_c,接 M1 栅"))
    d2 += flow.Arrow().at(b5.W).left().length(1.0)
    d2 += (b6 := flow.Box(w=5.6, h=1.5).anchor("E").label("发放事件:x 自下而上越过 1\n(论文输出 O = H(V_gs − 0.2 V))"))
    d2 += elm.Label().at((b1.W[0], b1.N[1] + 0.9)).label(
        "Table 1:μ 150, α 0.01, β 20, γ₁ 600, γ₂ 5, λ 0.2, R_on 1 MΩ, R_off 1 TΩ, C_g 10 pF, R_c 250 MΩ(确定性,σ₁ = σ₂ = 0)",
        loc="right", fontsize=10)
    save(d2, "05b_adm_state_equations")





# ----------------------------------------------------------------------------- 7. GDN 子块(gdn_layer.netlist)
def gdn_layer_figures():
    import experiments.spice.gdn_layer as gl
    # 7a 数据通路框图
    d = schemdraw.Drawing(unit=2.5); d.config(fontsize=10)
    d += (bx := flow.Box(w=3.6, h=1.3).label("块输入 x_τ\n16 路三值 PWL"))
    d += flow.Arrow().right().length(0.8)
    d += (bp := flow.Box(w=5.6, h=1.6).label("投影 + 神经元 ×3\nI_j = Σ_i W_ji·x_i 灌膜电容\nq 二值 / k、v 三值,锁存"))
    d += flow.Arrow().right().length(0.8)
    d += (bg := flow.Box(w=6.2, h=2.2).label("gain cell 阵列 16×16\nφ1 泄漏窗 t_gap = −ln α·RC\nφ3 r_j = Σ_i k_i S_ij → 采样保持\nφ4 ΔS_ij = k_i·β(v_j − r_j)\nφ5 o_j = Σ_i q_i S_ij"))
    d += flow.Arrow().right().length(0.8)
    d += (bo := flow.Box(w=4.2, h=1.6).label("SN_pre(三值)\no_j 灌膜电容\n阈值 0.3"))
    d += flow.Arrow().at(bo.S).down().length(0.9)
    d += (bw := flow.Box(w=4.2, h=1.6).anchor("N").label("W_O + SN_out(三值)\ny_j,阈值 0.5"))
    d += flow.Arrow().at(bw.W).left().length(0.8)
    d += (by := flow.Box(w=4.0, h=1.3).anchor("E").label("输出脉冲 y_τ\n(接累加器)"))
    d += elm.Label().at((bx.W[0], bx.N[1] + 1.1)).label(
        "单头 d = d_k = d_v = 16,L = 16 个 token,周期 T = 12 µs;门值 α_τ、β_τ 与 x_τ 由软件记录外部给定", loc="right", fontsize=10)
    save(d, "07a_gdn_layer_datapath")

    # 7b 一个神经元的电路
    d = schemdraw.Drawing(unit=3.0); d.config(fontsize=10)
    d += (Bi := elm.SourceControlledI().at((-4, 0)).up())
    d += elm.Ground().at((-4, 0))
    lbl(d, (-4.6, 1.5), "B_int", halign="right")
    lbl(d, (-4, -1.2), "I = u(φ_int)·(Σ_i W_ji x_i)·C_n/w_int\n积分窗 w_int = 1 µs → ΔU = Σ W x", halign="center", fontsize=9)
    d += elm.Line().at(Bi.end).right().tox(0)
    d += (Cn := elm.Capacitor().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (0.5, 1.5), "C_n 1 nF\n膜电位 U")
    d += elm.Dot().at(Cn.end).label("u", loc="top", ofst=0.15)
    d += (Rl := elm.Resistor().at(Cn.end).right().label("R_n", loc="top").label("RC = 1 µs", loc="bottom"))
    d += (Sl := elm.Switch().at(Rl.end).down())
    d += elm.Ground().at(Sl.end)
    lbl(d, (Rl.end[0] + 0.5, 1.5), "S_leak\n泄漏窗 0.105 µs\n→ ×0.9 每 token")
    # 复位电流源
    d += (Br := elm.SourceControlledI().at((7.5, 3.0)).down())
    d += elm.Ground().at(Br.end)
    d += elm.Line().at(Rl.end).right().tox(7.5)
    lbl(d, (8.1, 1.5), "B_reset\nI = u(φ_reset)·s·θ·C_n/w_r\n减阈值软复位")
    # 比较器与锁存
    d += elm.Line().at(Cn.end).up().length(1.8)
    d += elm.Line().right().tox(11)
    d += (Bc := elm.SourceControlledV().at((11, 3.0 + 1.8 - 3.0)).up())
    d += elm.Ground().at((11, 1.8))
    lbl(d, (10.4, 3.3), "B_cmp\ns = u(U − θ)\n[三值:−u(−U − θ)]", halign="right")
    d += (Sf := elm.Switch().at((11, 4.8)).right().label("S_fire(φ_fire)", loc="top", ofst=0.3))
    d += elm.Dot().at(Sf.end)
    lbl(d, (Sf.end[0] + 0.3, 5.45), "s:脉冲,保持到下一 token")
    d += (Ch := elm.Capacitor().at(Sf.end).down())
    d += elm.Ground().at(Ch.end)
    lbl(d, (Sf.end[0] + 0.5, 3.3), "C_h 1 pF")
    d += (Sc := elm.Switch().at(Sf.end).right().label("S_clr(清除)", loc="top", ofst=0.3))
    d += elm.Ground().at(Sc.end)
    lbl(d, (5.5, -2.3), "一个 token 内的顺序:泄漏窗 → 积分窗 → 比较并锁存 → 减阈值复位。对应软件 U = 0.9·U + I,s = fire(U),U ← U − θ·s",
        halign="center", fontsize=9.5)
    save(d, "07b_neuron_circuit")

    # 7c 一个 gain cell
    d = schemdraw.Drawing(unit=3.0); d.config(fontsize=10)
    d += (Cs := elm.Capacitor().at((0, 0)).up())
    d += elm.Ground().at((0, 0))
    lbl(d, (0.5, 1.5), "C_s 1 nF\n状态 S_ij")
    d += elm.Dot().at(Cs.end).label("s_ij", loc="top", ofst=0.15)
    d += (R := elm.Resistor().at(Cs.end).right().label("R_s", loc="top").label("RC = 1.44 µs", loc="bottom"))
    d += (S1 := elm.Switch().at(R.end).down())
    d += elm.Ground().at(S1.end)
    lbl(d, (R.end[0] + 0.5, 1.5), "φ1 泄漏窗\nt_gap = −ln α_τ · RC")
    d += (Bw := elm.SourceControlledI().at((-4, 0)).up())
    d += elm.Ground().at((-4, 0))
    lbl(d, (-4.6, 1.5), "B_write", halign="right")
    d += elm.Line().at(Bw.end).right().tox(0)
    lbl(d, (-4, -1.2), "I = u(φ4)·k_i·Δv_j·C_s/w_4\nΔv_j = β(v_j − r_j)", halign="center", fontsize=9)
    d += elm.Line().at(Cs.end).up().length(1.5)
    d += elm.Line().right().length(2.0)
    d += (Bo := elm.SourceControlledI().right().label("读出(列求和)\nI ∝ k_i·S_ij(φ3) 或 q_i·S_ij(φ5)", loc="top", ofst=0.3))
    d += elm.Dot().at(Bo.end).label("列线 j", loc="right", ofst=0.2)
    lbl(d, (1.5, -2.3), "16×16 个这样的 cell;行 i 的 k_i / q_i 选通读出,列 j 共享 Δv_j", halign="center", fontsize=9.5)
    save(d, "07c_gain_cell")

    # 7d 一个 token 的相位时序
    fig, ax = plt.subplots(figsize=(11, 5.2))
    rows = [("clr 清除锁存", gl.PH["clr"]), ("q/k/v 泄漏窗", gl.PH["nq"]), ("φ2 投影积分", gl.PH["p2"]), ("φ2s 发放锁存", gl.PH["p2s"]),
            ("φ2r 复位", gl.PH["p2r"]), ("φ1 阵列泄漏窗(α=0.5)", (gl.PH["p1"][0], gl.PH["p1"][0] + math.log(2) * gl.CELL["RC"])),
            ("φ3 检索", gl.PH["p3"]), ("φ3s 采样 r", gl.PH["p3s"]), ("φ4 写入", gl.PH["p4"]), ("SN_pre 泄漏窗", gl.PH["npre"]),
            ("φ5 读出积分", gl.PH["p5"]), ("φ5s/φ5r 发放、复位", (gl.PH["p5s"][0], gl.PH["p5r"][1])), ("SN_out 泄漏窗", gl.PH["nout"]),
            ("φ6 W_O 积分", gl.PH["p6"]), ("φ6s/φ6r 发放、复位", (gl.PH["p6s"][0], gl.PH["p6r"][1])), ("记录", (gl.T_REC, gl.T_REC + 0.05))]
    for i, (lab, (t0, t1)) in enumerate(rows):
        y = len(rows) - i
        ax.plot([0, t0, t0, t1, t1, gl.T], [y, y, y + 0.7, y + 0.7, y, y], lw=1.8)
        ax.text(-0.2, y + 0.3, lab, ha="right", va="center", fontsize=9.5)
    ax.set_xlim(-4.5, gl.T); ax.set_ylim(0.5, len(rows) + 1.2); ax.set_yticks([]); ax.set_xlabel("t (µs),一个 token 周期 T = 12 µs")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("GDN 子块一个 token 的相位序列(gdn_layer.PH)", fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "07d_gdn_layer_phases.png", dpi=200); fig.savefig(OUT / "07d_gdn_layer_phases.svg"); plt.close(fig)




# ----------------------------------------------------------------------------- 7e. 整层 GDN 子块的电路级原理图
from schemdraw.segments import Segment, SegmentCircle


class Cmp(elm.Element):
    """比较器(阈值单元):向右的小三角,in 在左、out 在右。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.segments.append(Segment([(0, 0.45), (0, -0.45), (0.9, 0), (0, 0.45)]))
        self.anchors["in"] = (0, 0)
        self.anchors["out"] = (0.9, 0)
        self.anchors["center"] = (0.45, 0)


def _neuron(d, x, y_in, name, ternary, theta_txt):
    """列线自上而下到达 (x, y_in):膜电容接地、泄漏开关电阻、比较器向下、锁存(采样开关+电容)、复位电流源。紧凑画法。"""
    d += elm.Dot().at((x, y_in))
    d += elm.Capacitor().at((x, y_in)).down().length(1.1)
    d += elm.Ground()
    d += elm.Resistor().at((x, y_in)).right().length(0.9)
    d += elm.Switch().down().length(0.9)
    d += elm.Ground()
    lbl(d, (x + 1.0, y_in - 0.25), "φ_leak", fontsize=7)
    d += elm.Line().at((x, y_in)).left().length(0.5)
    d += (c := Cmp().theta(-90).anchor("in").at((x - 0.5, y_in)))
    lbl(d, (x - 0.6, y_in - 0.45), theta_txt, halign="right", fontsize=7)
    d += elm.Switch().at(c.out).down().length(0.9)
    lbl(d, (x - 0.4, c.out[1] - 0.45), "φ_fire", fontsize=7)
    d += elm.Dot()
    y_s = c.out[1] - 0.9
    d += elm.Capacitor().at((x - 0.5, y_s)).right().length(0.9)
    d += elm.Ground().at((x + 0.4, y_s)).theta(90)
    lbl(d, (x - 0.5, y_s - 0.55), name, halign="center", fontsize=8)
    return (x - 0.5, y_s)


def _crossbar(d, x0, y_rows, cols, col_labels, row_labels, title):
    """横向字线 y_rows,纵向位线 cols;交叉点一个电阻(差分对以 G⁺−G⁻ 记)。返回各位线底端坐标。"""
    y_top, y_bot = y_rows[0] + 0.8, y_rows[-1] - 1.0
    for xc in cols:
        d += elm.Line().at((xc, y_top)).down().toy(y_bot)
    for yr in y_rows:
        d += elm.Line().at((x0, yr)).right().tox(cols[-1] + 0.6)
        for xc in cols:
            d += elm.Resistor().at((xc - 0.75, yr)).to((xc, yr - 0.75))
            d += elm.Dot(radius=0.06).at((xc - 0.75, yr))
            d += elm.Dot(radius=0.06).at((xc, yr - 0.75))
    for yr, rl in zip(y_rows, row_labels):
        lbl(d, (x0 - 0.2, yr), rl, halign="right", fontsize=8)
    for xc, cl in zip(cols, col_labels):
        lbl(d, (xc, y_top + 0.35), cl, halign="center", fontsize=8)
    lbl(d, (x0, y_top + 1.0), title, fontsize=9)
    return [(xc, y_bot) for xc in cols]


def _cell(d, x, y, write_lbl, read_lbl, x_dv, x_rd, y_row):
    """一个 gain cell:存储节点 (x, y)。电容接地;泄漏 R + 开关(φ1);写入受控电流源(由行 k_i 与列 Δv_j 控制);
    读出受控电流源灌入列读线 x_rd(由行 k_i/q_i 与 S_ij 控制)。"""
    d += elm.Dot().at((x, y))
    d += elm.Capacitor().at((x, y)).down().length(1.0)
    d += elm.Ground()
    d += elm.Resistor().at((x, y)).right().length(0.9)
    d += elm.Switch().down().length(0.9)
    d += elm.Ground()
    lbl(d, (x + 1.0, y - 0.3), "φ1", fontsize=7)
    # 写入源:从地向上注入存储节点,控制量 = 行 k_i × 列 Δv_j
    d += elm.SourceControlledI().at((x - 0.9, y - 1.0)).up().length(1.0)
    d += elm.Ground().at((x - 0.9, y - 1.0))
    d += elm.Line().at((x - 0.9, y)).right().tox(x)
    if write_lbl:
        lbl(d, (x - 1.45, y + 0.35), write_lbl, halign="right", fontsize=7)
    # 读出源:从地向上灌入列读线(位于 x_rd),控制量 = 行 × S_ij
    d += elm.SourceControlledI().at((x_rd, y - 1.0)).up().length(1.0)
    d += elm.Ground().at((x_rd, y - 1.0))
    d += elm.Dot(radius=0.06).at((x_rd, y))
    if read_lbl:
        lbl(d, (x_rd + 0.5, y + 0.4), read_lbl, fontsize=7)
    # 控制关系用细虚线示意:行线 → 写入源与读出源,列 Δv 线 → 写入源
    d += elm.Line().at((x - 0.9, y_row)).down().toy(y + 0.55).linestyle("--").linewidth(0.6)
    d += elm.Line().at((x_rd, y_row)).down().toy(y + 0.55).linestyle("--").linewidth(0.6)


def _xbar(d, x0, rows, row_labels, cols, col_labels, y_bot, title):
    """常规 crossbar 画法:横向字线、纵向位线,交叉点一个电阻(字线上的点 → 斜向电阻 → 位线上的点),字线与位线交叉处不连。"""
    y_top = rows[0] + 1.0
    for xc in cols:
        d += elm.Line().at((xc, y_top)).down().toy(y_bot)
    for yr in rows:
        d += elm.Line().at((x0, yr)).right().tox(cols[-1] + 0.8)
        for xc in cols:
            d += elm.Dot(radius=0.07).at((xc - 0.9, yr))
            d += elm.Resistor().at((xc - 0.9, yr)).to((xc, yr - 0.9))
            d += elm.Dot(radius=0.07).at((xc, yr - 0.9))
    for yr, rl in zip(rows, row_labels):
        lbl(d, (x0 - 0.25, yr), rl, halign="right", fontsize=11)
    for xc, cl in zip(cols, col_labels):
        lbl(d, (xc, y_top + 0.4), cl, halign="center", fontsize=11)
    lbl(d, (x0, y_top + 1.3), title, fontsize=11)


def _lif(d, x, y, out_name, theta):
    """神经元的简化符号:膜电容接地 + 比较器(θ)→ 输出脉冲节点。泄漏窗、锁存、复位见 07b。"""
    d += elm.Dot().at((x, y))
    d += elm.Capacitor().at((x, y)).down().length(1.3)
    d += elm.Ground()
    d += elm.Line().at((x, y)).right().length(0.35)
    d += (c := Cmp().anchor("in").at((x + 0.35, y)))
    lbl(d, (x + 0.8, y - 0.75), theta, halign="center", fontsize=9)
    d += elm.Dot().at(c.out)
    lbl(d, (c.out[0] + 0.2, y), out_name, fontsize=11)


def _cell2(d, x, y, x_rd, first):
    """gain cell:存储电容 S_ij,φ1 泄漏(电阻 + 开关),写入电流源,读出电流源(灌入读线 x_rd)。"""
    d += elm.Dot().at((x, y))
    d += elm.Capacitor().at((x, y)).down().length(1.3)
    d += elm.Ground()
    d += elm.Resistor().at((x, y)).right().length(1.2)
    d += elm.Switch().down().length(1.3)
    d += elm.Ground()
    d += elm.SourceControlledI().at((x - 1.3, y - 1.3)).up().length(1.3)
    d += elm.Ground().at((x - 1.3, y - 1.3))
    d += elm.Line().at((x - 1.3, y)).right().tox(x)
    d += elm.SourceControlledI().at((x_rd, y - 1.3)).up().length(1.3)
    d += elm.Ground().at((x_rd, y - 1.3))
    d += elm.Dot(radius=0.07).at((x_rd, y))
    if first:
        lbl(d, (x - 1.5, y + 0.55), "写入 φ4:I ∝ k_i·Δv_j", halign="right", fontsize=9)
        lbl(d, (x + 1.35, y - 0.7), "φ1", fontsize=9)
        lbl(d, (x_rd + 0.3, y + 0.55), "读出:I ∝ k_i·S_ij(φ3)或 q_i·S_ij(φ5)", fontsize=9)
        lbl(d, (x + 0.2, y - 0.75), "S_ij", fontsize=9)


def gdn_layer_full_schematic():
    d = schemdraw.Drawing(unit=1.0); d.config(fontsize=10)
    # ===== 左上:投影 crossbar(W_q | W_k | W_v 共用字线)+ 神经元 =====
    rows = [27.0, 25.0, 22.0]; row_labels = ["x₁", "x₂", "x₁₆"]
    cols, col_labels = [], []
    for g, (nm, x0) in enumerate([("q", 3.0), ("k", 13.2), ("v", 23.4)]):
        cs = [x0, x0 + 2.6, x0 + 6.4]
        cols += cs; col_labels += [f"{nm}₁", f"{nm}₂", f"{nm}₁₆"]
        lbl(d, (x0 + 4.5, 23.5), "…", halign="center", fontsize=14)
        lbl(d, (x0 + 1.3, 29.6), f"W_{nm}", halign="center", fontsize=11)
    _xbar(d, 1.2, rows, row_labels, cols, col_labels, 19.5, "投影 crossbar:字线 = 块输入 x_i(三值,电压 +V/0/−V),交叉点电阻 = 权重(差分对 G⁺−G⁻),位线电流 = Σ_i G_ji x_i")
    lbl(d, (16.0, 23.5), VDOTS, halign="center", fontsize=14)
    for xc, cl in zip(cols, col_labels):
        _lif(d, xc, 19.5, cl, "θ=1")
    lbl(d, (1.2, 16.6), "每条位线接一个神经元(简化符号:膜电容 + 比较器;泄漏窗、φ_fire 锁存、减阈值复位见 07b)。q 二值;k、v 三值。", fontsize=10)
    lbl(d, (1.2, 15.8), "去向:k_i × 1/√d → 阵列行线 i(φ3 检索、φ4 写入);q_i → 阵列行线 i(φ5 读出);v_j → 第 j 列的 delta 块", fontsize=10)

    # ===== 左下:W_O crossbar + SN_out =====
    rows2 = [12.0, 10.0, 7.0]
    cols2 = [8.0, 10.6, 14.4]
    _xbar(d, 6.2, rows2, ["o₁", "o₂", "o₁₆"], cols2, ["y₁", "y₂", "y₁₆"], 4.5, "W_O crossbar:字线 = SN_pre 的三值脉冲 o_i")
    lbl(d, (12.5, 8.5), "…", halign="center", fontsize=14)
    lbl(d, (9.3, 8.5), VDOTS, halign="center", fontsize=14)
    for xc, cl in zip(cols2, ["y₁", "y₂", "y₁₆"]):
        _lif(d, xc, 4.5, cl, "θ=0.5")
    lbl(d, (6.2, 1.6), "SN_out:三值,输出脉冲 y_τ → 累加器(FFN 与累加器本图未画)", fontsize=10)

    # ===== 右:gain cell 阵列(行 1、2、16;列 1、2、16)=====
    X0 = 39.5
    rows_a = [26.0, 20.5, 13.0]; row_names = ["1", "2", "16"]
    cols_a = [X0 + 4.0, X0 + 11.5, X0 + 21.0]; col_names = ["1", "2", "16"]
    x_top = rows_a[0] + 2.2
    lbl(d, (X0 - 1.0, 30.2), "gain cell 阵列 S(16×16,画行 1、2、16 与列 1、2、16):每个 cell = 存储电容 + φ1 泄漏窗(开关 + 电阻,t_gap = −ln α_τ·RC)+ 写入源 + 读出源", fontsize=11)
    for yr, rn in zip(rows_a, row_names):
        d += elm.Line().at((X0 - 1.0, yr + 1.4)).right().tox(cols_a[-1] + 4.0)
        lbl(d, (X0 - 1.2, yr + 1.4), f"行 {rn}:k{rn}/√d(φ3,φ4)|q{rn}(φ5)", halign="right", fontsize=9.5)
    lbl(d, (X0 + 12.5, 16.8), VDOTS, halign="center", fontsize=14)
    lbl(d, (X0 + 16.6, 26.0), "…", halign="center", fontsize=14)
    for xc, cn in zip(cols_a, col_names):
        x_dv, x_rd = xc - 2.6, xc + 2.9
        d += elm.Line().at((x_dv, x_top)).down().toy(rows_a[-1] - 2.0)
        d += elm.Line().at((x_rd, x_top)).down().toy(rows_a[-1] - 2.0)
        lbl(d, (x_dv, x_top + 0.4), f"Δv{cn}", halign="center", fontsize=10)
        lbl(d, (x_rd, x_top + 0.4), f"读线 {cn}", halign="center", fontsize=10)
        for yr, rn in zip(rows_a, row_names):
            _cell2(d, xc, yr, x_rd, first=(cn == "1" and rn == "1"))
        # 列底:采样保持 r_j、delta 块、SN_pre
        yb = rows_a[-1] - 2.0
        d += elm.Switch().at((x_rd, yb)).down().length(1.3)
        lbl(d, (x_rd + 0.25, yb - 0.65), "φ3s", fontsize=9)
        d += elm.Dot()
        d += elm.Capacitor().at((x_rd, yb - 1.3)).left().length(1.2)
        d += elm.Ground().at((x_rd - 1.2, yb - 1.3)).theta(90)
        lbl(d, (x_rd + 0.25, yb - 1.5), f"r{cn}", fontsize=10)
        d += elm.SourceControlledV().at((x_dv, yb - 3.6)).up().length(1.6)
        d += elm.Ground().at((x_dv, yb - 3.6))
        d += elm.Line().at((x_dv, yb - 2.0)).up().toy(yb)
        lbl(d, (x_dv - 0.5, yb - 2.8), f"Δv{cn} = β_τ(v{cn} − r{cn})", halign="right", fontsize=9.5)
        d += elm.Line().at((x_rd, yb - 1.3)).down().toy(yb - 3.4)
        _lif(d, x_rd, yb - 3.4, f"o{cn}", "θ=0.3")
    lbl(d, (X0 - 1.0, 4.6), "列底:φ3s 把读线上的 r_j = Σ_i k_i S_ij 采到保持电容;delta 块(受控电压源)给出 Δv_j 驱动写入;φ5 读线上的 o_j = Σ_i q_i S_ij 灌进 SN_pre_j,其脉冲 o_j 作 W_O 的字线", fontsize=10)
    lbl(d, (X0 - 1.0, 3.6), "所有开关由相位时钟驱动(07d);求和在网表里由 B 源完成,换成电阻阵列与 GF180 开关是下一步", fontsize=10)
    save(d, "07e_gdn_layer_schematic")


# ----------------------------------------------------------------------------- 6. crossbar 测试台(crossbar.bench_netlist)
def crossbar_bench():
    """差分 1T1R crossbar 的测试台:画 2 行(每行 ± 两条字线)× 2 列,其余省略。元件名与 crossbar.xbar_lines / bench_netlist 一致。"""
    import experiments.spice.crossbar as xb
    d = schemdraw.Drawing(unit=1.0); d.config(fontsize=10)
    rows = [(16.0, "rp1", "+V_read·x1"), (13.0, "rn1", "−V_read·x1"), (8.5, "rp16", "+V_read·x16"), (5.5, "rn16", "−V_read·x16")]
    cols = [9.0, 17.0]
    y_sel, y_end = 18.6, 3.4
    # 行驱动:Vrp_i(PWL)与 Ern_i = −1 × rp_i
    for k, (yr, node, val) in enumerate(rows):
        d += elm.Line().at((3.0, yr)).right().tox(cols[-1] + 2.2)
        src = elm.SourceV() if k % 2 == 0 else elm.SourceControlledV()
        d += src.at((1.6, yr - 1.3)).up().length(1.3)
        d += elm.Ground().at((1.6, yr - 1.3))
        d += elm.Line().at((1.6, yr)).right().tox(3.0)
        lbl(d, (0.9, yr - 0.65), ("Vrp" if k % 2 == 0 else "Ern") + node[2:] + "\n" + val, halign="right", fontsize=8.5)
        lbl(d, (3.1, yr + 0.3), node, fontsize=8.5)
    # 选通线、位线
    d += elm.Line().at((3.0, y_sel)).right().tox(cols[-1] + 2.2).linestyle("--")
    lbl(d, (3.0, y_sel + 0.45), f"sel:选通栅线,积分窗内 {xb.V_SEL} V,接全部选通管的栅(R 模式里换成理想开关,"
        f"导通电阻 = nmos_3p3 的 R_on ≈ 790 Ω)", fontsize=9)
    for j, xc in enumerate(cols):
        cn = "1" if j == 0 else "16"
        d += elm.Line().at((xc, 17.6)).down().toy(y_end)
        lbl(d, (xc, 17.9), f"bl{cn}", halign="center", fontsize=9)
        for k, (yr, node, val) in enumerate(rows):
            xm = xc - 3.2
            d += elm.Dot(radius=0.07).at((xm, yr))
            d += elm.Memristor().at((xm, yr)).down().length(0.9)
            d += (m := elm.NFet().right().anchor("drain").at((xm, yr - 0.9)).scale(0.6))   # 漏在上、源在下、栅在右
            d += elm.Line().at(m.source).down().toy(yr - 2.0)
            d += elm.Line().right().tox(xc)
            d += elm.Dot(radius=0.07).at((xc, yr - 2.0))
            d += elm.Line().at(m.gate).right().length(0.4).linestyle(":").linewidth(0.8)
            if j == 0 and k == 0:
                lbl(d, (xm - 0.5, yr - 0.45), "R_m = 1/G − R_on\nG = G_off + n·ΔG", halign="right", fontsize=8)
                lbl(d, (xm - 0.3, yr - 1.55), "M nmos_3p3 2/0.28 µm,栅 → sel", halign="right", fontsize=8)
        # 列底:0 V 电流计(位线虚地)+ 电流复制进积分电容
        d += (vs := elm.SourceV().at((xc, y_end)).down().length(1.2))
        d += elm.Ground().at(vs.end)
        lbl(d, (xc + 0.5, y_end - 0.6), f"Vs{cn} 0 V(位线钳在 0 V,测电流)", fontsize=8.5)
        xf = xc + 2.4
        y0 = y_end - 3.6
        d += (f := elm.SourceControlledI().at((xf, y0)).up().length(1.4))
        d += elm.Ground().at((xf, y0))
        d += elm.Line().at(f.end).right().length(1.4)
        d += elm.Dot().label(f"u{cn}", loc="top", ofst=0.1)
        d += elm.Capacitor().down().length(1.4)
        d += elm.Ground()
        d += elm.Switch().at((xf + 1.4, y0 + 1.4)).right().length(1.2)
        d += elm.Ground().theta(0)
        lbl(d, (xf - 0.5, y0 + 0.7), "F = 1 × I(Vs)", halign="right", fontsize=8.5)
        lbl(d, (xf + 1.65, y0 + 0.55), "C_m 10 pF", fontsize=8.5)
        lbl(d, (xf + 2.0, y0 + 1.85), "S_clr", halign="center", fontsize=8.5)
    lbl(d, (13.0, 11.0), VDOTS, halign="center", fontsize=14)
    lbl(d, (13.0, 2.4), "…", halign="center", fontsize=14)
    lbl(d, (0.0, -1.8), f"差分 1T1R crossbar(16×16 只画行 1、16 与列 1、16):权重 w = (n+ − n−)·w_max/{xb.LEVELS - 1},n 为 {xb.LEVELS} 级电导,"
        f"G_on {xb.G_ON * 1e6:.0f} µS / G_off {xb.G_OFF * 1e6:.0f} µS,V_read {xb.V_READ} V;", fontsize=9.5)
    lbl(d, (0.0, -2.6), f"行线、位线每个 cell 一段 R_line = {xb.R_LINE} Ω(图中未画)。位线电流 = Σ_i (G+_ji − G−_ji)·V_read·x_i,"
        f"积分窗 {xb.T_INT * 1e6:.0f} µs 后 C_m 上的电压与 W·x 成正比(1 单位 = V_read·ΔG_max/w_max·T_int / C_m)", fontsize=9.5)
    lbl(d, (0.0, -3.4), "完整一层(block_layer,xbar = R / 1T1R)里 7 个权重矩阵都用这个结构,电流传输器把位线电流灌进神经元膜电容,"
        "膜电容取 V_read·ΔG_max/w_max·T_int 使 1 单位 = 1 V", fontsize=9.5)
    save(d, "06_crossbar_1t1r_bench")


# ----------------------------------------------------------------------------- 8. 完整一层(block_layer.netlist)
STAGE_COLORS = dict(io="#6b7280", A="#1f5f99", gc="#4c8fd1", B="#2f7f7a", C="#5aa39a", D="#b5651d", E="#d9902f")


def block_layer_datapath():
    """08a:一层的数据通路。累加器是同一组电容:load 装入 h_in,acc1 注入 y1,acc2 注入 y2;两组比较器在 cmp1 / cmp2 读它。"""
    d = schemdraw.Drawing(unit=1.0); d.config(fontsize=10)
    def chain(x, y, items, gap=1.2):
        """在 y 上从 x 起向右排一串方框,相邻方框之间画箭头;返回最后一个方框。"""
        last = None
        for w, h, text in items:
            if last is not None:
                d.add(elm.Arrow().at(last.E).right().length(gap))
                x = last.E[0] + gap
            last = d.add(flow.Box(w=w, h=h).at((x, y)).anchor("W").label(text, fontsize=9.5))
        return last

    d += flow.Box(w=4.6, h=7.4).at((0, 0)).anchor("W").label(
        "累加器 h_i\n(C_acc 1 nF × 16)\n\nload:DAC 装入 h_in\nacc1:+ y1\nacc2:+ y2")
    yt, yb = 2.1, -2.1
    d += elm.Arrow().at((4.6, yt)).right().length(1.2)
    d += elm.Arrow().at((4.6, yb)).right().length(1.2)
    gdn = chain(5.8, yt, [(5.4, 1.9, "Q_in1:2T 个比较器\n±jθ/T,cmp1 锁存 → x1"),
                          (13.6, 2.6, "GDN 注意力子块(07a / 07e)\nW_q,W_k,W_v → q,k,v 神经元 → gain cell 16×16\n→ SN_pre → W_O → SN_out:y1")])
    f2 = chain(5.8, yb, [(5.4, 1.9, "Q_in2:2T 个比较器\n±jθ/T,cmp2 锁存 → x2"),
                         (5.6, 1.9, "W_1 | W_3 共用字线\ng 二值 / u 三值神经元"),
                         (4.0, 1.9, "锁存 + AND\np = g·u"),
                         (5.0, 1.9, "W_2 → y2 神经元\n(三值)")], gap=1.0)
    # y1 / y2 回到累加器
    xr = max(gdn.E[0], f2.E[0]) + 0.8
    d += elm.Line().at(gdn.E).right().tox(xr)
    d += elm.Line().up().toy(4.8)
    d += elm.Arrow().left().tox(2.3)
    d += elm.Arrow().down().toy(3.7)
    lbl(d, ((2.3 + gdn.E[0]) / 2, 5.2), "acc1:y1 的每个脉冲向 C_acc 注入一份电荷(1 单位 = 1 V)→ h_mid", halign="center", fontsize=9.5)
    d += elm.Line().at(f2.E).right().tox(xr)
    d += elm.Line().down().toy(-4.8)
    d += elm.Arrow().left().tox(2.3)
    d += elm.Arrow().up().toy(-3.7)
    lbl(d, ((2.3 + f2.E[0]) / 2, -5.2), "acc2:y2 注入 → h_out(下一层的入口;本次只有一层,下一 token 由 DAC 重新装入)", halign="center", fontsize=9.5)
    lbl(d, (0, 7.2), "完整一层的电路数据通路(block_layer.netlist):d = 16,d_ff = 32,单头,每 token 最多 T 个脉冲(T = 1、2);"
        "门值 α、β 外部给定", fontsize=10.5)
    lbl(d, (0, 6.4), "顺序:load → cmp1 → 注意力子块 → acc1 → cmp2 → 前馈子块 → acc2(相位见 08c)", fontsize=10)
    save(d, "08a_block_layer_datapath")


def _and_down(d, x, y):
    """朝下的二输入 AND(锁存后的 g_j、u_j 相乘):in1、in2 在上,out 在下。返回 (in1, in2, out)。"""
    import schemdraw.logic as logic
    g = logic.And(inputs=2).theta(-90).at((x, y)).anchor("out").scale(0.7)
    d += g
    return g.in1, g.in2, g.out


def block_layer_schematic():
    """08b:新增部分的电路级原理图(累加器、两组块入口比较器、前馈子块),注意力子块画成方框(细节见 07e)。"""
    d = schemdraw.Drawing(unit=1.0); d.config(fontsize=10)
    # ===== 左:累加器与比较器(通道 1、2、16)=====
    chans = [(27.0, "1"), (23.0, "2"), (17.0, "16")]
    x2_out = {}
    for k, (yc, cn) in enumerate(chans):
        first = k == 0
        d += elm.SourceV().at((0.6, yc - 1.6)).up().length(1.6)
        d += elm.Ground().at((0.6, yc - 1.6))
        d += elm.Switch().at((0.6, yc)).right().length(1.6)
        lbl(d, (1.4, yc + 0.45), "S_ld(load)" if first else "load", halign="center", fontsize=8)
        d += elm.Dot().at((2.2, yc)).label(f"h{cn}", loc="top", ofst=0.1)
        d += elm.Capacitor().at((2.2, yc)).down().length(1.6)
        d += elm.Ground()
        d += elm.SourceControlledI().at((3.6, yc - 1.6)).up().length(1.6)
        d += elm.Ground().at((3.6, yc - 1.6))
        d += elm.Line().at((2.2, yc)).right().tox(3.6)
        if first:
            lbl(d, (0.0, yc - 0.8), "V_hin\n(DAC)", halign="right", fontsize=8)
            lbl(d, (2.2, yc - 2.6), "C_acc", halign="center", fontsize=8)
            lbl(d, (3.6, yc - 2.6), "B_acc", halign="center", fontsize=8)
        # 两组比较器:Q_in1(cmp1 → x1)、Q_in2(cmp2 → x2),都读同一个 h_i
        d += elm.Line().at((3.6, yc)).right().tox(5.0)
        d += elm.Line().at((5.0, yc + 0.9)).down().toy(yc - 0.9)
        for side, (dy, ph, nm) in enumerate(((0.9, "cmp1", "Q_in1"), (-0.9, "cmp2", "Q_in2"))):
            y = yc + dy
            d += elm.Line().at((5.0, y)).right().length(0.3)
            d += (c := Cmp().right().anchor("in").at((5.3, y)))
            d += elm.Switch().at(c.out).right().length(1.2)
            xo = c.out[0] + 1.2
            d += elm.Dot().at((xo, y))
            d += elm.Capacitor().at((xo, y)).down().length(0.6)
            d += elm.Ground().scale(0.6)
            lbl(d, (c.out[0] + 0.6, y + 0.45), ph, halign="center", fontsize=7.5)
            if first:
                lbl(d, (5.75, y + 0.65), nm, halign="center", fontsize=7.5)
            if side == 0:
                d += elm.Line().at((xo, y)).right().length(0.5)
                lbl(d, (xo + 0.6, y), f"x1_{cn} → GDN", fontsize=8.5)
            else:
                x2_out[cn] = (xo, y)
    lbl(d, (2.2, 20.4), VDOTS, halign="center", fontsize=14)
    lbl(d, (0.0, 13.6), "C_acc 1 nF;B_acc:I = φacc1·y1_i·C_acc/0.1 µs + φacc2·y2_i·C_acc/0.1 µs(acc1 注入 y1,acc2 注入 y2,1 单位 = 1 V)", fontsize=9)
    lbl(d, (0.0, 12.9), "比较器:2T 个阈值 ±jθ/T(θ = 0.7),输出 (Σ_j 1[h ≥ jθ/T] − Σ_j 1[h ≤ −jθ/T]) / T,", fontsize=9)
    lbl(d, (0.0, 12.2), "在 cmp1 / cmp2 采样进 1 pF 锁存,保持到下一次采样(展开见 08d)。图中编号从 1 起,网表节点从 0 起(h0…h15)", fontsize=9)
    # ===== 左上:GDN 子块方框 =====
    d += flow.Box(w=11.0, h=2.4).at((0.0, 31.2)).anchor("W").label(
        "GDN 注意力子块(电路见 07e,相位 A、p1、p3、p4、B、C)\n输入:x1_i(cmp1 锁存)   输出:y1_i(SN_out 锁存)→ B_acc(acc1)", fontsize=9.5)
    # ===== 右上:W_1 | W_3 crossbar(共用字线 x2,列交错:g_j 与 u_j 相邻)=====
    X0 = 13.5
    rows = [x2_out["1"][1], x2_out["2"][1], x2_out["16"][1]]
    pairs = [(16.5, "1"), (23.5, "2"), (33.5, "32")]
    cols = []
    for xb_, pn in pairs:
        cols += [(xb_, f"g{pn}", "θ=1"), (xb_ + 2.6, f"u{pn}", "θ=1")]
    y_top, y_bot = rows[0] + 1.6, rows[-1] - 1.8
    for cn, (xo, yo) in x2_out.items():
        d += elm.Line().at((xo, yo)).right().tox(X0)
    for yr, cn in zip(rows, ("1", "2", "16")):
        d += elm.Line().at((X0, yr)).right().tox(cols[-1][0] + 0.8)
        lbl(d, (X0 - 0.15, yr + 0.35), f"x2_{cn}", halign="right", fontsize=8.5)
        for xc, _, _ in cols:
            d += elm.Dot(radius=0.07).at((xc - 0.9, yr))
            d += elm.Resistor().at((xc - 0.9, yr)).to((xc, yr - 0.9))
            d += elm.Dot(radius=0.07).at((xc, yr - 0.9))
    for xc, nm, th in cols:
        d += elm.Line().at((xc, y_top)).down().toy(y_bot)
        lbl(d, (xc, y_top + 0.4), nm, halign="center", fontsize=9)
    lbl(d, (X0, y_top + 1.4), "前馈子块:W_1(列 g_j)与 W_3(列 u_j)两个 crossbar 共用字线 x2_i,列交错画;d_ff = 32 列各画 1、2、32", fontsize=10)
    lbl(d, (29.0, rows[1] - 2.0), "…", halign="center", fontsize=14)
    lbl(d, (X0 + 0.6, (rows[1] + rows[2]) / 2 - 0.4), VDOTS, halign="center", fontsize=14)
    # 神经元 + 锁存 + AND
    y_and = y_bot - 3.6
    p_nodes = []
    for (xb_, pn) in pairs:
        _lif(d, xb_, y_bot, f"g{pn}", "θ=1")
        _lif(d, xb_ + 2.6, y_bot, f"u{pn}", "θ=1")
        og, ou = (xb_ + 1.25, y_bot), (xb_ + 2.6 + 1.25, y_bot)
        in1, in2, out = _and_down(d, xb_ + 4.4, y_and)
        d += elm.Line().at(og).down().toy(y_bot - 2.0)
        d += elm.Line().tox(in1[0]); d += elm.Line().toy(in1[1])
        d += elm.Line().at(ou).down().toy(y_bot - 1.6)
        d += elm.Line().tox(in2[0]); d += elm.Line().toy(in2[1])
        p_nodes.append((out, pn))
    lbl(d, (pairs[0][0] + 4.9, y_and + 0.9), "AND:p_j = g_j·u_j", fontsize=8.5)
    lbl(d, (X0, y_bot - 4.9), "g 二值、u 三值(简化符号;泄漏窗、子时隙锁存、减阈值复位见 08d);锁存输出相乘得 p_j ∈ {−1, 0, +1}(T = 2 时 c_g·c_u / 4)", fontsize=9)
    # ===== 右下:W_2 crossbar(字线 p_j)+ y2 神经元 =====
    rows2 = [y_and - 3.0, y_and - 4.8, y_and - 7.4]
    cols2 = [(41.0, "y2_1"), (43.6, "y2_2"), (47.4, "y2_16")]
    for (out, pn), yr in zip(p_nodes, rows2):
        d += elm.Line().at(out).down().toy(yr)
        d += elm.Line().right().tox(cols2[-1][0] + 0.8)
        lbl(d, (out[0] - 0.15, yr + 0.35), f"p{pn}", halign="right", fontsize=8.5)
        for xc, _ in cols2:
            d += elm.Dot(radius=0.07).at((xc - 0.9, yr))
            d += elm.Resistor().at((xc - 0.9, yr)).to((xc, yr - 0.9))
            d += elm.Dot(radius=0.07).at((xc, yr - 0.9))
    y_bot2 = rows2[-1] - 1.8
    for xc, nm in cols2:
        d += elm.Line().at((xc, rows2[0] + 1.2)).down().toy(y_bot2)
        lbl(d, (xc, rows2[0] + 1.6), nm, halign="center", fontsize=9)
        _lif(d, xc, y_bot2, nm, "θ=1")
    lbl(d, (45.5, rows2[1] - 1.3), "…", halign="center", fontsize=14)
    lbl(d, (39.5, rows2[0] + 2.6), "W_2 crossbar(字线 p_j,32 行;列 = y2 神经元,16 列)", fontsize=10)
    lbl(d, (39.5, y_bot2 - 3.0), "y2_i(三值)→ B_acc 的 acc2 项", fontsize=9.5)
    lbl(d, (0.0, y_bot2 - 4.4), "求和:ideal 模式由 B 源完成;R / 1T1R 模式下 7 个权重矩阵都是 06 的差分 crossbar,位线经电流传输器灌膜电容(1 单位 = 1 V)。"
        "所有开关由 08c 的相位时钟驱动。", fontsize=9.5)
    save(d, "08b_block_layer_schematic")


def block_layer_phases():
    """08c:一个 token 的相位时间线(block_layer.schedule),T = 1 与 T = 2 各一张。"""
    import experiments.spice.block_layer as bl
    from matplotlib.patches import Patch
    groups = [("clr / load", ["clr", "load"], "io"), ("cmp1(Q_in1 锁存)", ["cmp1"], "io"),
              ("A:q/k/v 泄漏 · 积分", ["Al", "Ai"], "A"), ("A:子时隙 采样 s_k · 计数 a_k", "As", "A"),
              ("p1 阵列泄漏(α = 0.5)", ["p1"], "gc"), ("p3 检索 / p3s 采样", ["p3", "p3s"], "gc"), ("p4 写入", ["p4"], "gc"),
              ("B:SN_pre 泄漏 · 积分", ["Bl", "Bi"], "B"), ("B:子时隙", "Bs", "B"),
              ("C:SN_out 泄漏 · 积分", ["Cl", "Ci"], "C"), ("C:子时隙", "Cs", "C"), ("acc1(y1 注入)", ["acc1"], "io"),
              ("cmp2(Q_in2 锁存)", ["cmp2"], "io"), ("D:g/u 泄漏 · 积分", ["Dl", "Di"], "D"), ("D:子时隙", "Ds", "D"),
              ("E:y2 泄漏 · 积分", ["El", "Ei"], "E"), ("E:子时隙", "Es", "E"), ("acc2(y2 注入)", ["acc2"], "io")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10.5), sharex=True)
    for ax, T in zip(axes, (1, 2)):
        PH, period, t_rec = bl.schedule(T)
        for r, (lab, keys, col) in enumerate(groups):
            y = len(groups) - r
            if isinstance(keys, str):                      # 子时隙:s_k 实心、a_k 斜线
                tag = keys[0]
                for k in range(T):
                    s0, s1 = PH[f"{tag}s{k}"]; a0, a1 = PH[f"{tag}a{k}"]
                    ax.broken_barh([(s0, s1 - s0)], (y - 0.35, 0.7), color=STAGE_COLORS[col])
                    ax.broken_barh([(a0, a1 - a0)], (y - 0.35, 0.7), facecolor="white", edgecolor=STAGE_COLORS[col], hatch="////")
            else:
                for i, key in enumerate(keys):
                    t0, t1 = PH[key]
                    if key == "p1":
                        t1 = t0 + math.log(2) * bl.CELL["RC"]
                    light = i == 0 and len(keys) > 1 and key not in ("clr", "p3")
                    ax.broken_barh([(t0, t1 - t0)], (y - 0.35, 0.7), color=STAGE_COLORS[col], alpha=0.45 if light else 1.0)
            ax.text(-0.15, y, lab, ha="right", va="center", fontsize=9)
        ax.axvline(t_rec, color="#9ca3af", lw=0.8, ls="--")
        ax.text(t_rec, len(groups) + 0.9, "记录", ha="center", fontsize=8.5, color="#6b7280")
        ax.set_ylim(0.3, len(groups) + 1.3); ax.set_yticks([]); ax.set_xlim(-0.05, period)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.set_title(f"T = {T}(每 token 最多 {T} 个脉冲):周期 {period:g} µs,每个神经元级 {T} 个子时隙", fontsize=10.5, loc="left")
    axes[-1].set_xlabel("t (µs)")
    axes[0].legend(handles=[Patch(color=STAGE_COLORS["A"], alpha=0.45, label="泄漏窗"), Patch(color=STAGE_COLORS["A"], label="积分窗 / 子时隙采样 s_k"),
                            Patch(facecolor="white", edgecolor=STAGE_COLORS["A"], hatch="////", label="子时隙计数 + 复位 a_k")],
                   loc="upper right", fontsize=8.5, frameon=False)
    fig.suptitle("完整一层一个 token 的相位序列(block_layer.schedule):注意力子块蓝 / 青,前馈子块橙,累加器与比较器灰", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "08c_block_layer_phases.png", dpi=200); fig.savefig(OUT / "08c_block_layer_phases.svg"); plt.close(fig)


def block_layer_multilevel_units():
    """08d:每 token 最多 T 个脉冲的神经元(T 个子时隙)与 Q_in 的 2T 比较器组。元件名与 block_layer.neurons / comparators 一致。"""
    d = schemdraw.Drawing(unit=1.0); d.config(fontsize=10)
    # ---- 左:神经元(下排:积分、膜电容、泄漏、复位;上排:比较、子时隙锁存、计数)----
    d += (bi := elm.SourceControlledI().at((0.0, 0.0)).up().length(2.2))
    d += elm.Ground().at((0.0, 0.0))
    lbl(d, (-0.6, 1.1), "B_int\nI = φ_i·(Σ_i W_ji x_i)·C_n / 1 µs", halign="right", fontsize=8.5)
    d += elm.Line().at(bi.end).right().tox(3.0)
    d += elm.Dot().at((3.0, 2.2)).label("U", loc="bottom", ofst=0.15)
    d += elm.Capacitor().at((3.0, 2.2)).down().length(2.2)
    d += elm.Ground()
    lbl(d, (3.35, 0.9), "C_n 1 nF", fontsize=8.5)
    d += elm.Line().at((3.0, 2.2)).right().tox(5.6)
    d += elm.Resistor().down().length(1.1)
    d += elm.Switch().down().length(1.1)
    d += elm.Ground()
    lbl(d, (5.95, 1.1), "泄漏窗 φ_l\n0.105 µs → ×0.9", fontsize=8.5)
    d += elm.Line().at((5.6, 2.2)).right().tox(9.4)
    d += elm.SourceControlledI().at((9.4, 2.2)).down().length(2.2)
    d += elm.Ground().at((9.4, 0.0))
    lbl(d, (10.1, 1.1), "B_reset(从 U 吸出)\nI = Σ_k φa_k·f·(θ/T)·C_n / 0.1 µs", fontsize=8.5)
    # 比较器 + T 个子时隙采样开关 → f
    d += elm.Line().at((3.0, 2.2)).up().toy(5.0)
    d += elm.Line().at((3.0, 5.0)).right().length(0.5)
    d += (c := Cmp().right().anchor("in").at((3.5, 5.0)))
    lbl(d, (1.0, 7.3), "B_cmp:f_in = 1[U ≥ θ/T] − 1[U ≤ −θ/T](二值只有前一项)", fontsize=8.5)
    xa, xb2 = c.out[0] + 0.4, c.out[0] + 2.0
    d += elm.Line().at(c.out).right().tox(xa)
    d += elm.Line().at((xa, 5.9)).down().toy(4.1)
    d += elm.Line().at((xb2, 5.9)).down().toy(4.1)
    for y, ph in ((5.9, "φs_1"), (4.1, "φs_T")):
        d += elm.Switch().at((xa, y)).right().tox(xb2)
        lbl(d, ((xa + xb2) / 2, y + (0.5 if y > 5 else -0.55)), ph, halign="center", fontsize=8)
    lbl(d, ((xa + xb2) / 2, 5.0), VDOTS, halign="center", fontsize=11)
    xf = xb2 + 0.8
    d += elm.Line().at((xb2, 5.0)).right().tox(xf)
    d += elm.Dot().at((xf, 5.0)).label("f", loc="top", ofst=0.1)
    d += elm.Capacitor().at((xf, 5.0)).down().length(1.3)
    d += elm.Ground()
    lbl(d, (xf + 0.35, 4.35), "C_f 1 pF", fontsize=8.5)
    # 计数到 s
    xs = xf + 4.2
    d += (bs := elm.SourceControlledI().at((xs, 3.6)).up().length(1.4))
    d += elm.Ground().at((xs, 3.6))
    lbl(d, (xs, 2.75), "B_s:I = Σ_k φa_k·f·C_h / (T·0.1 µs)", halign="center", fontsize=8.5)
    d += elm.Line().at(bs.end).right().length(1.5)
    d += elm.Dot().label("s", loc="top", ofst=0.1)
    d += elm.Capacitor().down().length(1.4)
    d += elm.Ground()
    d += elm.Switch().at((xs + 1.5, 5.0)).right().length(1.3)
    d += elm.Ground().theta(0)
    lbl(d, (xs + 1.85, 3.95), "C_h 1 pF", fontsize=8.5)
    lbl(d, (xs + 2.15, 5.55), "S_clr(clr)", halign="center", fontsize=8.5)
    lbl(d, (-4.5, -1.6), "一个 token:泄漏窗 → 积分窗 → T 个子时隙,每个子时隙先 φs_k 把比较结果采进 f,再 φa_k 让 s += f/T、U −= f·θ/T。", fontsize=9.5)
    lbl(d, (-4.5, -2.3), "T 个子时隙后 s = c/T(c ≤ T 个脉冲,三值时带符号),与 snn_spec 的 levels = T 相同:c = min(T, floor(U·T/θ))。T = 1 即原来的单脉冲神经元。", fontsize=9.5)
    # ---- 右:2T 比较器组 ----
    X = 21.0
    d += elm.Dot().at((X, 3.0)).label("h_i", loc="left", ofst=0.15)
    ths = ["+θ/T", "+2θ/T = +θ(T = 2)", "−θ/T", "−2θ/T = −θ"]
    ys = [5.6, 4.1, 1.9, 0.4]
    d += elm.Line().at((X, ys[0])).down().toy(ys[-1])
    for y, th in zip(ys, ths):
        d += elm.Line().at((X, y)).right().length(0.6)
        d += (cc := Cmp().right().anchor("in"))
        lbl(d, (cc.out[0] + 0.2, y + 0.4), th, fontsize=8)
        d += elm.Line().at(cc.out).right().tox(X + 4.6)
    d += flow.Box(w=2.4, h=6.0).at((X + 4.6, 3.0)).anchor("W").label("(Σ 上 − Σ 下)\n÷ T", fontsize=9.5)
    d += elm.Line().at((X + 7.0, 3.0)).right().length(0.4)
    d += elm.Switch().right().length(1.3)
    d += elm.Dot().label("x_i", loc="right", ofst=0.15)
    d += elm.Capacitor().down().length(1.3)
    d += elm.Ground()
    lbl(d, (X + 8.05, 3.6), "cmp1 / cmp2", halign="center", fontsize=8.5)
    lbl(d, (X - 0.5, -1.6), "块入口量化器 Q_in:每通道 2T 个比较器,阈值 ±jθ/T(j = 1…T);", fontsize=9.5)
    lbl(d, (X - 0.5, -2.3), "x_i = (Σ_j 1[h_i ≥ jθ/T] − Σ_j 1[h_i ≤ −jθ/T]) / T ∈ {−1, …, −1/T, 0, 1/T, …, 1}", fontsize=9.5)
    save(d, "08d_multilevel_neuron_and_comparators")


def block_layer_figures():
    block_layer_datapath(); block_layer_schematic(); block_layer_phases(); block_layer_multilevel_units()


if __name__ == "__main__":
    import sys as _sys
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    if only == "layer":
        gdn_layer_full_schematic()
    elif only == "block":
        block_layer_figures()
    elif only == "crossbar":
        crossbar_bench()
    else:
        gain_cell_single_step(); phase_timing(); leak_paths(); ideal_switch_neuron(); adm_1m1t1r(); gdn_layer_figures(); gdn_layer_full_schematic()
        crossbar_bench(); block_layer_figures()
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
