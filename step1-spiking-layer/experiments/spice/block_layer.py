"""完整一层(GDN 注意力子块 + SwiGLU 前馈子块 + 两个残差累加器与块入口比较器)的 ngspice 实现,按相位表逐 token 执行。

与 `gdn_layer.py`(只有注意力子块、块输入 x 由软件给定)相比,这里块输入来自电路自己:
  累加器:每通道一个电容,token 开始时由 DAC 装入 h_in;y1、y2 的每个脉冲向它注入一份电荷(1 个软件单位 = 1 V)。
  块入口比较器 Q_in:读累加器电压,输出三值脉冲 x1(注意力子块入口)与 x2(前馈子块入口),锁存到 token 结束。
  前馈子块:W_1 x2 → g 神经元(二值)、W_3 x2 → u 神经元(三值),锁存 + AND 得 p = g·u,W_2 p → y2 神经元(三值)。
每 token 最多 T 个脉冲(levels = T,语义与 snn_spec 的 levels 相同):
  神经元在积分后进入 T 个子时隙,每个子时隙先把比较结果采样进锁存 f(U ≥ θ/T 为 1,三值时 U ≤ −θ/T 为 −1),
  再把 f/T 累加到脉冲锁存 s、从膜电位减去 f·θ/T;T 个子时隙结束后 s = c/T。T = 1 时与 gdn_layer 的神经元等价。
  比较器 Q_in 是 2T 个比较器,阈值 ±j·θ/T;AND 的输出 p = g·u 是两个计数之积(c_g·c_u/T²)。
真的(电容、开关、相位、连续时间):全部神经元、累加器、比较器的锁存、16×16 gain cell 阵列、检索值采样保持、delta 块。
理想的(B 源):crossbar 列求和(xbar="ideal";xbar="R" / "1T1R" 时 7 个权重矩阵都换成电阻或 1T1R crossbar)、
阵列的列求和 Σ_i k_i S_ij 与 Σ_i q_i S_ij、比较器本身(理想阈值比较)、AND 的乘法、DAC。门值 α、β 是外部给定的
每 token 随机值(α ∈ [0.5, 0.99],β ∈ [0.1, 0.9]),软件参考用同一组(首个闭环的定义;门生成电路不在本次范围内)。

数值上要注意的三处(都在 4 token 的冒烟测试里暴露过):
  按窗口计量电荷的电流(积分、泄漏、复位、计数、累加器注入、gain cell 泄漏与写入)直接乘相位电压,相位用 1 ns 线性边沿
  且 ∫φ dt = 窗宽;用 u(φ − 0.5) 的硬阶跃时求解步跨过阶跃会让窗宽差半个步长(0.5 ns / 100 ns = 0.5% 的计数误差)。
  积分法用 Gear:采样开关 1 Ω × 1 pF 的时间常数远小于 1 ns 步长,梯形法在这个刚性模态上不衰减,锁存值残留 2e-4 的振铃。
  .options interp 把波形插值到 TSTEP_US 的网格上,记录时刻必须离相位边沿至少一个网格间隔。
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .runner import run_ngspice, parse_wrdata
from . import crossbar as xb
from .gdn_layer import _xbar_group
from .gf180 import HEADER as GF180_HEADER, LINKS as GF180_LINKS
from snn_spec.block import SpikingBlock
from snn_spec.gdn import ExternalGates

ROOT = Path(__file__).resolve().parents[2]
NEU = dict(RC=1.0, leak=0.9)                       # 神经元泄漏 RC(µs),每 token 泄漏因子
CELL = dict(RC=1.4427)                             # gain cell 泄漏 RC(µs):alpha=0.5 ↔ t_gap=1.0 µs
CN, CS, CH, CACC = 1e-9, 1e-9, 1e-12, 1e-9         # 膜电容、cell 电容、锁存/保持电容、累加器电容(F)
W_INT, W_SLOT, GAP = 1.0, 0.1, 0.1                 # 积分窗、子时隙的采样/作用窗、相位间隔(µs)
TMAX_NS = 1.0                                      # 瞬态最大步长(ns)
EDGE_US = 0.001                                    # 相位边沿(µs)
METHOD = "gear"                                    # 积分方法:采样开关的 RC(1 ps)远小于步长,梯形法在这个刚性模态上不衰减(锁存值残留振铃)
EXTRA_OPTIONS = ""                                 # 追加到 .options 的求解器选项(如 "reltol=1e-5");默认用 ngspice 的缺省容差
TSTEP_US = 0.05                                    # 输出网格(.options interp 把波形插值到这个间隔上;记录时刻离任何相位边沿 ≥ 0.05 µs)
# 神经元级:A = q/k/v 投影神经元,B = SN_pre(读出 o),C = SN_out(W_O,得 y1),D = g/u 神经元(W_1、W_3),E = y2 神经元(W_2)
DEFAULT_THR = dict(q_in1=0.7, q=1.0, k=1.0, v=1.0, pre=0.3, out=0.5, q_in2=0.7, g=1.0, u=1.0, y2=1.0)


# ----------------------------------------------------------------------------- 相位表

def schedule(levels):
    """按 T 排一个 token 周期的全部相位窗口(µs)。返回 (PH, 周期, 记录时刻)。
    每个神经元级 X 有:Xl 泄漏窗、Xi 积分窗,以及 T 个子时隙的 Xs{k}(比较采样)与 Xa{k}(计数 + 复位)。"""
    PH = dict(clr=(0.0, 0.15), load=(0.0, 0.10))
    t = 0.20

    def win(name, w):
        # 不能粗舍入:泄漏窗 −ln 0.9 = 0.1053605 µs 若舍到 0.1054,泄漏因子变成 0.899964,膜电位每 token 差 4e-5 的相对量
        nonlocal t
        PH[name] = (round(t, 7), round(t + w, 7))
        t = t + w + GAP

    win("cmp1", 0.10)

    def stage(tag):
        win(f"{tag}l", -math.log(NEU["leak"]) * NEU["RC"])
        win(f"{tag}i", W_INT)
        for k in range(levels):
            win(f"{tag}s{k}", W_SLOT)
            win(f"{tag}a{k}", W_SLOT)

    stage("A")
    PH["p1"] = (round(t, 7), None)                  # gain cell 泄漏窗,宽度按每 token 的 alpha(alpha = 0.5 时 1.0 µs)
    t += 1.0 + GAP
    win("p3", 1.0)                                  # 检索(r 连续),窗末采样保持
    PH["p3s"] = (round(PH["p3"][1] - 0.1, 7), PH["p3"][1])
    win("p4", 1.0)                                  # 写入
    stage("B")
    stage("C")
    win("acc1", 0.10)                               # y1 注入累加器
    win("cmp2", 0.10)                               # 前馈子块入口比较
    stage("D")
    stage("E")
    win("acc2", 0.10)                               # y2 注入累加器
    t_rec = round(t + 0.1, 7)
    period = math.ceil((t_rec + 0.3) * 10) / 10
    return PH, period, t_rec


def _pulse(name, node, t0, t1, period):
    """0→1 的相位窗口:EDGE_US 线性边沿、平顶 t1 − t0 − EDGE_US,∫V dt 正好是 t1 − t0。按窗口计量电荷的电流都直接乘相位电压
    (不用 u(V − 0.5) 的硬阶跃:求解步跨过阶跃时窗宽会差半个步长);边沿两端是求解器必经的断点。开关(阈值 0.5 V)在边沿中点动作。"""
    return f"V{name} {node} 0 PULSE(0 1 {t0:.7f}u {EDGE_US:.4g}u {EDGE_US:.4g}u {t1 - t0 - EDGE_US:.7f}u {period:.4f}u)\n"


def _pwl_leak_windows(alphas, t_p1, period):
    """gain cell 泄漏窗(每 token 宽度 −ln α · RC),同样是线性边沿、∫V dt = 窗宽。"""
    pts = ["0 0"]
    for tau, a in enumerate(alphas):
        t0 = tau * period + t_p1
        tg = -math.log(a) * CELL["RC"]
        pts += [f"{t0:.6f}u 0", f"{t0 + EDGE_US:.6f}u 1", f"{t0 + tg:.6f}u 1", f"{t0 + tg + EDGE_US:.6f}u 0"]
    return "PWL(" + "\n+ ".join(" ".join(pts[i:i + 6]) for i in range(0, len(pts), 6)) + ")"


def _pwl_token(values, period):
    """每 token 一个常数值、整个 token 保持的 PWL。"""
    pts = []
    for tau, v in enumerate(values):
        a, b = tau * period, (tau + 1) * period - 0.001
        pts += [f"{a:.4f}u {v:.9g}", f"{b:.4f}u {v:.9g}"]
    return "PWL(" + "\n+ ".join(" ".join(pts[i:i + 6]) for i in range(0, len(pts), 6)) + ")"


# ----------------------------------------------------------------------------- 电路块

def neurons(name, n, drive_terms, ternary, thr, stage, levels, cap=CN):
    """n 个 LIF 神经元(每 token 最多 levels 个脉冲)。drive_terms[j] 是积分窗里的输入(软件单位,B 源表达式);
    None 表示输入电流由 crossbar 经电流传输器灌入膜电容。节点:膜 u{name}{j}、比较 c{name}{j}、子时隙锁存 f{name}{j}、
    脉冲锁存 s{name}{j}(token 开始清零,T 个子时隙后为 c/T)。"""
    th = thr / levels
    g_int = cap / (W_INT * 1e-6)
    applies = "+".join(f"V({stage}a{k})" for k in range(levels))
    lines = []
    for j in range(n):
        u, c, f, s = f"u{name}{j}", f"c{name}{j}", f"f{name}{j}", f"s{name}{j}"
        fire = f"u(V({u})-{th:.6g})" + (f"-u(-V({u})-{th:.6g})" if ternary else "")
        drive = f"V({stage}i)*({drive_terms[j]})*{g_int:.6g}" if drive_terms is not None else "0"
        lines.append(f"C{name}{j} {u} 0 {cap:.4g} ic=0")
        lines.append(f"B{name}{j} 0 {u} I = {drive}"
                     f" - V({stage}l)*V({u})*{cap / (NEU['RC'] * 1e-6):.6g}"
                     f" - ({applies})*V({f})*{th:.6g}*{cap / (W_SLOT * 1e-6):.6g}")
        lines.append(f"Bc{name}{j} {c} 0 V = {fire}")
        for k in range(levels):
            lines.append(f"Sf{name}{j}_{k} {c} {f} {stage}s{k} 0 SW")
        lines.append(f"Cf{name}{j} {f} 0 {CH:.4g} ic=0")
        lines.append(f"Bs{name}{j} 0 {s} I = ({applies})*V({f})*{CH / (levels * W_SLOT * 1e-6):.6g}")
        lines.append(f"Cs{name}{j} {s} 0 {CH:.4g} ic=0")
        lines.append(f"Sc{name}{j} {s} 0 clr 0 SW")
    return "\n".join(lines) + "\n"


def comparators(name, src_nodes, thr, levels, phase):
    """块入口量化器 Q_in:每通道 2T 个比较器(阈值 ±j·θ/T),输出 (Σ_j 1[h ≥ jθ/T] − Σ_j 1[h ≤ −jθ/T])/T,在 phase 窗口采样锁存。"""
    lines = []
    for i, h in enumerate(src_nodes):
        pos = "+".join(f"u(V({h})-{j * thr / levels:.6g})" for j in range(1, levels + 1))
        neg = "+".join(f"u(-V({h})-{j * thr / levels:.6g})" for j in range(1, levels + 1))
        lines.append(f"Bc{name}{i} c{name}{i} 0 V = (({pos})-({neg}))/{levels}")
        lines.append(f"S{name}{i} c{name}{i} {name}{i} {phase} 0 SW")
        lines.append(f"C{name}{i} {name}{i} 0 {CH:.4g} ic=0")
    return "\n".join(lines) + "\n"


def netlist(ref, xbar="ideal"):
    W = ref["weights"]
    d, dff, L, T = ref["d"], ref["d_ff"], ref["L"], ref["levels"]
    thr = ref["thr"]
    PH, period, _ = schedule(T)
    ck = 1.0 / math.sqrt(d)
    n = [f"* one decoder layer: GDN (1 head, d={d}) + SwiGLU (d_ff={dff}) + 2 accumulators, L={L}, levels={T}, crossbar={xbar}\n"]
    if xbar != "ideal":
        n.append(GF180_HEADER)
    n.append(f".options method={METHOD} {EXTRA_OPTIONS}\n.model SW sw vt=0.5 vh=0.01 ron=1 roff=1e12\n")
    for key, (t0, t1) in PH.items():
        if key != "p1":
            n.append(_pulse(key, key, t0, t1, period))
    n.append(f"Vp1 p1 0 {_pwl_leak_windows(ref['alpha'], PH['p1'][0], period)}\n")
    n.append(f"Vbeta beta 0 {_pwl_token(ref['beta'], period)}\n")
    # 累加器:DAC 装入 h_in;acc1 注入 y1,acc2 注入 y2(每个脉冲单位 = 1 V)
    for i in range(d):
        n.append(f"Vhin{i} hin{i} 0 {_pwl_token(ref['h_in'][:, i].tolist(), period)}\n")
        n.append(f"Cacc{i} h{i} 0 {CACC:.4g} ic=0\nSld{i} h{i} hin{i} load 0 SW\n")
        n.append(f"Bacc{i} 0 h{i} I = V(acc1)*V(sy{i})*{CACC / 0.1e-6:.6g} + V(acc2)*V(sz{i})*{CACC / 0.1e-6:.6g}\n")
    n.append(comparators("x", [f"h{i}" for i in range(d)], thr["q_in1"], T, "cmp1"))
    n.append(comparators("w", [f"h{i}" for i in range(d)], thr["q_in2"], T, "cmp2"))

    def proj(Wm, src):
        return ["+".join(f"({Wm[j, i]:.9g})*V({src}{i})" for i in range(Wm.shape[1])) for j in range(Wm.shape[0])]

    def crossbar_stage(groups, rows_src, stage, n_rows):
        """R / 1T1R:行驱动 ±V_read·src、选通 = 该级积分窗;groups: [(名字, 权重键, 三值)]。"""
        rp, rn = [f"{rows_src}rp{i}" for i in range(n_rows)], [f"{rows_src}rn{i}" for i in range(n_rows)]
        for i in range(n_rows):
            n.append(f"E{rows_src}rp{i} {rp[i]} 0 {rows_src}{i} 0 {xb.V_READ:.4g}\nE{rows_src}rn{i} {rn[i]} 0 {rows_src}{i} 0 {-xb.V_READ:.4g}\n")
        t0, t1 = PH[f"{stage}i"]
        # 选通开关在边沿中点(V_SEL/2)动作:平顶取窗宽 − 1 个边沿,导通时间正好是积分窗宽
        n.append(f"Vsel{stage} sel{stage} 0 PULSE(0 {xb.V_SEL} {t0:.7f}u {EDGE_US:.4g}u {EDGE_US:.4g}u {t1 - t0 - EDGE_US:.7f}u {period:.4f}u)\n")
        for nm, key, tern in groups:
            cap = _xbar_group(n, nm, nm, W[key], ref["w_max"][key], rp, rn, f"sel{stage}", xbar)
            n.append(neurons(nm, W[key].shape[0], None, tern, thr[{"z": "y2", "p": "pre", "y": "out"}.get(nm, nm)], stage, T, cap=cap))

    # 注意力子块:投影神经元(q 二值,k、v 三值)
    if xbar == "ideal":
        n.append(neurons("q", d, proj(W["Wq"], "x"), False, thr["q"], "A", T))
        n.append(neurons("k", d, proj(W["Wk"], "x"), True, thr["k"], "A", T))
        n.append(neurons("v", d, proj(W["Wv"], "x"), True, thr["v"], "A", T))
    else:
        crossbar_stage([("q", "Wq", False), ("k", "Wk", True), ("v", "Wv", True)], "x", "A", d)
    for i in range(d):
        n.append(f"Bkk{i} kk{i} 0 V = {ck:.6g}*V(sk{i})\n")
    for i in range(d):
        for j in range(d):
            s = f"m{i}_{j}"
            n.append(f"Cm{i}_{j} {s} 0 {CS:.4g} ic=0\n")
            n.append(f"Bm{i}_{j} 0 {s} I = -V(p1)*V({s})*{CS / (CELL['RC'] * 1e-6):.6g}"
                     f" + V(p4)*V(kk{i})*V(dv{j})*{CS / 1e-6:.6g}\n")
    for j in range(d):
        n.append(f"Br{j} r{j} 0 V = " + "+".join(f"V(kk{i})*V(m{i}_{j})" for i in range(d)) + "\n")
        n.append(f"Sr{j} r{j} rh{j} p3s 0 SW\nCrh{j} rh{j} 0 {CH:.4g} ic=0\n")
        n.append(f"Bdv{j} dv{j} 0 V = V(beta)*(V(sv{j})-V(rh{j}))\n")
        n.append(f"Bo{j} o{j} 0 V = " + "+".join(f"V(sq{i})*V(m{i}_{j})" for i in range(d)) + "\n")
    n.append(neurons("p", d, [f"V(o{j})" for j in range(d)], True, thr["pre"], "B", T))
    if xbar == "ideal":
        n.append(neurons("y", d, proj(W["Wo"], "sp"), True, thr["out"], "C", T))
    else:
        crossbar_stage([("y", "Wo", True)], "sp", "C", d)
    # 前馈子块:g = LIF₂(W_1 x2),u = LIF₃(W_3 x2),p = g·u(锁存 + AND),y2 = LIF₃(W_2 p)
    if xbar == "ideal":
        n.append(neurons("g", dff, proj(W["W1"], "w"), False, thr["g"], "D", T))
        n.append(neurons("u", dff, proj(W["W3"], "w"), True, thr["u"], "D", T))
    else:
        crossbar_stage([("g", "W1", False), ("u", "W3", True)], "w", "D", d)
    for j in range(dff):
        n.append(f"Band{j} pa{j} 0 V = V(sg{j})*V(su{j})\n")
    if xbar == "ideal":
        n.append(neurons("z", d, proj(W["W2"], "pa"), True, thr["y2"], "E", T))
    else:
        crossbar_stage([("z", "W2", True)], "pa", "E", dff)
    rec = (" ".join(f"v(h{i}) v(x{i}) v(sq{i}) v(sk{i}) v(sv{i}) v(sp{i}) v(sy{i}) v(w{i}) v(sz{i})" for i in range(d)) + " "
           + " ".join(f"v(sg{j}) v(su{j}) v(pa{j})" for j in range(dff)) + " "
           + " ".join(f"v(m{i}_{j})" for i in range(d) for j in range(d)))
    n.append(f""".options interp
.control
set wr_singlescale
set wr_vecnames
tran {TSTEP_US:.3g}u {L * period:.4f}u 0 {TMAX_NS:.3g}n uic
wrdata wave.txt {rec}
.endc
.end
""")
    return "".join(n)


# ----------------------------------------------------------------------------- 软件参考与对照

def block_points(blk):
    """(名字, 模块) 的全部脉冲点,名字与 DEFAULT_THR 一致。"""
    a, f = blk.attn, blk.ffn
    return [("q_in1", blk.q_in1), ("q", a.sn_q), ("k", a.sn_k), ("v", a.sn_v), ("pre", a.sn_pre), ("out", a.sn_out),
            ("q_in2", blk.q_in2), ("g", f.sn_1), ("u", f.sn_3), ("y2", f.sn_2)]


def make_reference(L=16, d=16, d_ff=32, seed=0, w_scale=0.5, levels=1, thr=None, h_scale=1.0, quantize=False):
    """软件参考:随机权重的单头 SpikingBlock(外部门值),每 token 的块输入 h_in ~ N(0, h_scale²);逐 token 记录。
    levels = T 时把全部 10 个脉冲点设为每 token 最多 T 个脉冲(snn_spec 的 levels)。quantize=True 时 7 个权重矩阵
    先量化到 crossbar 的电导级(crossbar 模式下软件参考与电路用同一组量化权重)。"""
    torch.manual_seed(seed)
    thr = dict(DEFAULT_THR, **(thr or {}))
    blk = SpikingBlock(d, d_ff, 1, 1, d, d, gates=ExternalGates(), leak=NEU["leak"])
    w_max = {}
    mats = dict(Wq=blk.attn.Wq, Wk=blk.attn.Wk, Wv=blk.attn.Wv, Wo=blk.attn.Wo, W1=blk.ffn.W1, W3=blk.ffn.W3, W2=blk.ffn.W2)
    with torch.no_grad():
        for key, lin in mats.items():
            Wr = torch.randn_like(lin.weight) * w_scale
            if quantize:
                _, _, Wq_, wm = xb.quantize_weights(Wr.numpy())
                Wr = torch.tensor(Wq_, dtype=Wr.dtype)
                w_max[key] = wm
            lin.weight.copy_(Wr)
        for name, mod in block_points(blk):
            target = mod.log_thr if hasattr(mod, "log_thr") else mod._thr.log_thr
            target.fill_(math.log(thr[name]))
            mod.levels = int(levels)
    h_in = torch.randn(L, d) * h_scale
    g = torch.Generator().manual_seed(seed + 1)
    alpha = (0.5 + 0.49 * torch.rand(L, generator=g)).tolist()
    beta = (0.1 + 0.8 * torch.rand(L, generator=g)).tolist()
    st = blk.init_state(1, torch.device("cpu"))
    recs = []
    with torch.no_grad():
        for t in range(L):
            ext = (torch.tensor([[alpha[t]]]), torch.tensor([[beta[t]]]))
            _, st, r = blk.step(h_in[t:t + 1], st, ext_gates=ext, record=True)
            recs.append({k: v.clone() for k, v in r.items()})
    W = {k: lin.weight.detach().numpy() for k, lin in mats.items()}
    return dict(d=d, d_ff=d_ff, L=L, levels=int(levels), h_in=h_in.numpy(), alpha=alpha, beta=beta, thr=thr, weights=W,
                recs=recs, w_max=w_max, quantized=quantize)


SIGNALS = [  # (名字, 电路节点前缀, 软件记录键, 宽度键, 计数级数的倍数)
    ("x1", "x", "x_attn", "d", 1), ("q", "sq", "q", "d", 1), ("k", "sk", "k", "d", 1), ("v", "sv", "v", "d", 1),
    ("o_spk", "sp", "o_spk", "d", 1), ("y1", "sy", "y_attn", "d", 1), ("x2", "w", "x_ffn", "d", 1),
    ("g", "sg", "g", "d_ff", 1), ("u", "su", "u", "d_ff", 1), ("p", "pa", "p", "d_ff", 2), ("y2", "sz", "y_ffn", "d", 1)]


def simulate(ref, xbar="ideal"):
    if xbar != "ideal" and not ref.get("quantized"):
        raise ValueError("crossbar 模式需要 make_reference(quantize=True) 的量化权重参考")
    log, files = run_ngspice(netlist(ref, xbar), files=("wave.txt",), timeout=14400, links=GF180_LINKS if xbar != "ideal" else ())
    if "wave.txt" not in files:
        raise RuntimeError("ngspice 未输出波形:\n" + log[-3000:])
    w = parse_wrdata(files["wave.txt"])
    t = np.array(w["time"]) * 1e6
    PH, period, t_rec = schedule(ref["levels"])
    d, dff, L = ref["d"], ref["d_ff"], ref["L"]
    width = dict(d=d, d_ff=dff)
    at = lambda tau, tm: int(np.argmin(np.abs(t - (tau * period + tm))))  # noqa: E731
    out = {name: [] for name, *_ in SIGNALS}
    out.update(h_mid=[], h_out=[], S=[])
    for tau in range(L):
        i_rec = at(tau, t_rec)
        for name, node, _, wk, _ in SIGNALS:
            out[name].append([w[f"v({node}{j})"][i_rec] for j in range(width[wk])])
        i_mid = at(tau, PH["acc1"][1] + 0.05)
        out["h_mid"].append([w[f"v(h{j})"][i_mid] for j in range(d)])
        out["h_out"].append([w[f"v(h{j})"][at(tau, PH["acc2"][1] + 0.05)] for j in range(d)])
        out["S"].append([[w[f"v(m{i}_{j})"][i_rec] for j in range(d)] for i in range(d)])
    return {k: np.array(v) for k, v in out.items()}


def threshold_margins(ref):
    """软件参考里每个脉冲点离判决边界的最小距离:神经元是发放前的膜电位 |U| 到最近的 jθ/T,块入口比较器是 |h| 到最近的 jθ/T。
    电路的连续量误差(~1e-5)只会翻转距离与它同量级的判决;返回 {点: dict(min, token, channel)}(编号从 0 起)。"""
    d, L, T, W, thr = ref["d"], ref["L"], ref["levels"], ref["weights"], ref["thr"]
    recs = ref["recs"]
    vec = lambda t, k: recs[t][k].reshape(-1).numpy().astype(float)  # noqa: E731
    ident = np.eye(d)
    # (点, 输入矩阵, 输入记录键, 输出记录键, 输出缩放)
    pts = [("q", W["Wq"], "x_attn", "q", 1.0), ("k", W["Wk"], "x_attn", "k", math.sqrt(d)), ("v", W["Wv"], "x_attn", "v", 1.0),
           ("pre", ident, "o", "o_spk", 1.0), ("out", W["Wo"], "o_spk", "y_attn", 1.0),
           ("g", W["W1"], "x_ffn", "g", 1.0), ("u", W["W3"], "x_ffn", "u", 1.0), ("y2", W["W2"], "p", "y_ffn", 1.0)]
    res = {}

    def nearest(vals, th):
        return np.abs(np.abs(vals)[:, None] - th / T * np.arange(1, T + 1)[None, :]).min(1)

    for name, Wm, src, dst, sc in pts:
        U = np.zeros(Wm.shape[0])
        best = (np.inf, -1, -1)
        for t in range(L):
            U = NEU["leak"] * U + Wm @ vec(t, src)
            m = nearest(U, thr[name])
            j = int(m.argmin())
            best = min(best, (float(m[j]), t, j))
            U = U - thr[name] * vec(t, dst) * sc
        res[name] = dict(min=best[0], token=best[1], channel=best[2])
    for name, key in (("q_in1", "h_in"), ("q_in2", "h_mid")):
        best = min((float(nearest(vec(t, key), thr[name]).min()), t, int(nearest(vec(t, key), thr[name]).argmin())) for t in range(L))
        res[name] = dict(min=best[0], token=best[1], channel=best[2])
    return res


def compare(ref, out):
    """逐 token 比对:每个脉冲点按计数(c = round(|s|·T),正负分开)的 precision/recall、符号错误率、逐元素一致率;
    累加器 h_mid、h_out 与状态 S 的最大绝对误差。p = g·u 的计数级数是 T²。"""
    from snn_spec.metrics import sign_error_rate, spike_prf_by_sign
    d, L, T = ref["d"], ref["L"], ref["levels"]
    ck = 1.0 / math.sqrt(d)
    rep = {}
    for name, _, rkey, _, mult in SIGNALS:
        lv = T ** mult
        scale = 1.0 / ck if name == "k" else 1.0
        r = np.stack([ref["recs"][t][rkey].reshape(-1).numpy() * scale for t in range(L)])
        c = np.rint(out[name] * lv) / lv                       # 电路电压取最近的脉冲级
        cr = np.rint(r * lv) / lv
        prf = spike_prf_by_sign(torch.tensor(c), torch.tensor(cr), levels=lv)
        rep[name] = dict(levels=lv, agreement=float((np.abs(c - cr) < 0.5 / lv).mean()),
                         ref_spikes=int(np.abs(np.rint(cr * lv)).sum()), ckt_spikes=int(np.abs(np.rint(c * lv)).sum()),
                         pos_precision=prf["pos_precision"], pos_recall=prf["pos_recall"],
                         neg_precision=prf["neg_precision"], neg_recall=prf["neg_recall"],
                         sign_error_rate=sign_error_rate(torch.tensor(c), torch.tensor(cr)),
                         max_level_dev=float(np.abs(out[name] * lv - np.rint(out[name] * lv)).max()),
                         per_token_mismatch=[int((np.abs(c[t] - cr[t]) >= 0.5 / lv).sum()) for t in range(L)],
                         mismatch_sites=[[int(t), int(j)] for t, j in zip(*np.nonzero(np.abs(c - cr) >= 0.5 / lv))][:20])
    rep["margins"] = threshold_margins(ref)
    for key in ("h_mid", "h_out"):
        r = np.stack([ref["recs"][t][key].reshape(-1).numpy() for t in range(L)])
        e = np.abs(out[key] - r)
        t, i = np.unravel_index(int(e.argmax()), e.shape)
        rep[key] = dict(max_abs_err=float(e.max()), max_abs_ref=float(np.abs(r).max()), worst_site=[int(t), int(i)],
                        ckt_at_worst=float(out[key][t, i]), ref_at_worst=float(r[t, i]))
    S_ref = np.stack([ref["recs"][t]["S"][0, 0].numpy() for t in range(L)])
    bound = math.sqrt(d) / (1 - max(ref["alpha"]))
    rep["S"] = dict(max_abs_err=float(np.abs(out["S"] - S_ref).max()), max_abs_ref=float(np.abs(S_ref).max()),
                    err_over_bound=float(np.abs(out["S"] - S_ref).max() / bound),
                    err_over_scale=float(np.abs(out["S"] - S_ref).max() / np.abs(S_ref).max()))
    spk = [rep[n]["ref_spikes"] for n, *_ in SIGNALS]
    rep["spikes_total_ref"] = int(sum(spk))
    rep["worst"] = dict(pr=min(min(rep[n][k] for k in ("pos_precision", "pos_recall", "neg_precision", "neg_recall")) for n, *_ in SIGNALS),
                        sign_error=max(rep[n]["sign_error_rate"] for n, *_ in SIGNALS),
                        agreement=min(rep[n]["agreement"] for n, *_ in SIGNALS))
    return rep


def main(levels=1, xbar="ideal", L=16, seed=0):
    ref = make_reference(L=L, levels=levels, seed=seed, quantize=(xbar != "ideal"))
    t0 = time.time()
    out = simulate(ref, xbar)
    rep = compare(ref, out)
    rep["sim_seconds"] = round(time.time() - t0, 1)
    PH, period, t_rec = schedule(levels)
    rep["period_us"] = period
    name = f"spice-block-layer-T{levels}" + ("" if xbar == "ideal" else f"-{xbar}") + ("" if seed == 0 else f"-seed{seed}")
    (ROOT / "results" / f"{name}.json").write_text(
        json.dumps(dict(d=ref["d"], d_ff=ref["d_ff"], L=ref["L"], levels=levels, xbar=xbar, seed=seed, thr=ref["thr"],
                        report=rep), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("worst", "h_mid", "h_out", "S", "spikes_total_ref", "sim_seconds", "period_us")}, indent=1))
    return ref, out, rep


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    main(levels=int(a[0]) if len(a) > 0 else 1, xbar=a[1] if len(a) > 1 else "ideal",
         L=int(a[2]) if len(a) > 2 else 16, seed=int(a[3]) if len(a) > 3 else 0)
