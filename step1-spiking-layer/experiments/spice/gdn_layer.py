"""单头 GDN 子块的 ngspice 实现:按 S0.7 的相位表逐 token 执行。

电路原语是真的:神经元膜电容(泄漏窗 + 积分窗 + 比较 + 减阈值复位 + 脉冲锁存)、16×16 gain cell 电容阵列
(泄漏窗 + 行脉冲写入 + 列求和读出)、检索值采样保持、β·(v − r) 的 delta 块、W_O 后的两级神经元。
四个权重矩阵的列求和有三种实现(xbar 参数):ideal = B 源理想加法;R = 忆阻电阻 crossbar + 理想选通开关(导通电阻
取 GF180 选通管的 R_on)+ 线电阻;1T1R = 忆阻电阻 + 真实 GF180MCU nmos_3p3 选通管 + 线电阻(experiments.spice.crossbar)。
后两种模式里位线由电流传输器钳在 0 V 并把电流灌进膜电容,权重量化到 16 级电导,软件参考用同样的量化权重。
阵列的列求和(Σ_i k_i S_ij、Σ_i q_i S_ij)仍是 B 源。门值 α、β 与块输入脉冲 x 由软件记录外部给定(首个闭环)。
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from .runner import run_ngspice, parse_wrdata
from . import crossbar as xb
from .gf180 import HEADER as GF180_HEADER, LINKS as GF180_LINKS
from snn_spec.gdn import SpikingGDN, ExternalGates

ROOT = Path(__file__).resolve().parents[2]

# ---- 时序(µs);一个 token 周期 T,各相位窗口
T = 12.0
NEU = dict(RC=1.0, leak=0.9)                      # 神经元泄漏 RC(µs),每 token 泄漏因子
CELL = dict(RC=1.4427)                            # gain cell 泄漏 RC(µs):alpha=0.5 ↔ t_gap=1.0 µs
PH = dict(clr=(0.00, 0.15),                       # 清除脉冲锁存
          nq=(0.20, 0.20 - math.log(NEU["leak"]) * NEU["RC"]),   # q/k/v 神经元泄漏窗(0.1054 µs)
          p2=(0.50, 1.50), p2s=(1.60, 1.70), p2r=(1.80, 1.90),   # 投影积分、发放采样、复位
          p1=(2.00, None),                                        # gain cell 泄漏窗,宽度按 token 的 alpha
          p3=(3.20, 4.20), p3s=(4.10, 4.20),                      # 检索、采样 r
          p4=(4.40, 5.40),                                        # 写入
          npre=(5.60, 5.60 - math.log(NEU["leak"]) * NEU["RC"]),
          p5=(5.80, 6.80), p5s=(6.90, 7.00), p5r=(7.10, 7.20),    # 读出积分、发放、复位
          nout=(7.40, 7.40 - math.log(NEU["leak"]) * NEU["RC"]),
          p6=(7.60, 8.60), p6s=(8.70, 8.80), p6r=(8.90, 9.00))    # W_O 积分、发放、复位
T_REC = 11.0
TMAX_NS = 1.0                                      # 瞬态最大步长(ns):泄漏窗长度的时间量化决定 alpha 的精度
CN, CS, CH = 1e-9, 1e-9, 1e-12                     # 神经元膜电容、cell 电容、锁存/保持电容(F)


def _pulse(name, node, t0, t1):
    return f"V{name} {node} 0 PULSE(0 1 {t0:.4f}u 1p 1p {t1 - t0:.4f}u {T:.4f}u)\n"


def _pwl_hold(values, t_on=0.0, t_off=None):
    """每 token 一个常数值的 PWL(值从 token 起点保持到终点)。"""
    pts = []
    for tau, v in enumerate(values):
        a, b = tau * T + t_on, tau * T + (T if t_off is None else t_off) - 0.001
        pts += [f"{a:.4f}u {v:.6g}", f"{b:.4f}u {v:.6g}"]
    return "PWL(" + "\n+ ".join(" ".join(pts[i:i + 6]) for i in range(0, len(pts), 6)) + ")"


def _pwl_leak_windows(alphas):
    """gain cell 泄漏窗:每 token 从 p1 起点打开 t_gap = −ln(alpha)·RC。"""
    pts = ["0 0"]
    for tau, a in enumerate(alphas):
        t0 = tau * T + PH["p1"][0]
        tg = -math.log(a) * CELL["RC"]
        pts += [f"{t0 - 0.001:.4f}u 0", f"{t0:.4f}u 1", f"{t0 + tg:.4f}u 1", f"{t0 + tg + 0.001:.4f}u 0"]
    return "PWL(" + "\n+ ".join(" ".join(pts[i:i + 6]) for i in range(0, len(pts), 6)) + ")"


def _neuron(name, n, drive_terms, ternary, thr, ph_int, ph_leak, ph_fire, ph_reset, w_int, cap=CN):
    """n 个神经元:膜电容、积分电流(drive_terms[j] 是 B 源表达式,单位 = 软件里的输入值;None 表示电流由 crossbar
    经电流传输器灌入)、泄漏窗、比较、锁存、复位。脉冲节点 s{name}{j} 在发放采样后保持到下一 token 的清除脉冲。"""
    lines = []
    g_int = cap / (w_int * 1e-6)
    for j in range(n):
        u, s, cmp_ = f"u{name}{j}", f"s{name}{j}", f"c{name}{j}"
        fire = f"u(V({u})-{thr:.6g})" + (f"-u(-V({u})-{thr:.6g})" if ternary else "")
        drive = f"u(V({ph_int})-0.5)*({drive_terms[j]})*{g_int:.6g}" if drive_terms is not None else "0"
        lines.append(f"C{name}{j} {u} 0 {cap:.4g} ic=0")
        lines.append(f"B{name}{j} 0 {u} I = {drive}"
                     f" - u(V({ph_leak})-0.5)*V({u})*{cap / (NEU['RC'] * 1e-6):.6g}"
                     f" - u(V({ph_reset})-0.5)*V({s})*{thr:.6g}*{cap / 0.1e-6:.6g}")
        lines.append(f"Bc{name}{j} {cmp_} 0 V = {fire}")
        lines.append(f"Sf{name}{j} {cmp_} {s} {ph_fire} 0 SW")
        lines.append(f"Ch{name}{j} {s} 0 {CH:.4g} ic=0")
        lines.append(f"Sc{name}{j} {s} 0 clr 0 SW")
    return "\n".join(lines) + "\n"


def _xbar_group(n, tag, name, W, w_max, rowp, rown, sel, xbar):
    """一个权重矩阵的 crossbar(R / 1T1R)+ 每列的电流传输器(0 V 位线钳位、电流复制进膜电容)。返回该组神经元的膜电容值。"""
    npos, nneg, _, _ = xb.quantize_weights(W, w_max=w_max)
    lines, ends = xb.xbar_lines(tag, npos, nneg, rowp, rown, sel, xbar)
    n.append("\n".join(lines) + "\n")
    cap = xb.unit_scale(w_max) * xb.T_INT                  # 1 软件单位 = 1 V
    for j in range(W.shape[0]):
        n.append(f"Vs{tag}{j} {ends[j]} 0 0\nF{tag}{j} 0 u{name}{j} Vs{tag}{j} 1\n")
    return cap


def netlist(ref, xbar="ideal"):
    W = ref["weights"]; d, L = ref["d"], ref["L"]
    x = ref["x"]; alphas = ref["alpha"]; betas = ref["beta"]
    ck = 1.0 / math.sqrt(d)
    thr = ref["thr"]
    n = []
    n.append("* GDN sub-block, single head, d=%d, L=%d tokens, phase-sequenced, crossbar=%s\n" % (d, L, xbar))
    if xbar != "ideal":
        n.append(GF180_HEADER + ".options method=gear\n")
    n.append(".model SW sw vt=0.5 vh=0.01 ron=1 roff=1e12\n")
    # 相位时钟
    for key in ("clr", "nq", "p2", "p2s", "p2r", "p3", "p3s", "p4", "npre", "p5", "p5s", "p5r", "nout", "p6", "p6s", "p6r"):
        t0, t1 = PH[key]
        n.append(_pulse(key, key, t0, t1))
    n.append(f"Vp1 p1 0 {_pwl_leak_windows(alphas)}\n")
    n.append(f"Vbeta beta 0 {_pwl_hold(betas)}\n")
    # 输入脉冲 x_i(三值,每 token 常数)
    for i in range(d):
        n.append(f"Vx{i} x{i} 0 {_pwl_hold(x[:, i].tolist())}\n")
    # 投影 + 神经元(q 二值,k、v 三值)
    def proj(Wm):
        return ["+".join(f"({Wm[j, i]:.6g})*V(x{i})" for i in range(d)) for j in range(d)]
    if xbar == "ideal":
        n.append(_neuron("q", d, proj(W["Wq"]), False, thr["q"], "p2", "nq", "p2s", "p2r", 1.0))
        n.append(_neuron("k", d, proj(W["Wk"]), True, thr["k"], "p2", "nq", "p2s", "p2r", 1.0))
        n.append(_neuron("v", d, proj(W["Wv"]), True, thr["v"], "p2", "nq", "p2s", "p2r", 1.0))
    else:
        # 行驱动:x_i → ±V_read·x_i(理想驱动器);选通线 = φ2 / φ6 抬到 3.3 V
        for i in range(d):
            n.append(f"Exp{i} xp{i} 0 x{i} 0 {xb.V_READ:.4g}\nExn{i} xn{i} 0 x{i} 0 {-xb.V_READ:.4g}\n")
            n.append(f"Eop{i} op{i} 0 sp{i} 0 {xb.V_READ:.4g}\nEon{i} on{i} 0 sp{i} 0 {-xb.V_READ:.4g}\n")
        for key, node in (("p2", "sel2"), ("p6", "sel6")):
            t0, t1 = PH[key]
            n.append(f"V{node} {node} 0 PULSE(0 {xb.V_SEL} {t0:.4f}u 1n 1n {t1 - t0:.4f}u {T:.4f}u)\n")
        xp, xn = [f"xp{i}" for i in range(d)], [f"xn{i}" for i in range(d)]
        for nm, key, tern in (("q", "Wq", False), ("k", "Wk", True), ("v", "Wv", True)):
            cap = _xbar_group(n, nm, nm, W[key], ref["w_max"][key], xp, xn, "sel2", xbar)
            n.append(_neuron(nm, d, None, tern, thr[nm], "p2", "nq", "p2s", "p2r", 1.0, cap=cap))
    # 缩放后的 k
    for i in range(d):
        n.append(f"Bkk{i} kk{i} 0 V = {ck:.6g}*V(sk{i})\n")
    # gain cell 阵列:泄漏窗放电 + 行脉冲 × 列 Δv 写入
    for i in range(d):
        for j in range(d):
            s = f"s{i}_{j}"
            n.append(f"Cs{i}_{j} {s} 0 {CS:.4g} ic=0\n")
            n.append(f"Bs{i}_{j} 0 {s} I = -u(V(p1)-0.5)*V({s})*{CS / (CELL['RC'] * 1e-6):.6g}"
                     f" + u(V(p4)-0.5)*V(kk{i})*V(dv{j})*{CS / 1e-6:.6g}\n")
    # 检索 r_j = Σ_i kk_i S_ij,p3s 采样保持;delta Δv_j = beta (v_j − r_j)
    for j in range(d):
        n.append(f"Br{j} r{j} 0 V = " + "+".join(f"V(kk{i})*V(s{i}_{j})" for i in range(d)) + "\n")
        n.append(f"Sr{j} r{j} rh{j} p3s 0 SW\nCrh{j} rh{j} 0 {CH:.4g} ic=0\n")
        n.append(f"Bdv{j} dv{j} 0 V = V(beta)*(V(sv{j})-V(rh{j}))\n")
        n.append(f"Bo{j} o{j} 0 V = " + "+".join(f"V(sq{i})*V(s{i}_{j})" for i in range(d)) + "\n")
    # 读出 o → SN_pre(三值)→ W_O → SN_out(三值)
    n.append(_neuron("p", d, [f"V(o{j})" for j in range(d)], True, thr["pre"], "p5", "npre", "p5s", "p5r", 1.0))
    if xbar == "ideal":
        n.append(_neuron("y", d, ["+".join(f"({W['Wo'][j, i]:.6g})*V(sp{i})" for i in range(d)) for j in range(d)],
                         True, thr["out"], "p6", "nout", "p6s", "p6r", 1.0))
    else:
        cap = _xbar_group(n, "o", "y", W["Wo"], ref["w_max"]["Wo"], [f"op{i}" for i in range(d)], [f"on{i}" for i in range(d)], "sel6", xbar)
        n.append(_neuron("y", d, None, True, thr["out"], "p6", "nout", "p6s", "p6r", 1.0, cap=cap))
    # 输出
    rec = " ".join([f"v(sq{j}) v(sk{j}) v(sv{j}) v(sp{j}) v(sy{j})" for j in range(d)] +
                   [f"v(s{i}_{j})" for i in range(d) for j in range(d)])
    n.append(f""".options interp
.control
set wr_singlescale
set wr_vecnames
tran 1u {L * T:.4f}u 0 {TMAX_NS:.3g}n uic
wrdata wave.txt {rec}
.endc
.end
""")
    return "".join(n)


def make_reference(L=16, d=16, seed=0, w_scale=0.5, thr=None, quantize=False):
    """软件参考:随机权重的单头 SpikingGDN,外部门值,随机三值输入;逐 token 记录。
    quantize=True 时权重先量化到 crossbar 的电导级(crossbar 模式下软件参考与电路用同一组量化权重)。"""
    torch.manual_seed(seed)
    thr = thr or dict(q=1.0, k=1.0, v=1.0, pre=0.3, out=0.5)
    m = SpikingGDN(d, 1, 1, d, d, gates=ExternalGates(), leak=NEU["leak"])
    w_max = {}
    with torch.no_grad():
        for key, lin in (("Wq", m.Wq), ("Wk", m.Wk), ("Wv", m.Wv), ("Wo", m.Wo)):
            Wr = torch.randn_like(lin.weight) * w_scale
            if quantize:
                _, _, Wq_, wm = xb.quantize_weights(Wr.numpy())
                Wr = torch.tensor(Wq_, dtype=Wr.dtype)
                w_max[key] = wm
            lin.weight.copy_(Wr)
        for sn, key in ((m.sn_q, "q"), (m.sn_k, "k"), (m.sn_v, "v"), (m.sn_pre, "pre"), (m.sn_out, "out")):
            sn._thr.log_thr.fill_(math.log(thr[key]))
    x = torch.randint(-1, 2, (L, d)).float()
    g = torch.Generator().manual_seed(seed + 1)
    alpha = (0.5 + 0.49 * torch.rand(L, generator=g)).tolist()
    beta = (0.1 + 0.8 * torch.rand(L, generator=g)).tolist()
    st = m.init_state(1, torch.device("cpu"))
    recs = []
    with torch.no_grad():
        for t in range(L):
            ext = (torch.tensor([[alpha[t]]]), torch.tensor([[beta[t]]]))
            _, st, r = m.step(x[t:t + 1], st, ext_gates=ext, record=True)
            recs.append({k: v.clone() for k, v in r.items()})
    W = {k: getattr(m, k).weight.detach().numpy() for k in ("Wq", "Wk", "Wv", "Wo")}
    return dict(d=d, L=L, x=x.numpy(), alpha=alpha, beta=beta, thr=thr, weights=W, recs=recs, w_max=w_max, quantized=quantize)


def simulate_layer(ref, xbar="ideal"):
    if xbar != "ideal" and not ref.get("quantized"):
        raise ValueError("crossbar 模式需要 make_reference(quantize=True) 的量化权重参考")
    log, files = run_ngspice(netlist(ref, xbar), files=("wave.txt",), timeout=7200, links=GF180_LINKS if xbar != "ideal" else ())
    if "wave.txt" not in files:
        raise RuntimeError("ngspice 未输出波形:\n" + log[-3000:])
    w = parse_wrdata(files["wave.txt"])
    t = np.array(w["time"]) * 1e6
    d, L = ref["d"], ref["L"]
    out = dict(q=[], k=[], v=[], o_spk=[], y=[], S=[])
    for tau in range(L):
        idx = int(np.argmin(np.abs(t - (tau * T + T_REC))))
        out["q"].append([w[f"v(sq{j})"][idx] for j in range(d)])
        out["k"].append([w[f"v(sk{j})"][idx] for j in range(d)])
        out["v"].append([w[f"v(sv{j})"][idx] for j in range(d)])
        out["o_spk"].append([w[f"v(sp{j})"][idx] for j in range(d)])
        out["y"].append([w[f"v(sy{j})"][idx] for j in range(d)])
        out["S"].append([[w[f"v(s{i}_{j})"][idx] for j in range(d)] for i in range(d)])
    return {k: np.array(v) for k, v in out.items()}


def compare(ref, out):
    """逐 token 比对:一致率、按正负分的 precision/recall、符号错误率(snn_spec.metrics)、状态误差。"""
    from snn_spec.metrics import spike_prf_by_sign, sign_error_rate
    d, L = ref["d"], ref["L"]
    ck = 1.0 / math.sqrt(d)
    rep = {}
    total = 0
    for key, rkey, scale in (("q", "q", 1.0), ("k", "k", 1.0 / ck), ("v", "v", 1.0), ("o_spk", "o_spk", 1.0), ("y", "y", 1.0)):
        r = np.stack([ref["recs"][t][rkey].reshape(-1).numpy() * scale for t in range(L)])
        c = np.rint(out[key])
        prf = spike_prf_by_sign(torch.tensor(c), torch.tensor(r))
        rep[key] = dict(agreement=float((np.abs(c - r) < 0.5).mean()), ref_spikes=int((r != 0).sum()), ckt_spikes=int((c != 0).sum()),
                        pos_precision=prf["pos_precision"], pos_recall=prf["pos_recall"],
                        neg_precision=prf["neg_precision"], neg_recall=prf["neg_recall"],
                        sign_error_rate=sign_error_rate(torch.tensor(c), torch.tensor(r)),
                        per_token_matches=[int((np.abs(c[t] - r[t]) < 0.5).sum()) for t in range(L)])
        total += int((r != 0).sum())
    S_ref = np.stack([ref["recs"][t]["S"][0, 0].numpy() for t in range(L)])
    bound = math.sqrt(d) / (1 - max(ref["alpha"]))
    rep["S"] = dict(max_abs_err=float(np.abs(out["S"] - S_ref).max()), max_abs_ref=float(np.abs(S_ref).max()),
                    err_over_bound=float(np.abs(out["S"] - S_ref).reshape(L, -1).max(axis=1).max() / bound),
                    err_over_scale=float(np.abs(out["S"] - S_ref).max() / np.abs(S_ref).max()),
                    per_token_max_err=[float(np.abs(out["S"][t] - S_ref[t]).max()) for t in range(L)])
    rep["spikes_total_ref"] = total
    return rep


def main(xbar="ideal", L=16):
    import time
    ref = make_reference(L=L, quantize=(xbar != "ideal"))
    t0 = time.time()
    out = simulate_layer(ref, xbar)
    rep = compare(ref, out)
    rep["sim_seconds"] = round(time.time() - t0, 1)
    suffix = "" if xbar == "ideal" else f"-{xbar}"
    (ROOT / "results" / f"spice-gdn-layer{suffix}.json").write_text(
        json.dumps(dict(d=ref["d"], L=ref["L"], T_us=T, xbar=xbar, report=rep), indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "ideal", int(sys.argv[2]) if len(sys.argv) > 2 else 16)
