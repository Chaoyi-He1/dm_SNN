"""Table 1 物理模型(snn_spec/adm.py)与 Zhao et al. 2025 正文各图的对照。正文(Nat. Electron. 8, 1211–1221)不含
模型方程与参数表,能核对的是器件行为:Fig. 2e 的脉冲串发放、Fig. 3b 的泄漏积分、Fig. 3c/d 的发放阈值、Fig. 4a/b 的
内在可塑性、Fig. 4c/d 的 ISI 随控制电阻。事件按两种定义分别统计:
    导通事件:x = x₁ + x₂ 自下而上越过 1(现有定义,onsets(A))
    输出脉冲:Eq. S15 的 O = H(V_gs − V_th) 的上升沿(onsets(O)),对应实验里测到的输出电流尖峰
结果写入 results/adm-main-text.json;解释见 results/adm-neuron.md。"""
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snn_spec.adm import ADMNeuronModel, TABLE1, pulse_train, onsets, first_fire_time   # noqa: E402

DT = 2e-6
FIG2E = dict(amplitude=0.8, period=2e-3, duty=0.5)      # 正文 Fig. 2e:0.8 V、2 ms 周期、50% 占空比
ISI_MEAN_PAPER_MS = {"5e+07": 2.25, "1e+08": 2.3, "2.5e+08": 4.0, "5e+08": 8.0, "1e+09": 23.8}   # Fig. 4d(250/500 MΩ 读图)


def sim(vin, **over):
    return ADMNeuronModel(**dict(TABLE1, **over), dt=DT).run(vin)


def first_ms(A):
    t = first_fire_time(A, DT)
    return None if t is None else t * 1e3


def isi_stats(idx, t0, t1):
    t = idx.double() * DT
    t = t[(t >= t0) & (t <= t1)]
    if len(t) < 2:
        return dict(n=int(len(t)), mean_ms=None, median_ms=None, min_ms=None, max_ms=None)
    d = (t[1:] - t[:-1]) * 1e3
    return dict(n=int(len(t)), mean_ms=d.mean().item(), median_ms=d.median().item(), min_ms=d.min().item(), max_ms=d.max().item())


def fig2e(**over):
    """Fig. 2e 的刺激跑 1 s。正文:首次发放 ≈ 88 ms,之后为分立的电流尖峰,输出电流在尖峰之间回到基线。"""
    r = sim(pulse_train(**FIG2E, t_end=1.0, dt=DT), **over)
    on, out = onsets(r["A"]), onsets(r["O"])
    t_on = on.double() * DT
    late = r["V"][int(0.5 / DT):]
    win = (t_on >= 0.2) & (t_on < 0.4)
    stable = (t_on >= 0.6) & (t_on <= 0.9)
    return dict(first_turn_on_ms=first_ms(r["A"]), first_output_spike_ms=first_ms(r["O"]),
                n_turn_on_0_400ms=int((t_on < 0.4).sum()), n_output_spikes_0_400ms=int((out.double() * DT < 0.4).sum()),
                n_turn_on_200_400ms=int(win.sum()), pulses_with_turn_on_200_400ms=int(len(torch.unique((t_on[win] // FIG2E["period"]).long()))),
                V_min_after_0p5s=late.min().item(), V_max=r["V"].max().item(), frac_time_O_high_after_0p5s=(late >= TABLE1["Vth"]).double().mean().item(),
                V_at_turn_on_600_900ms_mean=(r["V"][on[stable] - 1].mean().item() if stable.any() else None),
                isi_turn_on_600_900ms=isi_stats(on, 0.6, 0.9), isi_output_600_900ms=isi_stats(out, 0.6, 0.9))


def main():
    out = dict(model="Zhao 2025 SI Table 1, deterministic (sigma=0), dt=2us", source="正文 Nat. Electron. 8, 1211–1221;不含 SI")
    # E0 恒压首次导通(回归值,adm-neuron.md:77.6 / 34.924 / 15.432 ms)
    out["E0_constant_first_turn_on_ms"] = {str(v): first_ms(sim(torch.full((int(0.3 / DT),), v, dtype=torch.float64))["A"]) for v in (0.6, 0.8, 1.0)}
    # E1 Fig. 2e
    out["E1_fig2e_Rc250M"] = fig2e()
    # E2 Fig. 4c/d:ISI 随 R_c(同一刺激,稳定发放段取 0.6–0.9 s;正文取稳定后 300 ms)
    e2 = {}
    for Rc in (50e6, 100e6, 250e6, 500e6, 1e9):
        f = fig2e(Rc=Rc)
        e2[f"{Rc:.2g}"] = dict(paper_isi_mean_ms=ISI_MEAN_PAPER_MS[f"{Rc:.2g}"], isi_turn_on=f["isi_turn_on_600_900ms"], isi_output=f["isi_output_600_900ms"],
                              V_min_after_0p5s=f["V_min_after_0p5s"], frac_time_O_high_after_0p5s=f["frac_time_O_high_after_0p5s"],
                              refire_gate_voltage_implied_by_paper_V=FIG2E["amplitude"] * math.exp(-ISI_MEAN_PAPER_MS[f"{Rc:.2g}"] * 1e-3 / (Rc * TABLE1["Cg"])),
                              V_at_turn_on_mean=f["V_at_turn_on_600_900ms_mean"])
    out["E2_isi_vs_Rc"] = e2
    # E3 Fig. 3c/d:单个 15 ms 脉冲;正文 0.8 V ≈ 5% 发放,0.9 V ≈ 76%(约 5 ms 内导通),1.0 V ≈ 95%
    e3 = {}
    for v in (0.7, 0.8, 0.9, 1.0, 1.2):
        for pw in (15e-3, 25e-3):
            vin = torch.zeros(int(0.04 / DT), dtype=torch.float64); vin[: int(pw / DT)] = v
            e3[f"{v}V_{pw * 1e3:.0f}ms"] = first_ms(sim(vin)["A"])
    out["E3_single_pulse_first_turn_on_ms"] = e3
    # E4 Fig. 3b:0.9 V、1 ms 脉冲,间隔 0.4–1.2 ms,首次发放前的脉冲数(正文 ≈ 140 / 160 / 180 / 215 / 260)
    e4 = {}
    for gap, paper in zip((0.4e-3, 0.6e-3, 0.8e-3, 1.0e-3, 1.2e-3), (140, 160, 180, 215, 260)):
        period = 1e-3 + gap
        t = first_ms(sim(pulse_train(0.9, period, 1e-3 / period, 1.0, DT))["A"])
        e4[f"{gap * 1e3:.1f}ms"] = dict(t_first_ms=t, n_pulses=None if t is None else int(t * 1e-3 // period) + 1, paper_n_pulses=paper)
    out["E4_fig3b_pulses_before_firing"] = e4
    # E5 Fig. 4a/b:成对 2 V、20 ms 脉冲;正文第一次积分 ≈ 13 ms,两次之差 5.7 ms(间隔 3 ms)→ 2.4 ms(200 ms)→ 0.2 ms(1 s)
    e5 = {}
    for gap, paper in zip((3e-3, 30e-3, 80e-3, 200e-3, 1.0), (5.7, 4.5, 3.2, 2.4, 0.2)):
        n_p, n_g = int(20e-3 / DT), int(gap / DT)
        vin = torch.cat([torch.full((n_p,), 2.0), torch.zeros(n_g), torch.full((n_p,), 2.0)]).double()
        A = sim(vin)["A"]
        t1, t2 = first_ms(A[:n_p]), first_ms(A[n_p + n_g:])
        e5[f"{gap * 1e3:.0f}ms"] = dict(t1_ms=t1, t2_ms=t2, diff_ms=None if (t1 is None or t2 is None) else t1 - t2, paper_diff_ms=paper)
    out["E5_fig4ab_paired_pulse"] = e5
    # E6 敏感性:单个参数改动对 Fig. 2e 首次发放(正文 ≈ 88 ms)的影响;只作线索,不作修订依据
    e6 = {}
    for name, over in (("table1", {}), ("mu=200", dict(mu=200.0)), ("mu=300", dict(mu=300.0)), ("gamma1=450", dict(gamma1=450.0)),
                       ("gamma1=300", dict(gamma1=300.0)), ("beta=40", dict(beta=40.0)), ("gamma2=2.5", dict(gamma2=2.5)), ("alpha=0.005", dict(alpha=0.005))):
        r = sim(pulse_train(**FIG2E, t_end=0.4, dt=DT), **over)
        e6[name] = dict(first_turn_on_ms=first_ms(r["A"]), n_turn_on_0_400ms=int(len(onsets(r["A"]))), n_output_spikes_0_400ms=int(len(onsets(r["O"]))))
    out["E6_sensitivity_fig2e"] = e6
    (ROOT / "results" / "adm-main-text.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
