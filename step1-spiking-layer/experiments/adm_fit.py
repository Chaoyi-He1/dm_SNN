"""把 S1.2 的六参数状态模型拟合流程用在 Zhao 2025 的 ADM 物理模型(Table 1,确定性)上,量化理想抽象与器件的差距。
周期 T_cyc 内输入电压恒定;记录周期边界栅电压 V 与周期内是否发生导通;拟合后在新序列上自由运行比对。"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snn_spec.adm import ADMNeuronModel, TABLE1          # noqa: E402
from snn_spec.circuit import DMNeuronModel, fit_dm_neuron  # noqa: E402
from snn_spec.metrics import spike_prf_by_sign             # noqa: E402


def record(model, vin, T_cyc, state="V"):
    """vin: [T, N] 每周期恒定输入电压。返回 (U: [T+1, N] 边界状态量, S: [T, N] 导通事件, R: [T+1, N] 残留)。
    state="V" 取栅电压;state="x" 取离子通道长度 x = F + R。"""
    st = model.init_state((vin.shape[1],))
    pick = (lambda st: st["V"]) if state == "V" else (lambda st: st["F"] + st["R"])
    U, S, Rs = [pick(st).clone()], [], [st["R"].clone()]
    for t in range(vin.shape[0]):
        s, st = model.step(vin[t], st, T_cyc)
        U.append(pick(st).clone()); S.append(s); Rs.append(st["R"].clone())
    return torch.stack(U), torch.stack(S), torch.stack(Rs)


def run(seed=0, N=16, n_train=300, n_test=150, T_cyc=2e-3, dt=5e-6, v_mean=0.5, v_std=0.3, state="V"):
    g = torch.Generator().manual_seed(seed)
    dev = ADMNeuronModel(**TABLE1, dt=dt)
    vin_tr = (v_mean + v_std * torch.randn(n_train, N, generator=g)).clamp(0.0, 1.0).double()
    U, S, R = record(dev, vin_tr, T_cyc, state)
    fit = fit_dm_neuron(vin_tr, U, S, ternary=False)
    vin_te = (v_mean + v_std * torch.randn(n_test, N, generator=g)).clamp(0.0, 1.0).double()
    _, S_te, R_te = record(dev, vin_te, T_cyc, state)
    m = DMNeuronModel(**fit, ternary=False)
    st = m.init_state((N,)); pred = []
    for t in range(n_test):
        s, st = m.step(vin_te[t].float(), st); pred.append(s)
    pred = torch.stack(pred)
    prf = spike_prf_by_sign(pred, S_te)
    # 隐状态的作用:导通事件与周期起始残留 R 的关系
    R0 = R_te[:-1]
    fire_rate_lowR = S_te[R0 < 0.5].mean().item() if (R0 < 0.5).any() else None
    fire_rate_highR = S_te[R0 >= 0.5].mean().item() if (R0 >= 0.5).any() else None
    return dict(seed=seed, state=state, T_cyc=T_cyc, dt=dt, N=N, n_train=n_train, n_test=n_test,
                train_fire_rate=S.mean().item(), test_fire_rate=S_te.mean().item(),
                fit={k: (float(v) if not isinstance(v, int) else v) for k, v in fit.items()},
                pos_precision=prf["pos_precision"], pos_recall=prf["pos_recall"],
                fire_rate_when_residue_below_0p5=fire_rate_lowR, fire_rate_when_residue_above_0p5=fire_rate_highR)


def main():
    torch.set_num_threads(1)
    rows = [run(seed, state=state) for state in ("V", "x") for seed in (0, 1)]
    out = dict(device="Zhao 2025 SI Table 1 ADM 1M1T1R, deterministic", protocol="每周期 2 ms 恒定电压 clamp(N(0.5,0.3),0,1)",
               note="六参数模型只有一个状态;分别以栅电压 V 和离子通道长度 x = F + R 作状态量拟合,比较哪一个是这个器件的膜电位",
               runs=rows)
    (ROOT / "results" / "adm-fit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
