"""每 token 能耗估算的总入口:三个训练好的脉冲模型在 memristor 电路、数字 ASIC 上的每 token 能量,
加上本机 GPU 的实测(读 energy_gpu.py 写的 gpu.json),并给出敏感性扫描与嵌入表量化的级数 × 缩放扫描(无 DAC 的嵌入 crossbar 要几级才不掉准确率)。
对应设计文档 docs/superpowers/specs/2026-10-28-energy-estimate-design.md。

做法
  对每个脉冲模型:在切片上用 record=True 的前向数出每 token 的事件(energy_events.count_events),
    取量化前的权重矩阵(model_matrices),然后
    - memristor 电路:两列(65nm 主对比、advanced 展望)× 两个输出头变体(crossbar+ADC / 数字事件驱动)的部件分解;
    - 数字事件驱动脉冲网络:两列 × batch 1/16/64;
    - 嵌入表量化到 16 级(全局 w_max)后重新评估 token 准确率(评估完还原权重),再扫 16/32/64/128/256 级 × global/per_column 缩放,
      找出掉幅不超过 0.5 个百分点的最小配置,按所需的 cell 对数(ceil(log 级数 / log 16))修正嵌入 crossbar 的能量;
    - 敏感性:65nm 主列、crossbar+ADC 头,分别改 t_read、G 缩放、E_adc、列积分器偏置 I_int。
  数字稠密 int8 用 A0 的尺寸(只计成本,不量化不测精度):两列 × batch 1/16/64,另给一行 65nm 下 batch 1、权重在 DRAM 的对照。
  全部原始数(含参数表、事件汇总)写 energy.json,五张 markdown 表写 tables.md(同时打印到标准输出)。
  energy.json 的 meta 记下运行时的 git HEAD 短 SHA、日期、torch 版本与 --device。
  前向只用来数事件与测准确率;GPU 能耗不在这里测,读 --gpu-json。

用法(在 step2-toy-demo 下;--device cuda 时 CUDA_VISIBLE_DEVICES 必须恰为 1,0 号卡归别的项目,否则退出码 2):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/energy_estimate.py \\
      --models T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt,T2=results/improve/T2_c/model.pt,TR2=results/improve/ternres_sat_T2_c/model.pt \\
      --a0 results/ckpt/a0.pt --out results/energy --gpu-json results/energy/gpu.json --device cuda
"""
import argparse
import dataclasses
import datetime
import json
import math
import os
import subprocess
import sys
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP2))

import toy_demo  # noqa: E402,F401  先导入它才能让 step1 的 snn_spec、experiments.spice 可见
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments import energy_model as em  # noqa: E402
from experiments.beam_eval import load_model  # noqa: E402  (导入时会 chdir 到 step2-toy-demo,相对路径以此为准)
from experiments.energy_events import count_events, model_kind, model_matrices  # noqa: E402
from experiments.energy_params import COLUMNS, DEVICE, v  # noqa: E402
from experiments.spice import crossbar as xb  # noqa: E402
from toy_demo.data import all_sequences, load_slice  # noqa: E402
from toy_demo.train import teacher_forced_metrics  # noqa: E402

HEADS = ("crossbar_adc", "digital_event")
BATCHES = (1, 16, 64)
SCAN_T_READ = (10e-9, 100e-9, 1e-6)                    # 主值 100 ns
SCAN_G_SCALE = (0.1, 1.0)                              # 同乘 G_on 与 G_off,1 是主值
SCAN_E_ADC = (1e-12, 5e-12, 10e-12)                    # 主值 5 pJ
SCAN_I_INT = (1e-6, 5e-6, 50e-6)                       # 列积分器偏置,主值 5 µA
_DIMS = ("d", "n_blocks", "n_heads", "dk", "dv", "d_ff")
SWEEP_LEVELS = (16, 32, 64, 128, 256)                  # 嵌入量化扫描的级数
SWEEP_SCALINGS = ("global", "per_column")              # 先 global 后 per_column,同级数下 global 优先
SWEEP_THRESHOLD_PT = 0.5                               # 设计文档的门槛:token 准确率掉幅不超过 0.5 个百分点


# ----------------------------------------------------------------------------- 嵌入量化检查与敏感性

def embedding_quantization_check(model, tokens, levels=16):
    """把 model.emb.weight 按 crossbar.quantize_weights(全局 w_max)量化到 levels 级,重新评估 token 准确率,最后还原权重。
    返回 acc_before(原权重)、acc_after(量化后)、delta = acc_after − acc_before(小数,不是百分点)。"""
    emb = model.emb.weight
    saved = emb.data.clone()
    acc_before = teacher_forced_metrics(model, tokens)["token_acc"]
    try:
        _, _, Wq, _ = xb.quantize_weights(saved.cpu().double().numpy(), levels)
        emb.data.copy_(torch.as_tensor(Wq, dtype=emb.dtype, device=emb.device))
        acc_after = teacher_forced_metrics(model, tokens)["token_acc"]
    finally:
        emb.data.copy_(saved)
    return dict(acc_before=acc_before, acc_after=acc_after, delta=acc_after - acc_before)


def embedding_w_max(model):
    """嵌入表量化用的全局 w_max(= 权重绝对值的最大值)。"""
    return float(model.emb.weight.detach().abs().max())


def _quantize_embedding(W, levels, scaling):
    """W: numpy [vocab, d] → (Wq, w_max)。scaling:
    "global"     整张表一个 w_max(crossbar.quantize_weights 的默认,即 Task 4 的 16 级检查);
    "per_column" 每个嵌入维度(W 的每一列)各用自己的 w_max,电路上就是每列各自的电导满量程 / 积分增益;
                 返回的 w_max 是各列 w_max 的最大值。"""
    if scaling == "global":
        _, _, Wq, w_max = xb.quantize_weights(W, levels)
        return Wq, w_max
    if scaling == "per_column":
        Wq, w_maxes = np.zeros_like(W), []
        for i in range(W.shape[1]):
            col = W[:, i:i + 1]
            if not col.any():                        # 全零列没有可缩放的量(quantize_weights 会除以 0),保持 0
                w_maxes.append(0.0)
                continue
            _, _, Wq[:, i:i + 1], w_i = xb.quantize_weights(col, levels)
            w_maxes.append(w_i)
        return Wq, max(w_maxes)
    raise ValueError(f"unknown scaling {scaling!r}")


def embedding_quantization_sweep(model, tokens, levels_list=SWEEP_LEVELS, scalings=SWEEP_SCALINGS):
    """嵌入表量化的级数 × 缩放扫描。键 f"{scaling}/{levels}"(levels 升序,同级数内按 scalings 给定的顺序)→
    dict(acc_before, acc_after, delta, delta_pt, levels, scaling, w_max);delta 是小数(与 embedding_quantization_check 一致),
    delta_pt = 100·delta 是百分点。每次评估后在 finally 里还原权重。
    另有 "best_within_0.5pt":best_within_threshold(按上述顺序第一个满足 delta_pt ≥ −0.5 的 dict(levels, scaling),没有则 None)。"""
    emb = model.emb.weight
    saved = emb.data.clone()
    W = saved.cpu().double().numpy()
    acc_before = teacher_forced_metrics(model, tokens)["token_acc"]
    out = {}
    for levels in sorted(levels_list):
        for scaling in scalings:
            Wq, w_max = _quantize_embedding(W, levels, scaling)
            try:
                emb.data.copy_(torch.as_tensor(Wq, dtype=emb.dtype, device=emb.device))
                acc_after = teacher_forced_metrics(model, tokens)["token_acc"]
            finally:
                emb.data.copy_(saved)
            delta = acc_after - acc_before
            out[f"{scaling}/{levels}"] = dict(acc_before=acc_before, acc_after=acc_after, delta=delta, delta_pt=100 * delta,
                                              levels=levels, scaling=scaling, w_max=float(w_max))
    out["best_within_0.5pt"] = best_within_threshold(out, scalings)
    return out


def best_within_threshold(sweep, scalings=SWEEP_SCALINGS):
    """扫描字典(键 f"{scaling}/{levels}",可含 "best_within_0.5pt",忽略)→ 第一个满足 100·delta ≥ −SWEEP_THRESHOLD_PT 的
    dict(levels, scaling),没有则 None。顺序:级数升序,同级数内按 scalings 的顺序(默认 global 优先),与字典的插入顺序无关。
    纯函数,不需要模型。"""
    entries = sorted((e for k, e in sweep.items() if k != "best_within_0.5pt"),
                     key=lambda e: (e["levels"], list(scalings).index(e["scaling"])))
    for e in entries:
        if 100 * e["delta"] >= -SWEEP_THRESHOLD_PT - 1e-9:
            return dict(levels=e["levels"], scaling=e["scaling"])
    return None


def cells_per_value(levels, base=16):
    """一个权重值需要的差分 cell 对数:ceil(log(levels)/log(base)),16 级 1 对,32–256 级 2 对(多对 cell 二进制加权)。
    用整数幂比较,避开浮点对数在整数边界上的误差。"""
    k = 1
    while base ** k < levels:
        k += 1
    return k


def embed_levels_needed(sweep, embed_j_16):
    """扫描结果 → 嵌入 crossbar 需要的级数与按此修正的能量(embed_j_16 = 16 级一对 cell 的 J/token,即 by_component["embed"])。
    best_within_0.5pt 为 None 时取扫描里最大的级数(缩放取该级数掉幅最小者),并置 meets_threshold = False。"""
    best = sweep["best_within_0.5pt"]
    entries = [e for k, e in sweep.items() if k != "best_within_0.5pt"]
    if best is None:
        top = max(e["levels"] for e in entries)
        pick = max((e for e in entries if e["levels"] == top), key=lambda e: e["delta"])
        best = dict(levels=top, scaling=pick["scaling"])
    cells = cells_per_value(best["levels"])
    return dict(levels=best["levels"], scaling=best["scaling"], cells_per_value=cells,
                meets_threshold=sweep["best_within_0.5pt"] is not None,
                embed_energy_nJ_at_levels=embed_j_16 * cells * 1e9)


def _key(x):
    """敏感性字典的键:数值的 %g 写法("1e-08"、"0.1"、"1"、"5e-12")。"""
    return f"{x:g}"


def sensitivity(events, mats, cfg, vocab, col="65nm"):
    """主列、crossbar+ADC 头的 memristor 总能量(J)随 t_read、G 缩放(G_on 与 G_off 同乘)、E_adc、列积分器偏置 I_int 的变化。
    返回 {"t_read": {键: J}, "G_scale": {...}, "E_adc": {...}, "I_int": {...}},键见 _key。覆盖只在单次调用里生效,不改 COLUMNS。"""
    def total(overrides):
        return em.memristor_energy(events, mats, cfg, vocab, col, "crossbar_adc", overrides=overrides)["total"]

    g_on, g_off = v(col, "G_on"), v(col, "G_off")
    return dict(
        t_read={_key(t): total({"t_read": t}) for t in SCAN_T_READ},
        G_scale={_key(s): total({"G_on": s * g_on, "G_off": s * g_off}) for s in SCAN_G_SCALE},
        E_adc={_key(e): total({"E_adc": e}) for e in SCAN_E_ADC},
        I_int={_key(i): total({"I_int": i}) for i in SCAN_I_INT})


# ----------------------------------------------------------------------------- 汇总计算

def events_summary(ev):
    """events 的汇总量(标量与每个矩阵的驱动脉冲总数),不含完整的逐行数组。"""
    return dict(n_tokens=ev["n_tokens"], levels=ev["levels"], spikes_lif=ev["spikes_lif"], n_lif=ev["n_lif"],
                cmp_decisions=ev["cmp_decisions"], gain=dict(ev["gain"]),
                rows_sum={k: float(np.sum(x)) for k, x in ev["rows"].items()},
                embed_rows_sum=float(np.sum(ev["embed_rows"])))


def dense_dram_row(cfg, vocab, col="65nm"):
    """稠密 int8、batch 1、权重放 DRAM 的对照:权重取数按 E_dram_bit,其余(含片上嵌入查表)不变。"""
    r = em.digital_dense_energy(cfg, vocab, col, 1, overrides={"E_sram_far_bit": v(col, "E_dram_bit")})
    r["embed"] = em.digital_dense_energy(cfg, vocab, col, 1)["embed"]          # 嵌入表仍在片上 SRAM
    r["total"] = sum(r[k] for k in ("mac", "act", "weight", "embed", "state"))
    return r


def model_report(model, cfg, vocab, tokens, dev, path):
    ev = count_events(model, tokens, dev)
    mats = model_matrices(model)
    levels = int(DEVICE["levels"]["value"])
    quant = embedding_quantization_check(model, tokens, levels)
    quant.update(levels=levels, w_max=embedding_w_max(model))
    sweep = embedding_quantization_sweep(model, tokens)
    memristor = {col: {head: em.memristor_energy(ev, mats, cfg, vocab, col, head) for head in HEADS} for col in COLUMNS}
    return dict(
        path=str(path), kind=model_kind(model), levels=ev["levels"], events_summary=events_summary(ev),
        memristor=memristor,
        digital_event={col: {str(b): em.digital_event_energy(ev, cfg, vocab, col, b) for b in BATCHES} for col in COLUMNS},
        embed_quant=quant, embed_quant_sweep=sweep,
        embed_levels_needed=embed_levels_needed(sweep, memristor["65nm"]["crossbar_adc"]["by_component"]["embed"]),
        sensitivity=sensitivity(ev, mats, cfg, vocab))


def git_short_sha():
    """运行时仓库 HEAD 的短 SHA(git rev-parse --short HEAD,在 step2-toy-demo 下运行);拿不到时为 "unknown"。"""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=STEP2, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def build_result(models, a0_path, slice_dir, dev, gpu_json, device_arg=None):
    sl = load_slice(slice_dir)
    tokens = all_sequences(sl)
    a0, cfg, _ = load_model(a0_path, sl, torch.device("cpu"))
    vocab = a0.emb.num_embeddings
    per_model = {}
    for label, path in models.items():
        model, mcfg, _ = load_model(path, sl, dev)
        if any(getattr(mcfg, k) != getattr(cfg, k) for k in _DIMS) or model.emb.num_embeddings != vocab:
            raise ValueError(f"{label}: 尺寸与 A0 不一致,数字稠密对照将不可比")
        print(f"[{label}] {path}: counting events and checking embedding quantisation on {len(tokens)} sentences", flush=True)
        per_model[label] = model_report(model, mcfg, vocab, tokens, dev, path)
        del model
    gpu = json.loads(Path(gpu_json).read_text(encoding="utf-8")) if gpu_json else None
    return dict(
        meta=dict(slice=str(slice_dir), n_sentences=len(tokens), vocab=vocab, a0=str(a0_path), cfg=dataclasses.asdict(cfg),
                  git_sha=git_short_sha(), run_date=datetime.date.today().isoformat(), torch=torch.__version__,
                  device=str(dev) if device_arg is None else device_arg),
        params=dict(columns=COLUMNS, device=DEVICE),
        mac_counts=em.mac_counts(cfg, vocab),
        models=per_model,
        dense={col: {str(b): em.digital_dense_energy(cfg, vocab, col, b) for b in BATCHES} for col in COLUMNS},
        dense_dram_b1_65nm=dense_dram_row(cfg, vocab),
        gpu=gpu)


# ----------------------------------------------------------------------------- markdown 表

def sig3(x):
    """三位有效数字,不用指数写法,千位以上加逗号。"""
    if x == 0:
        return "0"
    r = float(f"{x:.3g}")
    return f"{r:,.{max(0, 2 - math.floor(math.log10(abs(r))))}f}"


def nj(x):
    return sig3(x * 1e9)


def _md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


HEAD_LABEL = {"crossbar_adc": "输出头 crossbar+ADC", "digital_event": "输出头数字(事件驱动,batch 1)"}
COMPONENTS = [("crossbar_blocks", "突触 crossbar 导通(各块)"), ("crossbar_head", "输出头 crossbar 导通"),
              ("embed", "嵌入 crossbar 导通"), ("gate_wire", "选通栅驱动 + 行线/位线导线"),
              ("integrator", "列积分器偏置"), ("leak_off", "未选通 cell 漏电"),
              ("adm_spikes", "ADM 神经元发放"), ("adm_leak", "ADM 静态漏电"), ("comparators", "比较器"),
              ("gain_cells", "gain cell 阵列"), ("adc", "输出头 ADC"), ("head_digital", "输出头数字(事件驱动)"),
              ("control", "相位时钟与控制")]


GPU_FOOTNOTE = ("GPU 行是实测:小模型在 GPU 上被空载功耗(约 21 W)与逐 token 调度淹没,'增量'是扣掉深度空载后的值,不是边际计算能量;"
                "它反映'GPU 跑这种模型'而非 GPU 的峰值效率。")


def table1(res, ref_label):
    mem, ref = res["models"], res["models"][ref_label]["memristor"]["65nm"]["crossbar_adc"]["total"]
    rows = []

    def add(scheme, model, setting, joules, unit="nJ"):
        val = sig3(joules * (1e9 if unit == "nJ" else 1e3))
        rows.append((scheme, model, setting, f"{val} {unit}", f"{sig3(joules / ref)}×"))

    for label, m in mem.items():
        for head in HEADS:
            add("memristor 65 nm", label, HEAD_LABEL[head], m["memristor"]["65nm"][head]["total"])
    for b in BATCHES:
        add("数字稠密 int8 65 nm", "A0 尺寸", f"batch {b},权重在片上 SRAM", res["dense"]["65nm"][str(b)]["total"])
    add("数字稠密 int8 65 nm(对照)", "A0 尺寸", "batch 1,权重在 DRAM", res["dense_dram_b1_65nm"]["total"])
    for label, m in mem.items():
        for b in BATCHES:
            add("数字事件驱动 65 nm", label, f"batch {b},权重在片上 SRAM", m["digital_event"]["65nm"][str(b)]["total"])
    gpu = res["gpu"]
    if gpu:
        for label, by_batch in gpu["runs"].items():
            for b, r in by_batch.items():
                add(f"GPU {gpu['gpu']}(实测,8 nm)", label, f"batch {b},总功率 ÷ 吞吐", r["total_j_per_token"], "mJ")
                add(f"GPU {gpu['gpu']}(实测,8 nm)", label, f"batch {b},扣空载 {gpu['idle_w']:.1f} W", r["incremental_j_per_token"], "mJ")
    head = f"**表 1 每 token 能量总表**(相对基准 = memristor 65 nm {ref_label}、输出头 crossbar+ADC = {nj(ref)} nJ;GPU 行单位为 mJ)"
    if gpu:
        head += "\n\n" + GPU_FOOTNOTE
    return head + "\n\n" + _md(["方案", "模型", "设定", "能量/token", f"相对 memristor {ref_label}(crossbar+ADC)"], rows)


def table2(res):
    labels = list(res["models"])
    header = ["部件"] + [f"{lab} · {HEAD_LABEL[h]}" for lab in labels for h in HEADS]
    rows = []
    for key, name in COMPONENTS:
        row = [name]
        for lab in labels:
            for h in HEADS:
                r = res["models"][lab]["memristor"]["65nm"][h]
                row.append(f"{nj(r['by_component'][key])} ({100 * r['by_component'][key] / r['total']:.1f}%)")
        rows.append(row)
    rows.append(["**合计**"] + [f"**{nj(res['models'][lab]['memristor']['65nm'][h]['total'])}**" for lab in labels for h in HEADS])
    return "**表 2 memristor 65 nm 的部件分解**(nJ/token,括号内为占该列合计的百分比)\n\n" + _md(header, rows)


def table3(res, ref_label):
    mem = res["models"]
    ref = mem[ref_label]["memristor"]["advanced"]["crossbar_adc"]["total"]
    rows = []

    def add(scheme, model, setting, joules):
        rows.append((scheme, model, setting, nj(joules), f"{sig3(joules / ref)}×"))

    for label, m in mem.items():
        for h in HEADS:
            add("memristor 3 nm 投影", label, HEAD_LABEL[h], m["memristor"]["advanced"][h]["total"])
    for b in (1, 64):
        add("数字稠密 int8 7 nm", "A0 尺寸", f"batch {b}", res["dense"]["advanced"][str(b)]["total"])
    for label, m in mem.items():
        for b in (1, 64):
            add("数字事件驱动 7 nm", label, f"batch {b}", m["digital_event"]["advanced"][str(b)]["total"])
    return ("**表 3 展望列**(各自最优投影,不同节点:memristor 取 3 nm FinFET 投影,数字取 7 nm;不与表 1 的 65 nm 主列混比;"
            f"能量单位 nJ/token,相对基准 = memristor 3 nm {ref_label}、crossbar+ADC = {nj(ref)} nJ)\n\n"
            + _md(["方案", "模型", "设定", "能量/token (nJ)", f"相对 memristor 3 nm {ref_label}(crossbar+ADC)"], rows))


def table4(res):
    labels = list(res["models"])
    first = res["models"][labels[0]]["sensitivity"]
    rows = [("基准(主值)", "—")
            + tuple(nj(res["models"][lab]["memristor"]["65nm"]["crossbar_adc"]["total"]) for lab in labels)]
    for param, unit, scale, shown in (("t_read", "ns", 1e9, "读窗 t_read"), ("G_scale", "×", 1.0, "电导缩放(G_on 与 G_off 同乘)"),
                                      ("E_adc", "pJ", 1e12, "ADC 能量 E_adc"), ("I_int", "µA", 1e6, "列积分器偏置 I_int")):
        for k in first[param]:
            rows.append((shown, f"{float(k) * scale:g} {unit}")
                        + tuple(nj(res["models"][lab]["sensitivity"][param][k]) for lab in labels))
    return ("**表 4 敏感性**(memristor 65 nm、输出头 crossbar+ADC 的总能量,nJ/token;一次只改一个参数)\n\n"
            + _md(["参数", "取值"] + [f"{lab} (nJ)" for lab in labels], rows))


def table5(res):
    models = res["models"]
    first = next(iter(models.values()))["embed_quant_sweep"]
    entries = [e for k, e in first.items() if k != "best_within_0.5pt"]
    levels = sorted({e["levels"] for e in entries})
    scalings = list(dict.fromkeys(e["scaling"] for e in entries))
    rows = []
    for L in levels:
        row = [L]
        for m in models.values():
            for s in scalings:
                d = 100 * m["embed_quant_sweep"][f"{s}/{L}"]["delta"]
                row.append(f"{d:+.2f}" + (" ⚠" if d < -SWEEP_THRESHOLD_PT else ""))
        rows.append(row)
    before = "、".join(f"{lab} {100 * next(iter(m['embed_quant_sweep'].values()))['acc_before']:.2f}%" for lab, m in models.items())
    lines = []
    for lab, m in models.items():
        n = m["embed_levels_needed"]
        base = n["embed_energy_nJ_at_levels"] / n["cells_per_value"]
        if n["meets_threshold"]:
            lines.append(f"- {lab}:最小满足 {SWEEP_THRESHOLD_PT} 个百分点的配置 = {n['levels']} 级、{n['scaling']},每个值 {n['cells_per_value']} 对差分 cell,"
                         f"嵌入 crossbar 能量 {sig3(n['embed_energy_nJ_at_levels'])} nJ/token(16 级一对 cell 为 {sig3(base)} nJ)")
        else:
            d = 100 * m["embed_quant_sweep"][f"{n['scaling']}/{n['levels']}"]["delta"]
            lines.append(f"- {lab}:无配置满足 {SWEEP_THRESHOLD_PT} 个百分点的门槛;扫描上限 {n['levels']} 级、{n['scaling']} 仍掉 {-d:.2f} 个百分点,"
                         f"这个配置**不满足**门槛。若仍按该配置({n['cells_per_value']} 对差分 cell),嵌入 crossbar 能量为 "
                         f"{sig3(n['embed_energy_nJ_at_levels'])} nJ/token(16 级一对 cell 为 {sig3(base)} nJ);这不是 {lab} 满足门槛时的嵌入能量")
    head = ("**表 5 嵌入表量化的 token 准确率变化**(百分点;切片上的 teacher-forcing,只量化嵌入表;global = 整表一个 w_max,"
            "per_column = 每个嵌入维度各用自己的 w_max;⚠ = 掉幅超过设计文档的门槛 0.5 个百分点,要在 README 写明并给脉冲注入的退路;"
            f"量化前:{before})")
    return head + "\n\n" + _md(["级数"] + [f"{lab} {s}" for lab in models for s in scalings], rows) + "\n\n" + "\n".join(lines)


def render_tables(res, ref_label="T2"):
    return "\n\n".join([table1(res, ref_label), table2(res), table3(res, ref_label), table4(res), table5(res)]) + "\n"


# ----------------------------------------------------------------------------- 命令行

def _json_default(o):
    """numpy 标量转成原生数,json.dumps 才能写出原始浮点(不是格式化后的字符串)。"""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"{type(o).__name__} 不能写入 JSON")


def parse_models(spec):
    out = {}
    for item in spec.split(","):
        label, path = item.split("=", 1)
        out[label] = path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="标签=检查点路径,逗号分隔(脉冲模型)")
    ap.add_argument("--a0", default="results/ckpt/a0.pt", help="浮点 A0:提供稠密数字侧的尺寸")
    ap.add_argument("--out", default="results/energy")
    ap.add_argument("--gpu-json", default="results/energy/gpu.json", help="energy_gpu.py 写的实测结果;留空则表 1 不含 GPU 行")
    ap.add_argument("--slice", default="data/tinystories-slice")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ref", default="T2", help="表 1、3 的倍数以哪个模型的 memristor(crossbar+ADC)为分母")
    a = ap.parse_args()

    if a.device.startswith("cuda") and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        print(f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES')!r}, not '1': GPU 0 belongs to another project, "
              "run with CUDA_VISIBLE_DEVICES=1", file=sys.stderr)
        sys.exit(2)
    models = parse_models(a.models)
    if a.ref not in models:
        sys.exit(f"--ref {a.ref} 不在 --models 里: {list(models)}")
    res = build_result(models, a.a0, a.slice, torch.device(a.device), a.gpu_json or None, device_arg=a.device)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "energy.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tables = render_tables(res, a.ref)
    (out / "tables.md").write_text(tables, encoding="utf-8")
    print(tables)
    print(f"saved {out / 'energy.json'} and {out / 'tables.md'}")


if __name__ == "__main__":
    main()
