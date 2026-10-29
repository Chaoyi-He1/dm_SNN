# SNN 每 token 能耗估算 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 算出 toy demo 三个脉冲模型在 memristor 电路上每 token 的能量(按切片实测事件数、逐部件),与同一模型在数字 ASIC(稠密 int8 与事件驱动)和 GPU(实测)上的每 token 能量放进一张表。

**Architecture:** 四个小模块:`energy_params`(参数与来源)、`energy_events`(跑模型记录、数事件)、`energy_model`(纯公式:乘加数、行电导、memristor / 数字能量)、`energy_gpu`(nvidia-smi 采样实测);一个 CLI `energy_estimate.py` 把它们串起来,输出 `results/energy/energy.json` 与 markdown 表;README 写结论。事件计数复用 `toy_demo` 模型的 `record=True` 记录,权重量化复用 `experiments/spice/crossbar.py`。

**Tech Stack:** Python 3.10 venv `/data/chaoyi_he/dm_snn/step2-toy-demo/.venv`(torch 2.14 cu126、numpy),pytest;`nvidia-smi` 采样;GPU 1。

**Spec:** `docs/superpowers/specs/2026-10-28-energy-estimate-design.md`

## Global Constraints

- 代码树:worktree `/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep`,分支 `worktree-conversion-sweep`;所有命令在 `step2-toy-demo/` 下跑,`PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python`。`experiments` 是跨两棵树的命名空间包:`from experiments.spice import crossbar`(step1)与 `from experiments.beam_eval import load_model`(step2)都可用,但 `import toy_demo` 必须先于任何 `snn_spec` / `experiments.spice` 导入。
- 不改任何已有文件(`toy_demo/`、`snn_spec/`、`experiments/*.py` 现有脚本、`results/improve/*`);新文件只在 `experiments/`、`tests/`、`results/energy/`。
- 测试在 CPU(`CUDA_VISIBLE_DEVICES=""`);事件计数与 GPU 实测只用 GPU 1(`CUDA_VISIBLE_DEVICES=1`,`nvidia-smi -i 1`);GPU 0 不碰。
- 工艺对齐:主列 65 nm,memristor 侧 CMOS 周边与数字侧同节点;忆阻器件量(G_on 100 µS、G_off 1 µS、16 级、V_read 0.2 V、ADM 1.39 pJ/spike)不随节点缩;展望列 memristor 3 nm 投影 / 数字 7 nm,表头标"各自最优投影,不同节点";GPU 8 nm 实测单列。
- 数字侧权重流量 = 不含嵌入表的参数(0.74M)× 8 bit / batch;嵌入查表 128 × 8 bit 另计;batch ∈ {1, 16, 64}。
- memristor 侧嵌入 = crossbar 读(one-hot 一行 × 256 cell),无 SRAM 无 DAC;输出头两个变体:crossbar + 3226 路 ADC,或数字事件驱动(batch 1)。
- 每个参数在 `energy_params.py` 里带 `source` 字段;README 的每个数字只能来自 `energy.json` / `gpu.json`。
- 提交信息结尾两行:`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`、`Claude-Session: https://claude.ai/code/session_01WXSmKMbtt8Re7oceEtjWmJ`。只提交到分支,不推 main。
- 中文 docstring。

## Review Focus

1. T > 1 的脉冲计数:每行的事件数是 |s|·T,W2 的输入是 c_g·c_u(不是 |p|·T)→ Task 2 `test_count_events_levels_two_counts_pulses_and_w2_products`。
2. 差分对:一个权重的两个 cell 在同一行,行电导必须是 Σ_j (G⁺ + G⁻)→ Task 1 `test_row_conductance_sums_both_cells_of_each_weight`。
3. 数字侧权重流量不含嵌入表、输出头只算一次 → Task 1 `test_mac_counts_match_analytic_shapes`。
4. batch 只摊销权重取数,乘加与激活访存不变 → Task 1 `test_digital_dense_batch_amortises_only_weight_fetch`。
5. 填充位置不能计入事件 → Task 2 `test_count_events_ignores_padding`。

---

## File Structure

- Create `step2-toy-demo/experiments/energy_params.py`:`DEVICE`、`COLUMNS["65nm"]`、`COLUMNS["advanced"]`,每项 `dict(value, unit, source)`;`v(col, name)` 取值。
- Create `step2-toy-demo/experiments/energy_model.py`:`mac_counts`、`row_conductance`、`crossbar_energy`、`memristor_energy`、`digital_dense_energy`、`digital_event_energy`。
- Create `step2-toy-demo/experiments/energy_events.py`:`count_events(model, tokens, dev, batch=64) -> dict`、`model_matrices(model) -> dict`。
- Create `step2-toy-demo/experiments/energy_gpu.py`:`sample_power(...)`、`measure(...)`、CLI。
- Create `step2-toy-demo/experiments/energy_estimate.py`:CLI,整合 + 敏感性 + 嵌入量化检查 + 表。
- Create `step2-toy-demo/tests/test_energy_estimate.py`。
- Create `step2-toy-demo/results/energy/{energy.json, tables.md, gpu.json, README.md}`。

---

### Task 1: 参数表与纯公式(`energy_params.py`、`energy_model.py`)

**Files:**
- Create: `step2-toy-demo/experiments/energy_params.py`
- Create: `step2-toy-demo/experiments/energy_model.py`
- Test: `step2-toy-demo/tests/test_energy_estimate.py`

**Interfaces:**
- Consumes: `experiments.spice.crossbar.quantize_weights(W, levels, w_max=None) -> (npos, nneg, Wq, w_max)`、`conductance_targets(npos, nneg, levels) -> (Gp, Gn)`(两者都按 `W: [d_out, d_in]`)。
- Produces:
  - `energy_params.DEVICE = dict(G_on=100e-6, G_off=1e-6, levels=16, V_read=0.2)`,每项也带 source(测试台器件)。
  - `energy_params.COLUMNS: dict[str, dict[str, dict]]`,列名 `"65nm"`、`"advanced"`;每列的键与**主列默认值**(单位 SI):
    `t_read 100e-9`,`t_token 1e-3`,`V_sel 1.0`,`C_gate 1.5e-15`,`C_wire 0.15e-15`,`I_off 10e-12`,`I_int 5e-6`,`V_dd 1.0`,
    `E_spike 1.39e-12`,`R_c 1e9`,`V_gs 0.15`,`E_cmp 50e-15`,`E_adc 5e-12`,`C_s 50e-15`,`V_s 0.5`,`E_ctrl 100e-12`,
    `node_scale 2.0`(Horowitz 45 nm → 65 nm 的 CV² 换算;source 写明依据),`E_mul8 0.2e-12·node_scale`,`E_add8 0.03e-12·node_scale`,`E_add16 0.05e-12·node_scale`,`E_add32 0.1e-12·node_scale`,
    `E_sram_near_bit 10e-12/64·node_scale`(8 KB),`E_sram_far_bit 100e-12/64·node_scale`(1 MB),`E_dram_bit 2e-9/64·node_scale`。
    `"advanced"` 列:memristor 侧按 paper Table 8 投影(`t_read 10e-9`,`t_token 0.1e-6`,`E_spike 1e-18`,`R_c 242e9`,`V_sel 0.7`,`C_gate 0.1e-15`,`C_wire 0.05e-15`,`E_cmp 5e-15`,`E_adc 0.5e-12`,`C_s 10e-15`,`V_s 0.3`,`E_ctrl 10e-12`,`I_int 1e-6`,`V_dd 0.7`),数字侧 7 nm(`E_mul8 0.03e-12`,`E_add8 0.005e-12`,`E_add16 0.008e-12`,`E_add32 0.015e-12`,`E_sram_near_bit 0.02e-12`,`E_sram_far_bit 0.3e-12`,`E_dram_bit 10e-12`,`node_scale 1.0`);这些展望值的 source 一律标 "估计(文献量级),见 README";builder 若能给出具体出处则替换值并写出处。
  - `v(col: str, name: str) -> float`。
  - `energy_model.mac_counts(cfg, vocab) -> dict`:键 `proj_qkv, proj_o, ffn_13, ffn_2, gates, core, head, total_macs, params_stream, params_embedding`,按 d、n_blocks、n_heads、dk、dv、d_ff、vocab 解析:每块 `proj_qkv = 3·d·d`(单头等宽时 n_heads·dk = d),`proj_o = d·d`,`ffn_13 = 2·d·d_ff`,`ffn_2 = d_ff·d`,`gates = 2·d·n_heads`,`core = n_heads·(4·dk·dv)`(检索、写入、读出各 dk·dv,衰减 dk·dv);`head = d·vocab`;`params_stream = Σ 块权重(含门偏置 2·n_heads)+ head`;`params_embedding = vocab·d`。
  - `energy_model.row_conductance(W: np.ndarray, levels: int, G_on: float, G_off: float) -> np.ndarray`:形状 `[d_in]`,`Σ_j (Gp + Gn)[j, i]`。
  - `energy_model.crossbar_energy(c_rows: np.ndarray, grow: np.ndarray, d_out: int, col: str, n_pulses_window: int = 1) -> dict`:`c_rows[i]` = 该行每 token 平均脉冲数;键 `conduction = V_read²·t_read·Σ_i c_i·grow_i`,`gate = Σ_i c_i·2·d_out·C_gate·V_sel²`,`wire = Σ_i c_i·2·d_out·C_wire·V_read²`,`leak_off = (2·d_out·len(c_rows) − Σ_i c_i·2·d_out)·V_read·I_off·t_read`(负则取 0),`integrator = d_out·I_int·V_dd·t_read·n_pulses_window`,`total`。
  - `energy_model.memristor_energy(events: dict, mats: dict, cfg, vocab: int, col: str, head: str) -> dict`:`head ∈ {"crossbar_adc", "digital_event"}`;返回 `dict(by_component={...}, by_matrix={...}, total=float)`,部件键:`crossbar_blocks, crossbar_head, embed, gate_wire, integrator, leak_off, adm_spikes, adm_leak, comparators, gain_cells, adc, head_digital, control`。公式见 spec 的部件表与"审稿后定下的细节":ADM `events["spikes_lif"]·E_spike + events["n_lif"]·V_gs²/R_c·t_token`;比较器 `events["cmp_decisions"]·E_cmp`;gain cell 每块每头 dk·dv 个 cell:`leak = cells·(1 − α_mean²)·½·C_s·V_s²·s_rms_norm²`,`retrieve + write = 2·k_rows·dv·½·C_s·V_s²`,`readout = q_rows·dv·½·C_s·V_s²`;ADC `vocab·E_adc`;`head_digital`(batch 1)= `rows_head·vocab·E_add16 + rows_head·vocab·8·E_sram_far_bit + rows_head·vocab·2·8·E_sram_near_bit`;`control = E_ctrl`。
  - `energy_model.digital_dense_energy(cfg, vocab: int, col: str, batch: int) -> dict`:`mac = total_macs·(E_mul8 + E_add32)`,`act = total_macs·2·8·E_sram_near_bit`,`weight = params_stream·8·E_sram_far_bit / batch`,`embed = d·8·E_sram_far_bit`,`state = n_blocks·n_heads·dk·dv·2·8·E_sram_near_bit`,`total`。
  - `energy_model.digital_event_energy(events: dict, cfg, vocab: int, col: str, batch: int) -> dict`:`add = Σ_mat Σ_i c_i·d_out(mat)·E_add16`,`weight = Σ_mat Σ_i c_i·d_out(mat)·8·E_sram_far_bit / batch`,`act` 同上按加法数,`neurons = n_lif·(E_mul8 + E_add16)`(泄漏乘法 + 比较),`state` 与 `core` 同稠密,`embed` 同稠密,`total`。
  - `events` 字典的约定(Task 2 产出,此处消费):`n_tokens:int`,`levels:int`,`rows: dict[str, np.ndarray]`(键 `blocks.{i}.qkv`、`blocks.{i}.gates`、`blocks.{i}.o`、`blocks.{i}.ffn13`、`blocks.{i}.w2`、`head`,值 = 每行每 token 平均脉冲数),`embed_rows: np.ndarray[vocab]`(每 token 平均 one-hot),`spikes_lif: float`,`n_lif: int`,`cmp_decisions: float`,`gain: dict(k_rows, q_rows, alpha_mean, s_rms_norm)`(k_rows、q_rows 为每 token 每块每头的平均驱动行数,对块与头求和后的总数)。`mats: dict[str, np.ndarray]` 同键,值 `[d_out, d_in]`(`gates` 为 `[2·n_heads, d + 1]`,偏置作最后一列、对应一个恒为 1 的输入行;`embed` 为 `[d, vocab]`)。

- [ ] **Step 1: 写失败的测试**

```python
"""能耗估算:参数表、纯公式、事件计数、数字侧乘加数。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import numpy as np
import torch

from experiments import energy_model as em
from experiments.energy_params import COLUMNS, DEVICE, v
from experiments.spice import crossbar as xb
from toy_demo.config import ToyConfig


def test_every_parameter_has_a_source():
    for col in ("65nm", "advanced"):
        for name, p in COLUMNS[col].items():
            assert set(p) >= {"value", "unit", "source"} and p["source"], (col, name)
    assert v("65nm", "E_spike") == 1.39e-12 and v("65nm", "t_read") == 100e-9 and DEVICE["levels"]["value"] == 16


def test_mac_counts_match_analytic_shapes():
    cfg, V = ToyConfig(), 3226
    m = em.mac_counts(cfg, V)
    assert m["head"] == 128 * 3226 and m["proj_qkv"] == 2 * 3 * 128 * 128 and m["ffn_13"] == 2 * 2 * 128 * 256
    assert m["core"] == 2 * 4 * 4 * 32 * 32 and m["gates"] == 2 * 2 * 128 * 4
    assert m["params_embedding"] == 3226 * 128
    assert m["params_stream"] == 2 * (4 * 128 * 128 + 3 * 128 * 256 + 2 * 128 * 4 + 2 * 4) + 128 * 3226   # 不含嵌入表
    assert m["total_macs"] == sum(m[k] for k in ("proj_qkv", "proj_o", "ffn_13", "ffn_2", "gates", "core", "head"))


def test_row_conductance_sums_both_cells_of_each_weight():
    W = np.array([[1.0, -0.5], [0.0, 1.0]])                   # d_out=2, d_in=2;列 0:+1 与 0;列 1:−0.5 与 +1
    npos, nneg, _, _ = xb.quantize_weights(W, 16)
    Gp, Gn = xb.conductance_targets(npos, nneg, 16)
    g = em.row_conductance(W, 16, DEVICE["G_on"]["value"], DEVICE["G_off"]["value"])
    assert np.allclose(g, (Gp + Gn).sum(0)) and g.shape == (2,)
    assert g[1] > g[0]                                         # 行 1 有一个负权重与一个正权重,两个 cell 都导通


def test_crossbar_energy_formula_on_one_row():
    c = np.array([2.0, 0.0]); grow = np.array([1e-4, 1e-4])
    e = em.crossbar_energy(c, grow, d_out=4, col="65nm")
    V, t = v("65nm", "V_read"), v("65nm", "t_read")
    assert np.isclose(e["conduction"], V * V * t * 2.0 * 1e-4)
    assert np.isclose(e["gate"], 2.0 * 8 * v("65nm", "C_gate") * v("65nm", "V_sel") ** 2)
    assert np.isclose(e["wire"], 2.0 * 8 * v("65nm", "C_wire") * V * V)
    assert np.isclose(e["leak_off"], (16 - 16) * V * v("65nm", "I_off") * t) and e["leak_off"] >= 0
    assert np.isclose(e["integrator"], 4 * v("65nm", "I_int") * v("65nm", "V_dd") * t)
    assert np.isclose(e["total"], sum(e[k] for k in ("conduction", "gate", "wire", "leak_off", "integrator")))


def test_digital_dense_batch_amortises_only_weight_fetch():
    cfg, V = ToyConfig(), 3226
    e1, e64 = em.digital_dense_energy(cfg, V, "65nm", 1), em.digital_dense_energy(cfg, V, "65nm", 64)
    assert np.isclose(e64["weight"], e1["weight"] / 64) and e64["mac"] == e1["mac"] and e64["act"] == e1["act"]
    assert np.isclose(e1["weight"], em.mac_counts(cfg, V)["params_stream"] * 8 * v("65nm", "E_sram_far_bit"))
    assert e1["weight"] > 10 * e1["mac"]                       # batch 1 时权重取数支配


def _synthetic_events(cfg, V):
    d, dff, H = cfg.d, cfg.d_ff, cfg.n_heads
    rows = {}
    for i in range(cfg.n_blocks):
        rows[f"blocks.{i}.qkv"] = np.full(d, 0.5); rows[f"blocks.{i}.gates"] = np.full(d + 1, 0.5)
        rows[f"blocks.{i}.o"] = np.full(d, 0.25); rows[f"blocks.{i}.ffn13"] = np.full(d, 0.5); rows[f"blocks.{i}.w2"] = np.full(dff, 0.1)
    rows["head"] = np.full(d, 0.5)
    return dict(n_tokens=100, levels=1, rows=rows, embed_rows=np.full(V, 1.0 / V), spikes_lif=1700.0, n_lif=3200,
                cmp_decisions=5 * d * 2, gain=dict(k_rows=2 * H * 10.0, q_rows=2 * H * 6.0, alpha_mean=0.8, s_rms_norm=0.3))


def _synthetic_mats(cfg, V):
    rng = np.random.default_rng(0); d, dff, H = cfg.d, cfg.d_ff, cfg.n_heads
    mats = {}
    for i in range(cfg.n_blocks):
        mats[f"blocks.{i}.qkv"] = rng.normal(0, 0.5, (3 * d, d)); mats[f"blocks.{i}.gates"] = rng.normal(0, 0.5, (2 * H, d + 1))
        mats[f"blocks.{i}.o"] = rng.normal(0, 0.5, (d, d)); mats[f"blocks.{i}.ffn13"] = rng.normal(0, 0.5, (2 * dff, d)); mats[f"blocks.{i}.w2"] = rng.normal(0, 0.5, (d, dff))
    mats["head"] = rng.normal(0, 0.5, (V, d)); mats["embed"] = rng.normal(0, 0.5, (d, V))
    return mats


def test_memristor_energy_breakdown_adds_up_and_head_variants_differ():
    cfg, V = ToyConfig(), 3226
    ev, mats = _synthetic_events(cfg, V), _synthetic_mats(cfg, V)
    a = em.memristor_energy(ev, mats, cfg, V, "65nm", head="crossbar_adc")
    b = em.memristor_energy(ev, mats, cfg, V, "65nm", head="digital_event")
    for r in (a, b):
        assert np.isclose(r["total"], sum(r["by_component"].values())) and all(x >= 0 for x in r["by_component"].values())
    assert np.isclose(a["by_component"]["adc"], V * v("65nm", "E_adc")) and a["by_component"]["head_digital"] == 0
    assert b["by_component"]["adc"] == 0 and b["by_component"]["head_digital"] > 0 and b["by_component"]["crossbar_head"] == 0
    assert np.isclose(a["by_component"]["adm_spikes"], 1700.0 * v("65nm", "E_spike"))
    assert np.isclose(a["by_component"]["adm_leak"], 3200 * v("65nm", "V_gs") ** 2 / v("65nm", "R_c") * v("65nm", "t_token"))
    assert np.isclose(a["by_component"]["comparators"], ev["cmp_decisions"] * v("65nm", "E_cmp"))
    assert set(a["by_matrix"]) == set(mats)


def test_digital_event_energy_scales_with_driven_rows():
    cfg, V = ToyConfig(), 3226
    ev = _synthetic_events(cfg, V)
    e = em.digital_event_energy(ev, cfg, V, "65nm", batch=1)
    adds = sum(ev["rows"][k].sum() * {"qkv": 3 * cfg.d, "gates": 2 * cfg.n_heads, "o": cfg.d, "ffn13": 2 * cfg.d_ff, "w2": cfg.d}[k.split(".")[-1]]
               for k in ev["rows"] if k != "head") + ev["rows"]["head"].sum() * V
    assert np.isclose(e["add"], adds * v("65nm", "E_add16")) and np.isclose(e["weight"], adds * 8 * v("65nm", "E_sram_far_bit"))
    assert np.isclose(em.digital_event_energy(ev, cfg, V, "65nm", batch=16)["weight"], e["weight"] / 16)
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_energy_estimate.py -q`
Expected: `ModuleNotFoundError: experiments.energy_model`(或 `energy_params`)。

- [ ] **Step 3: 实现 `energy_params.py` 与 `energy_model.py`**

按 Interfaces 的公式;`memristor_energy` 里每个 crossbar(含 `embed`、`head`)都经 `crossbar_energy`,`n_pulses_window = events["levels"]`;`gate_wire` 部件 = 各矩阵 gate + wire 之和,`integrator`、`leak_off` 同理汇总,`crossbar_blocks` = 块内矩阵的 conduction 之和,`crossbar_head` = 头的 conduction(`digital_event` 时头不走 crossbar,四项都为 0),`embed` = 嵌入 crossbar 的 conduction。

- [ ] **Step 4: 运行,确认通过**

Run: 同 Step 2。Expected: `7 passed`。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/energy_params.py step2-toy-demo/experiments/energy_model.py step2-toy-demo/tests/test_energy_estimate.py
git commit -m "energy: parameter table with sources and the per-token energy formulas (memristor, digital dense, digital event-driven)"
```

---

### Task 2: 事件计数(`energy_events.py`)

**Files:**
- Create: `step2-toy-demo/experiments/energy_events.py`
- Test: `step2-toy-demo/tests/test_energy_estimate.py`(追加)

**Interfaces:**
- Consumes: `toy_demo.data.split_inputs_targets(tokens) -> (inputs, targets, mask)`;模型 `model(inputs, record=True) -> (logits, recs)`,`recs[t][i]` 为块 i 的记录(键 `x_attn, q, k, v, o, o_spk, y_attn, x_ffn, g, u, p, y_ffn, alpha, beta, S`),`recs[t][-1]["h_final"]`;`SpikingLM`:`model.q_out`(头的输入 = `model.q_out(h_final)`)、`blk.q_in1/q_in2`(`.levels`)、`blk.attn.{Wq,Wk,Wv,Wo,gates.Wa,gates.Wb,c_k,sn_q..}`、`blk.ffn.{W1,W3,W2}`;`TernaryResidualLM`:`model.q_emb`、`blk.m1/m2`,头的输入 = `h_final`;`experiments.beam_eval.load_model`。
- Produces:
  - `count_events(model, tokens: torch.Tensor, dev, batch: int = 64) -> dict`,返回 Task 1 约定的 `events`(全部为每 token 平均,按 `mask` 只计有效位置;`n_tokens` 为有效 token 数)。计数规则:`levels = T`;每个脉冲向量 s 的行脉冲数 `c = round(|s|·T)`;`blocks.i.qkv` 与 `blocks.i.gates` 都来自 `x_attn`(`gates` 向量长 d + 1,最后一项 = 1.0,对应偏置行);`blocks.i.o` 来自 `o_spk`;`blocks.i.ffn13` 来自 `x_ffn`;`blocks.i.w2` 来自 `g` 与 `u`:`c = round(|g|·T)·round(|u|·T)`;`head` 来自 `q_out(h_final)`(SpikingLM)或 `h_final`(ternres);`embed_rows[tok] += 1` 按输入 token;`spikes_lif` = 8 个 LIF 点(`q, k/c_k, v, o_spk, y_attn, g, u, y_ffn`)的 c 之和;`n_lif = n_blocks·(3·d + d + d + 2·d_ff + d)`(= 每块 8 个点的神经元数);`cmp_decisions`:SpikingLM `(n_blocks·2·d + d)·2T`,ternres `(d + n_blocks·2·d)·2T`;`gain.k_rows`:每块每头 `k` 的非零计数之和(c 求和)、`q_rows` 同理(对块、头求和,每 token 平均);`alpha_mean`、`s_rms_norm = rms(S) / max|S|`(全程)。
  - `model_matrices(model) -> dict[str, np.ndarray]`:Task 1 约定的 `mats`(`gates` 把 `Wa`、`Wb` 的权重竖着拼成 `[2H, d]` 再拼偏置列;`embed = model.emb.weight.T`;`head = model.head.weight`)。
  - `model_kind(model) -> str`:`"spiking"` 或 `"ternres"`(按 `hasattr(model, "q_emb")`)。

- [ ] **Step 1: 追加失败的测试**

```python
from experiments.energy_events import count_events, model_matrices, model_kind
from experiments.multilevel import apply_levels
from toy_demo.data import all_sequences, split_inputs_targets
from toy_demo.model import SpikingLM
from toy_demo.ternary_residual import TernaryResidualLM


def _lower_thresholds(m):
    with torch.no_grad():
        for lif in m.lif_modules():
            lif._thr.log_thr.fill_(-1.0)


def test_count_events_matches_manual_recount(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    tok = all_sequences(sl)[:12]
    ev = count_events(m, tok, torch.device("cpu"), batch=5)
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)[mask]
    assert ev["n_tokens"] == int(mask.sum()) and ev["levels"] == 1
    assert np.allclose(ev["rows"]["blocks.0.qkv"], x.abs().mean(0).numpy())
    assert np.allclose(ev["rows"]["blocks.0.gates"][:-1], x.abs().mean(0).numpy()) and ev["rows"]["blocks.0.gates"][-1] == 1.0
    g = torch.stack([recs[t][0]["g"] for t in range(inputs.shape[1])], 1)[mask]; u = torch.stack([recs[t][0]["u"] for t in range(inputs.shape[1])], 1)[mask]
    assert np.allclose(ev["rows"]["blocks.0.w2"], (g.abs() * u.abs()).mean(0).numpy())
    hf = torch.stack([recs[t][-1]["h_final"] for t in range(inputs.shape[1])], 1)[mask]
    assert np.allclose(ev["rows"]["head"], m.q_out(hf).abs().mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.n_blocks * 2 * cfg.d + cfg.d) * 2 and ev["n_lif"] == cfg.n_blocks * (6 * cfg.d + 2 * cfg.d_ff)
    assert np.isclose(ev["embed_rows"].sum(), 1.0) and ev["embed_rows"].shape == (sl.vocab_size,)
    assert 0 < ev["gain"]["alpha_mean"] < 1 and 0 <= ev["gain"]["s_rms_norm"] <= 1 and ev["gain"]["k_rows"] >= 0
    assert ev["spikes_lif"] > 0


def test_count_events_levels_two_counts_pulses_and_w2_products(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); apply_levels(m, 2); _lower_thresholds(m)
    tok = all_sequences(sl)[:8]
    ev = count_events(m, tok, torch.device("cpu"))
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)[mask]
    g = torch.stack([recs[t][0]["g"] for t in range(inputs.shape[1])], 1)[mask]; u = torch.stack([recs[t][0]["u"] for t in range(inputs.shape[1])], 1)[mask]
    assert ev["levels"] == 2
    assert np.allclose(ev["rows"]["blocks.0.qkv"], (x.abs() * 2).round().mean(0).numpy())
    assert np.allclose(ev["rows"]["blocks.0.w2"], ((g.abs() * 2).round() * (u.abs() * 2).round()).mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.n_blocks * 2 * cfg.d + cfg.d) * 4


def test_count_events_ignores_padding(cfg, sl):
    import pytest
    torch.manual_seed(1)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    tok = all_sequences(sl)[:6]
    inputs, _, mask = split_inputs_targets(tok)
    if mask.all():
        pytest.skip("切片里这 6 句等长,没有填充可测")
    ev = count_events(m, tok, torch.device("cpu"))
    _, recs = m(inputs, record=True)
    x = torch.stack([recs[t][0]["x_attn"] for t in range(inputs.shape[1])], 1)
    assert ev["n_tokens"] == int(mask.sum())
    assert np.allclose(ev["rows"]["blocks.0.qkv"], x[mask].abs().mean(0).numpy())                      # 只计有效位置
    assert not np.allclose(ev["rows"]["blocks.0.qkv"], x.reshape(-1, cfg.d).abs().mean(0).numpy())    # 含填充的平均不同


def test_ternres_events_use_h_final_as_head_input(cfg, sl):
    torch.manual_seed(0)
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.9, merge="sat", levels=1); _lower_thresholds(m)
    tok = all_sequences(sl)[:8]
    ev = count_events(m, tok, torch.device("cpu"))
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = m(inputs, record=True)
    hf = torch.stack([recs[t][-1]["h_final"] for t in range(inputs.shape[1])], 1)[mask]
    assert model_kind(m) == "ternres" and np.allclose(ev["rows"]["head"], hf.abs().mean(0).numpy())
    assert ev["cmp_decisions"] == (cfg.d + cfg.n_blocks * 2 * cfg.d) * 2


def test_model_matrices_shapes(cfg, sl):
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9)
    mats = model_matrices(m)
    d, dff, H, V = cfg.d, cfg.d_ff, cfg.n_heads, sl.vocab_size
    assert mats["blocks.0.qkv"].shape == (3 * d, d) and mats["blocks.0.gates"].shape == (2 * H, d + 1)
    assert mats["blocks.0.o"].shape == (d, d) and mats["blocks.0.ffn13"].shape == (2 * dff, d) and mats["blocks.0.w2"].shape == (d, dff)
    assert mats["head"].shape == (V, d) and mats["embed"].shape == (d, V)
    assert np.allclose(mats["blocks.0.gates"][:H, -1], m.blocks[0].attn.gates.Wa.bias.detach().numpy())
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_energy_estimate.py -q -k "count_events or ternres_events or model_matrices"`
Expected: `ModuleNotFoundError: experiments.energy_events`。

- [ ] **Step 3: 实现 `energy_events.py`**

分批前向(`batch` 句一批,`torch.no_grad()`),累加各行脉冲数与 token 数,最后除以 `n_tokens`;`k` 的计数先除以 `blk.attn.c_k`;`gain` 的 k_rows / q_rows 按头切片 `k.view(-1, H, dk)`。

- [ ] **Step 4: 运行,确认通过**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_energy_estimate.py -q`。Expected: `12 passed`。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/energy_events.py step2-toy-demo/tests/test_energy_estimate.py
git commit -m "energy: per-token event counts from the models' records (driven rows, LIF spikes, comparators, gain cells, embedding rows)"
```

---

### Task 3: GPU 实测(`energy_gpu.py`)

**Files:**
- Create: `step2-toy-demo/experiments/energy_gpu.py`
- Test: `step2-toy-demo/tests/test_energy_estimate.py`(追加,只测纯函数)
- Create: `step2-toy-demo/results/energy/gpu.json`(运行产物)

**Interfaces:**
- Consumes: `experiments.beam_eval.load_model`、`toy_demo.data.{load_slice, all_sequences, batches, split_inputs_targets}`、`toy_demo.config.ToyConfig`。
- Produces:
  - `parse_power_lines(lines: list[str]) -> list[float]`(`nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits` 的行 → 瓦)。
  - `joules_per_token(mean_w: float, idle_w: float, seconds: float, tokens: int) -> dict(total=..., incremental=..., ms_per_token=...)`。
  - `sample_power(gpu_index: int, stop_event) -> list[float]`:子进程 `nvidia-smi -i {gpu_index} --query-gpu=power.draw --format=csv,noheader,nounits -lms 100`,线程里读行直到 `stop_event`。
  - `measure(model, tokens, dev, batch: int, seconds: float, gpu_index: int, idle_w: float) -> dict(batch, seconds, tokens, mean_w, idle_w, total_j_per_token, incremental_j_per_token, ms_per_token, samples)`:`batch == 1` 时逐句 `model(sent[None, :-1])`,否则用 `batches(sl, batch, g)` 的批;循环直到 ≥ `seconds`;token 数按有效位置计。
  - CLI:`--models A0=results/ckpt/a0.pt,T2=results/improve/T2_c/model.pt --batches 1,64 --seconds 60 --idle-seconds 20 --gpu-index 1 --save results/energy/gpu.json`;先测空载 `idle_w`,再每个模型每个 batch 测一次;JSON 顶层 `dict(gpu=名字, idle_w, runs={模型: {batch: 结果}})`。

- [ ] **Step 1: 追加失败的测试**

```python
from experiments.energy_gpu import joules_per_token, parse_power_lines


def test_parse_power_lines_and_joules_per_token():
    assert parse_power_lines(["20.08", " 25.9 ", "", "N/A"]) == [20.08, 25.9]
    r = joules_per_token(mean_w=60.0, idle_w=20.0, seconds=10.0, tokens=4000)
    assert np.isclose(r["total"], 600.0 / 4000) and np.isclose(r["incremental"], 400.0 / 4000) and np.isclose(r["ms_per_token"], 2.5)
```

- [ ] **Step 2: 运行,确认失败** — `-k joules`,Expected `ModuleNotFoundError: experiments.energy_gpu`。

- [ ] **Step 3: 实现脚本**

- [ ] **Step 4: 单元测试通过;在 GPU 1 上实测**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_energy_estimate.py -q` → `13 passed`。
Run: `CUDA_VISIBLE_DEVICES=1 $PY experiments/energy_gpu.py --models A0=results/ckpt/a0.pt,T2=results/improve/T2_c/model.pt --batches 1,64 --seconds 60 --idle-seconds 20 --gpu-index 1 --save results/energy/gpu.json`
Expected:约 5 分钟;`gpu.json` 四个结果,`idle_w` 约 20 W,batch 64 的 `incremental_j_per_token` 明显小于 batch 1;报告里写下四个 J/token 与 ms/token。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/energy_gpu.py step2-toy-demo/tests/test_energy_estimate.py step2-toy-demo/results/energy/gpu.json
git commit -m "energy: measured GPU energy per token for A0 and T=2 C on an RTX A4500 (batch 1 and 64)"
```

---

### Task 4: 整合 CLI、敏感性、嵌入量化检查(`energy_estimate.py`)

**Files:**
- Create: `step2-toy-demo/experiments/energy_estimate.py`
- Create: `step2-toy-demo/results/energy/energy.json`、`results/energy/tables.md`(运行产物)
- Test: `step2-toy-demo/tests/test_energy_estimate.py`(追加)

**Interfaces:**
- Consumes: Task 1–3 的全部函数;`experiments.beam_eval.load_model`;`toy_demo.train.teacher_forced_metrics(model, tokens)`;`experiments.spice.crossbar.quantize_weights`。
- Produces:
  - `embedding_quantization_check(model, tokens, levels=16) -> dict(acc_before, acc_after, delta)`:把 `model.emb.weight` 按 `quantize_weights`(全局 w_max)量化到 `levels` 级后重新评估(改完要**还原**权重)。
  - `sensitivity(events, mats, cfg, vocab) -> dict`:主列上分别改 `t_read ∈ {10e-9, 100e-9, 1e-6}`、`G 缩放 ∈ {0.1, 1}`(对 `G_on`、`G_off` 同乘)、`E_adc ∈ {1e-12, 5e-12, 10e-12}` 的 memristor 总能量(`crossbar_adc` 头);实现方式:`energy_model` 的公式接受可选 `overrides: dict` 覆盖列参数(Task 1 的函数签名末尾加 `overrides=None`;builder 在 Task 4 里补这个参数并保持 Task 1 测试通过)。
  - `main()`:`--models T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt,T2=results/improve/T2_c/model.pt,TR2=results/improve/ternres_sat_T2_c/model.pt --a0 results/ckpt/a0.pt --out results/energy --gpu-json results/energy/gpu.json --device cuda`;对每个脉冲模型:`count_events`(GPU 1)、`model_matrices`、两列 × 两个头变体的 `memristor_energy`、`digital_event_energy`(两列 × batch 1/16/64)、`embedding_quantization_check`、`sensitivity`;`digital_dense_energy` 用 A0 的尺寸(两列 × batch 1/16/64);读入 `gpu.json`;写 `energy.json`(全部原始数,含 events 的汇总量与参数表)和 `tables.md`:
    表 1 主表(行:memristor 65 nm × {T1, T2, TR2} × {头 crossbar+ADC, 头数字};数字稠密 65 nm × batch {1,16,64},外加一行"batch 1、权重在 DRAM"(`weight` 项用 `E_dram_bit`)作对照;数字事件驱动 65 nm × {T1,T2,TR2} × batch {1,16,64};GPU A0 / T2 × batch {1,64} 的 total 与 incremental;列:J/token、相对 memristor T2(crossbar+ADC)的倍数);
    表 2 memristor 65 nm 的部件分解(每个模型一列,部件一行,nJ 与百分比);
    表 3 展望列(memristor 3 nm 投影 vs 数字 7 nm,batch 1/64);
    表 4 敏感性;表 5 嵌入 16 级量化的准确率变化。
    单位:nJ/token 三位有效数字;GPU 用 mJ/token。

- [ ] **Step 1: 追加失败的测试**

```python
from experiments.energy_estimate import embedding_quantization_check, sensitivity


def test_embedding_quantization_check_restores_weights(cfg, sl):
    torch.manual_seed(0)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9); _lower_thresholds(m)
    before = m.emb.weight.detach().clone()
    r = embedding_quantization_check(m, all_sequences(sl)[:16], levels=16)
    assert set(r) == {"acc_before", "acc_after", "delta"} and torch.equal(m.emb.weight.detach(), before)


def test_sensitivity_moves_with_t_read_and_adc():
    cfg, V = ToyConfig(), 3226
    s = sensitivity(_synthetic_events(cfg, V), _synthetic_mats(cfg, V), cfg, V)
    assert s["t_read"]["1e-06"] > s["t_read"]["1e-07"] > s["t_read"]["1e-08"]
    assert s["E_adc"]["1e-11"] > s["E_adc"]["1e-12"]
    assert s["G_scale"]["0.1"] < s["G_scale"]["1"]
```

- [ ] **Step 2: 运行,确认失败** — Expected `ModuleNotFoundError: experiments.energy_estimate`。

- [ ] **Step 3: 实现脚本(与 Task 1 的 `overrides` 参数)**

- [ ] **Step 4: 测试通过;真实数据运行**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_energy_estimate.py -q` → `15 passed`;并跑 step2 全套 `CUDA_VISIBLE_DEVICES="" $PY -m pytest -q`(46 + 15 passed)。
Run: `CUDA_VISIBLE_DEVICES=1 $PY experiments/energy_estimate.py --models T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt,T2=results/improve/T2_c/model.pt,TR2=results/improve/ternres_sat_T2_c/model.pt --a0 results/ckpt/a0.pt --out results/energy --gpu-json results/energy/gpu.json --device cuda`
Expected:几分钟;`energy.json` 与 `tables.md` 写出;抽查:T2 的 `events["spikes_lif"]` 与 `results/improve/profile_final.json` 里 T2_c 的 `total_spikes_per_token`(2996)减去三个比较器点后的量级一致(报告里写两个数);数字稠密 batch 1 的 `weight` ≈ 0.74M × 8 × E_sram_far_bit。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/energy_estimate.py step2-toy-demo/experiments/energy_model.py step2-toy-demo/tests/test_energy_estimate.py step2-toy-demo/results/energy/energy.json step2-toy-demo/results/energy/tables.md
git commit -m "energy: end-to-end estimate on the three spiking models, sensitivity scan, embedding-quantisation check"
```

---

### Task 5: README 与结论

**Files:**
- Create: `step2-toy-demo/results/energy/README.md`
- Modify: `step2-toy-demo/results/improve/README.md`(文末一行指向 `../energy/README.md`)

**Interfaces:**
- Consumes: `results/energy/energy.json`、`tables.md`、`gpu.json`;`energy_params.py` 的参数与 source。

- [ ] **Step 1: 写 README**

内容(中文,`results/improve/README.md` 的风格):问题与对象;工艺对齐的说明(65 nm 主列、器件量不缩、展望列不同节点、GPU 单列);参数表(名、值、单位、来源,直接从 `energy_params.py` 生成);表 1–5(从 `tables.md` 粘);结论:memristor 相对数字 ASIC 的倍数在 batch 1 和 64 下各是多少、优势来自哪里(事件驱动数字行与稠密行的差 = 稀疏 + 三值;事件驱动数字行与 memristor 行的差 = 存内计算省掉的权重取数)、memristor 的开销在哪里(输出头 ADC、栅驱动 / 导线、crossbar 导通的占比)、"输出头数字"的混合方案为何不划算、GPU 数字的含义;嵌入 16 级量化对准确率的影响与(若 > 0.5 个百分点)脉冲注入退路的能耗;没做的(spec 的非目标 + 未验证的 int8 精度 + 展望列为估计)。每个数字标明出自哪个 JSON 键。

- [ ] **Step 2: 自查数字** — 抽 5 个 README 数字对照 `energy.json`,写进报告。

- [ ] **Step 3: 提交并推送分支**

```bash
git add step2-toy-demo/results/energy/README.md step2-toy-demo/results/improve/README.md docs/superpowers/specs/2026-10-28-energy-estimate-design.md docs/superpowers/plans/2026-10-29-energy-estimate.md
git commit -m "energy: results README — memristor vs digital ASIC vs GPU per-token energy for the toy GDN SNN"
git push origin worktree-conversion-sweep
```
