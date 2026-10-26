# 三值残差流 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建一个残差流全程三值的脉冲块与语言模型,在 toy demo 上按现有 A0 → A1 → B → C 链训练 {sat, and} × {T=1, T=2} 四条,和现有 C 比精度。

**Architecture:** `snn_spec.ternary_residual.TernaryResidualBlock` 复用现有 `SpikingGDN`、`SpikingSwiGLU`,用两个 `ResidualMerge`(三值加法 + 阈值)代替实值累加器与 Q_in。`toy_demo.ternary_residual.TernaryResidualLM` 在嵌入后加一个 `QuantizerIn`,输出头直接读三值残差。`experiments/ternary_residual.py` 复用 `convert_lab` 的训练循环、评估与保存,提供 a1 / c / iscan / bscan / chain。现有模型、转换、流水线代码不改。

**Tech Stack:** Python 3.10 venv `/data/chaoyi_he/dm_snn/step2-toy-demo/.venv`(torch 2.14 cu126),pytest,`snn_spec`(由 `toy_demo/__init__.py` 指向同一代码树的 `step1-spiking-layer`)。

**Spec:** `docs/superpowers/specs/2026-10-25-ternary-residual-design.md`

## Global Constraints

- 代码树:worktree `/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep`,分支 `worktree-conversion-sweep`。step1 命令在 `step1-spiking-layer/` 下跑,step2 命令在 `step2-toy-demo/` 下跑,都用 `PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python`。
- 不改:`snn_spec/block.py`、`snn_spec/gdn.py`、`snn_spec/neurons.py`、`toy_demo/model.py`、`toy_demo/convert.py`、`toy_demo/run.py`、`toy_demo/config.py`、`experiments/convert_lab.py`、`experiments/multilevel.py`。`experiments/beam_eval.py` 只加识别新模型的分支。
- 测试在 CPU 上跑(`CUDA_VISIBLE_DEVICES=""`);训练只用 GPU 1(`CUDA_VISIBLE_DEVICES=1`),GPU 0 留给用户的扩散项目。
- 合并规则阈值:`sat` 1.0 定死,`and` 1.5 定死,`learn` 初值 1.0 可学;M 不做分位数标定。T = `levels`,残差与脉冲同级数。
- 训练配方:lr 3e-3、上限 12000 步、`eval_every` 100、停止规则不变;fq 候选 `0.5,0.75,0.9`;LIF 倍数候选 `1,1.5,2,2.25,2.5,3,4`。
- 结果目录 `step2-toy-demo/results/improve/`,前缀 `ternres_`。
- 每个任务结束提交到 `worktree-conversion-sweep` 并推送该分支;不推 main。提交信息结尾两行:`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` 与 `Claude-Session: https://claude.ai/code/session_01WXSmKMbtt8Re7oceEtjWmJ`。
- 中文 docstring,与现有代码一致。

## Review Focus

1. T = 2 时两个 0.5 步长的量相加落在 0.5 网格上,`sat` 必须正好是 clamp、`and` 的 1.5 要恰好触发第二级 → Task 1 `test_merge_levels_two_clamps_half_steps`。
2. 残差不再是恒等通道,梯度只能穿过代理函数;若到不了嵌入和 W_q,训练根本不会动 → Task 1 `test_gradient_reaches_input_and_weights_through_the_ternary_residual`、Task 2 `test_gradient_flows_to_embedding_through_ternary_residual`。
3. A1(leak 0)的 state_dict 必须严格装进 C(leak 0.9)的同构模型,含 levels = 2 → Task 2 `test_a1_state_loads_into_leaky_c_model`。
4. 共用的 profile / beam 脚本按 result.json 的 args 认模型,认不出就把新检查点当旧模型装载而报键不匹配 → Task 3 `test_describe_and_load_checkpoint_recognise_ternary_residual`。
5. 子块不发放时 `sat` 应把残差原样传过去(否则"残差"名不副实)→ Task 1 `test_silent_subblocks_pass_the_residual_through`。

---

## File Structure

- Create `step1-spiking-layer/snn_spec/ternary_residual.py`:`ResidualMerge`(两输入三值合并)、`TernaryResidualBlock`(一层)。
- Create `step1-spiking-layer/tests/test_ternary_residual.py`。
- Create `step2-toy-demo/toy_demo/ternary_residual.py`:`TernaryResidualLM`、`block_points`、`named_points`、`calibrate_thresholds`、`spike_counts`。
- Create `step2-toy-demo/tests/test_ternary_residual.py`。
- Modify `step2-toy-demo/experiments/beam_eval.py`:`describe_checkpoint`、`load_model` 各加一个分支。
- Create `step2-toy-demo/experiments/ternary_residual.py`:子命令 a1 / c / iscan / bscan / chain。
- Create `step2-toy-demo/run_ternres.sh`:4 条链的后台启动脚本。
- Create `step2-toy-demo/experiments/ternres_tables.py`:汇总表。
- Modify `step2-toy-demo/results/improve/README.md`:新节"三值残差流"。

---

### Task 1: `ResidualMerge` 与 `TernaryResidualBlock`(step1)

**Files:**
- Create: `step1-spiking-layer/snn_spec/ternary_residual.py`
- Test: `step1-spiking-layer/tests/test_ternary_residual.py`

**Interfaces:**
- Consumes: `snn_spec.neurons.level_ternary_fire(U, thr, levels, sg_width)`、`QuantizerIn`、`snn_spec.gdn.SpikingGDN`、`snn_spec.block.SpikingSwiGLU`。
- Produces:
  - `ResidualMerge(merge="sat", levels=1, sg_width=2.0)`:`nn.Module`,属性 `merge: str`、`levels: int`、`log_thr: nn.Parameter`(`requires_grad` 仅 `learn` 为 True)、`thr` 属性;`forward(a, b) -> Tensor` = `level_ternary_fire(a + b, thr, levels, sg_width)`。`merge` 不在 `("sat", "and", "learn")` 内抛 `ValueError`。
  - `TernaryResidualBlock(d, d_ff, n_k_heads, n_v_heads, dk, dv, gates=None, fir_taps=0, alpha_rng=(0.5, 0.99), leak=0.9, thr=1.0, learn_thr=False, merge="sat", levels=1)`:属性 `attn`、`ffn`、`m1`、`m2`、`merge`、`levels`;构造时把 8 个 LIF 的 `.levels` 设为 `levels`。`init_state(B, device) -> dict(attn=..., ffn=...)`;`step(r, st, ext_gates=None, record=False) -> (r_out, st, rec)`,rec 键与 `SpikingBlock` 相同(`h_in, x_attn, alpha, beta, q, k, v, S, o, o_spk, y_attn, h_mid, x_ffn, g, u, p, y_ffn, h_out`)另加 `s_mid = r + y1`、`s_out = h_mid + y2`;`lif_modules() -> list` 按 `sn_q, sn_k, sn_v, sn_pre, sn_out, sn_1, sn_3, sn_2` 顺序。

- [ ] **Step 1: 写失败的测试**

```python
"""残差流全程三值的一层:两输入合并 M 的真值表、块的输出取值、梯度能否穿过。"""
import itertools
import torch

from snn_spec.neurons import QuantizerIn
from snn_spec.ternary_residual import ResidualMerge, TernaryResidualBlock

GRID = torch.tensor(list(itertools.product((-1.0, 0.0, 1.0), repeat=2)))
A, B = GRID[:, 0], GRID[:, 1]


def test_sat_merge_is_saturating_add():
    assert torch.equal(ResidualMerge("sat")(A, B), torch.sign(A + B))


def test_and_merge_keeps_only_agreeing_signs():
    expected = torch.where((A + B).abs() == 2, torch.sign(A + B), torch.zeros_like(A))
    assert torch.equal(ResidualMerge("and")(A, B), expected)


def test_merge_levels_two_clamps_half_steps():
    grid = torch.tensor(list(itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0), repeat=2)))
    a, b = grid[:, 0], grid[:, 1]
    assert torch.equal(ResidualMerge("sat", levels=2)(a, b), torch.clamp(a + b, -1.0, 1.0))
    m = ResidualMerge("and", levels=2)                      # 阈值 0.75、1.5
    pairs = [((0.5, 0.0), 0.0), ((0.5, 0.5), 0.5), ((1.0, 0.5), 1.0), ((1.0, 1.0), 1.0), ((-1.0, -0.5), -1.0), ((1.0, -0.5), 0.0)]
    for (x, y), want in pairs:
        assert m(torch.tensor([x]), torch.tensor([y])).item() == want, (x, y)


def test_fixed_merges_have_frozen_threshold_and_learn_does_not():
    assert ResidualMerge("sat").thr.item() == 1.0 and not ResidualMerge("sat").log_thr.requires_grad
    assert ResidualMerge("and").thr.item() == 1.5 and not ResidualMerge("and").log_thr.requires_grad
    assert ResidualMerge("learn").thr.item() == 1.0 and ResidualMerge("learn").log_thr.requires_grad
    try:
        ResidualMerge("max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown merge must raise")


def _block(**kw):
    torch.manual_seed(0)
    return TernaryResidualBlock(d=16, d_ff=32, n_k_heads=2, n_v_heads=2, dk=4, dv=4, **kw)


def test_block_residual_stream_is_ternary_and_merges_are_used():
    blk = _block(merge="sat")
    st = blk.init_state(3, torch.device("cpu"))
    r = torch.randint(-1, 2, (3, 16)).float()
    r_out, st, rec = blk.step(r, st, record=True)
    for key in ("x_attn", "h_mid", "x_ffn", "h_out", "y_attn", "y_ffn"):
        assert set(rec[key].unique().tolist()) <= {-1.0, 0.0, 1.0}, key
    assert torch.equal(rec["x_attn"], r) and torch.equal(rec["x_ffn"], rec["h_mid"]) and torch.equal(r_out, rec["h_out"])
    assert torch.equal(rec["h_mid"], blk.m1(rec["x_attn"], rec["y_attn"]))
    assert torch.equal(rec["h_out"], blk.m2(rec["h_mid"], rec["y_ffn"]))
    assert torch.equal(rec["s_mid"], r + rec["y_attn"]) and torch.equal(rec["s_out"], rec["h_mid"] + rec["y_ffn"])
    assert len(blk.lif_modules()) == 8


def test_block_levels_two_sets_every_spike_point_and_residual_to_half_steps():
    blk = _block(merge="sat", levels=2)
    assert all(m.levels == 2 for m in blk.lif_modules()) and blk.m1.levels == 2 and blk.m2.levels == 2
    r = torch.randint(-2, 3, (3, 16)).float() / 2
    _, _, rec = blk.step(r, blk.init_state(3, torch.device("cpu")), record=True)
    for key in ("h_mid", "h_out", "y_attn", "y_ffn", "q", "v"):
        v = rec[key]
        assert torch.equal((v * 2).round(), v * 2) and v.abs().max() <= 1.0, key


def test_silent_subblocks_pass_the_residual_through():
    blk = _block(merge="sat")
    with torch.no_grad():
        blk.attn.Wo.weight.zero_()
        blk.ffn.W2.weight.zero_()
    r = torch.randint(-1, 2, (3, 16)).float()
    r_out, _, rec = blk.step(r, blk.init_state(3, torch.device("cpu")), record=True)
    assert torch.all(rec["y_attn"] == 0) and torch.all(rec["y_ffn"] == 0)
    assert torch.equal(r_out, r)


def test_gradient_reaches_input_and_weights_through_the_ternary_residual():
    for merge in ("sat", "learn"):
        blk = _block(merge=merge, learn_thr=True)
        e = (torch.randn(3, 16) * 2).requires_grad_()
        r = QuantizerIn(0.5)(e)
        r_out, _, _ = blk.step(r, blk.init_state(3, torch.device("cpu")))
        (r_out ** 2).sum().backward()
        assert e.grad is not None and e.grad.abs().sum() > 0
        assert blk.attn.Wq.weight.grad.abs().sum() > 0 and blk.ffn.W1.weight.grad.abs().sum() > 0
        assert (blk.m1.log_thr.grad is not None) == (merge == "learn")
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step1-spiking-layer && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q`
Expected: 全部 ERROR/FAIL,`ModuleNotFoundError: snn_spec.ternary_residual`。

- [ ] **Step 3: 实现 `snn_spec/ternary_residual.py`**

`ResidualMerge.__init__`:`thr0 = {"sat": 1.0, "and": 1.5, "learn": 1.0}[merge]`,`self.log_thr = nn.Parameter(torch.log(torch.tensor(thr0)), requires_grad=(merge == "learn"))`。
`TernaryResidualBlock.step` 的流程:`y1, st["attn"], r1 = self.attn.step(r, st["attn"], ext_gates=ext_gates, record=record)`;`h_mid = self.m1(r, y1)`;`y2, st["ffn"], r2 = self.ffn.step(h_mid, st["ffn"], record=record)`;`r_out = self.m2(h_mid, y2)`。模块 docstring 写明与 `SpikingBlock` 的差别和电路含义(见 spec)。

- [ ] **Step 4: 运行,确认通过**

Run: 同 Step 2。Expected: `8 passed`。再跑 `CUDA_VISIBLE_DEVICES="" $PY -m pytest -q` 确认原有 130 项仍通过、1 项 xfail(SPICE 项约 5 分钟;可先加 `-k "not spice"` 看其余)。

- [ ] **Step 5: 提交**

```bash
git add step1-spiking-layer/snn_spec/ternary_residual.py step1-spiking-layer/tests/test_ternary_residual.py docs/superpowers/specs/2026-10-25-ternary-residual-design.md docs/superpowers/plans/2026-10-25-ternary-residual.md
git commit -m "snn_spec: TernaryResidualBlock, a layer whose residual stream is ternary (ResidualMerge sat/and/learn)"
```

---

### Task 2: `TernaryResidualLM`、脉冲点登记、标定、脉冲计数(step2)

**Files:**
- Create: `step2-toy-demo/toy_demo/ternary_residual.py`
- Test: `step2-toy-demo/tests/test_ternary_residual.py`

**Interfaces:**
- Consumes: `toy_demo.model._LMBase`(`init_state`、`forward`、`generate`、`device`)、`snn_spec.gdn.SigmoidGates(d, n_heads, alpha_rng)`、`toy_demo.convert._set_thr`、`toy_demo.data.split_inputs_targets`、Task 1 的 `TernaryResidualBlock`。
- Produces:
  - `TernaryResidualLM(cfg, vocab, leak, merge="sat", levels=1)`:`emb`、`q_emb = QuantizerIn(1.0, levels=levels)`、`blocks`(每块 `TernaryResidualBlock(cfg.d, cfg.d_ff, cfg.n_heads, cfg.n_heads, cfg.dk, cfg.dv, gates=SigmoidGates(cfg.d, cfg.n_heads, rng), alpha_rng=rng, leak=leak, thr=1.0, learn_thr=True, merge=merge, levels=levels)`)、`head = nn.Linear(cfg.d, vocab, bias=False)`、属性 `merge`、`levels`、`cfg`、`vocab`。覆写 `step(tok, st, record=False)`:`h = self.q_emb(self.emb(tok))` → 各块 → `logits = self.head(h)`;`record` 时 recs 末尾追加 `dict(h_final=h)`。`lif_modules()` 逐块 yield 8 个 LIF。
  - `block_points(blk) -> list[(name, module, fn(rec) -> preact, ternary)]`:`q, k, v, pre, out, g, u, y2`,lambda 与 `convert._block_points` 对应项相同。
  - `named_points(model) -> list[(full, name, module)]`:`("q_emb", "q_emb", model.q_emb)`,然后每块按 `q, k, v, pre, out, m1, g, u, y2, m2`,full 为 `blocks.{i}.{name}`。
  - `calibrate_thresholds(model, tokens, quantile=0.75) -> dict`:`q_emb` 用 `model.emb(inputs)[mask].abs()` 的分位数;每块 `block_points` 按前向顺序用 `convert.calibrate_thresholds` 同样的 `quantile_of` 逻辑;不碰 `m1`、`m2`。
  - `spike_counts(model, tokens, dev) -> dict`:与 `experiments.multilevel.spike_counts` 同样的字段(`points`、`mean_spikes_per_neuron_token`、`weighted_spikes_per_neuron_token`、`total_spikes_per_token`、`n_spiking_neurons`、`w2_events`、`w2_events_per_token`),点到记录键:`q_emb → blocks[0] 的 h_in`,`m1 → h_mid`,`m2 → h_out`,其余与 multilevel 相同(`k` 除以 `c_k`)。

- [ ] **Step 1: 写失败的测试**

```python
"""三值残差流语言模型:与 A0 的权重对接、前向与记录、脉冲点登记与标定、A1 → C 的装载、梯度。"""
import sys
sys.dont_write_bytecode = True

import toy_demo  # noqa: E402,F401
import torch
import torch.nn.functional as F

from toy_demo.convert import set_leak, transfer_weights
from toy_demo.data import EOS, all_sequences
from toy_demo.model import FloatLM
from toy_demo.ternary_residual import TernaryResidualLM, calibrate_thresholds, named_points, spike_counts

POINTS = ("q", "k", "v", "pre", "out", "m1", "g", "u", "y2", "m2")
CALIB = ("q", "k", "v", "pre", "out", "g", "u", "y2")


def test_weights_transfer_from_float_and_extra_keys_are_thresholds(cfg):
    f, m = FloatLM(cfg, 20), TernaryResidualLM(cfg, 20, leak=0.0, merge="sat")
    transfer_weights(f, m)
    extra = set(m.state_dict()) - set(f.state_dict())
    assert all("log_thr" in k for k in extra)
    assert {"q_emb.log_thr", "blocks.0.m1.log_thr", "blocks.0.m2.log_thr"} <= extra
    assert not any(k.startswith(("q_out", "blocks.0.q_in")) for k in m.state_dict())


def test_forward_record_generate_and_residual_is_ternary(cfg, sl):
    tok = all_sequences(sl)[:4]
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.9)
    logits = m(tok[:, :-1])
    assert logits.shape == (4, tok.shape[1] - 1, sl.vocab_size)
    logits2, recs = m(tok[:, :-1], record=True)
    assert torch.equal(logits, logits2) and len(recs) == tok.shape[1] - 1
    assert len(recs[0]) == cfg.n_blocks + 1 and "h_final" in recs[0][-1]
    for rec in recs:
        for i in range(cfg.n_blocks):
            for key in ("h_in", "h_mid", "h_out"):
                assert set(rec[i][key].unique().tolist()) <= {-1.0, 0.0, 1.0}, key
        assert set(rec[-1]["h_final"].unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert torch.equal(rec[1]["h_in"], rec[0]["h_out"])          # 层间传的就是三值 r_out
    out = m.generate(sl.ids[0][:3], max_new=6)
    assert len(out) <= 6 and EOS not in out and all(0 <= t < sl.vocab_size for t in out)


def test_named_points_and_calibration_skip_the_merges(cfg, sl):
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="and")
    transfer_weights(FloatLM(cfg, sl.vocab_size), m)
    names = [n for n, _, _ in named_points(m)]
    assert names[0] == "q_emb" and names[1:11] == [f"blocks.0.{p}" for p in POINTS]
    assert len(names) == 1 + 10 * cfg.n_blocks
    thr = calibrate_thresholds(m, all_sequences(sl)[:16], 0.75)
    assert set(thr) == {"q_emb"} | {f"blocks.{i}.{p}" for i in range(cfg.n_blocks) for p in CALIB}
    assert all(v >= 1e-3 for v in thr.values())
    assert abs(m.q_emb.thr.item() - thr["q_emb"]) < 1e-5
    assert m.blocks[0].m1.thr.item() == 1.5 and m.blocks[0].m2.thr.item() == 1.5


def test_spike_counts_cover_every_point_including_merges(cfg, sl):
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, levels=2)
    sc = spike_counts(m, all_sequences(sl)[:8], torch.device("cpu"))
    assert set(sc["points"]) == {n for n, _, _ in named_points(m)}
    assert sc["points"]["blocks.0.m1"]["levels"] == 2 and sc["points"]["q_emb"]["neurons"] == cfg.d
    for key in ("mean_spikes_per_neuron_token", "total_spikes_per_token", "w2_events_per_token"):
        assert sc[key] >= 0


def test_a1_state_loads_into_leaky_c_model(cfg, sl):
    a = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="sat", levels=2)
    c = TernaryResidualLM(cfg, sl.vocab_size, leak=cfg.leak, merge="sat", levels=2)
    c.load_state_dict(a.state_dict())
    set_leak(c, 0.5)
    assert all(lif.leak == 0.5 for lif in c.lif_modules()) and len(list(c.lif_modules())) == 8 * cfg.n_blocks


def test_gradient_flows_to_embedding_through_ternary_residual(cfg, sl):
    tok = all_sequences(sl)[:4]
    for merge in ("sat", "learn"):
        m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge=merge)
        logits = m(tok[:, :-1])
        F.cross_entropy(logits.reshape(-1, sl.vocab_size), tok[:, 1:].reshape(-1)).backward()
        assert m.emb.weight.grad.abs().sum() > 0 and m.blocks[0].attn.Wq.weight.grad.abs().sum() > 0
        assert (m.blocks[0].m1.log_thr.grad is not None) == (merge == "learn")
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q`
Expected: `ModuleNotFoundError: toy_demo.ternary_residual`。

- [ ] **Step 3: 实现 `toy_demo/ternary_residual.py`**

`spike_counts` 的 `keys` 映射:`{"q_emb": "h_in", "q": "q", "k": "k", "v": "v", "pre": "o_spk", "out": "y_attn", "m1": "h_mid", "g": "g", "u": "u", "y2": "y_ffn", "m2": "h_out"}`,`q_emb` 取 `recs[t][0]`。其余照 `multilevel.spike_counts` 的算法。

- [ ] **Step 4: 运行,确认通过**

Run: 同 Step 2。Expected: `6 passed`。再跑 `CUDA_VISIBLE_DEVICES="" $PY -m pytest -q` 确认原有 28 项仍通过。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/toy_demo/ternary_residual.py step2-toy-demo/tests/test_ternary_residual.py
git commit -m "toy_demo: TernaryResidualLM with spike-point registry, calibration and spike counts"
```

---

### Task 3: 共用脚本认出新模型(`beam_eval.describe_checkpoint` / `load_model`)

**Files:**
- Modify: `step2-toy-demo/experiments/beam_eval.py:43-74`
- Test: `step2-toy-demo/tests/test_ternary_residual.py`(追加)

**Interfaces:**
- Consumes: `beam_eval.SIZE_KEYS`(`(args 键, ToyConfig 字段)` 对的列表)、Task 2 的 `TernaryResidualLM`。
- Produces: `describe_checkpoint(path)` 在 `args.get("residual") == "ternary"` 时返回 `dict(kind="ternres", leak=0.0 if stage == "a1" else None, levels=int(a["levels"]), merge=a["merge"], sizes=sizes)`;`load_model` 对 `kind == "ternres"` 构造 `TernaryResidualLM(cfg, sl.vocab_size, leak=leak, merge=info["merge"], levels=info["levels"])`,不调用 `apply_levels`。

- [ ] **Step 1: 追加失败的测试**

```python
def test_describe_and_load_checkpoint_recognise_ternary_residual(cfg, sl, tmp_path):
    import json
    from experiments.beam_eval import SIZE_KEYS, describe_checkpoint, load_model
    m = TernaryResidualLM(cfg, sl.vocab_size, leak=0.0, merge="and", levels=2)
    d = tmp_path / "ternres_and_T2_a1"
    d.mkdir()
    torch.save(m.state_dict(), d / "model.pt")
    args = dict(stage="a1", residual="ternary", merge="and", levels=2)
    args.update({k: getattr(cfg, key) for k, key in SIZE_KEYS})
    (d / "result.json").write_text(json.dumps(dict(args=args)), encoding="utf-8")
    info = describe_checkpoint(d / "model.pt")
    assert info["kind"] == "ternres" and info["merge"] == "and" and info["levels"] == 2 and info["leak"] == 0.0
    mm, _, _ = load_model(d / "model.pt", sl, torch.device("cpu"))
    assert isinstance(mm, TernaryResidualLM) and mm.blocks[0].m1.thr.item() == 1.5 and mm.blocks[0].m1.levels == 2
    assert torch.equal(mm(all_sequences(sl)[:2, :-1]), m(all_sequences(sl)[:2, :-1]))
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q -k describe`
Expected: FAIL(`kind == "spiking"` 或 `load_state_dict` 键不匹配)。若 `experiments.beam_eval` 导入即报错(导入时副作用),在测试里改为 `importlib` 按路径加载,并记在提交信息里。

- [ ] **Step 3: 在 `beam_eval.py` 加两个分支**

`describe_checkpoint` 的 `if meta.exists()` 内、`stage == "a0"` 判断之后加 `residual` 分支;`load_model` 的 `else` 前加 `elif info["kind"] == "ternres"`。导入放在函数内(`from toy_demo.ternary_residual import TernaryResidualLM`)或模块顶部,与文件现有风格一致。

- [ ] **Step 4: 运行,确认通过**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q`
Expected: `7 passed`。再跑 `$PY experiments/beam_eval.py --selftest`(若该选项不依赖 GPU/真实切片)确认旧路径未坏。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/beam_eval.py step2-toy-demo/tests/test_ternary_residual.py
git commit -m "beam_eval: load ternary-residual checkpoints (kind ternres) so profile/beam scripts accept them"
```

---

### Task 4: 实验脚本 `experiments/ternary_residual.py`(a1 / c / iscan / bscan / chain)

**Files:**
- Create: `step2-toy-demo/experiments/ternary_residual.py`
- Test: `step2-toy-demo/tests/test_ternary_residual.py`(追加)

**Interfaces:**
- Consumes: `experiments.convert_lab` 的 `get_device, load_env, load_a0, train, _save, scale_thresholds, lr_schedule`;`toy_demo.train.evaluate, teacher_forced_metrics`;`toy_demo.convert.transfer_weights, set_leak`;Task 2 的 `TernaryResidualLM, calibrate_thresholds, spike_counts`。
- Produces:
  - `make_model(cfg, sl, dev, merge, levels, leak) -> TernaryResidualLM`
  - `build_a1(a0, cfg, sl, tokens, dev, fq, merge, levels, seed=0, leak=0.0) -> (model, thresholds)`:`torch.manual_seed(seed)` → `make_model` → `transfer_weights` → `calibrate_thresholds(m, tokens[:cfg.calib_sentences].to(dev), fq)`。
  - `describe_init(init_path) -> dict(merge, levels, fq, sizes)`:读同目录 `result.json` 的 `args`。
  - 子命令与参数(公共:`--name`、`--out results/improve`、`--a0 results/ckpt/a0.pt`、`--seed 0`、`--eval-every 100`、`--no-stop`、`--lr`、`--steps`):
    - `a1 --merge {sat,and,learn} --levels T --fq 0.75`(默认 lr 3e-3、steps 12000)
    - `c --init <a1 dir>/model.pt --lif-scale 1.0`(merge/levels/fq 从 init 继承;默认 lr 3e-3、steps 12000;训练前 `scale_thresholds(m, lif_scale, only=list(m.lif_modules()))`;最终以 `leak=cfg.leak` 评估)
    - `iscan --merge --levels --fq-list 0.5,0.75,0.9 --save path`:每个 fq `build_a1` + `teacher_forced_metrics`,行 `dict(fq, token_acc, ppl, spikes=spike_counts(...)["mean_spikes_per_neuron_token"])`
    - `bscan --init --scale-list 1,1.5,2,2.25,2.5,3,4 --save path`:leak=cfg.leak 装载,逐倍数缩放 LIF 阈值,`teacher_forced_metrics`
    - `chain --merge --levels --fq-list --scale-list --lr --steps --seed`:进程内依次 iscan(取 token_acc 最高的 fq)→ a1(目录 `<name>_a1`)→ bscan(`<name>_bscan.json`,取最高倍数)→ c(目录 `<name>_c`);每步把选择打印并写进 `<name>_chain.json`。
  - 每次 `_save` 的 `args` 里含 `stage`(`"a1"` / `"c"`)、`residual="ternary"`、`merge`、`levels`、`fq`、以及 `SIZE_KEYS` 需要的尺寸键(用 `cfg` 的值填)。

- [ ] **Step 1: 追加失败的测试**

```python
def test_build_a1_calibrates_and_trains_a_few_steps(cfg, sl, demo):
    from experiments.convert_lab import train
    from experiments.ternary_residual import build_a1
    tokens = all_sequences(sl)
    m, thr = build_a1(FloatLM(cfg, sl.vocab_size), cfg, sl, tokens, torch.device("cpu"), fq=0.75, merge="sat", levels=2, seed=0)
    assert m.merge == "sat" and m.levels == 2 and "q_emb" in thr and m.blocks[0].m1.thr.item() == 1.0
    res = train(m, sl, demo, cfg, tokens, lr=1e-3, steps=3, seed=0, eval_every=3, stop=False, log=lambda *_: None)
    assert res["history"][-1]["step"] == 3 and 0.0 <= res["final"]["token_acc"] <= 1.0
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q -k build_a1`
Expected: `ModuleNotFoundError: experiments.ternary_residual`。

- [ ] **Step 3: 实现脚本**

照 `convert_lab.cmd_a1 / cmd_c / cmd_iscan / cmd_bscan` 的结构,把 `SpikingLM`+`apply_levels`+`calibrate_thresholds` 换成本任务的 `make_model`/`build_a1`/`toy_demo.ternary_residual.calibrate_thresholds`,`lif_modules(m)` 换成 `list(m.lif_modules())`,`safe_spike_counts` 换成 `toy_demo.ternary_residual.spike_counts`。模块 docstring 给出 chain 的用法一行。

- [ ] **Step 4: 运行,确认通过;CLI 冒烟**

Run: `cd step2-toy-demo && CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_ternary_residual.py -q` → `8 passed`。
Run: `$PY experiments/ternary_residual.py --help` → 列出 `a1, c, iscan, bscan, chain`。
Run(GPU 1,真实数据,约 2 分钟):`CUDA_VISIBLE_DEVICES=1 $PY experiments/ternary_residual.py a1 --name ternres_smoke --merge sat --levels 1 --fq 0.75 --steps 100 --eval-every 50 --no-stop --out results/improve`
Expected: 打印 `[a1_init:ternres_smoke] ... acc ...`,100 步后 `token_acc` 高于初始值,`results/improve/ternres_smoke/{model.pt,result.json,history.jsonl}` 存在;然后 `CUDA_VISIBLE_DEVICES=1 $PY experiments/ternary_residual.py bscan --init results/improve/ternres_smoke/model.pt --scale-list 1,2` 打印两行。做完删除 `results/improve/ternres_smoke*`。

- [ ] **Step 5: 提交**

```bash
git add step2-toy-demo/experiments/ternary_residual.py step2-toy-demo/tests/test_ternary_residual.py
git commit -m "experiments: ternary_residual a1/c/iscan/bscan/chain on the convert_lab training loop"
```

---

### Task 5: 启动 4 条链

**Files:**
- Create: `step2-toy-demo/run_ternres.sh`

**Interfaces:**
- Consumes: Task 4 的 `chain` 子命令。
- Produces: `results/improve/ternres_{sat,and}_T{1,2}_{a1,c}/`、`..._bscan.json`、`..._chain.json`、`..._chain.log`。

- [ ] **Step 1: 写启动脚本**

```bash
#!/bin/bash
# 三值残差实验:{sat, and} × {T=1, T=2} 四条链,GPU 1 上同一 T 的两条并行,T=1 完成后再跑 T=2。
cd /data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
for T in 1 2; do
  for M in sat and; do
    nohup $PY experiments/ternary_residual.py chain --name ternres_${M}_T${T} --merge $M --levels $T \
      --fq-list 0.5,0.75,0.9 --scale-list 1,1.5,2,2.25,2.5,3,4 --lr 3e-3 --steps 12000 --seed 0 \
      > results/improve/ternres_${M}_T${T}_chain.log 2>&1 &
  done
  wait
done
```

- [ ] **Step 2: 后台启动并确认开跑**

Run: `chmod +x run_ternres.sh && nohup ./run_ternres.sh > results/improve/ternres_launcher.log 2>&1 &`
Expected: 1 分钟后 `nvidia-smi` 显示 GPU 1 上两个 python 进程;`results/improve/ternres_sat_T1_chain.log` 出现 iscan 的三行与 `[a1_init:...]`。

- [ ] **Step 3: 提交脚本**

```bash
git add step2-toy-demo/run_ternres.sh
git commit -m "run_ternres.sh: launch the four ternary-residual chains on GPU 1"
```

- [ ] **Step 4: 等待(约 16–20 小时)**

每条链的 `<name>_chain.json` 出现 `c` 字段即完成。期间不改 `experiments/ternary_residual.py` 的行为(运行中的进程已装载旧代码,但 c 阶段由同一进程调用,改动会生效于尚未开始的阶段)。

---

### Task 6: 汇总表、README 与种子补跑

**Files:**
- Create: `step2-toy-demo/experiments/ternres_tables.py`
- Modify: `step2-toy-demo/results/improve/README.md`(文末新节)
- Modify: `stage1-progress.md`(可选,一句话指向 README)

**Interfaces:**
- Consumes: 各 `ternres_*_{a1,c}/result.json`、`history.jsonl`、`*_chain.json`;基线 `results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/result.json`(T=1 C)与 `results/improve/T2_c{,_s1,_s2}/result.json`(T=2 C)。
- Produces: `ternres_tables.py --runs ternres_sat_T1,ternres_and_T1,ternres_sat_T2,ternres_and_T2 --baselines T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k,T2=results/improve/T2_c` 打印 markdown 表并写 `results/improve/ternres_tables.json`。列:规则、T、选中的 fq、A1 初始 acc、A1 acc(步数)、B acc、选中的 LIF 倍数、C acc、C ppl、demo 最后 10 次 5/5 次数、C 收敛步数、每 token 脉冲总数、W2 事件、用时。

- [ ] **Step 1: 写汇总脚本并运行**

Run: `cd step2-toy-demo && $PY experiments/ternres_tables.py --runs ... --baselines ...`
Expected: 表里 4 行新模型 + 2 行基线,数字与各 `result.json` 一致(抽查 `ternres_sat_T1_c/result.json` 的 `final.token_acc`)。

- [ ] **Step 2: 整句背对率与画像(补充指标)**

Run: `CUDA_VISIBLE_DEVICES=1 $PY experiments/beam_eval.py --models sat_T1=results/improve/ternres_sat_T1_c/model.pt,and_T1=...,sat_T2=...,and_T2=... --beams 1,4 --alphas 0 --save results/improve/beam/ternres.json`
Expected: 四个模型都能装载(Task 3)并给出贪心 / K=4 的整句背对率。

- [ ] **Step 3: 写 README 新节"三值残差流(2026-10-2x)"**

内容:改动是什么(两句)、表、结论(每种规则相对现有 C 掉多少;哪种好;T=2 下的情况)、电路含义(层间 / GDN→FFN 连线三值,省掉累加器电容与 DAC,每通道剩一个三值加法 + 比较)、没做的(行为级电路、SPICE)。数字只从 `ternres_tables.json` 取。

- [ ] **Step 4: 胜出规则补种子 1、2**

Run:对胜出的 (规则, T) 各起两条 `chain --seed 1` / `--seed 2`(名字加 `_s1` / `_s2`),完成后重跑 Step 1 把种子行加进表。

- [ ] **Step 5: 提交并推送分支**

```bash
git add step2-toy-demo/experiments/ternres_tables.py step2-toy-demo/results/improve/README.md step2-toy-demo/results/improve/ternres_*  stage1-progress.md
git commit -m "ternary residual: results of sat/and x T=1/2 against the real-valued residual baseline"
git push origin worktree-conversion-sweep
```

检查点 `model.pt` 按现有做法不入库(`results/improve/*/model.pt` 是否被 `.gitignore` 排除,提交前用 `git status` 确认;若未排除,只加 `result.json`、`history.jsonl`、`*_chain.json`、`*_bscan.json`)。
