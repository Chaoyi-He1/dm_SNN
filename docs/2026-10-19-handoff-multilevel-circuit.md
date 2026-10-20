# 交接：电路模型支持每 token 多个脉冲（T > 1），并对 T = 2 的模型做电路验收

日期 2026-10-19。本文给接手的 session 用：先讲项目在哪、要做什么、已经改到哪一步，再给逐项的做法和判据。读完本文不需要再翻聊天记录。

## 1. 项目现状与任务来源

- 仓库 `Chaoyi-He1/dm_SNN`，本地 Dropbox checkout 在 `~/Library/CloudStorage/Dropbox/Research/SNN/`。2026-10-19 已把本地 `main` 快进到 `origin/main`（10fd1d4，2026-10-16）。GPU 训练在另一台双 RTX A4500 的机器上做，结果只经 git 回到 Dropbox，所以开工前先 `git fetch origin` 并比较 `main..origin/main`（这台 Mac 没有 `timeout` 命令）。
- 模型侧自 2026-09-21 起是从零训练的玩具模型（`toy-demo-plan.md`，代码 `step2-toy-demo/`）：116 万参数，d = 128，2 个纯 GDN 块，TinyStories 切片 4003 句，流水线 A0 浮点 → A1 脉冲（leak 0）→ B（leak 0.9）→ C 微调 → 电路（`toy_demo/circuit.py` 的 `CircuitLM`，每块一个 `snn_spec.circuit.BehavioralBlock`）。
- 最新结论在报告页 https://claude.ai/artifact/S7uy6zb59fYLeSpZuoq48T（源文件 `step2-toy-demo/results/conversion-sweep/gdn-snn-report.html`）和两份记录 `step2-toy-demo/results/conversion-sweep/README.md`、`step2-toy-demo/results/improve/README.md`：
  - 每 token 1 个脉冲（T = 1）：A0 84.2%，C 75.2%，5 道题 1/5；理想电路与 C 逐脉冲一致；标称电路 teacher forcing 不通过（4 个输出的符号错误率 2.0% / 7.2% / 4.2% / 10.9%，上限 2%）。
  - 每 token 最多 2 个脉冲（T = 2）：C 82.5% 到 83.0%，3 个种子都 5/5；T = 4 为 83.2% 到 83.6%。结论：T = 1 是主要瓶颈。
  - 报告写明的下一步，也就是本文的任务：**电路模型支持 T > 1（多脉冲神经元、门与 W2 的事件计数），对 T = 2 的模型跑理想电路和标称电路验收；再按 FFN 子块、GDN 子块、状态阵列的顺序单独对照，找标称电路的误差来源。**
- 检查点已入库（提交 dc62d9c，2026-10-19）：21 个 `model.pt`，清单与 sha256 在 `step2-toy-demo/results/CHECKPOINTS.md`。本任务要用的是 `results/improve/<run>/model.pt`，`<run>` 为 `T2_c`、`T2_c_s1`、`T2_c_s2`、`T4_c`、`T4_c_s1`、`T4_c_s2`（以及对应的 `*_a1`），浮点 A0 在 `results/ckpt/a0.pt`。每个运行目录里有 `result.json`（含 `args.levels`、`args.level_points`、尺寸参数、`args.lif_scale`）和 `history.jsonl`。现成的装载函数是 `experiments/beam_eval.py` 的 `load_model(path, sl, device)`，它按同目录的 `result.json` 建模型、套 `apply_levels`、装 `state_dict`，返回 `(model, cfg, info)`，`info["levels"]` 给出 T。

## 2. 工作区里未提交的改动（接手时的起点）

`git status` 应与下面一致。两组改动互不相关，都未提交。

**A. 本任务，已做一半（第 3 节详述）**

- `step1-spiking-layer/snn_spec/neurons.py`、`circuit.py`、`metrics.py`、`__init__.py`
- `step1-spiking-layer/tests/test_neurons.py`、`tests/test_circuit.py`（新增测试）
- `step2-toy-demo/toy_demo/circuit.py`、`step2-toy-demo/tests/test_circuit.py`（新增测试）
- 当前测试状态：step1 新增 10 项里 9 项过、1 项失败；step2 新增 2 项里 1 项失败。两处失败的原因和改法见第 3.5 节，都是一两行的事。

**B. 同一天上午的 SPICE 工作，与本任务无关，不要动、不要删**

- `step1-spiking-layer/experiments/spice/crossbar.py`（1T1R 差分 crossbar 测试台）、`gdn_layer.py`（GDN 子块闭环，新增 `xbar` 参数 ideal / R / 1T1R）、`draw_schematics.py`、`results/spice-crossbar.json`、`results/spice-gdn-layer.{json,md}`、`schematics/`、`tests/test_spice_gdn_layer.py`，以及 `stage1-progress.md`、`step1-spiking-layer/README.md` 的相应段落。需要 ngspice；`tests/test_spice_gdn_layer.py` 约 70 s。

**环境**

- `step2-toy-demo/.venv` 是指向 `../step1-spiking-layer/.venv` 的符号链接（torch 2.14.0 CPU，Python 3.12），`snn_spec` 已以可编辑方式装进去。它在 `git status` 里显示为未跟踪，是因为 `.gitignore` 写的是 `.venv/`；要么改成 `.venv`，要么不管它。
- 跑测试：`cd step2-toy-demo && .venv/bin/python -m pytest -q`（改动前 24 项，7 s）；`cd step1-spiking-layer && .venv/bin/python -m pytest -o addopts='' -q`（改动前 110 项通过、1 项 xfail，80 s，含 SPICE）。

## 3. T > 1 的语义与已完成的代码

### 3.1 必须逐位一致的定义（来自 `step2-toy-demo/experiments/multilevel.py`）

软件侧训练 T = 2 / 4 / 8 模型用的是 `multilevel.py`，电路必须与它逐位一致：

- 每个脉冲点有级数 T（`levels`）。LIF：token 开始时 U ← leak·U + I（电流一次注入）；token 内 T 个子步不泄漏，每个子步 U ≥ θ/T 就发一个脉冲并减 θ/T。总脉冲数 c = min(⌊U·T/θ⌋, T)，输出 s = c/T ∈ {0, 1/T, …, 1}，软复位 U ← U − θ·s（正好减 c 个 θ/T）。三值点对负侧对称，s ∈ {−1, …, −1/T, 0, 1/T, …, 1}。θ 仍是满量程。
- 前向写成 T 个阶跃的平均 s = (1/T)·Σ_{j=1..T} H(U − j·θ/T)，每个阶跃用原来的 arctan 代理梯度，总积分与 T = 1 相同。**T = 1 时与原实现逐位相同。**
- 块输入量化器 Q_in 和输出头前的 q_out 同样推广（无状态，2T 个比较器，阈值 ±j·θ/T）。
- k 仍乘固定的 1/√dk；|s_k| ≤ 1，所以命题上界 `state_bound` 不变。
- SwiGLU 的门 p = g·u：T > 1 时是两个脉冲计数之积 c_g·c_u/(T_g·T_u)，W2 每通道最多收到 T_g·T_u 个事件，仍然只做加法（`spike_counts` 里的 `w2_events`）。
- `apply_levels(model, T, points)` 把选中的脉冲点就地换成 ML 类并设 `mod.levels = T`；`points` 可用别名 `all`、`mem`（q、k、v）、`attn`、`ffn`、`qin`、`head`。参数名不变，`state_dict` 可互相装载。

### 3.2 `snn_spec/neurons.py`（已改）

- 新增 `level_steps(U, thr, levels, sg_width)`、`level_binary_fire`、`level_ternary_fire`，算式与 `multilevel._steps` / `ml_ternary` 相同（`j = arange(1..T)/T`，`SpikeFn.apply(U.unsqueeze(-1) − thr·j).mean(-1)`）。
- `_LIFBase.__init__` 增加 `levels=1`，存为普通属性 `self.levels`（不进 `state_dict`）。`BinaryLIF.fire`、`TernaryLIF.fire`、`QuantizerIn.forward` 在 `levels == 1` 时走原代码路径，否则走多级函数。`QuantizerIn(thr0, sg_width, levels=1)`。
- 于是 `multilevel.py` 的类替换仍可用，也可以直接对原类设 `levels`。
- `__init__.py` 导出了三个新函数。

### 3.3 `snn_spec/circuit.py`（已改）

- `_level_count(U, theta, levels)`：`(U.unsqueeze(-1) − theta·j >= 0).to(dtype).mean(-1)`，与 `level_steps` 的比较顺序一致。`_fire(U, theta, ternary, levels=1)`。
- `IdealLIFModel(thr, leak, ternary=False, levels=1)`。
- `ComparatorModel(thr, offset=0.0, hysteresis=0.0, levels=1)`：T > 1 时为 2T 个比较器（同一失调），与迟滞同时给出时抛 `ValueError`。
- `latch_and` 数值不变（g·u），docstring 写明事件计数。
- `BehavioralBlock`：从软件模块读 `levels`（`_levels_of`），T > 1 的点以 `neuron_factory(thr, leak, ternary, levels=T)` 调用，T = 1 的点仍以三参数调用（旧的三参数工厂仍可用于 T = 1 模型）；新增 `self.levels` 字典（键 q_in1、q、k、v、pre、out、q_in2、g、u、y2）；`cmp1`、`cmp2` 按 `q_in1`、`q_in2` 的级数建。
- `GainCellArray`、`CrossbarModel` 不需要改：多级值只是幅度在 {0, ±1/T, …} 的线性输入。

### 3.4 `snn_spec/metrics.py` 与 `toy_demo/circuit.py`（已改）

- `spike_prf_by_sign(ckt, ref, levels=1)`：T > 1 时按脉冲计数统计，c = round(|s|·T)，正负分开，tp = Σ min(c_ckt, c_ref)，pred = Σ c_ckt，ref = Σ c_ref；T = 1 时与原逐事件统计相同。`sign_error_rate` 不变（按符号）。
- `CircuitLM`：输出头比较器 `head_cmp` 带 `q_out` 的级数；新增 `CircuitLM.levels()` 返回 `{"blocks.i.<点名>": T, "q_out": T}`。`teacher_forced_compare` 对每块的 `y_attn`（`attn.sn_out` 的级数）与 `y_ffn`（`ffn.sn_2` 的级数）按计数算 P/R，结果里每个 part 多一个 `levels` 字段，顶层多一个 `levels` 字典。`passes()` 判据不变。

### 3.5 两个待修的测试失败

1. `step1-spiking-layer/tests/test_neurons.py::test_levels_surrogate_gradient_integral_is_independent_of_T`：断言在 [−6, 8] 上数值积分等于 1 的误差小于 0.02，实际差 0.029。这是积分区间截掉了 Lorentzian 尾巴（尺度 1/π，左右尾各约 0.015），不是代码错。改法：把三种 T 的积分互相比较（相差 < 1e-3），或把区间放宽到 [−60, 60]。
2. `step2-toy-demo/tests/test_circuit.py::test_ideal_circuit_matches_multilevel_model`：理想电路 logits 与软件不等。原因是 `CircuitLM.__init__` 在建好 `BehavioralBlock` 之后用 `setattr(sim, name, ComparatorModel(thr, offset=frac * thr))` 重建了 `cmp1`、`cmp2`，丢掉了级数。改法：`ComparatorModel(thr, offset=frac * thr, levels=getattr(sim, name).levels)`。修完这两项，再把两边的全部测试跑一遍。

## 4. 待做事项（按顺序）

### 4.1 修测试，跑全量

修第 3.5 节两处，然后：

```sh
cd step1-spiking-layer && .venv/bin/python -m pytest -o addopts='' -q
cd ../step2-toy-demo && .venv/bin/python -m pytest -q
```

两边全绿才往下走。T = 1 的旧测试（`test_ideal_circuit_matches_spiking_model`、`test_behavioral_block_with_ideal_primitives_reproduces_software_block`、`tests/test_run.py` 的 tiny 流水线）必须不受影响。

### 4.2 新脚本 `step2-toy-demo/experiments/circuit_levels.py`：对一个运行目录做电路验收

不改 `toy_demo/run.py` 与 `config.py`（扫描以来的约定：默认流水线保持计划原值，改进配方只经 `experiments/` 复现）。脚本做 `experiments/assemble.py` 电路阶段同样的事，但针对任意 T 的检查点：

- 参数：`--run results/improve/T2_c`（目录，含 `model.pt` 与 `result.json`）；`--tiny` 走合成切片与 `tiny_config()`，自建一个 T = 2 的小模型（用 `tests/test_circuit.py` 的 `_model` 加 `apply_levels`），几秒跑完，用于冒烟测试和 pytest。
- 载入：直接用 `experiments/beam_eval.py` 的 `load_model(path, sl, torch.device("cpu"))`，它从 `result.json` 的 `args` 读 `levels`、`level_points`、尺寸与 leak，返回 `(model, cfg, info)`；之后 `model.eval()`。不要自己再拼一遍 `SpikingLM` + `apply_levels` + `load_state_dict`。`convert_lab.py` 里的 `load_env`、`get_device` 可复用（它会 `os.chdir(STEP2)`）。
- 评估：与 `assemble.py` 第 80 行起相同——`compare_seed` 抽 `n_compare_sentences = 200` 句加 5 句 demo 作 `cmp_tokens`；`circuit_ideal`：`evaluate(ideal, ...)`、`logits_max_abs_diff`、`argmax_equal_to_software`、`teacher_forced_compare`；`circuit_nominal`：`evaluate` 与 `teacher_forced_compare`；`sigma_g_sweep`：0 / 0.02 / 0.05 / 0.10 的 `token_acc`、`ppl`、`n_exact`。另记 `multilevel.spike_counts(m, cmp_tokens, dev)`（每 token 脉冲数与 W2 事件数）和 `CircuitLM.levels()`。
- 输出：`<run>/circuit.json`（含 `torch`、设备、用时、cfg、上面全部数字）和 `<run>/circuit.md`（一张验收表：理想 argmax 一致、理想逐脉冲（P/R、符号错误率、状态误差）、标称自由运行 acc 与 5/5、标称 teacher forcing 四个输出的 P/R、符号错误率、状态误差、argmax 一致率，每项标通过 / 不通过）。
- 测试：`tests/test_circuit.py` 加一项，调用脚本的主函数走 `--tiny`，断言输出文件存在、`circuit_ideal` 的 `compare.pass` 为真、`sigma_g_sweep` 长度为 4。
- CPU 单线程（`torch.set_num_threads(1)`，与 `assemble.py` 一致）；T = 1 时 205 句的 teacher forcing 与自由运行评估在报告里是分钟到小时级，T = 2 的前向代价相近（只是 fire 多一维），先在 `--tiny` 上确认，再跑真模型。

### 4.3 检查点到位后跑真模型

```sh
cd step2-toy-demo
for r in T2_c T2_c_s1 T2_c_s2 T4_c T4_c_s1 T4_c_s2; do .venv/bin/python experiments/circuit_levels.py --run results/improve/$r; done
```

判据（`toy-demo-plan.md` 第 7 节，P/R 改按脉冲计数）：理想电路与 C 逐脉冲相同、逐 token argmax 相同；标称自由运行 5 句全对；标称 teacher forcing 每块两个输出的正负 P/R ≥ 0.9、符号错误率 ≤ 2%、状态误差最大值 ≤ 0.1、argmax 一致率 ≥ 99%。T = 1 的基线（`results/conversion-v2-fq50-12k/`）：标称一致率 0.625，P/R 0.67–0.78，符号错误率 2.0 / 7.2 / 4.2 / 10.9%。

### 4.4 记录

- `step2-toy-demo/results/improve/README.md` 加一节"电路模型的 T > 1 支持与 T = 2 验收"：语义（第 3.1 节）、改了哪些文件、每个运行的验收表、与 T = 1 基线的对照、结论。按用户的文档要求写：面向外人，先给具体数字表再给定义，不留"(变了)/(不变)"之类的修订标记和评审口气。
- `step2-toy-demo/README.md` 的"模块"与"本次结果"各加一行；`step1-spiking-layer/README.md` 加一句"`snn_spec` 的脉冲点支持 `levels=T`，T = 1 不变"。
- 报告页 `results/conversion-sweep/gdn-snn-report.html` 第 7 节"代价"里"电路模型还没跟上"一条和验收表最右列的"未测"要按结果更新；发布的 artifact 由用户自己更新或按其要求更新。

### 4.5 标称电路的误差隔离（验收不过时做）

按计划第 7 节的顺序：先只让 FFN 子块用标称参数（其余理想），再只让 GDN 子块，再只让状态阵列读写。`CircuitLM` 目前只有整体的 `ideal` 开关，需要加一个按原语的开关：例如 `CircuitLM(..., nominal_parts=("ffn",))`，内部把 `crossbar_kw` 按矩阵名（Wq/Wk/Wv/Wo 对 GDN，W1/W3/W2 对 FFN，输出头单独）、`gain_cell_kw`（状态阵列）、比较器失调分开传。`BehavioralBlock` 的 `crossbar_kw` 现在对 7 个矩阵统一，需要改成可按矩阵名给出。每种隔离配置各跑一次 teacher forcing，报告哪一部分把符号错误率推过 2%。再往后是 σ_G 之外的旋钮扫描（读噪声、写入/读出误差、失调），得到容限。

### 4.6 不要做的事

- 不改 `toy_demo/run.py`、`toy_demo/config.py` 的默认值，不改 `toy-demo-plan.md` 的判据。
- 不删、不提交第 2 节 B 组的 SPICE 文件；提交本任务前先问用户。
- 不要重建数据切片（`data/tinystories-slice/` 已入库，`load_slice` 会核对 SHA-256）。
- 本机只有 CPU，不要尝试训练；检查点等用户拷贝。

## 5. 相关文件速查

| 路径 | 作用 |
|---|---|
| `toy-demo-plan.md` | 当前有效的计划；第 6 节电路配置与模式，第 7 节验收判据 |
| `step2-toy-demo/toy_demo/circuit.py` | `CircuitLM`、`teacher_forced_compare`、`passes`、`primitive_kwargs` |
| `step2-toy-demo/experiments/multilevel.py` | T > 1 的软件定义、`apply_levels`、`spike_counts`、`named_points` |
| `step2-toy-demo/experiments/convert_lab.py` | `load_env`、`build_a1`、`size_overrides`、`scale_thresholds`、`lif_modules`；`c` 子命令怎么保存 `result.json` |
| `step2-toy-demo/experiments/assemble.py` | 电路阶段的参考实现（A1/C 装配页） |
| `step2-toy-demo/toy_demo/train.py` | `evaluate`（统一评估）、`teacher_forced_metrics` |
| `step2-toy-demo/toy_demo/convert.py` | `_block_points`（脉冲点顺序与名字）、`calibrate_thresholds`、`set_leak` |
| `step1-spiking-layer/snn_spec/circuit.py` | `BehavioralBlock`、`IdealLIFModel`、`ComparatorModel`、`CrossbarModel`、`GainCellArray` |
| `step1-spiking-layer/snn_spec/neurons.py` | `BinaryLIF`、`TernaryLIF`、`QuantizerIn`、多级函数 |
| `step1-spiking-layer/snn_spec/metrics.py` | `spike_prf_by_sign(levels)`、`sign_error_rate`、`state_error` |
| `step2-toy-demo/results/improve/README.md` | T = 2/4/8 实验记录与全部运行的表 |
| `step2-toy-demo/results/conversion-sweep/README.md` | T = 1 扫描与标称电路基线数字 |
