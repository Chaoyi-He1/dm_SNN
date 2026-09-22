# Stage 0 软件参考实现

Stage 2 起的玩具模型 demo 在 `../step2-toy-demo/`,它通过可编辑安装引用本目录的 `snn_spec`(`pyproject.toml`)。

接口基线：上级目录 `folded-T1-spec.md`，版本 0.1.3，尚未冻结。

在本目录执行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -c "from snn_spec.stress import main; main()"
```

`gdn.py` 定义形式 A 核心与门；`neurons.py` 定义阈值、软复位和代理梯度；`block.py` 定义完整 GDN + SwiGLU + 残差单层及分段执行；`metrics.py` 提供软件/电路对照指标；`circuit.py` 是行为级电路原语（六参数 DM 神经元及其拟合、分相位的 gain cell 阵列、crossbar、比较器、锁存 AND）和按 φ0–φ8 相位表执行的行为级仿真器，理想参数下与软件层逐脉冲一致。

现有模型使用随机初始化权重。没有下载真实 Qwen 模型，没有进行真实文本评估或 SPICE 仿真。除本地手工转写外，现已直接执行固定提交的 HF Qwen3.5 官方 CPU 递推进行独立对照；不包含 FLA GPU 内核或完整预训练层。

本次验收结果见上级目录 `stage0-validation.md`。`.venv` 仅用于本地运行；依赖快照记录在 `requirements-lock.txt`。

## 固定版本上游对照

执行 `.venv/bin/python experiments/check_upstream.py`。解释见 `results/upstream-comparison.md`，原始数值见 `results/upstream-comparison.json`。独立参考源与哈希见 `tests/vendor/README.md`；离线可跑，无需下载模型或安装 Transformers。

## S0.10 学习探针

计划中的两个可选探针均已完成，配置同为 d=16、单头、完整 `SpikingBlock`。

### 延迟复制

执行 `.venv/bin/python experiments/delayed_copy.py`。结果与解释见 `results/delayed-copy.json` 和 `results/delayed-copy.md`。

### 关联检索

执行 `.venv/bin/python experiments/associative_recall.py`。结果与解释见 `results/associative-recall.json` 和 `results/associative-recall.md`。数据隔离与类别均衡有自动测试；记录包含全部种子及记忆消融。

## 目标层形状

`tests/test_target_shapes.py` 按 Qwen3.5-0.8B 的 `config.json`（24 层，18 层 GDN + 6 层全注意力；线性注意力 16 头、dk=dv=128；hidden 1024、FFN 3584）实例化 `SpikingBlock` 并核对元件数量。规范的"目标层的实例化参数"一节记录了这组数字，以及与 project-plan.md 中旧数字的差异。

## Stage 1：SPICE 测试台

需要 ngspice（macOS：`brew install ngspice`）。`experiments/spice/` 有单 cell 时序判别与泄漏管 α 扫描（`gain_cell.py`）和神经元六参数拟合（`dm_neuron.py`），结果与发现见 `results/spice-stage1.md`，原始数值见 `results/spice-gain-cell.json`、`results/spice-dm-neuron.json`，进展记录见上级目录 `stage1-progress.md`。没有 ngspice 时 `tests/test_spice.py` 自动跳过。

## 真实器件

- 晶体管：GF180MCU 开源 PDK（`pdk/gf180mcu/`，固定提交与哈希见其 README）。`experiments/spice/gf180.py` 比较四种 gain cell 泄漏路径，结果 `results/spice-gf180-leak.md`。
- 扩散忆阻器：`snn_spec/adm.py` 是 Zhao et al. 2025 补充材料 Note 4/6 的 1M1T1R 物理模型（Python，含 token 接口 `step`）；`experiments/spice/adm_spice.py` 是同一模型的 ngspice 实现并与 Python 交叉验证；`experiments/adm_fit.py` 检验六参数抽象对它是否成立。结果与结论见 `results/adm-neuron.md`。
- 正文对照：`experiments/adm_main_text.py` 把 Table 1 模型对着正文 Fig. 2e / 3b / 3c,d / 4a,b / 4c,d 跑一遍，结果 `results/adm-main-text.json`，断言在 `tests/test_adm_main_text.py`（1 项 xfail 记录模型与正文的不符），解释见 `results/adm-neuron.md` 末节。
- 依赖：ngspice，`pypdf`（仅用于提取补充材料文本，不入库）。

## Stage 0 完成状态

当前软件工作、固定版本上游对照与两个可选学习探针已完成；正式冻结仍需 Stage 2 真实层结果。完整验收记录见上级目录 `stage0-validation.md`。2026-09-14 复跑三项实验，数值与记录完全一致；加入行为级电路原语、SPICE 测试台、GF180MCU 与 ADM 物理模型后 pytest 105 项通过。

```sh
.venv/bin/python -m pytest -o addopts='' -q
.venv/bin/python experiments/check_upstream.py
.venv/bin/python experiments/delayed_copy.py
.venv/bin/python experiments/associative_recall.py
```

## Windows

`D:\APP\anaconda\New\python.exe`（Python 3.14，torch 2.14.0+cu126）。在本目录执行 `set PYTHONUTF8=1` 后 `python -m pytest -q`（含中文的 JSON 输出需要 UTF-8）；ngspice 未安装，SPICE 测试自动跳过。`.venv` 是 macOS 的，不能在 Windows 上用。
