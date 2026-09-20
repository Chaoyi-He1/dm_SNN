# Stage 0 验收记录

更新时间：2026-09-13。规范版本：0.1.0。

**Stage 0 当前软件工作已完成，允许继续 Stage 1；规范尚未正式冻结。** S0.10 的两个可选学习探针也已完成。正式冻结依然依赖计划明确安排在 Stage 2 的真实层结果。

## 正确性与稳定性

macOS ARM64、Python 3.12、PyTorch 2.14.0，CPU 执行。依赖快照见 `step1-spiking-layer/requirements-lock.txt`。

完整 pytest：**62 项通过**。覆盖递推 A/B/C、闭式、编码、软复位、跨 token/分段状态、残差、SwiGLU、代理梯度、稳定性、固定版本上游对照，以及关联检索数据隔离。

核心压力测试 dk=dv=16、seed=0：全零 key 200 步、全一 key 200 步、稠密随机 key 1000 步。

| alpha | beta | 零初始状态理论界 | 测试 2、3 最大范数/界 | 结果 |
|---|---|---|---|---|
| 0.99 | 0.50 | 400 | 0.018 | 三项通过 |
| 0.90 | 0.90 | 40 | 0.167 | 三项通过 |
| 0.50 | 0.99 | 8 | 0.533 | 三项通过 |

全零 key 使用非零初始状态验证几何衰减；该项使用 float64 避免快速衰减的 float32 下溢。之前依赖随机输出的跨 token 测试已改为可手算的阈下积累测试。

## 固定版本独立对照

[Hugging Face 官方 Qwen3.5 源码](https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，Transformers v5.3.0。保存原始文件、许可证及 SHA-256，抽取并执行原始函数 AST，不手工重写函数体。

18 组 CPU float32 对照：序列长度 1/7/65，零/非零初始状态，归一化连续输入/未归一化连续输入/脉冲输入；batch=2、heads=3、dk=8、dv=5。逐 token 输出最大绝对误差 **9.54e-07**，末端状态最大绝对误差 **1.19e-07**，均通过 atol=rtol=1e-5。

额外验证官方 chunk 函数跨 64-token 边界与官方 recurrent 一致。脉冲路径明确适配官方固定 query 缩放，未将 ANN 归一化混进脉冲规范。摘要见 `step1-spiking-layer/results/upstream-comparison.md`，原始数值见 `upstream-comparison.json`。

范围：验证的是官方 CPU 参考递推，不是 FLA GPU 内核，也不是完整 Qwen3.5 预训练层。原有 `hf_reference_recurrence` 手工转写仅作为辅助测试。

## 学习探针

延迟复制：三个种子均 100%，清空全部状态后均 25%。两个种子清空 S 后仍为 100%，因此该简单任务不足以验证 S 的贡献。详见 `step1-spiking-layer/results/delayed-copy.md`。

关联检索：3 个 key、4 类 value、随机写入顺序、按完整映射隔离训练/留出组合。以下为 288 个留出 episode 的准确率：

| 种子 | 正常状态 | 清空全部状态 | 清空 S | 清空膜电位 |
|---|---|---|---|---|
| 0 | 97.22% | 25.00% | 51.74% | 92.36% |
| 1 | 97.92% | 25.00% | 49.31% | 98.61% |
| 2 | 97.92% | 25.00% | 50.00% | 86.11% |

关联检索表明此配置可学会未训练映射组合的查询，且 S 对准确率有明显贡献。详见 `step1-spiking-layer/results/associative-recall.md`；原始数据保留三个种子和消融，没有只汇报最优种子。

## 后续阶段的条件

以下不记为 Stage 0 当前软件任务未完成，但不得省略：

- Stage 1：实测硬件 alpha 区间、保持误差、SPICE 时序；目前 [0.5,0.99] 仍为暂定。
- Stage 2：真实层在 4096 token 真实文本上的稳定性，以及固定 k 缩放的 ΔPPL；这些结果决定规范是否正式冻结或需要修订。

尚未下载真实模型、测量 PPL 或执行电路仿真。当前完成结论仅针对软件基线、独立递推验收和两个小任务学习探针。

## 2026-09-14 复跑与增补

同一环境（macOS ARM64、PyTorch 2.14.0、CPU 单线程）复跑 `check_upstream.py`、`delayed_copy.py`、`associative_recall.py`，输出与 2026-09-13 记录逐位一致：上游对照三种模式输出最大误差 8.94e-08 / 3.58e-07 / 9.54e-07，末端状态 1.19e-07；延迟复制三个种子 100%；关联检索留出集 97.22% / 97.92% / 97.92%。

新增 `tests/test_target_shapes.py`（3 项），按 Qwen3.5-0.8B 的 `config.json` 实例化目标层并核对元件数量；完整 pytest 现为 **65 项通过**。规范升至 0.1.1，增补尺度折算表、"一层"的定义、记录与回放规则、目标层实例化参数；递推、编码、时序不变。

核对 `config.json` 后发现基座配置与 project-plan.md 不一致：实际为 24 层、18 层 GDN 加 6 层全注意力（索引 3, 7, 11, 15, 19, 23），线性注意力 16 头、dk = dv = 128；一层的 gain cell 为 262,144 个、注意力忆阻器 8,388,608 个、DM 神经元 17,408 个。project-plan.md 的 0.3 节元件规模、S2.1、S3、S4、时间线与交付物中的层数已于同日据此更正。
