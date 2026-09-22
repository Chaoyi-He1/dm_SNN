# Stage 1 进展记录

更新时间：2026-09-14（同日两次：占位器件；真实器件）。对应 project-plan.md 的 Stage 1（电路原语的表征与时序验证）与 Stage 2.3（行为级仿真器）的骨架。规范版本 0.1.3。

**状态：Stage 1 的软件侧原语与相位级仿真器已完成并与 Stage 0 软件层逐脉冲一致；SPICE 侧先用占位元件完成单 cell 时序判别、泄漏管 α 扫描、神经元六参数拟合，随后换成真实器件：晶体管用 GlobalFoundries GF180MCU 开源 PDK（65 nm 无公开模型），扩散忆阻器用 Zhao et al. 2025 补充材料的物理模型。S1.5、S1.6、16×16 分块未做。**

## 已完成

**行为级电路原语（`snn_spec/circuit.py`，19 项测试）。** 六参数 DM 神经元状态模型 `DMNeuronModel` 与其拟合函数 `fit_dm_neuron`；按 φ1 衰减、φ3 检索、φ4 写入、φ5 读出分相位工作的 `GainCellArray`，带写入误差、读误差、保持因子，并能配置泄漏窗位置以复现形式 B、C 和"泄漏常开"三种时序错误；带电导量化、编程误差、读噪声的 `CrossbarModel`；带失调与迟滞的 `ComparatorModel`；锁存 AND；把 `SpikingBlock` 逐原语替换后按 S0.7 相位表逐 token 执行的 `BehavioralBlock`。理想参数下，行为级仿真器与 Stage 0 软件层在同一权重、同一输入上输出逐 token 相等、状态矩阵一致。

**SPICE 测试台（`experiments/spice/`，ngspice 47，8 项测试）。** 结果与数字见 `step1-spiking-layer/results/spice-stage1.md`。

| 项 | 做法 | 结果 |
|---|---|---|
| S1.4 单 cell 时序判别 | 1 nF 电容、相位开关泄漏、采样保持检索、受控电流源写入 | A / B / C = 1.2502 / 1.0004 / 0.7505，与闭式偏差 ≤ 0.001；第二判别点 α=0.25、β=1 给 2.000 / 1.250 / 0.501 |
| S1.3 α 可调区间 | 1 pF 存储电容，level-1 通用 NMOS 泄漏管，t_gap 1 µs | α 从 1 降到 0 只用了 0.45–0.65 V 的栅压窗口；同一栅压下存储值 1.0 / 0.5 / 0.25 V 的 α 为 0.543 / 0.122 / 0.011 |
| S1.2 神经元拟合 | RC 膜 + 带迟滞阈值开关的理想稳定 DM，400 周期训练、200 周期验证、3 个种子 | β_m、g_m、θ、U_hold 与电路理论值的偏差 ≤ 1%、≤ 0.2%、≤ 0.2%、≤ 0.03 V；自由运行 precision / recall 为 1.00 / 1.00、0.86 / 0.88、0.98 / 0.98 |

## 四条发现

1. **单步判别需要两个点。** α=β=0.5 时，"泄漏管整周期常开、周期中点检索写入"给出约 0.97，与形式 B 的 1.00 只差 0.03；补测 α=0.25、β=1 后四种时序两两距离都超过 0.15。已写进计划书 S1.4。
2. **gain cell 的泄漏必须是欧姆型。** MOSFET 在饱和区或亚阈区的泄漏电流与存储电压几乎无关，放电是恒流的，α 随存储值变化，GDN 的衰减乘子不再成立。泄漏管必须向 V_mid 放电、小摆幅、工作在三极管区。已写进规范 0.1.2 和计划书 S1.3。
3. **六参数模型的复位律要写成 U ← U_hold·sgn + ρ(U − θ·sgn)。** 原来"回到 U_hold 后部分放电"的写法在这类器件上回归退化。新写法以 ρ=1、U_hold=0 包含软件参考实现的软复位。单个 ρ 表达不了越阈时刻对再充电的影响，是 precision / recall 没有三个种子都过 0.95 的原因。已写进计划书 S1.2 和规范 0.1.2。
4. **神经元有两个输入上界。** 每周期电荷小于 C_m(θ − U_hold)，否则一个周期内发放两次（本例 0.8 mA，不限幅时 400 周期里 38 个双发放）；导通期间电流小于保持电流 U_hold/R_on，否则器件锁定在导通态（R_on = 100 Ω 时上限 2 mA，确实观察到锁定）。规范"每 token 至多一个脉冲"依赖这两条。已写进规范 0.1.2 和计划书 S1.2。

## Gate 状态

- S1.2：拟合流程成立，参数回到理论值；precision / recall 的 0.95 门三个序列过了两个。
- S1.3：流程成立；[α_min, α_max] 的具体数值依赖真实泄漏管，占位器件的数字不能用于钳位。
- S1.4：单 cell、理想元件下落在形式 A；16×16 分块未做。
- S1.5、S1.6：未做。

## 下一步

- Stage 1 剩余：S1.5 组合级 SPICE（crossbar 列 + 神经元、累加器 + 比较器、锁存 + AND、W_O 双神经元）、S1.6 门生成电路、16×16 分块的两点判别、行为级仿真器与 SPICE 在分块上的对照。
- 器件组需提供：HfO₂ ADM 的阈值/保持电压分布（补充材料只有图）、器件间差异，以及是否接受 Table 8 的快速动力学假设；GF 65LPe 的 NDA 模型文件。
- GF180 上的输出级（R_l、V_dd 按 3.3 V 重设）、开关 + PDK 多晶硅电阻的泄漏路径映射、按 ms 周期重做单 cell 两点判别。
- 模型侧改为从零训练的玩具模型(`toy-demo-plan.md`,代码在 `step2-toy-demo/`),不再下载 Qwen 权重或做 LoRA。

## 真实器件（2026-09-14 下半）

GF 65 nm（65LPe）只有 NDA 途径。经确认后改用 GF180MCU 开源 PDK（`step1-spiking-layer/pdk/gf180mcu/`，提交 9f992d5a，Apache 2.0，ngspice 原生 BSIM4）；65LPe 到手后换两个模型文件和模型名即可重跑。扩散忆阻器参数取自 Zhao et al. 2025 的补充材料 Note 4/6 与 Table 1/7/8（正文付费墙后）。全部测试 105 项通过。

| 项 | 做法 | 结果 |
|---|---|---|
| GF180 泄漏路径（S1.3 第 5 项） | nmos_3p3 四种泄漏配置对比 | 只有"开关 + 电阻、α 由泄漏窗长度定"是欧姆型：与 exp(−t/RC) 差 ≤ 0.3%，四种存储值下 α 离散 ≤ 1.4×10⁻⁴；接地泄漏 α 随存储值 1.0/0.5/0.25 V 为 0.42/0.22/0.016；亚阈区与三极管区到 V_mid 在 ±0.1 V 摆幅下离散 0.13–0.48 |
| ADM 物理模型（Python + ngspice） | Eq. S1–S5 的确定性实现，两份代码交叉验证 | 首次导通时刻一致到 10 µs（0.8 V 34.92 ms，1.0 V 15.43 ms）；复现补充材料的脉冲串发放、泄漏积分、内在可塑性三个趋势 |
| 六参数模型对 ADM 的拟合 | S1.2 流程，周期 2 ms | precision/recall 为 0（栅电压作状态）或 0–0.47（通道长度作状态）；发放率随残留 x₂ 从 0.11 跳到 0.53。抽象不成立 |

详见 `step1-spiking-layer/results/spice-gf180-leak.md` 与 `results/adm-neuron.md`。

## 又三条发现

5. **gain cell 的 α 用泄漏窗长度设定，不用栅压。** 真实晶体管的三种栅控泄漏都不是欧姆型；开关串电阻按占空比泄漏是唯一与存储值无关的做法。规范 S0.7 的 α 映射改为 α = exp(−t_gap/RC)。
6. **这个器件的膜电位不是栅电压，是离子残留 x₂。** 发放由 x 越过 1 触发，增长是加速的；线性 LIF 抽象（六参数模型）对它失效。行为级仿真器直接用物理模型 `ADMNeuronModel.step`（状态 x₁、x₂、V_gs）替换 LIF。
7. **时间尺度与脉冲约束。** Table 1 器件的积分需要 15–200 ms，token 周期只能是 ms 级；50% 占空比下 V_gs 在两次导通间降不到 0.2 V，持续输入下器件以 2–3 kHz 松弛振荡，"每 token 至多一个脉冲"要靠短输入脉冲、导通后撤除和合适的 R_c 来保证。

## 复现

```sh
cd step1-spiking-layer
.venv/bin/python -m pytest -q                      # 105 项
.venv/bin/python -m experiments.spice.gain_cell    # 需要 ngspice
.venv/bin/python -m experiments.spice.dm_neuron
.venv/bin/python -m experiments.spice.gf180        # GF180MCU 泄漏路径
.venv/bin/python -m experiments.spice.adm_spice    # ADM 1M1T1R
.venv/bin/python experiments/adm_fit.py
```

## 2026-09-19 ADM 模型与正文对照

正文 PDF 到手(`paper/Crossbar/diffusive.pdf`,仅正文,无 SI)。用 `experiments/adm_main_text.py` 把 Table 1 模型对着正文 Fig. 2e、3b、3c/d、4a/b、4c/d 跑了一遍(`results/adm-main-text.json`;`tests/test_adm_main_text.py` 4 项通过、1 项 xfail 记录不符),解释见 `results/adm-neuron.md` 末节。

- 正文能确认的参数(R_c 250 MΩ、R_load 47 kΩ、晶体管 0.22 V、HRS 量级、能耗量级)与我们的取值一致;V_dd 正文为 0.5 V(`adm_spice.py` 用 1 V,只影响输出级)。μ、β、γ、λ、C_g 等只在 SI 里,未核对。
- 残留寿命 γ₂ = 5 s⁻¹、泄漏随脉冲间隔的比例与正文一致;绝对积分时间落在正文两个实验器件相差 5 倍的范围内。
- **发现 6、7 需要更正。** Table 1 模型在 R_c ≥ 250 MΩ 下首次发放后锁死(x 在 1 附近每脉冲抖动约 7 次、V_gs ≥ 0.35 V、Eq. S15 输出恒高、没有输出脉冲),与正文 Fig. 2c/2e/4d(分立尖峰,250 MΩ 时 ISI ≈ 4 ms、由 R_c C_g 放电设定)定性不符:“2–3 kHz 松弛振荡”是抖动;“只能按导通事件计”反了,输出脉冲才是正确的事件定义。原因是模型对残留的增强太强——再次导通只需 V_in − V_gs ≈ 0.05 V,正文反推需 ≈ 0.65 V。规范“实现配置与接口约定”里关于 ADM 短脉冲/撤除输入的要求依据的是这个锁死行为,待 SI 核对后重写(未改规范)。R_c ≤ 100 MΩ 时模型与正文一致(每脉冲一个输出脉冲)。
- 环境:本次在 Windows 上用 `D:\APP\anaconda\New` 的 Python 3.14 + torch 2.14.0+cu126 跑,pytest 全部通过(SPICE 项因无 ngspice 跳过);写含中文的 JSON 需 `PYTHONUTF8=1`。GPU 为 RTX 5070 Ti(sm_120),cu126 版 torch 的架构表不含它,Stage 2 用 GPU 前要换 cu128 及以上的 wheel。
