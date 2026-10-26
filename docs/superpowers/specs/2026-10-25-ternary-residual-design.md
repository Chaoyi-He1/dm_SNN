# 三值残差流:设计(2026-10-25,已在对话中批准)

## 意图

现有 `SpikingBlock` 的残差流 h 是实值累加器:`h_mid = h + y1`、`h_out = h_mid + y2`,各层入口用 Q_in 把它读成三值。
实值 h 从不进任何权重矩阵,但它要跨层传输,电路上是每通道一个电容和一个 DAC。
这次试验把残差流本身变成三值:GDN 的输入、输出,GDN 到前馈子块之间的 h',以及层与层之间传的量,全都是 ±1/0。
SoC 上层间连线、GDN 到 FFN 的连线就只有三个电平(2 根线或 2 bit),累加器电容与 DAC 都省掉,只剩每通道一个"三值加法 + 比较"。

要回答的问题:残差三值化后,toy demo 的精度比现有模型掉多少;两种合并规则哪种好。

## 模型

新类,不改 `SpikingBlock`、`SpikingLM`、`toy_demo/run.py`、`toy_demo/config.py`、`toy_demo/convert.py`。

- `r`(三值)直接进 GDN:`y1 = GDN(r)`;不再有 Q_in1。
- `h' = M1(r + y1)`,三值,直接进前馈子块:`y2 = FFN(h')`;不再有 Q_in2。
- `r_out = M2(h' + y2)`,三值,就是下一层的输入。
- 第 1 层输入 `r = Q_emb(emb(tok))`,Q_emb 是阈值可学的三值量化器(和现有 q_in1 同类)。
- 输出头直接读最后一层的 `r_out`,不再有 q_out。

**合并规则 M**(两个三值量的和在 {−2, −1, 0, 1, 2},M 把它压回三值):

| 名字 | 阈值 θ | 含义 |
|---|---|---|
| `sat` | 1.0,定死 | 饱和加法:同号 → ±1,异号 → 0,一个为 0 → 取另一个 |
| `and` | 1.5,定死 | 只有两个同号时才输出 ±1,否则 0 |
| `learn` | 初值 1.0,可学 | 训练自己在两者之间选(可选,多一条链) |

`levels = T` 时(每 token 最多 T 个脉冲),残差默认也用 T 级:和在 {−2 … 2} 步长 1/T,M 用 2T 个阈值 ±jθ/T 压回 2T+1 级。
T = 1 就是严格三值。M 的阈值不做分位数标定(输入已经是脉冲单位)。

## 实验

- 训练链与现有模型完全相同:A0 权重转移 → 阈值标定(分位数 fq;Q_emb 和 8 个 LIF 点按前向顺序标,M 不标)→ A1 微调(leak 0)
  → B(leak 0.9,LIF 阈值倍数按 bscan 选)→ C 微调。配方沿用 improve:lr 3e-3、上限 12000 步、停止规则不变(5 句全对且 500 步内提升 < 0.1 个百分点)。
- fq 用 iscan(前向)在 {0.5, 0.75, 0.9} 里选,LIF 倍数用 bscan 在 {1, 1.5, 2, 2.25, 2.5, 3, 4} 里选,都按 teacher forcing 准确率最高取。
- 先跑种子 0 的 4 条链:{sat, and} × {T=1, T=2}。GPU 1 上两条并行。胜出的规则补种子 1、2。
- 对照:现有 C,T=1 是 0.752(`results/conversion-sweep/CL2_l1_lif225_lr3e3_12k`),T=2 是 0.825 / 0.825 / 0.830(`results/improve/T2_c*`)。
- 指标同一套:切片 token 准确率、PPL、demo 5/5 的稳定次数(最后 10 次评估)、每 token 脉冲数;整句背对率(贪心 / 束搜索)作补充。
- 结果写到 `results/improve/ternres_*`,表与结论进 `results/improve/README.md` 新的一节。

## 非目标

行为级电路(`snn_spec.circuit`、`toy_demo/circuit.py`)与 SPICE 这次不做,只在结论里写明线路三值化后电路省掉了什么。
不动 toy-demo-plan.md 的判据。

## 代码位置

- `step1-spiking-layer/snn_spec/ternary_residual.py`:`ResidualMerge`、`TernaryResidualBlock`
- `step2-toy-demo/toy_demo/ternary_residual.py`:`TernaryResidualLM`、脉冲点登记、阈值标定、脉冲计数
- `step2-toy-demo/experiments/ternary_residual.py`:a1 / c / iscan / bscan / chain 子命令,复用 `convert_lab` 的训练循环
- `step2-toy-demo/experiments/beam_eval.py`:`describe_checkpoint` / `load_model` 认出新模型(model_profile、beam_eval 共用)
- 测试放两边的 `tests/test_ternary_residual.py`
