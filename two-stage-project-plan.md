# 计划书:小 LLM → 线性注意力 SNN → 扩散忆阻器电路 Demo

**版本** v1.2 · 2026-09-14(v1.1:基座由 OPT-125M 改为 Qwen3.5-0.8B,Phase A/B 相应重写;v1.2:层数与层索引按 `config.json` 更正为 24 层、18 层 GDN 加 6 层全注意力)
**定位** 想法验证(idea demo)。性能下降不是问题,**能跑通、各环节可解释、软件模型与电路模型对得上**才是目标。
**两个阶段** 阶段一是纯软件的转换流水线;阶段二是电路仿真。阶段一的最终产物(折叠时间轴、T=1 的 SNN 线性注意力模型)就是阶段二电路的功能规范。

---

## 目录

- [0. 总览与"完成"的定义](#0)
- [阶段一:软件转换流水线](#s1)
  - [1.0 基座模型选择](#1-0)
  - [1.1 训练动作总表(冻结 / 微调 / 重训)](#1-1)
  - [1.2 Phase A:softmax 注意力 → Gated DeltaNet](#1-2)
  - [1.3 Phase B:把 ANN 改造成"可脉冲化"](#1-3)
  - [1.4 Phase C:转成 SNN](#1-4)
  - [1.5 Phase D(可选):MoE 上采样](#1-5)
  - [1.6 代码骨架](#1-6)
- [阶段二:电路仿真](#s2)
  - [2.0 仿真栈](#2-0)
  - [2.1 扩散忆阻器脉冲神经元](#2-1)
  - [2.2 SNN 线性注意力 demo 电路](#2-2)
  - [2.3 Crossbar FFN](#2-3)
  - [2.4 Crossbar MoE](#2-4)
  - [2.5 软件-电路闭环验证](#2-5)
- [3. 时间线与里程碑](#3)
- [4. 风险登记](#4)
- [5. 交付物清单](#5)

---

<a name="0"></a>
## 0. 总览与"完成"的定义

整个计划回答一个问题:**一个训练好的小 LLM,能否被改造成一个用扩散忆阻器神经元 + 忆阻器阵列跑的、注意力为线性递归形式的脉冲网络,并且软件模型与电路模型在同一段输入上给出一致的输出?**

阶段一的"完成"定义:一个 Qwen3.5-0.8B 模型,注意力全部换成 Gated DeltaNet(GDN),所有激活换成脉冲神经元,在 T=1 折叠模式下能在 WikiText-2 上给出有限的困惑度(PPL),且相对 ANN teacher 的退化被完整记录。

阶段二的"完成"定义:一个单头、d=16、序列长 16 的 SPICE 电路,其脉冲输出与阶段一软件模型在同一权重、同一输入下逐 token 一致(容差内);并给出 energy/token 与面积估计。

两个阶段串联,但 2.1(神经元)可以与阶段一并行启动。

---

<a name="s1"></a>
## 阶段一:软件转换流水线

<a name="1-0"></a>
### 1.0 基座模型选择

**推荐 Qwen3.5-0.8B 作主线。** 决定性的理由只有一条:**它的 24 层里有 18 层已经是原生训练好的 Gated DeltaNet。**

Qwen3.5 系列(2026 年 3 月)采用 GDN 线性注意力与 softmax 全注意力 3:1 交替的混合架构,0.8B 是其中最小的 dense 模型,24 层、隐层 1024、FFN 中间维 3584,Apache 2.0。每第 4 层是 softmax(第 4、8、12、16、20、24 层,即 0 基索引 3、7、11、15、19、23),其余 18 层是 GDN;GDN 层为 16 个 key 头、16 个 value 头、头维 128,conv1d 核长 4(`config.json` 的 `text_config`)。

这对阶段一的影响是结构性的。整条流水线里唯一需要"从头训"的东西是 GDN 的门投影,而在一个从零线性化的模型上,这些门没有任何参照、只能靠 Taylor-Calibrate 那类方法猜初值。换成 Qwen3.5-0.8B 之后,**需要转换的层从 12 层降到 6 层,而且每一个待转换层的上下都有原生 GDN 层可以直接抄门的统计量作初始化。** Phase A 从整个计划里风险最高的一步,变成"照着 18 个现成样本补 6 层"。

**必须说清楚的代价,共四条。**

第一,**"原生 GDN"不等于"不用动"。** Qwen3.5 的 GDN 层带三样脉冲不友好的组件:q/k 的 L2 归一化、SiLU 门控的输出门、以及 q/k/v 上核长为 4 的因果 conv1d。这三样在 Phase B 要统一改造并微调,是新增的工作量(见 1.3 节改造零)。

第二,**FFN 是 SwiGLU,不是 ReLU。** SwiGLU 是 $W_2(\mathrm{SiLU}(W_1x)\odot W_3x)$,里面有一个逐元素的激活乘激活。在折叠模式(T=1)下,$W_1x$ 和 $W_3x$ 各过一个 LIF 变成单比特脉冲,逐元素乘退化成一个 AND 门,没有跨维度求和,不存在 matmul 里那种交叉项问题;丢失的只是 $W_3x$ 的符号,与 $v, o$ 上已经接受并用 ternary 处理的问题同类。**所以 SwiGLU 是中等成本,不是拦路虎。** 在嵌套模式(rate-coded)下逐元素乘有 $\mathbb{E}[ab]\neq\mathbb{E}[a]\mathbb{E}[b]$ 的近似误差,但因为没有求和维度,比 matmul 的情况温和得多。

第三,**规模是 OPT-125M 的 6.4 倍。** BPTT 显存随 T 倍增,嵌套 T=4 下 0.8B 相当于训一个约 3.2B 的 ANN,V100 32GB 要开梯度检查点、batch 压到 2–4。折叠 T=1 不吃这个倍数。**2B 及以上在单卡 V100 上不要碰。**

第四,**丢掉 BiSpikCLM 的逐数字对照**,它只在 OPT 上有基线。换来的是 18 个原生 GDN 层作内部参照和一个强得多的 teacher,这笔账划算。

**两个小事。** Qwen3.5 小模型原生多模态,文本 demo 直接砍掉视觉编码器走纯文本路径;工具链需要 transformers ≥ 5.2 和 FLA 库的 GDN kernel。

**OPT-125M 降为可选旁路。** 只在需要与 BiSpikCLM 逐数字对比时才跑一遍,不是主线。Qwen3.6 与 Qwen3.8 架构相同但尺寸从 27B 起,对单卡 demo 没有意义,不必追新。

<a name="1-1"></a>
### 1.1 训练动作总表(冻结 / 微调 / 重训)

这是本阶段最核心的一张表。每个模块在每个 Phase 的动作都列清楚,**"新建"意味着从头训,"微调"意味着从 teacher 权重出发小步更新,"冻结"意味着直接搬**。

| 模块 | Phase A:转 6 个 softmax 层 | Phase B:可脉冲化 | Phase C1:纯转换 | Phase C2:蒸馏压 T |
|---|---|---|---|---|
| Embedding | 冻结 | 冻结 | 冻结(保持 FP) | 冻结 |
| 18 个原生 GDN 层的 $W_{Q,K,V,O}$ | **冻结** | 冻结 + LoRA | 冻结(直接搬) | LoRA |
| 18 个原生 GDN 层的门 $W_\alpha, W_\beta$ | **冻结(原生)** | 冻结 | 冻结(保持 FP) | 微调 |
| 18 个原生 GDN 层的 L2 归一化 / 输出门 / conv1d | 保留 | **改造:归一化→常数缩放,输出门→去掉,conv1d→4 抽头 FIR;微调** | 折叠 | 折叠 |
| 6 个转换层的 $W_{Q,K,V,O}$ | 从 softmax 层复制 + **LoRA** | 冻结 | 冻结 | LoRA |
| 6 个转换层的门 $W_\alpha, W_\beta$ | **新建,从相邻原生 GDN 层初始化后训练** | 冻结 | 冻结(保持 FP) | 微调 |
| FFN $W_1, W_2, W_3$(SwiGLU) | 冻结 | 冻结 + LoRA | 冻结 | LoRA |
| FFN 的 SiLU 与门控乘 | 保留 | **SiLU 换 QCFS,$W_3x$ 后插 QCFS,训 λ;乘法保留** | → IF;乘法 → 逐元素乘 / AND | LIF + 代理梯度;折叠下为 AND |
| q/k/v/o 处的激活 | 无(线性输出) | **插入 QCFS,训 λ** | → IF | LIF + 代理梯度 |
| RMSNorm | 保留 | **冻结统计量 → 缩放,微调** | 折进相邻线性层 | 折叠;可选换自适应阈值 |
| LM head | 冻结 | 冻结 | 冻结(FP) | 冻结 |
| **损失** | 注意力迁移 + LM | LM | 无 | SpAD 风格多项 |
| **Token 量** | 10M–50M | 30M–100M | 0(几百条校准) | 100M–500M |
| **单卡 V100 估时** | 0.5–1 天 | 1–2 天 | 分钟级 | 5–10 天 |

**关键判断一句话:整条流水线里只有两处是"新建"——6 个转换层的门投影(且从邻居初始化,不是真正从零),以及 Phase D 的 MoE 路由器。其余全部是冻结或微调。** 相比 OPT 路线,Phase A 的工作量和风险大幅下降,Phase B 因为要改造 18 个原生 GDN 层而略有上升,总量仍在一张卡两到三周之内。

<a name="1-2"></a>
### 1.2 Phase A:softmax 注意力 → Gated DeltaNet

#### 要解决的问题

softmax 注意力需要一个随序列增长的 KV cache,以及一个数据相关的 $L\times L$ 矩阵。这两样在阶段二的忆阻器阵列上都没有对应物。Gated DeltaNet 把注意力改成一个**固定尺寸、原地更新的状态矩阵**,它的三种操作(读、rank-1 写、衰减)恰好对应阵列的三种原生原语。所以 Phase A 不只是"换个高效注意力",它是在把模型改造成阵列能承载的形式。

#### 机理:GDN 的递推

采用误差校正形式,每个 token $t$ 做四步。$S_t \in \mathbb{R}^{d_h \times d_h}$ 是状态矩阵,$q_t, k_t, v_t \in \mathbb{R}^{d_h}$ 是投影,$\alpha_t, \beta_t \in (0,1)$ 是两个标量门。

$$r_t = S_{t-1}^\top k_t \qquad\text{(检索:用 key 读状态)}$$
$$\Delta v_t = \beta_t\,(v_t - r_t) \qquad\text{(delta 校正:新值减去已存的值)}$$
$$S_t = \alpha_t S_{t-1} + k_t \Delta v_t^\top \qquad\text{(更新:衰减 + rank-1 外积写入)}$$
$$o_t = S_t^\top q_t \qquad\text{(读出:用 query 读状态)}$$

$\alpha_t$ 控制遗忘,$\beta_t$ 控制写入强度。两者都由输入经一个小线性层加 sigmoid 得到。

#### 本 Phase 的范围

Qwen3.5-0.8B 的 18 个 GDN 层**原样保留、整层冻结**,本 Phase 只处理第 4、8、12、16、20、24 层这 6 个 softmax 层。

#### 转换配方(Liger 路线 + 邻居初始化)

1. **复制权重。** 每个待转换层的 $W_Q, W_K, W_V, W_O$ 从该 softmax 层原样搬过来。Qwen3.5 的全注意力层是带输出门的 Gated Attention,输出门丢弃。若该层的头数或 kv 头数与 GDN 层不一致(GQA 配置可能不同),用 head-wise 映射或一个小线性投影对齐,做法同 BiSpikCLM 处理 teacher/student 结构失配的方式。
2. **门从邻居初始化。** 这是与 OPT 路线的本质区别。待转换层 $\ell$ 的 $W_\alpha, W_\beta$ 以及 conv1d 权重,**直接从第 $\ell-1$ 层(原生 GDN)复制**,或取 $\ell\pm1$ 两层的平均。Taylor-Calibrate 指出"把 teacher 的注意力投影抄进 GDN student 时,衰减、写入、输出门的动态没被指定,student 从糟糕的动态区起步";邻居初始化直接绕过了这个问题——第 $\ell-1$ 层的门已经在同一个模型、同一套表示上训练收敛,是现成的"正确动态区"。Taylor-Calibrate 的解析初始化降级为备选。
3. **两阶段训练。** 第一阶段冻结所有原权重,只训这 6 层的门,损失是逐层隐状态的 MSE(注意力迁移),teacher 是原 Qwen3.5-0.8B;第二阶段加 LoRA(秩 8–16)到这 6 层的 $W_{Q,K,V,O}$,损失换成 LM 交叉熵加逐层 MSE。
4. **全线性化。** 6 层全换,不留 softmax。已知全线性化会损失召回能力(recall-intensive 任务掉得多),这是接受的代价;但 Qwen3.5 本身只靠这 6 层做全局召回,损失会比 OPT 路线的"12 层全换"温和。

#### 三个必须明确的设计决定

**第一,GDN 里 $k$ 的 L2 归一化改成常数缩放。** 原版 GDN 每个 token 对 $k$ 做 L2 归一化。Phase C 之后 $k$ 会变成二值脉冲,逐 token 归一化会破坏二值性。所以要改成一个固定的常数缩放。**注意这条现在同时作用于 6 个转换层和 18 个原生层**,但为了让 Phase A 的转换目标与原生层一致,Phase A 里 6 个转换层先保留 L2 归一化(和邻居一样),统一在 Phase B 的改造零里把 24 层一起改掉。常数的取值在 1.4 节由稳定性条件确定。

**第二,去掉 GDN 的输出 RMSNorm 和输出门。** 同上,Phase A 保留,Phase B 统一去掉。

**第三,$\alpha_t, \beta_t$ 在所有 Phase 都保持浮点。** 它们是每头一个的标量,在阶段二对应的是模拟控制电压(泄漏管栅压、电流镜比例),不是脉冲。所以软件模型里没必要把它们脉冲化。

#### Gate 判据

先在原 Qwen3.5-0.8B 上测出 8 项 zero-shot 平均 $A_0$ 和 WikiText-2 PPL $P_0$ 作基准。本 Phase 的 gate:zero-shot 平均 $\ge 0.85\,A_0$,PPL $\le 1.5\,P_0$。达不到先查头数对齐和门的邻居初始化是否生效,而不是加 token。

<a name="1-3"></a>
### 1.3 Phase B:把 ANN 改造成"可脉冲化"

#### 要解决的问题

Phase A 的产物是一个 24 层全 GDN 的普通 ANN:q/k/v/o 是可正可负的实数,GDN 层里还带着 L2 归一化、输出门和 conv1d,FFN 是 SwiGLU,RMSNorm 需要算均方根。要让它能被"直接换成 IF 神经元",需要先在 ANN 里就把这些地方改成脉冲神经元能表示的形式,并微调回来。**这一步的所有训练都是在为 Phase C 的零训练转换铺路。** 相比 OPT 路线,本 Phase 多了一个"改造零",因为原生 GDN 层不是为脉冲设计的。

#### 四个改造

**改造零:24 个 GDN 层的统一脉冲化改造。** 三样东西,每样一个决定。

- **q/k 的 L2 归一化 → 常数缩放。** 换成 $k \leftarrow k/\sqrt{r_k d_h}$、$q$ 同理,常数由校准数据上的发放率估计确定,取值依据在 1.4 节。
- **输出门 → 去掉。** Qwen3.5 GDN 的输出 $o$ 经一个 SiLU 门控做逐元素乘再送 $W_O$。demo 直接去掉这个门。若去掉后退化超预期,备选方案是保留为"$o$ 脉冲 AND 门脉冲",与 SwiGLU 的处理同构。
- **conv1d(核长 4)→ 保留为神经元前的 4 抽头 FIR。** 它是线性的因果滤波,作用在 $W_q x$ 等 FP 投影输出上,放在 LIF 之前不影响脉冲化;阶段二对应 4 个延迟单元。简化备选:去掉 conv1d,让 LIF 的膜泄漏吸收其时域滤波作用,靠微调补偿。

这三处改动作用在全部 24 层上,扰动不小,是本 Phase 微调 token 量比 OPT 路线高的原因。

**改造一:在所有未来的脉冲点插入 QCFS。** QCFS(Quantization Clip-Floor-Shift)是一个"长得像 IF 神经元"的激活函数:

$$a = \lambda\,\mathrm{clip}\!\left(\frac{1}{L}\left\lfloor \frac{z L}{\lambda} + \frac{1}{2}\right\rfloor,\,0,\,1\right)$$

其中 $z$ 是输入,$L$ 是量化级数(对应未来的时间步数),$\lambda$ 是可学习阈值,$1/2$ 的平移让向下取整变成四舍五入、误差期望归零。插入位置有六处:SwiGLU 里 $\mathrm{SiLU}(W_1x)$ 的位置(用 QCFS 替换 SiLU)、$W_3x$ 之后(新插)、FFN 输出、以及 GDN 层里 $q, k, v, o$ 四个投影的输出。

**SwiGLU 的处理要单独说。** $W_2(\mathrm{SiLU}(W_1x)\odot W_3x)$ 改成 $W_2(\mathrm{QCFS}(W_1x)\odot\mathrm{QCFS}(W_3x))$。逐元素乘保留;C1 嵌套模式下它是两个 rate-code 的乘积(近似),折叠 T=1 下它是一个 AND 门(精确)。$W_3x$ 被强制非负,丢失符号,与下面 $v, o$ 的问题同类。

后四处要特别说明。ANN-GDN 里 $q, k, v, o$ 本来没有激活函数,直接插 QCFS 等于强加了"非负 + 有界"的约束。对 $q, k$ 这是无害的——非负特征映射(ReLU 特征)本来就是线性注意力的标准做法。**对 $v$、$o$ 和 SwiGLU 的 $W_3x$ 这是有损的,因为它们的符号携带内容信息。** demo 阶段先接受;Phase C2 里再把这三处升级成 ternary 神经元恢复符号。

**改造二:RMSNorm 冻结成缩放。** 用几百条校准数据跑一遍,记录每个 RMSNorm 的输入均方根并冻结。冻结后 RMSNorm 退化成逐通道缩放,可以折进相邻的线性层。RMSNorm 没有均值减法,比 LayerNorm 更容易折。精度损失靠本 Phase 的微调恢复。

**改造三:LoRA 微调。** 上面三个改造都扰动了模型,用 30M–100M token 的 LM 损失微调,LoRA 放在全部 24 层的 $W_{Q,K,V,O}$ 和 FFN 上,QCFS 的 $\lambda$ 全量可训。teacher 仍是原 Qwen3.5-0.8B,加一项逐层特征 MSE 防止漂太远。

#### Gate 判据

PPL 相对 Phase A 退化 $\le 20\%$。$L$ 取 8 起步(对应 Phase C1 的 T=8 到 32),后续按 T 需求调。

<a name="1-4"></a>
### 1.4 Phase C:转成 SNN

分两个子阶段。C1 零训练,目的是"先跑通";C2 蒸馏,目的是把 T 压到 1。

#### 先解决一个结构性问题:两根时间轴

SNN 有 sub-timestep 轴 $t = 1..T$(每个 token 内部跑 $T$ 步),GDN 有 token 轴 $\tau = 1..L$(状态逐 token 递推)。两种折叠方式:

**嵌套模式。** 每个 token 跑 $T$ 个 sub-step,得到脉冲串;对脉冲串做时间平均,得到 rate-coded 的 $\bar q_\tau, \bar k_\tau, \bar v_\tau \in \{0, 1/T, \dots, 1\}^{d_h}$;用这些多值向量做一次 GDN 状态更新。精度高,但硬件上要在 $T$ 个 sub-step 期间冻结状态衰减,破坏了"衰减免费"的性质。

**折叠模式。** 令 $T=1$,token 轴就是时间轴。每个神经元每 token 只有一次二值发放机会,膜电位跨 token 携带。精度低,但这是**阶段二电路的实际工作方式**——一个 token 就是一个物理时钟周期,状态阵列在这个周期内自然衰减一次,$\alpha_\tau$ 与器件的衰减时间常数直接对应。

**计划:C1 和 C2 前半段用嵌套模式(先把精度立住),C2 后半段切到折叠模式 + ternary(对齐硬件)。** 阶段二只认折叠模式的软件模型。

#### C1:纯转换(零训练)

把 Phase B 里每个 QCFS 换成 IF 神经元,阈值 $\theta = \lambda$(学到的值),初始膜电位 $v(0) = \theta/2$(对应 QCFS 的 1/2 平移)。其余权重原样搬。取 $T = 32$ 起步,再试 16 和 64。

**C1 什么都不训。** 它的意义是给出一个"纯转换能到什么程度"的基线,并验证整条链没有 bug。Gate 判据:PPL 有限,且 $\le 5\times$ Phase B。达不到通常是阈值量程出了问题,先查 GDN 读出 $S^\top q$ 的数值范围是否落在神经元窗口内。

#### 必须处理的稳定性条件:delta 规则与二值 key

这是 SNN 版 GDN 独有的问题,不处理会直接发散。

GDN 原始形式里,状态乘以 $(I - \beta_t k_t k_t^\top)$。这个矩阵沿 $k_t$ 方向的特征值是 $1 - \beta_t \|k_t\|^2$。要不发散,需要 $0 \le \beta_t\|k_t\|^2 \le 2$。ANN 里 $k$ 做了 L2 归一化,$\|k\|^2 = 1$,条件自动满足。**但二值脉冲 $k$ 的 $\|k\|^2$ 等于其中 1 的个数 $n$。** 取 $d_h = 64$、发放率 0.18,$n \approx 11$;$\beta = 0.5$ 时 $\beta n = 5.5 > 2$,状态沿 $k$ 方向翻号并指数增长。

解法是 1.2 节预留的那个常数缩放:

$$k \leftarrow \frac{k}{\sqrt{r_k\, d_h}}$$

其中 $r_k$ 是校准得到的 $k$ 神经元平均发放率。这样 $\|k\|^2 = n/(r_k d_h) \approx 1$。**为防尾部 token 的 $n$ 达到均值三倍,额外把 $\beta$ 的 sigmoid 输出限制在 $(0, 0.6]$。** 这个缩放在阶段二就是写入电压的幅度,天然可实现。

#### C2:蒸馏压 T

Teacher 用 Phase A 的 24 层全 GDN ANN 模型(不是原 Qwen3.5-0.8B,它有 6 层是 softmax),因为架构相同,逐层特征对齐才有意义。Student 用 LIF 神经元($\beta = 0.9$)、arctan 代理梯度、BPTT,从 Phase B 权重初始化。

损失按 BiSpikCLM 的 SpAD 思路,但 GDN 没有注意力图,把"注意力图对齐"换成**状态矩阵 $S_t$ 或读出 $o_t$ 的对齐**:

- 逐层特征:Rate-MSE(teacher 特征喂进同一个 LIF 跑 T 步取时间平均,与 student 时间平均做 MSE)+ 直接 MSE
- 输出:软标签 KL(温度 2)+ 硬标签交叉熵
- 权重沿用 BiSpikCLM 的 0.2 / 0.1 / 0.1 / 0.3 / 0.3

训练顺序:嵌套 T=4 → 折叠 T=1(binary)→ 折叠 T=1 + $v, o$ 换 ternary。每一步从上一步初始化。

**显存预算。** BPTT 的显存随 $T$ 倍增。0.8B 在嵌套 T=4 下相当于训一个约 3.2B 的 ANN,V100 32GB 单卡的做法是:梯度检查点必开,batch 压到 2–4,序列长 512,LoRA 而非全量。若仍不够,把嵌套 T 降到 2 作为过渡,或只对 GDN 层展开时间轴、FFN 用 online 近似。折叠 T=1 不吃这个倍数,是最省的一步。**这也是不选 2B 的原因:2B 在 T=4 下相当于 8B,单卡 V100 做不了 BPTT。**

#### Gate 判据

嵌套 T=4:PPL $\le 2\times$ Phase A。折叠 T=1 + ternary:PPL 有限且 $\le 4\times$ Phase A。**折叠 T=1 的模型是阶段二的功能规范,它的每一个前向细节(缩放常数、阈值、复位方式)都要冻结并写进电路设计文档。**

<a name="1-5"></a>
### 1.5 Phase D(可选):MoE 上采样

阶段二要 demo crossbar MoE,软件侧得有个 MoE 目标。Qwen3.5-0.8B 是 dense 模型(Qwen3.5/3.6 的 MoE 版本从 35B-A3B 起,单卡跑不了),用**稀疏上采样**(sparse upcycling)造一个:

1. 把 FFN 复制 $E = 4$ 份作专家,各自加微小扰动打破对称。
2. 新建路由器 $W_r \in \mathbb{R}^{d \times E}$,**从头训**。这是本阶段第二处也是最后一处"新建"。
3. 路由方式用 SEMM 的思路:路由器输出过一个脉冲神经元,发放即选中,不用 softmax 也不用 top-k;负载均衡靠一个辅助损失。
4. 专家 LoRA 微调,50M token。
5. 然后按 Phase C 的流程脉冲化。

**为什么可选。** MoE 转换的误差是跳变而非渐变(路由错一个 token 就走向完全不同的专家),demo 里它是最容易出幺蛾子的部分。建议阶段一主线先不做,等 C2 稳定后再加。

<a name="1-6"></a>
### 1.6 代码骨架

折叠模式(T=1)的 SpikingGDN 层,以及 C2 的训练步。**loss 只写成函数,模型调用、loss 计算、参数更新全部放在 `train_one_step` 里。**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    """前向阶跃,反向 arctan 代理梯度(alpha=2 对齐 BiSpikCLM)。"""
    @staticmethod
    def forward(ctx, u, thr, alpha):
        ctx.save_for_backward(u)
        ctx.thr, ctx.alpha = thr, alpha
        return (u >= thr).float()

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        x = (math.pi / 2) * ctx.alpha * (u - ctx.thr)
        return g * (ctx.alpha / 2) / (1 + x * x), None, None


class LIF(nn.Module):
    """单步 LIF。膜电位 U 作为显式状态传入传出,折叠模式下跨 token 携带。
    复位用 S_t(当前步),与 BiSpikCLM Eq (1) 的 S_{t-1} 差一步,更常规。"""
    def __init__(self, thr=1.0, beta=0.9, alpha=2.0):
        super().__init__()
        self.thr, self.beta, self.alpha = thr, beta, alpha

    def forward(self, I, U):
        U = self.beta * U + I
        S = SpikeFn.apply(U, self.thr, self.alpha)
        U = U - self.thr * S          # 软复位
        return S, U


class SpikingGDN(nn.Module):
    """折叠模式(T=1)的脉冲 Gated DeltaNet。
    q/k/v/o 为二值脉冲;S 为模拟状态(阶段二对应 gain-cell 阵列);
    alpha/beta 为浮点标量门(阶段二对应模拟控制电压)。"""
    def __init__(self, d, n_heads, r_k, beta_max=0.6):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.Wa = nn.Linear(d, n_heads)    # 18 个原生层沿用 Qwen3.5 权重;
        self.Wb = nn.Linear(d, n_heads)    # 6 个转换层从相邻原生层初始化后训练
        self.sn_q, self.sn_k = LIF(), LIF()
        self.sn_v, self.sn_o = LIF(), LIF()
        self.k_scale = 1.0 / math.sqrt(r_k * self.dh)   # 稳定性缩放
        self.beta_max = beta_max

    def init_state(self, B, device):
        z = lambda *s: torch.zeros(*s, device=device)
        d = self.h * self.dh
        return dict(S=z(B, self.h, self.dh, self.dh),
                    Uq=z(B, d), Uk=z(B, d), Uv=z(B, d), Uo=z(B, d))

    def forward(self, x, st):
        # x: [B, L, d] 二值脉冲;逐 token 递推
        B, L, d = x.shape
        S = st["S"]
        Uq, Uk, Uv, Uo = st["Uq"], st["Uk"], st["Uv"], st["Uo"]
        outs = []
        for t in range(L):
            xt = x[:, t]
            q, Uq = self.sn_q(self.Wq(xt), Uq)
            k, Uk = self.sn_k(self.Wk(xt), Uk)
            v, Uv = self.sn_v(self.Wv(xt), Uv)
            a = torch.sigmoid(self.Wa(xt))                        # [B, h]
            b = self.beta_max * torch.sigmoid(self.Wb(xt))        # 限幅
            q, k, v = (z.view(B, self.h, self.dh) for z in (q, k, v))
            k = k * self.k_scale
            r = torch.einsum("bhij,bhi->bhj", S, k)               # 检索 S^T k
            dv = b[..., None] * (v - r)                           # delta
            S = a[..., None, None] * S \
                + torch.einsum("bhi,bhj->bhij", k, dv)            # 衰减 + rank-1
            o = torch.einsum("bhij,bhi->bhj", S, q)               # 读出 S^T q
            o, Uo = self.sn_o(self.Wo(o.reshape(B, d)), Uo)
            outs.append(o)
        st.update(S=S, Uq=Uq, Uk=Uk, Uv=Uv, Uo=Uo)
        return torch.stack(outs, dim=1), st


def spad_loss(s_feats, t_feats, s_logits, t_logits, labels, lam, tau=2.0):
    """loss 只写成函数。s_feats/t_feats 是逐层特征列表(已做时间平均)。"""
    l_feat = sum(F.mse_loss(s, t) for s, t in zip(s_feats, t_feats))
    l_soft = tau ** 2 * F.kl_div(
        F.log_softmax(s_logits / tau, dim=-1),
        F.softmax(t_logits / tau, dim=-1),
        reduction="batchmean",
    )
    l_hard = F.cross_entropy(s_logits.flatten(0, 1), labels.flatten())
    return lam[0] * l_feat + lam[1] * l_soft + lam[2] * l_hard


def train_one_step(student, teacher, batch, opt, lam, clip=0.7):
    x, labels = batch
    with torch.no_grad():
        t_feats, t_logits = teacher(x, return_feats=True)
    s_feats, s_logits = student(x, return_feats=True)
    loss = spad_loss(s_feats, t_feats, s_logits, t_logits, labels, lam)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), clip)
    opt.step()
    return loss.item()
```

`r_k` 从 Phase B 校准数据统计得到。`beta_max=0.6` 是 1.4 节稳定性条件的落地。

---

<a name="s2"></a>
## 阶段二:电路仿真

<a name="2-0"></a>
### 2.0 仿真栈

沿用《研究计划与设计文档》§5.1 的五层结构,不重复。这里只强调本阶段的桥接点:**L1(SPICE)扫出的神经元 f–I 曲线,做成查找表包进 1.6 节的 `LIF` 类替换掉 `SpikeFn`;L3 网络级用替换后的模型重跑阶段一的 C2,得到"器件精度"的 PPL。** 这一步把两个阶段焊在一起。

<a name="2-1"></a>
### 2.1 扩散忆阻器脉冲神经元

电路本身已在《设计文档》§2 完整给出(1M1T1R、f–I 曲线、四个设计参数),不重复。本阶段的具体任务:

1. **Verilog-A 紧凑模型。** 理想稳定 DM:确定性阈值 $V_{\rm th}$、固定 $\tau_{\rm relax}$、固定 $G_{\rm on}$。
2. **单神经元 SPICE 测试台。** 扫 $(R_{\rm leak}, C_{\rm mem}, V_{\rm th})$,提取 $g(a)$ 曲线族、死区 $I_{\min}$、饱和率 $1/\tau_{\rm ref}$、energy/spike。
3. **量程匹配。** 这是本节的 gate。阶段一折叠模型里,四个脉冲点的输入分布(q/k/v 投影输出、GDN 读出 $S^\top q$)各不相同,**每个点的 $g(a)$ 可用窗口 $[I_{\min}, I_{\rm sat}]$ 必须罩住对应输入分布的中心 ±2σ。** 用 $R_{\rm leak}$ 和一级电流镜缩放来调。
4. **ternary 变体。** 互补双 DM 支路,供 $v, o$ 使用。
5. **拟合并回灌。** $g(a)$ 做查找表,替换 1.6 节 `SpikeFn` 的前向,重跑 C2 的 eval。**器件精度 PPL 相对理想 LIF 的退化 $\le 30\%$ 为 gate。**

<a name="2-2"></a>
### 2.2 SNN 线性注意力 demo 电路

#### Demo 规模

单头,$d = d_h = 16$,序列长 $L = 16$。这是能在 SPICE 里做完整瞬态仿真的最小有意义规模。元件清单:

| 模块 | 器件 | 规模 |
|---|---|---|
| $W_Q, W_K, W_V, W_O$ | Drift NVM crossbar(1T1R,差分列) | 4 × (16×32) |
| $W_\alpha, W_\beta$ | 小 crossbar + 压缩电路 | 2 × (16×1) |
| 状态 $S$ | 3T1C gain-cell 阵列 | 16×16(每 cell 带中点参考) |
| 脉冲神经元 | 1M1T1R DM 神经元 | q/k/v/o 各 16 = 64,加 2 个作门的压缩函数 |
| 脉冲整形 | CMOS 缓冲 | 每神经元一个 |

#### 每个 token 的操作时序

一个 token 对应一个物理周期 $\Delta t_{\rm tok}$,由 DM 的 refractory 决定,µs 量级。周期内串行执行六步。

**① 投影。** $x_\tau$ 的脉冲打到 $W_Q/W_K/W_V$ crossbar 的 wordline(0 或 $V_{\rm rd}$,免 DAC)。bitline 电流直接灌进 DM 神经元的膜电容,越阈发放,得到 $q, k, v$ 脉冲。**神经元即 ADC,没有列 ADC。** 同时 $x_\tau$ 打到 $W_\alpha, W_\beta$ 的小 crossbar,输出电流经一个 DM 神经元的 $g(\cdot)$ 做 sigmoid 近似,得到模拟电压 $V_\alpha, V_\beta$。

**② 检索。** $k$ 脉冲打到 gain-cell 阵列的行读线。只有 $k_i = 1$ 的行被选通,列电流是这些行上电荷的和,即 $r = S^\top k$。这一步是纯行选通求和,和 crossbar 读完全同构。

**③ Delta。** $v$ 脉冲经一个参考电导变成电流,与 $r$ 的电流做差分相减得 $v - r$;再经一个由 $V_\beta$ 控制比例的电流镜,得 $\Delta v = \beta(v - r)$,以电压形式挂到阵列的列写线上。$\Delta v$ 有符号,用中点参考电压 $V_{\rm mid}$ 上下偏置表示。

**④ 写入。** $k$ 脉冲打开对应行的写晶体管,列写线上的 $\Delta v_j$ 注入该行各 cell。**只有 $k_i = 1$ 的行被写,写入量正比于 $\Delta v_j$**——这正是 rank-1 外积 $k \Delta v^\top$。二值 $k$ 让"外积"退化成"按行选通的列写入",不需要任何乘法器。1.4 节的 $k$ 缩放在这里就是写入电压的幅度。

**⑤ 衰减。** 泄漏晶体管栅压设为 $V_{\rm leak}(V_\alpha)$,所有 cell 在本周期内按 $\alpha_\tau$ 对应的时间常数自然放电。**这一步零操作,让时间流逝即可。** 这是整个电路最漂亮的一处:$\alpha_\tau$ 由一根模拟控制线实现,数字实现里的 $d_h^2$ 次乘法消失了。

**⑥ 读出。** $q$ 脉冲打到行读线,列电流是 $o = S^\top q$。电流灌进 $W_O$ 之前的 DM 神经元变成脉冲,再打 $W_O$ crossbar,再过 DM 神经元,得输出脉冲。

**②和⑥读同一个阵列,串行两次。** gain-cell 的读是非破坏性的,所以可行;代价是周期内两次读的时序要分开。

#### 三个电路层的设计要点

**有符号状态。** cell 电容以 $V_{\rm mid}$ 为零点,高于为正、低于为负;读出用差分放大器对 $V_{\rm mid}$ 求差。比差分双 cell 省一半面积。

**门的物理映射。** $\alpha_\tau \leftrightarrow$ 泄漏时间常数 $\tau_{\rm leak} = -\Delta t_{\rm tok}/\ln\alpha_\tau$,由泄漏管栅压设定。$\alpha$ 的可调范围受泄漏管亚阈值特性限制,**要早做一个"$V_{\rm leak}$ 能覆盖的 $\alpha$ 区间"的物理上界分析**,这决定 GDN 遗忘门的表达力上界。

**膜电位跨 token 携带。** 折叠模式下,DM 神经元的膜电容在 token 之间不清零,这是软件模型 `LIF` 类里 `U` 显式传递的物理对应。电路上就是不加复位脉冲,让它自然保持。

<a name="2-3"></a>
### 2.3 Crossbar FFN

SwiGLU 结构:三个 drift NVM crossbar,差分列。$W_1$ 和 $W_3$ 各为 $16 \times 64$,并行接同一组输入 wordline;各自的 bitline 电流灌进各自的 64 个 DM 神经元,得到两组脉冲;两组脉冲逐位过 **AND 门**(折叠模式下 SwiGLU 的逐元素乘就是它);AND 输出打到 $W_2$($64 \times 16$)的 wordline,bitline 电流灌末端 16 个 DM 神经元。脉冲入、电流出、神经元即 ADC,全程无浮点乘。残差连接就是把上一级的脉冲电流也接到同一个膜节点上,电流相加,零额外电路。

**相比 ReLU FFN 的额外成本:** 多一个 $16\times64$ crossbar 和 64 个神经元(约 +50% 面积),外加 64 个 AND 门(可忽略)。这是选 Qwen3.5 而非 OPT 在电路侧付出的全部代价。

Demo 里 FFN 的价值不在结构,在**验证 ADC-free 的量程闭环**:$W_1$ 的 bitline 电流分布必须落在 64 个神经元的可用窗口里,$W_2$ 同理。这是 2.1 节 gate 在整层规模上的复验。

<a name="2-4"></a>
### 2.4 Crossbar MoE

对应 Phase D。结构在《设计文档》§3.5 已完整给出(TTFS 路由 + 侧向抑制),不重复。本阶段具体做:

1. 路由 crossbar $16 \times 4$,4 个 DM 神经元,一条共享抑制线,一个计数到 $k=1$ 的锁存。
2. 4 个专家 tile,每个是 2.3 节的 FFN 结构。
3. **未选中的 tile 完全不驱动 wordline**,SPICE 里直接验证其动态能耗为零。
4. 路由一致率:同一输入下电路的首发放专家与软件模型的选中专家一致的比例,**gate 为 $\ge 95\%$**。

**面积账在这里必须算。** 4 个专家常驻意味着 FFN 的阵列面积 ×4,demo 规模下不是问题,但要把"每增加一个专家的面积成本"作为结论量化出来。

<a name="2-5"></a>
### 2.5 软件-电路闭环验证

这是阶段二的总 gate,也是整个计划的收口。

1. 从阶段一折叠 T=1 模型里取一层(GDN + FFN)的权重和 16 个 token 的输入脉冲。
2. 把权重按差分电导写进 SPICE 的 crossbar 模型,$S$ 初始化为零。
3. 跑 16 个 token 周期的瞬态仿真,记录每个 DM 神经元的发放。
4. 与软件模型逐 token、逐神经元比对脉冲。

**判据:脉冲一致率 $\ge 90\%$**(容差来自电路的连续时间与软件的离散步之间的固有差异)。低于 90% 的话,先查 2.1 的量程匹配,再查 ⑤ 衰减的 $\alpha$ 映射。

同时输出:每 token 能耗(pJ,分项到 crossbar 读、gain-cell 读写、神经元、缓冲)、面积估计、以及 ADC 能耗占比(目标为零,这是要写进论文摘要的数字)。

---

<a name="3"></a>
## 3. 时间线与里程碑

| 阶段 | 内容 | 估时 | Gate |
|---|---|---|---|
| **1A** | Qwen3.5-0.8B:转 6 个 softmax 层,门从邻居初始化 | 1 周 | zero-shot ≥ 0.85 A₀,PPL ≤ 1.5 P₀ |
| **1B** | 改造 24 个 GDN 层、插 QCFS、冻 RMSNorm、微调 | 2 周 | PPL 相对 1A 退化 ≤ 20% |
| **1C1** | 纯转换,T=32 | 3 天 | PPL 有限且 ≤ 5× 1B |
| **1C2** | 蒸馏:嵌套 T=4 → 折叠 T=1 → ternary | 4–5 周 | T=1 PPL ≤ 4× 1A |
| **2.1** | DM 神经元 SPICE + 量程匹配 + 回灌 | 3 周(可与 1A/1B 并行) | 器件精度 PPL 退化 ≤ 30% |
| **2.2** | 线性注意力 demo 电路 | 4 周 | 六步时序在 SPICE 跑通 |
| **2.3** | Crossbar FFN(SwiGLU:三 crossbar + AND) | 1.5 周 | 量程闭环 |
| **2.5** | 软件-电路闭环 | 2 周 | 脉冲一致率 ≥ 90% |
| **1D + 2.4** | MoE(可选) | 3 周 | 路由一致率 ≥ 95% |

主线(不含 MoE)约 **4 个月**;含 MoE 约 5 个月。与 OPT 路线相比,1A 缩短一周、1B 和 1C2 各延长约一周,总时长基本持平,但 1A 的风险大幅下降。2.1 与 1A/1B 并行可以省三周。

**两个必须先做的事,决定后面所有参数:** 一是 1B 的校准数据要同时统计出六个脉冲点的输入分布(含 SwiGLU 的两处),2.1 的量程匹配靠它;二是 2.2 要先做 $\alpha$ 可调范围的物理上界分析——**这一条对 Qwen 路线尤其重要**,因为 18 个原生层的 $\alpha$ 分布是训好的、不由你选,若器件可调范围罩不住它,1B 就得把这些层的 $\alpha$ 钳进可实现区间再微调。

---

<a name="4"></a>
## 4. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| **delta 规则在二值 $k$ 下发散** | 高,1C 直接跑不出有限 PPL | 1.4 节的 $k$ 缩放 + $\beta \le 0.6$;C1 先用小 $\beta$ 验证稳定 |
| **$v$、$o$、SwiGLU 的 $W_3x$ 符号丢失导致 1B 退化远超 20%** | 中 | 提前到 1B 就引入 ternary 的 QCFS 双支路 |
| **改造零(去 L2 归一化 / 输出门 / conv1d)让 18 个原生层退化超预期** | 中,1B 达不到 gate | 逐项 ablation 定位;输出门改为 AND 门保留;conv1d 保留为 FIR |
| **6 个转换层与原生层头数 / kv 头数不一致** | 低 | head-wise 映射或线性投影对齐 |
| **全线性化损失召回能力** | 低(只换 6 层) | 接受;必要时保留第 24 层 softmax 作对照 |
| **原生层 $\alpha$ 分布超出 gain-cell 可调范围** | 中,GDN 遗忘门被钳 | 1A 前做物理上界分析;不够就在 1B 把 $\alpha$ 钳进可实现区间微调 |
| **折叠 T=1 精度崩塌** | 高 | ternary 是必需品;不够就退回嵌套 T=2 并让硬件加门控衰减 |
| **周期内两次读 + 一次写的时序冲突** | 中 | 2.2 用三相时钟显式分开;或用双端口 gain cell |
| **BPTT 显存(0.8B,T=4 ≈ 3.2B ANN)** | 中 | 梯度检查点、batch 2–4、seq 512、LoRA;不够降 T=2 过渡 |
| **软件-电路一致率 < 90%** | 高,闭环失败 | 逐步隔离:先只比 FFN 层,再比 GDN 层,定位到具体模块 |

---

<a name="5"></a>
## 5. 交付物清单

**阶段一**
- 1A:24 层全 GDN 的 ANN 权重 + 6 个转换层的邻居初始化配置与头数对齐方案 + 训练日志
- 1B:可脉冲化 ANN 权重 + 改造零的逐项 ablation + 六个脉冲点的输入分布统计 + 24 层原生 $\alpha$ 分布(供 2.2 的可调范围分析)
- 1C1:纯转换 SNN + T 扫描曲线
- 1C2:折叠 T=1 SNN(**冻结,作为电路功能规范**)+ 各模式 PPL/zero-shot 对照表
- 1D(可选):脉冲 MoE 权重 + 路由一致率

**阶段二**
- 2.1:Verilog-A 模型、SPICE 测试台、$g(a)$ 曲线族、查找表、器件精度 PPL
- 2.2:demo 电路 netlist、六步时序波形、$\alpha$ 可调范围分析
- 2.3:FFN netlist + 量程闭环报告
- 2.4(可选):MoE netlist + 路由一致率
- 2.5:闭环比对报告、energy/token 分项、面积估计、ADC 占比

**跨阶段**
- 一份"折叠 T=1 前向规范"文档,冻结所有数值细节(缩放常数、阈值、复位方式、门的限幅),是软件与电路之间的唯一接口定义。
