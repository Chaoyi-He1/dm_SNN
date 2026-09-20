# 计划书:小 LLM → 线性注意力 SNN → 扩散忆阻器电路 Demo

**日期** 2026-09-08(2026-09-14 修订:层数、层索引、头数与元件规模按 `config.json` 更正为 24 层、18 层 GDN 加 6 层全注意力;S1.2 复位律与拟合流程、S1.3 第 5 项、S1.4 第二判别点按 SPICE 初测结果补充;S1.2、S1.3 补入 GF180MCU 与 Zhao 2025 物理模型的真实器件结论)
**定位** 想法验证。性能下降可以接受;目标是整条链路能跑通、每个环节可解释、软件模型与电路模型在同一段输入上给出一致的输出。电路侧的目标是**一层真实的网络层**,不是整个模型。

---

## 目录

- [0. 总览](#0)
- [Stage 0:单 token 递推规范](#s0)
- [Stage 1:电路原语的 SPICE 表征与时序验证](#s1)
- [Stage 2:一层真实 GDN 层的脉冲化与行为级电路闭环](#s2)
- [Stage 3:验证 6 个 softmax 层的转换](#s3)
- [Stage 4:扩到整模型](#s4)
- [扩展:MoE](#ext)
- [时间线](#tl)
- [风险登记](#risk)
- [验收指标](#metrics)
- [交付物](#deliv)

---

<a name="0"></a>
## 0. 总览

### 0.1 目标

一个训练好的小语言模型,能否被改造成一个用扩散忆阻器神经元加忆阻器阵列执行的脉冲网络,其注意力为 Gated DeltaNet 线性递归形式,并且**其中一层**在校准过的电路模型上与软件模型给出一致的输出。

### 0.2 术语与器件

**SNN(Spiking Neural Network,脉冲神经网络)。** 激活值不是实数而是脉冲(0/1 或 −1/0/+1),神经元有膜电位状态,输入积分到阈值时发放。

**GDN(Gated DeltaNet)。** 一种线性注意力。它不维护随序列增长的 KV cache,而是维护一个固定尺寸的状态矩阵 $S$,每个 token 用 delta 规则做一次原地更新,再用 query 读出。Qwen3.5 系列的主要注意力层就是 GDN。

**折叠模式(T=1)。** SNN 通常在每个 token 内跑 $T$ 个子时间步(嵌套模式)。折叠模式令 $T=1$,让 token 轴成为唯一的时间轴,神经元的膜电位跨 token 携带。这是本项目电路的实际工作方式:一个 token 对应一个物理时钟周期。

**DM 神经元。** 用扩散忆阻器(diffusive memristor,DM)构成的脉冲神经元,结构为一个扩散忆阻器、一个晶体管、一个电阻(1M1T1R),具备泄漏积分、阈值发放、不应期等特性(Zhao et al., *Nature Electronics* 2025)。本项目采用其理想化稳定模型:衰减速率与脉冲幅度恒定。它承担网络中有状态的非线性,并兼作模数转换。

**Drift 忆阻器 crossbar(1T1R)。** 非易失忆阻器阵列,电导存静态权重,输入脉冲加在行(wordline),输出电流在列(bitline)求和,即模拟域的矩阵向量乘。承载所有线性投影和 FFN 权重。

**Gain-cell 阵列(3T1C)。** 电容型模拟存储单元,写入快、耐久无限、存在可控泄漏。承载 GDN 的状态矩阵 $S$;其泄漏正好实现 GDN 的衰减门。

**残差累加器。** 每通道一个电容,沿网络深度累加各块的输出,不进入任何 wordline。块入口由比较器读它并产生脉冲。

**行为级仿真器。** 用 SPICE 表征出的元件模型(神经元、gain cell、crossbar 列、比较器、门电路)替换规范软件模型中的理想原语,得到的一个能在几秒内跑完一层全宽网络的仿真器。SPICE 负责标定原语,行为级仿真器负责跑整层。

### 0.3 为什么电路只做一层,以及一层为什么不能全在 SPICE 里

递推、每个原语、残差累加器、SwiGLU 门控,在一层里全部出现;仿真更多层不增加任何新的验证内容。所以电路侧的目标是一层。

但一层 Qwen3.5-0.8B(隐层 1024,FFN 中间维 3584,线性注意力 16 头、$d_k=d_v=128$)的元件规模是:四个注意力投影 8,388,608 个忆阻器($3\times1024\times2048+2048\times1024$),三个 FFN 矩阵 11,010,048 个,状态阵列 262,144 个 gain cell(16 头 $\times128\times128$),17,408 个 DM 神经元($q,k,v,o_{\rm pre}$ 各 2048、$o_{\rm out}$ 1024、$g,u$ 各 3584、FFN 输出 1024)。SPICE 瞬态仿真跑几十个 token 周期,能承受的元件数在 $10^5$ 到 $10^6$ 量级,差一到两个数量级。因此保真度分两级:**SPICE 只表征原语并验证时序;一层全宽的仿真在行为级仿真器里做。**

### 0.4 执行顺序

| 阶段 | 内容 | 回答的问题 |
|---|---|---|
| **Stage 0** | 单 token 递推规范 | 软件和电路到底在实现哪一个递推 |
| **Stage 1** | 电路原语的 SPICE 表征与时序验证 | 每个原语的行为是什么;电路时序是否落在规范的递推上 |
| **Stage 2** | 一层真实 GDN 层的脉冲化 + 行为级电路闭环 | 一层真实层能否在校准过的电路模型上正确执行 |
| **Stage 3** | 验证 6 个 softmax 层的转换 | 原生权重能否迁移、哪种初始化可行 |
| **Stage 4** | 扩到整模型 | 整模型的退化有多大、能否回灌器件模型 |
| **扩展** | MoE | 留作扩展 |

顺序的原则是先回答最可能否定整个方案的问题。**最早的硬结论在约第 9 周出现,内容是"给定门值时,一层真实 Qwen 层在校准过的电路模型上跑通"**,此时语言模型侧只投入了一层的脉冲化改造和一层的 softmax 转换实验。Stage 2 用的是 Qwen 的原生 GDN 层,不依赖 Stage 3。

### 0.5 各阶段完成的定义

- **Stage 1:** DM 神经元的状态模型对 SPICE 的发放序列按正负分别统计 precision/recall $\ge0.95$;gain cell 的 $\alpha$ 可调区间和保持误差测出;16×16 分块的单步判别测试落在规范递推的闭式上;crossbar 列、神经元、锁存的组合与规范定义一致。
- **Stage 2:** 一层原生 GDN 层完成脉冲化改造并记录逐类 ΔPPL;行为级闭环中,该层输出脉冲按正负分别统计的 precision/recall $\ge0.9$、符号错误率 $\le2\%$,状态误差的最大值与末端值在容限内,把电路层放回模型后的 PPL 相对软件层 $\le1.1\times$。
- **Stage 3:** 单层转换的最优初始化方式确定,单层替换后 PPL $\le1.15\times$ 原模型。
- **Stage 4:** 24 层全 GDN 的折叠 T=1 模型 PPL 有限,相对 ANN 的退化被逐阶段记录;换入器件状态模型后 PPL 退化 $\le30\%$。

---

<a name="s0"></a>
## Stage 0:单 token 递推规范

### S0.1 为什么需要唯一的一份规范

同一个 GDN 递推可以写成几种形式,它们的差别在于每一步读取的是哪个时刻的状态。这些形式在数学上不等价,原生权重是在其中一种下训练的,电路时序也只能实现其中一种。软件和电路如果各自按自己的理解实现,即使都"正确",也对不上。本 Stage 只做一件事:写下唯一的一份规范,明确每个操作读取哪个时刻的状态、每个信号的编码、每个接口的物理载体。之后所有软件实现和电路时序都从这份规范推出。

### S0.2 GDN 的递推:先衰减,再检索

约定:状态 $S\in\mathbb{R}^{d_k\times d_v}$,$k,q\in\mathbb{R}^{d_k}$,$v\in\mathbb{R}^{d_v}$,门 $\alpha_\tau,\beta_\tau\in(0,1)$ 为标量。每个 token $\tau$ 按四步执行,每步读取的状态版本用不同符号标出:

$$\tilde S_\tau = \alpha_\tau\, S_{\tau-1} \qquad\text{(衰减:读 } S_{\tau-1}\text{)}$$
$$r_\tau = \tilde S_\tau^{\top} k_\tau \qquad\text{(检索:读衰减后的 } \tilde S_\tau\text{)}$$
$$S_\tau = \tilde S_\tau + \beta_\tau\, k_\tau\,(v_\tau - r_\tau)^{\top} \qquad\text{(写入:产生 } S_\tau\text{)}$$
$$o_\tau = S_\tau^{\top} q_\tau \qquad\text{(读出:读更新后的 } S_\tau\text{)}$$

**推导它与 GDN 原式一致。** 原式(Yang, Kautz, Hatamizadeh, ICLR 2025)是 $S_\tau = S_{\tau-1}\,\alpha_\tau(I-\beta_\tau k_\tau k_\tau^\top) + \beta_\tau v_\tau k_\tau^\top$。转到本文约定后展开:

$$S_\tau = \alpha_\tau(I-\beta_\tau k_\tau k_\tau^\top)S_{\tau-1} + \beta_\tau k_\tau v_\tau^\top = \alpha_\tau S_{\tau-1} + \beta_\tau k_\tau\big(v_\tau - \alpha_\tau S_{\tau-1}^\top k_\tau\big)^\top$$

括号里的检索项是 $\alpha_\tau S_{\tau-1}^\top k_\tau = \tilde S_\tau^\top k_\tau$,用的是衰减后的状态。这与 FLA 库 `fused_recurrent_gated_delta_rule` 内核的执行顺序一致(先 `h *= exp(g)`,再 `v -= h·k`,再写入,再读出),也是 HuggingFace `modeling_qwen3_5.py` 的顺序。本文称它为**形式 A**。

**两种容易混淆的写法。** 若从衰减前的状态检索(形式 B):$r_\tau = S_{\tau-1}^\top k_\tau$,展开得 $S_\tau=(\alpha_\tau I-\beta_\tau k_\tau k_\tau^\top)S_{\tau-1}+\beta_\tau k_\tau v_\tau^\top$。若先检索写入、再对整个结果衰减(形式 C):$S_\tau=\alpha_\tau(I-\beta_\tau k_\tau k_\tau^\top)S_{\tau-1}+\alpha_\tau\beta_\tau k_\tau v_\tau^\top$。三者沿 $k_\tau$ 方向的遗忘系数分别为 $\alpha_\tau(1-\beta_\tau\|k_\tau\|^2)$、$\alpha_\tau-\beta_\tau\|k_\tau\|^2$、$\alpha_\tau(1-\beta_\tau\|k_\tau\|^2)$,新值存入增益分别为 $\beta_\tau$、$\beta_\tau$、$\alpha_\tau\beta_\tau$。Qwen 的原生门是在形式 A 下训练的,迁移时不能混用。一个电路若在写入之后仍让状态衰减,实现的就是形式 C。

### S0.3 折叠 T=1 的脉冲化递推

折叠模式下 token 轴就是唯一的时间轴。每个 token 只有一次发放机会,膜电位跨 token 携带。以 $q$ 为例,其它有状态脉冲点同构:

$$U^{(q)}_\tau = \beta_m U^{(q)}_{\tau-1} + W_q x_\tau,\qquad q_\tau = \mathrm{fire}(U^{(q)}_\tau),\qquad U^{(q)}_\tau \leftarrow U^{(q)}_\tau - \theta\, q_\tau$$

$\beta_m$ 是膜泄漏系数,$\theta$ 是阈值,$x_\tau$ 是块输入脉冲(S0.6),$\mathrm{fire}$ 按 S0.5 的编码是二值或三值阶跃。膜电位跨 token 携带意味着 $q_\tau$ 不只取决于 $x_\tau$,还取决于最近几个 token 的输入。这是一个新的记忆通道,与 GDN 的状态 $S$ 并存,也是折叠模式区别于"把 $T$ 从 4 降到 1"的本质:它改变了模型的记忆机制,Stage 2 在真实层上验证它可学。

脉冲化后的 GDN 四步与 S0.2 完全相同,只是 $q_\tau,k_\tau,v_\tau$ 取离散值、$S$ 保持连续(软件里是浮点,电路里是电荷),$\alpha_\tau,\beta_\tau$ 保持浮点标量。

### S0.4 稳定性:条件与保证

**转移算子。** 由 S0.2 的闭式,$S_\tau = A_\tau S_{\tau-1} + \beta_\tau k_\tau v_\tau^\top$,其中 $A_\tau=\alpha_\tau(I-\beta_\tau k_\tau k_\tau^\top)$ 是对称矩阵,特征值为 $\alpha_\tau$(在 $k_\tau$ 的正交补上)和 $\alpha_\tau(1-\beta_\tau\|k_\tau\|^2)$(沿 $k_\tau$)。状态不发散要求沿 $k_\tau$ 的系数绝对值小于 1。

**离散 $k$ 带来的问题。** 连续值的 GDN 对 $k$ 做 L2 归一化,$\|k\|^2=1$,条件自动满足。二值或三值的 $k$ 无法逐 token 归一化(归一化会破坏离散性)。若只按平均发放率做缩放,$\|k_\tau\|^2$ 会随该 token 的非零个数 $n_\tau$ 波动,全发放的 token 会让 $\beta\|k\|^2$ 超过 2,状态沿 $k$ 方向发散。

**固定缩放(本规范采用)。** 令

$$k_\tau = \frac{s_\tau}{\sqrt{d_k}},\qquad s_\tau\in\{-1,0,+1\}^{d_k}$$

则 $\|k_\tau\|^2 = n_\tau/d_k \le 1$ 对任意输入成立,不需要任何竞争选择电路。

> **命题(状态一致有界).** 设 $S_0=0$。若对所有 $\tau$ 有 $\|k_\tau\|^2\le1$、$\beta_\tau\in(0,1)$、$\alpha_\tau\le\alpha_{\max}<1$,且 $v_\tau$ 为三值向量,则 $\sup_\tau\|S_\tau\|_F\le\sqrt{d_v}/(1-\alpha_{\max})$。若 $S_0\ne0$,上界为 $\alpha_{\max}^{\tau}\|S_0\|_F+\sqrt{d_v}/(1-\alpha_{\max})$。
>
> *证明.* $\beta_\tau\|k_\tau\|^2\in[0,1)$,故 $\alpha_\tau(1-\beta_\tau\|k_\tau\|^2)\in(0,\alpha_\tau]$,于是 $\|A_\tau\|_2=\alpha_\tau$。由三角不等式 $\|S_\tau\|_F\le\alpha_\tau\|S_{\tau-1}\|_F+\beta_\tau\|k_\tau\|\|v_\tau\|\le\alpha_{\max}\|S_{\tau-1}\|_F+\sqrt{d_v}$,展开几何级数即得。$\square$

**代价与实验问题。** 沿 $k_\tau$ 方向的有效写入强度是 $\beta_\tau n_\tau/d_k$。稀疏 token($n_\tau\approx0.2\,d_k$)的写入只有 $0.2\beta_\tau$,同一个 key 要出现约 5 次才能完全覆盖旧值。这是否损害语言建模,由 Stage 2 在真实层上直接测量;$\beta$ 的门可以学习补偿,读出神经元的阈值可以吸收尺度。

**k-WTA 作为增强实验,不进首版。** 用 k-winners-take-all 侧向抑制把 $n_\tau$ 封顶到 $n_{\max}$、幅度取 $1/\sqrt{n_{\max}}$,可以把写入强度提到 $\beta_\tau$ 量级。但要注意软件和电路的选择语义不同:软件若按周期末 $|U|$ 取 top-$n_{\max}$,电路的共享抑制线按先发放者选中,而膜电位跨 token 携带后,先发放者由 $U_{\tau-1}$ 和当前输入共同决定,两者选出的集合一般不同。若做这个实验,软件必须按到达阈值的时间排序建模,而不是按 $|U|$。

**四项压力测试(直接向递推核心注入 $k,v,q$ 和门值,绕过编码器;编码器另测):**

1. 全零 key 连续 200 token,$\alpha_\tau$ 取允许区间内任意固定值:$\|S_\tau\|_F$ 应等于 $\big(\prod_{s\le\tau}\alpha_s\big)\|S_0\|_F$,相对偏差 $\le1\%$。$\alpha=0.99$ 时 200 步后剩余约 13.4%,这是正常衰减,不是故障。
2. 全一 key($s=\mathbf 1$,$\|k\|^2=1$)连续 200 token:$\|S_\tau\|_F$ 不超过命题的界。
3. 连续稠密随机 key($n_\tau=d_k$)1000 token:$\|S_\tau\|_F$ 有界。
4. 真实层在 4096 token 的真实文本上:$\|S_\tau\|_F$ 有界且有限。

### S0.5 编码:每个脉冲点一次定死

每个脉冲点的编码(二值或三值)按最终电路一次确定,之后所有阶段不再改。先训非负表示、再在后期恢复符号的做法等于换了两次问题,不采用。

| 脉冲点 | 编码 | 有无状态 | 理由 |
|---|---|---|---|
| 块输入 $x=Q_{\rm in}(h)$ | 三值 $\{-1,0,+1\}$ | **无状态**(阈值逐层可学) | 残差累加器的读出,见 S0.6 |
| $q$ | 二值 $\{0,1\}$ | 有 | 读出地址,选行求和,符号信息在 $S$ 里 |
| $k$ | 三值 $\{-1,0,+1\}/\sqrt{d_k}$ | 有 | 负分量提供负向召回;固定缩放保证 $\|k\|^2\le1$ |
| $v$ | 三值 $\{-1,0,+1\}$ | 有 | 内容,符号必须保留 |
| $W_O$ 前的 $o$ | 三值 | 有 | 内容 |
| $W_O$ 后的输出 | 三值 | 有 | 内容 |
| SwiGLU 的 $W_1x$(门) | 二值 | 有 | SiLU 门主要为非负 |
| SwiGLU 的 $W_3x$(内容) | 三值 | 有 | 内容 |
| FFN 输出 | 三值 | 有 | 内容 |

有状态的脉冲点由 DM 神经元实现,膜电位跨 token 携带;无状态的 $Q_{\rm in}$ 由比较器实现。这个划分的理由在 S0.6。$k$ 取二值作为 Stage 2 的一个消融项保留,默认三值。

### S0.6 残差、逐元素乘、$W_O$、门的精确定义

**残差:累加器,不上 wordline。** 残差流 $h$ 是一个沿深度累加的实值向量。它的初值是该 token 的嵌入,每个块把自己的三值输出加到它上面。$h$ 本身永远不进入任何线性层;每个块的输入是

$$x = Q_{\rm in}(h) = \mathrm{sign}(h)\cdot\mathbb{1}\big[|h|\ge\theta_{\rm in}\big]\in\{-1,0,+1\}^d$$

一个无状态的三值阈值,$\theta_{\rm in}$ 逐层可学。于是线性层只看到三值脉冲,不存在"整数怎么编码上 wordline"的问题。$h$ 的取值范围随深度线性增长:$L$ 个块、每块两个残差,$\|h\|_\infty\le\|\mathrm{emb}\|_\infty+2L$。

电路上,$h$ 是每通道一个电容;块输出 $y$ 的每个分量注入 $\pm Q_0$ 或不注入;$Q_{\rm in}$ 是两个比较器(阈值 $\pm\theta_{\rm in}$)。嵌入的写入需要一个 DAC,这是整个电路里唯一的 DAC,在输入端。单层仿真时,该层入口的累加器值由软件记录并写入,量程按该层实际统计值定。

**为什么 $Q_{\rm in}$ 无状态。** 累加器已经沿深度保存了该 token 的全部信息;若 $Q_{\rm in}$ 也带膜电位跨 token 携带,就是在累加器之上再做一次积分,引入一个没有明确作用的记忆通道。有状态的非线性只放在投影之后的 DM 神经元上。

**逐元素乘(SwiGLU 门控)。** 软件里 $p = g\odot u$,$g\in\{0,1\}$,$u\in\{-1,0,+1\}$,$p\in\{-1,0,+1\}$。电路上,同一 token 内两路神经元的发放时刻不同,裸 AND 会漏掉不重叠的脉冲。规范规定:每路各一个"本周期已发放"锁存,周期末对锁存做 AND;三值路的符号由其 $(S^+,S^-)$ 哪一路锁存决定;乘积的符号取 $u$ 的符号(因 $g\ge0$)。锁存结果在下一个脉冲槽驱动 $W_2$。

**$W_O$ 前后各一个神经元。** $o_\tau=S_\tau^\top q_\tau$ 是连续值(电路里是模拟电流),必须先脉冲化才能进 $W_O$ 的 1-bit wordline。规范规定:$o\to\mathcal{SN}_{\rm pre}\to W_O\to\mathcal{SN}_{\rm out}$,两个神经元,软件和电路一致。这与 BiSpikCLM(Guo et al., 2026)Algorithm 1 的 `SN_AttnOut → Linear_out → SN_Out` 相同。

**门:首个闭环用外部给定值。** 一个 token 只有一次发放机会时,单个脉冲神经元给不出连续的 $\alpha_\tau,\beta_\tau$;若用多次发放估计频率,测量时间和对神经元状态的影响都要计入周期。规范因此分两步:

- **首个闭环(S2.4):** $\alpha_\tau,\beta_\tau$ 由软件模型算出,作为外部输入施加(泄漏管栅压、电流镜比例)。该实验的结论边界是"给定门值时,递推核心在电路上成立"。
- **门生成电路(S1.6、S2.5):** 单独实验。首选模拟通路——门 crossbar 的输出电流经一个差分对做挤压,直接产生 $V_{\rm leak}$ 和镜像比例,不经过脉冲;备选是子 token 窗口内多次发放的频率估计,此时须把 $m$ 次 ISI 计入 token 周期。软件的门函数在门电路测定后拟合其传输特性,此前用 sigmoid,输入为块输入脉冲 $x_\tau$。$\alpha$ 钳到硬件可实现区间 $[\alpha_{\min},\alpha_{\max}]$(S1.3 测出;此前暂用 $[0.5,0.99]$)。

### S0.7 电路时序:占空比泄漏

S0.2 要求衰减只作用于 $S_{\tau-1}$,新写入的增量不被衰减。gain cell 的泄漏在墙钟时间里连续发生,所以必须把泄漏限定在一个窗口内。

一个 token 周期的相位:

| 相位 | 动作 | 泄漏管 | 读写的状态版本 |
|---|---|---|---|
| φ0 | 累加器写入(单层仿真:软件记录值;整模型:上一块输出);$Q_{\rm in}$ 读累加器得 $x_\tau$;由 $x_\tau$ 得门 $\alpha_\tau,\beta_\tau$(首个闭环:外部给定),设定 $V_{\rm leak}$ | 关 | — |
| φ1 泄漏窗 $t_{\rm gap}$ | 什么都不做,让 $S_{\tau-1}\to\tilde S_\tau$ | **开** | 读 $S_{\tau-1}$ |
| φ2 投影 | $x_\tau$ 打 $W_{q,k,v}$ crossbar,神经元积分发放,锁存 $q,k,v$ | 关 | — |
| φ3 检索 | $k$ 打阵列行读线,列电流 $=\tilde S_\tau^\top k$ | 关 | 读 $\tilde S_\tau$ |
| φ4 写入 | $v-r$ 差分、$\beta$ 缩放得 $\Delta v$;先写 $k=+1$ 的行,再反相写 $k=-1$ 的行 | 关 | 产生 $S_\tau$ |
| φ5 读出 | $q$ 打行读线,列电流 $=S_\tau^\top q$,灌 $\mathcal{SN}_{\rm pre}$ | 关 | 读 $S_\tau$ |
| φ6 | $W_O$、$\mathcal{SN}_{\rm out}$ 得 $y$;累加器 $+y$;$Q_{\rm in}$ 读累加器得 $x'$ | 关 | — |
| φ7 | $x'$ 打 $W_1,W_3$;$\mathcal{SN}_1,\mathcal{SN}_3$;锁存;AND | 关 | — |
| φ8 | $W_2$、$\mathcal{SN}_2$ 得 $y'$;累加器 $+y'$ | 关 | — |

$\alpha_\tau$ 与器件的映射为 $\alpha_\tau=\exp(-t_{\rm gap}/\tau_{\rm leak}(V_{\rm leak},\tau))$。φ2–φ8 期间泄漏管关断,但 gain cell 仍有本征亚阈值泄漏,它在这几个相位内造成的误差记为 $\epsilon_{\rm hold}$,Stage 1 要测出来并计入误差预算。

### S0.8 规范的地位与冻结条件

S0.2–S0.7 合起来写成一份独立文档《folded-T1-spec.md》,带版本号。所有软件实现、行为级仿真器和电路时序都引用它。任何一方要改递推、编码或时序,先改规范再改实现。

三个接口——残差的累加器模型、门值的来源、$k$ 的固定缩放——已在本规范中确定。规范在 Stage 2 给出固定缩放下真实层的 ΔPPL 后冻结;若该结果要求引入 WTA,则按 S0.4 的语义要求修订后再冻结。

### S0.9 规范的软件参考实现

下面是规范的直接翻译。它是 Stage 2 脉冲化真实层的构件,也是行为级仿真器替换原语的基线。压力测试直接调用 `SpikingGDN.core()`。

```python
import math
import torch
import torch.nn as nn


class SpikeFn(torch.autograd.Function):
    """前向:u >= 0 发放。反向:arctan 代理梯度。"""
    @staticmethod
    def forward(ctx, u, alpha):
        ctx.save_for_backward(u)
        ctx.alpha = alpha
        return (u >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        x = (math.pi / 2) * ctx.alpha * u
        return g * (ctx.alpha / 2) / (1 + x * x), None


def ternary_fire(U, thr, alpha):
    """无状态三值阈值:U >= thr 得 +1,U <= -thr 得 -1。"""
    return SpikeFn.apply(U - thr, alpha) - SpikeFn.apply(-U - thr, alpha)


class BinaryLIF(nn.Module):
    """二值 LIF,有状态:膜电位 U 显式传入传出,跨 token 携带。"""
    def __init__(self, thr=1.0, beta=0.9, alpha=2.0):
        super().__init__()
        self.thr, self.beta, self.alpha = thr, beta, alpha

    def forward(self, I, U):
        U = self.beta * U + I
        S = SpikeFn.apply(U - self.thr, self.alpha)
        U = U - self.thr * S
        return S, U


class TernaryLIF(nn.Module):
    """三值 LIF,有状态。"""
    def __init__(self, thr=1.0, beta=0.9, alpha=2.0):
        super().__init__()
        self.thr, self.beta, self.alpha = thr, beta, alpha

    def forward(self, I, U):
        U = self.beta * U + I
        S = ternary_fire(U, self.thr, self.alpha)
        U = U - self.thr * S
        return S, U


class QuantizerIn(nn.Module):
    """块输入量化器:无状态三值阈值,阈值逐层可学。电路上是两个比较器。"""
    def __init__(self, thr0=1.0, alpha=2.0):
        super().__init__()
        self.thr = nn.Parameter(torch.tensor(float(thr0)))
        self.alpha = alpha

    def forward(self, h):
        return ternary_fire(h, self.thr, self.alpha)


class SpikingGDN(nn.Module):
    """按规范:先衰减 → 检索 → 写入 → 读出;W_O 前后各一个神经元。
    多头:头维 dk, dv,头数 H。core() 独立于门的来源和编码器。"""
    def __init__(self, d, H, dk, dv, alpha_rng=(0.5, 0.99)):
        super().__init__()
        self.H, self.dk, self.dv = H, dk, dv
        self.Wq = nn.Linear(d, H * dk, bias=False)
        self.Wk = nn.Linear(d, H * dk, bias=False)
        self.Wv = nn.Linear(d, H * dv, bias=False)
        self.Wo = nn.Linear(H * dv, d, bias=False)
        self.Wa = nn.Linear(d, H)
        self.Wb = nn.Linear(d, H)
        self.sn_q, self.sn_k, self.sn_v = BinaryLIF(), TernaryLIF(), TernaryLIF()
        self.sn_pre, self.sn_out = TernaryLIF(), TernaryLIF()
        self.c_k = 1.0 / math.sqrt(dk)                # 固定缩放:||k||^2 = n/dk <= 1
        self.a_lo, self.a_hi = alpha_rng

    def init_state(self, B, device):
        z = lambda *s: torch.zeros(*s, device=device)
        H, dk, dv = self.H, self.dk, self.dv
        return dict(S=z(B, H, dk, dv), Uq=z(B, H * dk), Uk=z(B, H * dk),
                    Uv=z(B, H * dv), Upre=z(B, H * dv), Uout=z(B, self.Wo.out_features))

    def gates(self, x):
        """软件门:sigmoid,输入为块输入脉冲。门电路测定后换成其传输特性。"""
        a = self.a_lo + (self.a_hi - self.a_lo) * torch.sigmoid(self.Wa(x))   # [B,H]
        b = torch.sigmoid(self.Wb(x))                                          # [B,H]
        return a, b

    @staticmethod
    def core(k, v, q, a, b, S):
        """递推核心。k: [B,H,dk] 已缩放三值;v: [B,H,dv] 三值;q: [B,H,dk] 二值;
        a, b: [B,H];S: [B,H,dk,dv]。返回读出 o: [B,H,dv] 和新状态。"""
        S_tilde = a[..., None, None] * S                       # 衰减:读 S_{t-1}
        r = torch.einsum("bhij,bhi->bhj", S_tilde, k)          # 检索:读 S_tilde
        dv = b[..., None] * (v - r)
        S = S_tilde + torch.einsum("bhi,bhj->bhij", k, dv)     # 写入:产生 S_t
        o = torch.einsum("bhij,bhi->bhj", S, q)                # 读出:读 S_t
        return o, S

    def step(self, x, st):
        """x: [B, d] 三值脉冲(块输入)。"""
        B = x.shape[0]
        H, dk, dv = self.H, self.dk, self.dv
        a, b = self.gates(x)
        q, st["Uq"] = self.sn_q(self.Wq(x), st["Uq"])
        k, st["Uk"] = self.sn_k(self.Wk(x), st["Uk"])
        v, st["Uv"] = self.sn_v(self.Wv(x), st["Uv"])
        o, st["S"] = self.core(k.view(B, H, dk) * self.c_k, v.view(B, H, dv),
                               q.view(B, H, dk), a, b, st["S"])
        o_spk, st["Upre"] = self.sn_pre(o.reshape(B, H * dv), st["Upre"])
        y, st["Uout"] = self.sn_out(self.Wo(o_spk), st["Uout"])
        return y, st


class SpikingSwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W3 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)
        self.sn_1, self.sn_3, self.sn_2 = BinaryLIF(), TernaryLIF(), TernaryLIF()

    def init_state(self, B, d_ff, d, device):
        z = lambda *s: torch.zeros(*s, device=device)
        return dict(U1=z(B, d_ff), U3=z(B, d_ff), U2=z(B, d))

    def step(self, x, st):
        g, st["U1"] = self.sn_1(self.W1(x), st["U1"])
        u, st["U3"] = self.sn_3(self.W3(x), st["U3"])
        p = g * u                                     # 折叠下即"锁存 + AND",符号取 u
        y, st["U2"] = self.sn_2(self.W2(p), st["U2"])
        return y, st


class SpikingBlock(nn.Module):
    """一层:GDN + SwiGLU,两个累加器残差。"""
    def __init__(self, d, d_ff, H, dk, dv):
        super().__init__()
        self.q_in1, self.attn = QuantizerIn(), SpikingGDN(d, H, dk, dv)
        self.q_in2, self.ffn = QuantizerIn(), SpikingSwiGLU(d, d_ff)

    def step(self, h, st):
        """h: 残差累加器(实值),不进任何线性层。逐 token 调用。"""
        x = self.q_in1(h)
        y, st["attn"] = self.attn.step(x, st["attn"])
        h = h + y
        x = self.q_in2(h)
        y, st["ffn"] = self.ffn.step(x, st["ffn"])
        h = h + y
        return h, st


@torch.no_grad()
def core_stress_tests(dk=16, dv=16, alpha=0.99, beta=0.5, device="cpu"):
    """S0.4 的测试 1–3:直接向递推核心注入 k, v, q 和门值,绕过编码器。单头。"""
    B, H = 1, 1
    bound = math.sqrt(dv) / (1 - alpha)
    ck = 1.0 / math.sqrt(dk)
    a = torch.full((B, H), alpha, device=device)
    b = torch.full((B, H), beta, device=device)
    q = torch.ones(B, H, dk, device=device)
    rnd_v = lambda: torch.randint(-1, 2, (B, H, dv), device=device).float()
    res = {}
    S = torch.randn(B, H, dk, dv, device=device); S0 = S.norm().item(); ok = True
    for t in range(1, 201):
        _, S = SpikingGDN.core(torch.zeros(B, H, dk, device=device), rnd_v(), q, a, b, S)
        expect = (alpha ** t) * S0
        ok &= abs(S.norm().item() - expect) <= 0.01 * expect
    res["zero_key_follows_alpha_product"] = ok
    S = torch.zeros(B, H, dk, dv, device=device); ok = True
    k = torch.ones(B, H, dk, device=device) * ck
    for _ in range(200):
        _, S = SpikingGDN.core(k, rnd_v(), q, a, b, S)
        ok &= S.norm().item() <= bound * 1.05
    res["all_one_key_bounded"] = ok
    S = torch.zeros(B, H, dk, dv, device=device); ok = True
    for _ in range(1000):
        s = torch.randint(0, 2, (B, H, dk), device=device).float() * 2 - 1
        _, S = SpikingGDN.core(s * ck, rnd_v(), q, a, b, S)
        ok &= S.norm().item() <= bound * 1.05
    res["dense_key_bounded"] = ok
    return res
```

编码器(各 `sn_*`)的正确性单独做单元测试:给定设计好的输入,检查输出脉冲与预期一致;不与稳定性测试混在一起。

### S0.10 可选探针:d=16 玩具模型

用上面的构件搭一个 d=16、单头、1 层的模型,在关联检索和延迟复制上从零训几分钟,可以提前几周看到折叠 T=1 是否可学。它不进入电路验证,不是任何阶段的前提;不做的话,这个问题在 S2.1 的真实层脉冲化里得到回答。

---

<a name="s1"></a>
## Stage 1:电路原语的 SPICE 表征与时序验证

### S1.1 目的

用 SPICE 回答两个问题:每个原语的行为是什么(给行为级仿真器提供标定模型);电路时序是否落在规范的形式 A 上。本 Stage 不需要任何训练好的模型。

### S1.2 DM 神经元:带状态的单步映射

静态发放率曲线 $g(a)$(恒定输入下的发放率对输入)描述不了折叠模式下起作用的跨 token 膜电位、复位深度和不应期,不能用它替换软件模型里的发放函数。

**做法。** 从 SPICE 瞬态里拟合一个带状态的单步映射:

$$(U_{\tau-1},\ I_\tau,\ c_{\tau-1})\ \longmapsto\ (S_\tau,\ U_\tau,\ c_\tau)$$

其中 $c$ 是不应期计数器。参数化形式:

$$U_\tau = \beta_m U_{\tau-1} + g_m I_\tau,\qquad S_\tau=\mathbb{1}[c_{\tau-1}=0]\cdot\mathrm{fire}(U_\tau)$$
$$\text{发放时:}\quad U_\tau\leftarrow U_{\rm hold}\cdot\mathrm{sign}(S_\tau)+\rho\,\big(U_\tau-\theta\cdot\mathrm{sign}(S_\tau)\big),\qquad c_\tau=\begin{cases}n_{\rm ref} & S_\tau\ne0\\ \max(c_{\tau-1}-1,0)&\text{else}\end{cases}$$

六个参数 $(\beta_m, g_m, \theta, U_{\rm hold}, \rho, n_{\rm ref})$:膜泄漏、电荷增益、阈值、保持电压、越阈剩余的回充比例、不应期长度(以 token 周期计)。复位的物理图像是:越阈后器件导通,把膜电容放电到保持电压 $U_{\rm hold}$ 时关断(放电时间远短于周期),之后输入电流在周期剩余时间里继续积分;越阈时刻之后本会积累的部分是 $U_\tau-\theta$,其中比例 $\rho$ 留在电容上。$\rho=0$ 是硬复位到 $U_{\rm hold}$;$\rho=1$ 且 $U_{\rm hold}=0$ 就是"减阈值"的软复位,所以软件参考实现的 LIF 是它的特例。三值神经元的负支路对称。把复位写成"回到 $U_{\rm hold}$ 后部分放电"的形式在这类器件上回归会退化($\rho\to1$ 时 $U_{\rm hold}$ 不可辨),不采用。

从 SPICE 里用一串随机电流序列(一个周期一个值)驱动 1M1T1R,记录周期边界上的膜电压和发放事件,拟合:非发放周期做两遍最小二乘得 $(\beta_m,g_m)$,剔除边界采样落在放电中途的离群点;$n_{\rm ref}$ 取最小发放间隔减一;$\theta$ 在非不应期样本上按"发放 ⇔ $|U_{\rm pre}|\ge\theta$"分类错误最少来选;发放周期再做两遍最小二乘得 $(\rho,U_{\rm hold})$。发放事件按导通的起始边沿计数。

**真实器件上的结果(2026-09-14)。** 用 Zhao et al. 2025 补充材料 Note 4 的物理模型(细丝 $x_1$、残留 $x_2$、栅电压 $V_{\rm gs}$,Table 1 参数)代替理想阈值开关后,上述六参数拟合的 precision/recall 为 0:该器件的积分量是离子残留而非栅电压,且增长加速,线性 LIF 抽象不成立。Stage 2.3 的行为级仿真器直接用物理模型替换 LIF(`snn_spec/adm.py`),六参数模型仅保留给理想开关类器件。详见 `step1-spiking-layer/results/adm-neuron.md`。

**输入量程两个上界(规范"每 token 至多一个脉冲"依赖它们)。** 每周期输入电荷小于 $C_m(\theta-U_{\rm hold})$,否则复位后同一周期内会再次越阈;导通期间输入电流小于保持电流 $U_{\rm hold}/R_{\rm on}$,否则器件锁定在导通态不再发放。两者一起给出神经元前电流镜比例的取值范围,是 S1.5 量程匹配的输入。

**验证。** 在另一组随机序列上,拟合模型预测的发放序列与 SPICE 的发放序列比对:正、负事件分别算 precision/recall,均 $\ge0.95$;符号错误率 $\le1\%$。静态 $g(a)$ 曲线只作校验:恒定输入下拟合模型的稳态发放率应与 SPICE 一致。

### S1.3 gain cell 与 16×16 阵列

3T1C gain cell,电容以 $V_{\rm mid}$ 为零点表示有符号电荷,泄漏管栅压 $V_{\rm leak}$ 可编程,读写分相。SPICE 要测四件事:

1. **rank-1 写入。** 行加 $k$(三值:先写 $+1$ 行,再反相写 $-1$ 行),列加 $\Delta v$,测各 cell 电荷增量与 $k_i\Delta v_j$ 的偏差。
2. **读。** 行加 $q$ 或 $k$,测列电流与 $\sum_i q_iS_{ij}$ 的偏差。
3. **可调衰减范围。** 扫 $V_{\rm leak}$,测 $t_{\rm gap}$ 内的 $\alpha$,得 $[\alpha_{\min},\alpha_{\max}]$。这个区间回填 S0.6 的 $\alpha$ 钳位,并直接决定 Stage 2 里原生层的 $\alpha$ 分布是否需要钳位。
4. **保持误差。** 泄漏管关断时,φ2–φ8 时长内的本征泄漏 $\epsilon_{\rm hold}$。
5. **泄漏是否欧姆型。** 同一泄漏设定下分别以不同存储值测 $\alpha$。GF180MCU 真实晶体管上,栅压调节的三种泄漏(接地、亚阈区到 $V_{\rm mid}$、三极管区到 $V_{\rm mid}$)都随存储值变化;只有"泄漏窗内全开的开关串电阻"给出与存储值无关的 $\alpha=\exp(-t_{\rm gap}/RC)$(偏差 $\le0.3\%$,离散 $\le1.4\times10^{-4}$)。因此 $\alpha$ 按头用泄漏窗长度 $t_{\rm gap}$ 设定,S0.7 的映射改为 $\alpha_\tau=\exp(-t_{\rm gap}/RC)$。见 `step1-spiking-layer/results/spice-gf180-leak.md`。

### S1.4 时序验证:单步判别测试

这是 Stage 0 那个问题在电路上的直接检验,不需要模型。在 16×16 分块上注入一组已知量,跑一个完整的 φ0–φ8 周期,把 $S_1$ 与三个闭式比对。

取 $S_0$ 只在 cell (1,1) 有值 1,其余为零;$k=e_1$(行 1 正脉冲),$v$ 的第 1 分量为 2,$q=e_1$,$\alpha=0.5$(对应的 $V_{\rm leak}$ 由 S1.3 的标定查得),$\beta=0.5$(镜像比例)。三个闭式给出的 $S_1[1,1]$:

| 形式 | $S_1[1,1]$ |
|---|---|
| A(规范) | $0.5\times(1-0.5)\times1+0.5\times2=$ **1.25** |
| B | $(0.5-0.5)\times1+0.5\times2=$ **1.00** |
| C | $0.5\times(1-0.5)\times1+0.5\times0.5\times2=$ **0.75** |

**判据:** 测得的 $S_1[1,1]$ 与 1.25 的偏差 $\le0.05$,且与 1.00、0.75 的距离都大于 0.15。全零 key 测试分不开三者(它们在 $k=0$ 时都退化为纯衰减),所以必须用这个单步测试。S0.4 的压力测试 1–3 也在分块上直接注入复跑。

**第二判别点。** 在 $\alpha=\beta=0.5$ 上,"泄漏管整周期常开、周期中点检索并写入"的电路给出约 0.97,与形式 B 的 1.00 只差 0.03:单步测试能拒绝它,却会把原因误判为检索顺序反了。补测 $\alpha=0.25$、$\beta=1$($S_0$、$k$、$v$、$q$ 不变):A 2.00、B 1.25、C 0.50,常开约 1.02,与三者距离都超过 0.15。两点都落在 A 上才算时序正确。单 cell 的 SPICE 已按两点验证(`results/spice-stage1.md`),16×16 分块待做。

### S1.5 crossbar 列、神经元、累加器、锁存的组合

**crossbar 列 + 神经元。** 一列 1T1R(差分对,含线电阻)加一个 DM 神经元:输入一串 wordline 脉冲,测 bitline 电流和神经元发放,与"理想电流 × 拟合神经元模型"的预测比对。这一步给出 IR drop 与量化级数对发放的影响,进入行为级仿真器的 crossbar 模型。

**累加器 + 比较器。** 一个电容、$\pm Q_0$ 注入、两个比较器:测线性度、比较器失调、$\theta_{\rm in}$ 可编程范围、所需分辨率随注入次数的关系。

**乘法锁存 + AND。** 两路神经元、两个周期锁存、一个 AND:验证不重叠脉冲被正确捕获,三值符号逻辑正确。

**$W_O$ 双神经元。** 列电流 → $\mathcal{SN}_{\rm pre}$ → 一列 crossbar → $\mathcal{SN}_{\rm out}$,验证与软件的两级脉冲化一致。

### S1.6 门生成电路

门 crossbar 输出电流 → 差分对挤压 → $V_{\rm leak}$ 与镜像比例。测传输特性 $V_{\rm leak}(I)$、$\beta(I)$,拟合成函数。备选是子 token 窗口内 $m$ 次发放的频率估计,须把 $m$ 次 ISI 计入周期。

### S1.7 Gate

S1.2 拟合达标;S1.3 得到 $[\alpha_{\min},\alpha_{\max}]$、$\epsilon_{\rm hold}$、读写误差;S1.4 落在形式 A 上;S1.5 各组合与规范定义一致;S1.6 门函数拟合完成。所有标定结果打包成行为级仿真器的元件模型。

---

<a name="s2"></a>
## Stage 2:一层真实 GDN 层的脉冲化与行为级电路闭环

### S2.1 选层与逐类脉冲化改造

从 Qwen3.5-0.8B 的 18 个原生 GDN 层里选一层(建议取中间层;`config.json` 给出全注意力层索引为 3、7、11、15、19、23(0 基),GDN 为 16 个 key 头、16 个 value 头、头维 128)。其余 23 层保持 ANN 不动。对这一层按下面的顺序**每次只改一类组件,改后短微调(LoRA,5M token),记录增量 PPL**:

1. RMSNorm 冻结成缩放;
2. conv1d:保留为 4 抽头 FIR / 去掉,二选一并记录;
3. q/k 的 L2 归一化 → 固定缩放 $1/\sqrt{d_k}$;**这一步的 ΔPPL 就是 S0.8 冻结条件要的数**;
4. 输出门:去掉 / 改为锁存 AND,二选一并记录;
5. 残差改为累加器 + 无状态 $Q_{\rm in}$(三值,阈值可学);
6. QCFS 插入:$q$ 二值,$k,v$ 三值,$o$ 三值且 $W_O$ 前后各一;
7. SwiGLU:$W_1x$ 二值 QCFS,$W_3x$ 三值 QCFS,输出三值 QCFS。

QCFS(Quantization Clip-Floor-Shift)是一个"长得像 IF 神经元"的激活函数,$a=\lambda\,\mathrm{clip}(\frac{1}{L}\lfloor \frac{zL}{\lambda}+\frac12\rfloor,0,1)$,$\lambda$ 可学习、之后直接映射为 SNN 阈值,$\frac12$ 的平移使转换误差期望为零(Bu et al., ICLR 2022)。

改完 7 类后,该层是一个"可脉冲化的 ANN 层"。接着做**折叠转换**:QCFS 换成 LIF,阈值取学到的 $\lambda$,膜电位跨 token 携带,该层逐 token 顺序执行,其余层照常并行。先零训练看 PPL,再对该层做短 BPTT 微调(只有一层、序列长 512,显存可忽略)。这一步同时回答"折叠 T=1 在真实层上是否可学"。

微调的 loss 只写成函数,模型调用、loss 计算、参数更新放在一个训练步里:

```python
import torch.nn.functional as F


def layer_loss(logits, labels, y_spk, y_ref, lam=0.1):
    """LM 交叉熵 + 该层输出与改造前参考输出的 MSE。"""
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    return ce + lam * F.mse_loss(y_spk, y_ref)


def train_one_step(model, ref_layer_out, batch, opt, clip=0.7):
    tokens, labels = batch
    logits, y_spk = model(tokens, return_layer_out=True)
    loss = layer_loss(logits, labels, y_spk, ref_layer_out(tokens))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return loss.item()
```

**Gate:** 7 类累计 ΔPPL $\le30\%$;折叠转换后 PPL 有限,短微调后相对折叠前 $\le1.2\times$;S0.4 的测试 4(4096 token 真实文本)通过。

### S2.2 记录该层在真实文本上的输入

在 2048 token 的真实文本上,记录该层每个 token 的:入口累加器值 $h_\tau$、门值 $\alpha_\tau,\beta_\tau$、以及软件模型的 $q_\tau,k_\tau,v_\tau,S_\tau,o_\tau$、两个残差后的 $y_\tau,y'_\tau$。这是行为级仿真器的输入和比对基准。

### S2.3 行为级仿真器:规范模型 + 标定原语

行为级仿真器就是 S0.9 的软件模型,把每个理想原语换成 Stage 1 标定出的模型:

| 理想原语 | 替换为 |
|---|---|
| `nn.Linear`(投影、FFN) | crossbar 模型:电导量化到 $N$ 级、编程误差 $\sigma_G$、按列的 IR drop、读噪声;输出为 bitline 电流 |
| `BinaryLIF` / `TernaryLIF` | S1.2 的六参数状态模型;输入电流按相位时长积分到膜电容 |
| `core()` 的衰减、检索、写入、读出 | gain-cell 阵列模型:S1.3 的写入误差、$\alpha(V_{\rm leak})$ 查表、$\epsilon_{\rm hold}$、读误差;衰减只在 φ1 窗口内发生 |
| `QuantizerIn` | 比较器模型:失调、迟滞 |
| `g * u` | 锁存 + AND(理想数字) |
| `gates()` | 首个闭环:直接读 S2.2 记录的 $\alpha_\tau,\beta_\tau$;S2.5 起换成 S1.6 的传输函数 |

```python
class CrossbarModel:
    """替换 nn.Linear。W 量化到 N 级电导,含编程误差与列 IR drop。"""
    def __init__(self, W, n_levels, sigma_g, r_wire): ...
    def __call__(self, x_spk):            # x_spk: {-1,0,+1} 脉冲 → bitline 电流
        ...

class DMNeuronModel:
    """替换 BinaryLIF / TernaryLIF。S1.2 的六参数状态模型。"""
    def __init__(self, beta_m, g_m, theta, U_hold, rho, n_ref, ternary): ...
    def step(self, I, state): ...           # 返回 (S, state)

class GainCellArray:
    """替换 core()。衰减只在 phi1 发生;写入含误差;读含误差。"""
    def __init__(self, dk, dv, alpha_lut, write_err, read_err, eps_hold): ...
    def phi1_decay(self, V_leak): ...
    def phi3_retrieve(self, k): ...
    def phi4_write(self, k, dv): ...
    def phi5_readout(self, q): ...

class ComparatorModel:
    """替换 QuantizerIn。"""
    def __init__(self, thr, offset, hysteresis): ...
```

仿真器逐 token、逐相位执行 S0.7 的表。一层全宽(1024 维、约 1500 万个 crossbar 元件)在这个抽象层次上是每 token 毫秒级。

### S2.4 首个闭环:给定门值下的真实层

把 S2.1 的层权重装进 S2.3 的仿真器,用 S2.2 记录的 $h_\tau$ 和 $\alpha_\tau,\beta_\tau$ 驱动,跑 2048 token。比对:

- 该层输出 $y_\tau,y'_\tau$ 的脉冲事件,按正负分别统计 precision/recall,均 $\ge0.9$;符号错误率 $\le2\%$;
- 状态误差 $e_\tau=\|S^{\rm ckt}_\tau-S^{\rm sw}_\tau\|_F/B$($B$ 为命题上界),$\max_\tau e_\tau\le0.1$,$e_L\le0.1$;
- **电路层在环 PPL:** 把仿真器输出的 $y'_\tau$ 送回模型上方的 ANN 层,算整模型 PPL,相对软件层的 PPL $\le1.1\times$;
- S0.4 的测试 4 在仿真器里复跑。

**结论边界:** 证明"给定门值时,一层真实层的递推核心、投影、累加器、乘法锁存在校准过的电路模型上与软件一致"。它的可信度取决于 Stage 1 原语标定的质量;门的生成不在其中。

### S2.5 门电路接入

把 `gates()` 换成 S1.6 拟合的传输函数,重跑 S2.4;同时把该传输函数换进软件模型做短微调,记录退化。门值复现误差 $\le0.02$。**结论边界:** "门电路在容限内复现软件门值"。两条边界合起来才是完整的一层闭环。

### S2.6 可选:一个头的 SPICE 级仿真

若有商用 SPICE(FastSPICE 模式),可以对该层的一个头做 SPICE 级仿真:三个 $1024\times d_h$ 投影、一个 $d_h\times1024$ 的 $W_O$ 子块、$d_h\times d_h$ 状态阵列,$d_h=128$,共 524,288 个忆阻器加 16,384 个 gain cell。这是一个带真实权重、真实输入宽度的完整 GDN 递推,SPICE 保真度。用它校验行为级仿真器对完整递推的建模,而不只是原语。ngspice 跑不动这个规模。

### S2.7 Gate

S2.1 达标;S2.4 四项全过;S2.5 门值误差达标。S2.4 不过时先隔离:只跑 FFN 子块比对,再只跑 GDN 子块,再只跑状态阵列的读写,逐级定位。

---

<a name="s3"></a>
## Stage 3:验证 6 个 softmax 层的转换

### S3.1 基座与配置

Qwen3.5-0.8B,24 层 dense,隐层 1024,FFN 中间维 3584,GDN 与 softmax 全注意力 3:1 交替(`full_attention_interval` = 4),即 18 层 GDN、6 层全注意力,全注意力层索引为 3、7、11、15、19、23(0 基)。GDN 层为 16 个 key 头、16 个 value 头、头维 128(q、k、v 投影各 2048 维),conv1d 核长 4;全注意力层为 8 头、2 个 kv 头、头维 256、带输出门(`config.json` 的 `text_config`,2026-09-14 读取)。全注意力层的具体索引、GDN 的 key/value 头数与头维、conv1d 核长,全部以 `config.json` 的 `layer_types`、`linear_num_key_heads`、`linear_key_head_dim` 等字段为准。文本 demo 砍掉视觉编码器。

选它的理由是 18 层 GDN 已经原生训练好,需要转换的只有 6 层,且每个待转换层的相邻层都有原生 GDN 可作参照。规模是 0.8B,嵌套 T=4 下的 BPTT 显存相当于约 3.2B 的 ANN。

### S3.2 先量化 6 个全注意力层的重要性

在转换任何一层之前,做 6 次消融:依次把每个全注意力层替换成恒等映射,测 PPL。只剩 6 层全注意力的模型,这 6 层很可能承担了集中的全局召回功能;消融直接量化这一点。若某一层的消融导致 PPL 翻倍以上,它就是转换风险最高的层,也是最该先做的层。

### S3.3 只转一层,三种初始化对比

选 S3.2 里影响最大的那一层先转。三种初始化:

| 初始化 | 做法 |
|---|---|
| 邻层复制 | 门投影和 conv1d 从相邻原生 GDN 层复制 |
| 固定保守值 | $\alpha_0=0.95$,$\beta_0=0.3$,conv1d 近似恒等 |
| 校准 | Taylor-Calibrate 式解析初始化(为 GDN student 的衰减、写入、输出门指定初始动态) |

$W_{Q,K,V,O}$ 从该全注意力层复制;头数或头维不一致时用 head-wise 映射或小线性投影对齐,并记录哪些参数能直接搬、哪些不能。训练:先冻结全部原权重,只训该层的门,损失为该层输出与原层输出的 MSE,10M token;再加 LoRA(秩 8–16)到该层 $W_{Q,K,V,O}$,损失为 LM 交叉熵加逐层 MSE,10M token。

**结论形式。** 三种初始化的单层替换 PPL、收敛 token 数、门的最终分布。邻层的门是否适配当前层的投影和表示,在这里是一个被测量的量,不是假设。

### S3.4 对转换层做脉冲化改造

对 S3.3 转好的层,按 S2.1 的 7 类顺序做脉冲化改造并记录增量 PPL。与 S2.1 原生层的增量表并列,看转换层与原生层对脉冲化的敏感度是否不同。

### S3.5 Gate

最优初始化的单层替换 PPL $\le1.15\times$ 原模型;转换层的 7 类累计 ΔPPL $\le30\%$。

---

<a name="s4"></a>
## Stage 4:扩到整模型

### S4.1 转 6 层

用 S3.3 的最优初始化转全部 6 个全注意力层。先各自单独转并单独验证,再合并。合并后 LoRA + LM 微调 30M token。Gate:zero-shot 8 项平均 $\ge0.85A_0$,PPL $\le1.5P_0$,其中 $A_0,P_0$ 是原模型基准。全线性化会损失召回能力,这是接受的代价。

### S4.2 24 层逐类改造

按 S2.1 得到的顺序,对全部 24 层逐类改造,每类之后微调 10M–30M token。编码严格按 S0.5,从一开始就用最终编码。累加器量程为嵌入加 48(24 层、每层两个残差),$\theta_{\rm in}$ 逐层可学。Gate:累计 PPL 退化 $\le40\%$。

### S4.3 转 SNN:纯转换,再蒸馏

**纯转换(零训练)。** QCFS 换 IF 神经元,阈值等于学到的 $\lambda$,初始膜电位 $\theta/2$。嵌套 T=32。Gate:PPL 有限且 $\le5\times$ S4.2。

**蒸馏。** Teacher 是 S4.2 的 24 层全 GDN 可脉冲化 ANN。Student 用 LIF、arctan 代理梯度、BPTT,从 S4.2 权重初始化。损失按 BiSpikCLM 的 SpAD(Spike-Aware Alignment Distillation)框架:逐层特征的 Rate-MSE(把 teacher 特征喂进同一个 LIF 跑 T 步取时间平均,与 student 时间平均做 MSE)加直接 MSE、软标签 KL、硬标签 CE;GDN 没有注意力图,对齐状态 $S_\tau$ 或读出 $o_\tau$。顺序:嵌套 T=4 → 折叠 T=1。折叠 T=1 的模型是最终产物,它的每个前向细节都已由 S0 规范冻结。Gate:嵌套 T=4 PPL $\le2\times$ S4.2;折叠 T=1 PPL 有限且 $\le4\times$;四项压力测试通过。

**器件回灌。** 把 S1.2 的状态模型和 S1.6 的门函数换进折叠 T=1 的 student,短微调后 PPL 退化 $\le30\%$。

### S4.4 显存预算

0.8B 在嵌套 T=4 下相当于训约 3.2B 的 ANN。单卡 V100 32GB:梯度检查点必开,batch 2–4,序列长 512,LoRA。不够则嵌套 T 降到 2 过渡。折叠 T=1 不吃倍数。2B 及以上不做。

---

<a name="ext"></a>
## 扩展:MoE

稀疏上采样造 $E=4$ 个专家,路由器从头训,SEMM 式阈值路由(路由器输出过脉冲神经元,发放即选中,不用 softmax 也不用 top-k);电路侧用首次发放时间做 argmax 加侧向抑制做 top-k。留到 Stage 4 稳定后。

---

<a name="tl"></a>
## 时间线

| 周 | Stage | 内容 | 并行 |
|---|---|---|---|
| 1 | S0 | 递推规范文档 | 可选探针 S0.10 |
| 2–5 | S1 | 神经元拟合(2–3)、gain cell 与 $\alpha$ 区间(2–3)、单步判别(4)、组合与门电路(4–5) | S2.1 从第 3 周开始 |
| 3–6 | S2.1–2.2 | 一层原生 GDN 层逐类脉冲化、折叠转换、记录输入 | 与 S1 并行 |
| 6–9 | S2.3–2.5 | 行为级仿真器、首个闭环、门电路接入 | S3.1–S3.2 消融同步做 |
| 6–8 | S3 | 一层 softmax 转换三种初始化 | 与 S2 并行 |
| 8–10 | S2.6 | 可选:一个头的 SPICE 级仿真 | 视仿真器资源 |
| 10–17 | S4 | 转 6 层、24 层改造、纯转换、蒸馏、回灌 | — |
| 18–20 | 扩展 | MoE | 可选 |

主线约 **17 周**,含 MoE 约 20 周。第 9 周末拿到 S2.4 的结论。

**三件先做的事:** S1.3 的 $\alpha$ 可调区间在第 3 周就要出来,它决定 S2.1 原生层的 $\alpha$ 是否需要钳;S1.2 的神经元状态模型在第 4 周出来,S2.1 的折叠转换用它;S2.1 第 3 类改造(固定缩放)的 ΔPPL 在第 5 周出来,决定 S0 是否需要引入 WTA 后再冻结。

---

<a name="risk"></a>
## 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| **折叠 T=1 在真实层上学不动** | 高,整条路线的前提 | S2.1 第 5–6 周测出;可选探针可提前到第 1 周;不行则退到嵌套 T=2 加门控泄漏,规范同步改 |
| **固定缩放下 ΔPPL 过大** | 中 | S2.1 第 5 周测出;$\beta$ 门补偿;不够则按 S0.4 的语义要求引入 WTA 并修订规范 |
| **$[\alpha_{\min},\alpha_{\max}]$ 太窄** | 高,原生层 $\alpha$ 分布被钳 | 第 3 周测出;S2.1 前用它评估该层 $\alpha$ 直方图落在区间内的比例 |
| **单步判别落在形式 C** | 高,时序设计错误 | S1.4 直接暴露;调整泄漏窗口的相位边界 |
| **行为级仿真器与 SPICE 不一致** | 高,闭环结论不可信 | S1.5 的组合级 SPICE 校验;S2.6 一个头的 SPICE 级校验 |
| **累加器分辨率不足** | 中 | S1.5 测出规格;$\theta_{\rm in}$ 可学;不够则在规范中定义再量化深度(规范变更) |
| **门电路精度不足** | 中 | S2.5 记录退化;首个闭环不依赖它;备选频率估计 |
| **$\epsilon_{\rm hold}$ 过大** | 中,φ2–φ8 内状态漂移 | 缩短相位或加保持电路;计入误差预算 |
| **S2.4 闭环不过** | 高 | 逐级隔离定位(FFN → GDN → 阵列原语) |
| **邻层初始化不如固定值** | 低,只影响 S3.3 选择 | 三种全测,按结果选 |
| **某个全注意力层承担集中功能** | 中,S4.1 达不到 gate | S3.2 先量化;必要时保留该层 softmax 作对照,记录代价 |
| **逐类改造累计超 30%/40%** | 中 | 每类可单独回滚;找出损伤最大的一类单独研究 |
| **头数/头维不匹配** | 低 | head-wise 映射;S3.3 记录哪些能搬 |
| **BPTT 显存** | 中 | 检查点、batch 2–4、seq 512、LoRA、T=2 过渡 |
| **器件状态模型拟合不达标** | 中 | 增加参数(双时间常数)或改用分段拟合 |

---

<a name="metrics"></a>
## 验收指标

| 指标 | 定义 | 容限 | 用在 |
|---|---|---|---|
| 发放事件 precision / recall,按正负分 | 三值神经元:对 $+1$ 事件和 $-1$ 事件分别统计;二值只统计 $+1$。逐神经元算,报告均值与最小值 | 均 $\ge0.9$(S2.4)/ $\ge0.95$(S1.2) | S1.2、S2.4 |
| 符号错误率 | 软硬件都发放但符号相反的事件占全部发放事件的比例 | $\le2\%$(S2.4)/ $\le1\%$(S1.2) | S1.2、S2.4 |
| 状态矩阵误差 | $e_\tau=\|S^{\rm ckt}_\tau-S^{\rm sw}_\tau\|_F/B$,$B=\sqrt{d_v}/(1-\alpha_{\max})$ 为命题上界;分母恒正 | $\max_\tau e_\tau\le0.1$,$e_L\le0.1$ | S2.4、S2.6 |
| 误差增长斜率 | $e_\tau$ 对 $\tau$ 的线性回归斜率,辅助观察量,不作 gate | — | S2.4 |
| 单步判别 | 16×16 分块单步后 $S_1[1,1]$ 与形式 A 闭式的偏差;与 B、C 的距离 | 偏差 $\le0.05$;距 B、C $>0.15$ | S1.4 |
| 状态范数上界 | $\sup_\tau\|S_\tau\|_F\le1.05B$ | — | 压力测试 2–4 |
| 衰减跟随 | 全零 key 下 $\|S_\tau\|_F$ 与 $(\prod\alpha_s)\|S_0\|_F$ 的相对偏差 | $\le1\%$ | 压力测试 1 |
| 电路层在环 PPL 比 | 仿真器输出送回模型上方各层后的 PPL / 软件层的 PPL | $\le1.1$ | S2.4 |
| 门值复现误差 | 门电路输出的 $\alpha,\beta$ 与软件值的最大绝对偏差 | $\le0.02$ | S2.5 |
| 逐类增量 PPL | 每类改造后的 PPL 相对前一步的变化 | 逐类记录;累计 $\le30\%$ | S2.1、S3.4、S4.2 |
| 单层替换 PPL 比 | 替换一层后 PPL / 原 PPL | $\le1.15$ | S3.3 |
| 稳态发放率一致性 | 恒定输入下拟合模型 vs SPICE 的发放率 | 校验用 | S1.2 |

**为什么这样定义。** 脉冲一致率在输出 90% 为零时会被永不发放的电路"通过";以发放为正例的 precision/recall 不会,但若不分正负,全部正脉冲翻成负脉冲仍能满分,所以三值必须分符号统计并报符号错误率。状态误差用命题上界做分母,避免软件状态为零时分母为零。最大误差和末端误差直接规定容限,斜率不能替代它们。层在环 PPL 是唯一能把电路误差换算成语言建模损失的指标。

---

<a name="deliv"></a>
## 交付物

**Stage 0**
- 《folded-T1-spec.md》:S0.2–S0.7 的完整规范,带版本号,是软硬件唯一接口;附冻结条件的满足记录;S0.9 参考实现与压力测试

**Stage 1**
- DM 神经元的六参数状态模型、拟合数据、按正负分的 precision/recall 与符号错误率;gain cell 的 $[\alpha_{\min},\alpha_{\max}]$、$\epsilon_{\rm hold}$、读写误差;单步判别测试结果;crossbar 列 + 神经元、累加器 + 比较器、锁存 + AND、$W_O$ 双神经元的 SPICE 验证;门电路传输特性;**打包好的行为级元件模型**

**Stage 2**
- 一层原生 GDN 层的逐类改造增量表、折叠转换结果;2048 token 的输入与基准记录;行为级仿真器代码;S2.4 闭环报告(四项指标,标明"给定门值"边界);S2.5 门电路接入报告;S2.6(可选)一个头的 SPICE 级校验;能耗与面积的分项清单(含 DAC、比较器、锁存、同步、参考、门电路)

**Stage 3**
- 6 层消融表;三种初始化对比表;转换层的逐类改造增量表;头数/头维对齐记录

**Stage 4**
- 24 层全 GDN ANN 权重;可脉冲化 ANN 权重;纯转换的 T 扫描;折叠 T=1 SNN(冻结,按 S0 规范);器件与门函数回灌结果;各阶段 PPL/zero-shot 退化总表

**跨阶段**
- 一份结论边界说明:S1.4(时序)、S2.4(给定门值的一层)、S2.5(门生成)、S2.6(SPICE 级的一个头)、S4(整模型)各自证明了什么、适用范围是什么
