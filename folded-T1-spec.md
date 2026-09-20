# 折叠 T=1 单 token 前向规范

版本：0.1.3（Stage 0 草案，未冻结；0.1.1 增补尺度折算表、一层的定义、记录与回放规则、目标层实例化参数；0.1.2 增补器件神经元的复位律与输入量程上界、gain cell 泄漏路径须为欧姆型；0.1.3 按真实器件结果改写 α 的实现方式与器件神经元模型；递推、编码、时序不变）  
日期：2026-09-14  
来源：project-plan.md 的 S0.2–S0.7。旧 two-stage-project-plan.md 不作为实现依据。

本文为软件参考实现与后续电路时序的接口基线。递推、编码或时序变更必须先修订本文并递增版本。

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


## 实现配置与接口约定

参考实现位于 `step1-spiking-layer/snn_spec/`。默认 float32；状态矩阵按 `[batch, value_heads, dk, dv]` 排列。q/k 头数可少于 value 头数，按连续分组 repeat_interleave 扩展，要求后者为前者整数倍。每个 value head 独立持有标量 alpha、beta。

- `SpikingGDN.core` 接收已编码、已缩放的 k/v/q，返回 `(o, S_new)`；不做额外归一化或门限幅。
- 默认 q 保持单位二值幅度；k 的 `1/sqrt(dk)` 在检索和写入各使用一次，不可遗漏或重复折算。连续 ANN 参考递推中的 q 归一化及 `dk^-1/2` 不属于脉冲默认路径。
- LIF 默认阈值 1、膜泄漏 0.9；先泄漏积分、再发放、当前步减阈值软复位。恰好达到正/负阈值时发放；每 token 每通道至多一个带符号脉冲。即使超阈多倍也不循环发放，剩余膜电位带到下一 token。
- 块入口阈值采用 exp(log_thr) 的正值可学参数。注意力、FFN 分别量化各自入口累加器。残差沿层深累加，不作为上一 token 的残差输入。
- 新序列显式清零 S、所有膜电位及可选 FIR 缓存；同一序列分段时传回完整状态。`step` 会更新传入状态字典，独立分支须自行复制。
- 首个电路闭环显式构造 `ExternalGates()`，门张量为 `[B,Hv]`，序列接口为 `[B,L,Hv]`。调用方保证门有限且落在所声明区间；传入外部门值不能自动覆盖其他门模块。
- 现有实现默认 `QwenNativeGates`，即 alpha=clamp(exp(-exp(A_log)*softplus(a_proj(x)+dt_bias)))，beta=sigmoid(b_proj(x))；这是函数形式占位，尚未装载真实权重。`SigmoidGates` 为从零训练选项，alpha=lo+(hi-lo)*sigmoid(Wa*x)。两者不可在实验记录中混称；电路拟合门待 Stage 1 标定。
- FIR 默认关闭；启用时为神经元前的逐通道因果线性 FIR，缓存跨 token 携带，不包含原生 conv 的额外激活。属于待消融扩展，不代表已复现完整 Qwen 层。
- 器件神经元按六参数状态模型 $(\beta_m, g_m, \theta, U_{\rm hold}, \rho, n_{\rm ref})$ 建模，复位律为 $U\leftarrow U_{\rm hold}\,\mathrm{sgn}+\rho\,(U-\theta\,\mathrm{sgn})$：$\rho=0$ 为硬复位到保持电压，$\rho=1$ 且 $U_{\rm hold}=0$ 即软件参考实现的"减阈值"软复位。行为级仿真器用它替换 `BinaryLIF` / `TernaryLIF`（`snn_spec/circuit.py` 的 `DMNeuronModel`）。
- 有状态神经元的输入量程有两个上界，规范"每 token 每通道至多一个脉冲"依赖它们：每周期注入的电荷小于 $C_m(\theta-U_{\rm hold})$，否则复位后同一周期内会再次越阈；器件导通期间的输入电流小于保持电流 $U_{\rm hold}/R_{\rm on}$，否则器件锁定在导通态不再发放。神经元前电流镜的比例必须让对应输入分布落在这两个上界之下。
- gain cell 的泄漏路径必须是欧姆型。GF180MCU 真实晶体管上的验证（`results/spice-gf180-leak.md`）表明用栅压调节 MOSFET 电导的做法（接地、亚阈区到 $V_{\rm mid}$、三极管区到 $V_{\rm mid}$）都不是欧姆型；采用"泄漏窗内全开的开关串电阻到 $V_{\rm mid}$"，$\alpha_\tau=\exp(-t_{\rm gap}/RC)$，按头用泄漏窗长度 $t_{\rm gap}$ 设定。S0.7 中 $\alpha$ 到 $V_{\rm leak}$ 的映射相应改为到 $t_{\rm gap}$ 的映射。
- 真实器件神经元按 Zhao et al. 2025 补充材料 Note 4 的物理模型建模（状态 $x_1$ 细丝、$x_2$ 残留、$V_{\rm gs}$），发放事件定义为忆阻器导通（$x$ 自下而上越过 1）；行为级仿真器用 `snn_spec/adm.py` 的 `ADMNeuronModel.step` 替换 LIF。六参数 LIF 抽象只对理想阈值开关成立，对该物理模型 precision/recall 为 0（`results/adm-neuron.md`）。Table 1 参数下积分需要 15–200 ms，token 周期为 ms 级；每 token 至多一个脉冲要求短输入脉冲、导通后撤除输入，并让 $V_{\rm gs}$ 在下一 token 前回到阈值以下（$R_c$ 可调）。

## 尺度折算表

Stage 2 在 ANN 的每个未来脉冲点插入 QCFS，其输出是 $a=\lambda\,s$：$s$ 是单位幅度的码（$\{0,1\}$ 或 $\{-1,0,+1\}$），$\lambda$ 是学到的尺度。折叠转换保留 $s$、去掉 $\lambda$，所以每个 $\lambda$ 必须有明确去处；连续参考递推里作用在 $q$ 上的 $d_k^{-1/2}$ 和作用在 $q,k$ 上的 L2 归一化同样必须有去处。规则只有三条：

1. 位于线性算子之前的尺度，折进该线性算子的权重（电路上是 crossbar 电导）。
2. 位于神经元之前的尺度，折进该神经元的输入增益 $g_m$；等价地，把该神经元的阈值除以这个尺度（电路上是神经元前的电流镜比例）。
3. $k$ 上的任何尺度都不折。$k$ 固定为 $s/\sqrt{d_k}$；ANN 里 $k$ 的 L2 归一化和 $\lambda_k$ 在 Stage 2 的第 3 类改造中被移除，其代价由 ΔPPL 直接测量，由 $\beta$ 门和微调补偿。

| 脉冲点 | ANN 侧的尺度 | 去处 | 电路对应 |
|---|---|---|---|
| $x=Q_{\rm in}(h)$ | 无（$\theta_{\rm in}$ 只决定是否发放，输出单位幅度） | 不需要；$W_{q,k,v}$、$W_{1,3}$ 直接接单位脉冲 | — |
| $q$ | $\lambda_q$（按头为标量）；参考递推的 $d_k^{-1/2}$ | $o=S^\top q$ 对 $q$ 线性 → 乘进 $\mathcal{SN}_{\rm pre}$ 的输入增益 | $\mathcal{SN}_{\rm pre}$ 前的电流镜 |
| $k$ | $\lambda_k$；L2 归一化 | 不折（规则 3） | 检索、写入的行脉冲幅度 $1/\sqrt{d_k}$ |
| $v$ | $\lambda_v$（可逐通道） | $S_0=0$ 时 $S$ 对 $v$ 齐次线性，$o$ 亦然 → 乘进 $\mathcal{SN}_{\rm pre}$ 的输入增益 | 同上 |
| $o_{\rm spk}$（$W_O$ 前） | $\lambda_o$（可逐通道） | 折进 $W_O$ 对应列 | crossbar 电导 |
| $y$（$W_O$ 后） | $\lambda_y$（可逐通道） | 折进累加器的注入电荷 $Q_0$ | 注入电荷量 |
| $g=\mathcal{SN}_1(W_1x)$ | $\lambda_g$（可逐通道） | $p=g\odot u$ 后接 $W_2$ → $\lambda_g\lambda_u$ 折进 $W_2$ 对应行 | crossbar 电导 |
| $u=\mathcal{SN}_3(W_3x)$ | $\lambda_u$（可逐通道） | 同上 | 同上 |
| $y'$（FFN 输出） | $\lambda_{y'}$（可逐通道） | 折进累加器的注入电荷 $Q_0'$ | 注入电荷量 |

于是 $\mathcal{SN}_{\rm pre}$ 相对参考实现的输入增益为 $\lambda_q\lambda_v d_k^{-1/2}$（$\lambda_v$ 逐通道时按通道），gain cell 存的是"单位 $v$"坐标下的 $S$。$\lambda_q$ 必须按头为标量：逐通道的 $\lambda_q$ 会给 $S$ 的不同行不同权重，无法折进读出侧。冻结的 RMSNorm 缩放折进紧随其后的 $W_{q,k,v}$ 或 $W_{1,3}$；被去掉的输出门 RMSNorm 若保留其冻结缩放，折进 $W_O$。

阈值侧：折叠转换把 QCFS 换成 LIF 时，神经元阈值取 $\theta=\lambda$。QCFS 的半步平移（等价于初始膜电位 $\lambda/2$）在折叠模式下没有对应物，因为膜电位跨 token 携带、初值只在序列开头出现；这一差异由 Stage 2 的短微调吸收。软件参考实现中所有神经元默认 $\theta=1$、单位增益；装载真实层时按本表设置各神经元的阈值（软件用阈值实现增益：$\theta_{\rm eff}=\theta/\text{增益}$），电路用电流镜比例实现增益。

## "一层"的定义与在环 PPL

"一层"指一个完整的 decoder block：$Q_{\rm in}$ → GDN → 累加 → $Q_{\rm in}$ → SwiGLU → 累加，即参考实现的 `SpikingBlock`，含两次残差。电路侧仿真一层时，该层入口的累加器值 $h_{\rm in}$ 由软件记录写入，出口 $h_{\rm out}$ 送回模型上方的 ANN 层计算 PPL；在环 PPL 替换的是完整的 $h_{\rm out}$，不是只替换 GDN 子块的输出。子块单独比对（只跑 FFN、只跑 GDN、只跑状态阵列读写）用于定位误差，不作验收。

## 记录与回放

输入只有两类：每个 token 的入口累加器值 $h_{\rm in}$；首个闭环中外部给定的门值 $\alpha_\tau,\beta_\tau$。其余记录量 $q,k,v,S,o,o_{\rm spk},y_{\rm attn},h_{\rm mid},x_{\rm ffn},g,u,p,y_{\rm ffn}$ 只作比对参考，不得用来逐 token 修正电路内部状态。单层仿真自行维持全部状态：$S$、各神经元膜电位、器件模型的不应期计数、启用时的 FIR 历史。新序列：全部状态清零。长文本分段：状态跨段传递、不清零，与软件 `run_sequence(block, h_seq, st)` 的语义一致。`SpikingBlock.step(..., record=True)` 返回的字典就是记录格式，字段名以它为准。

## 目标层的实例化参数

据 Hugging Face `Qwen/Qwen3.5-0.8B` 的 `config.json`（`text_config`，2026-09-14 读取）：

| 项 | 值 |
|---|---|
| `num_hidden_layers` | 24 |
| `layer_types` | 每 4 层一个 `full_attention`：索引 3, 7, 11, 15, 19, 23（0 基）；其余 18 层为 `linear_attention`（GDN） |
| `hidden_size` / `intermediate_size` | 1024 / 3584，`hidden_act` = silu |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 16 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128（q、k、v 投影输出均为 2048 维） |
| `linear_conv_kernel_dim` | 4 |
| 全注意力层 | 8 头、2 kv 头、`head_dim` 256、带输出门 |

对应的实例化为 `SpikingBlock(d=1024, d_ff=3584, n_k_heads=16, n_v_heads=16, dk=128, dv=128)`。一层的元件规模：

| 元件 | 数量 |
|---|---|
| 注意力投影忆阻器 $W_{q,k,v,o}$ | 8,388,608（$3\times1024\times2048+2048\times1024$） |
| FFN 忆阻器 $W_{1,3,2}$ | 11,010,048（$3\times1024\times3584$） |
| gain cell（状态 $S$） | 262,144（16 头 $\times128\times128$） |
| DM 神经元 | 17,408（$q,k,v,o_{\rm pre}$ 各 2048；$o_{\rm out}$ 1024；$g,u$ 各 3584；FFN 输出 1024） |
| 累加器电容 / 比较器 | 1024 / 2048 |

原生 GDN 层里 Stage 2 要改造的组件及其形状：q/k/v 投影后的 depthwise conv1d（核长 4、6144 通道、无偏置）加 SiLU；q、k 的 L2 归一化；门 `a_proj`、`b_proj`（1024→16）与 `dt_bias`、`A_log`（各 16）；读出的门控 RMSNorm（门 $z$ 由 1024→2048 的投影给出）；`out_proj`（2048→1024）。project-plan.md 的层数、层索引、头数与元件规模已于 2026-09-14 按本节更正。`tests/test_target_shapes.py` 按这组参数实例化并核对以上数量。

## 验收与冻结记录

Stage 0 验证覆盖：A/B/C 单步判别（1.25/1.00/0.75）、闭式与逐步递推一致性、编码和软复位、状态跨段传递、残差和 SwiGLU、核心压力测试 1–3。单步判别直接注入连续测试值 v=2、k=e0，绕过脉冲编码器，不受三值 v 或固定脉冲幅度限制。

本地 `hf_reference_recurrence` 是手工转写辅助参考。另已直接运行固定提交的 HF Qwen3.5 官方 CPU 递推，完成 18 组输出/状态对照以及 chunk 边界对照，详见 stage0-validation.md 与 `step1-spiking-layer/results/upstream-comparison.md`。此结论不包含 FLA GPU 内核。

冻结条件仍未满足：Stage 2 需提供真实层固定缩放的 ΔPPL；4096 token 真实文本稳定性测试也需真实层和输入。合成测试不得替代这两项。若需引入 WTA，先修订选择语义、编码和时序后再冻结。硬件 alpha 区间目前暂用 [0.5,0.99]，待 Stage 1 实测更新。

运行方式和本次结果见 `step1-spiking-layer/README.md` 与 `stage0-validation.md`。
