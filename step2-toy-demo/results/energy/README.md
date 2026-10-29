# 每 token 能耗估算:memristor 电路 vs 数字 ASIC vs GPU(2026-10-29)

toy demo 的脉冲 GDN 语言模型(d = 128、2 块、4 头、dk = dv = 32、d_ff = 256、词表 3226、116 万参数)放在 memristor 电路上,每 token 要多少能量?同一个模型放在数字 ASIC 和 GPU 上各是多少?memristor 的优势有多大、来自哪里、开销在哪里?这是量级对比,不是版图级或 SPICE 级功耗仿真。设计文档 `docs/superpowers/specs/2026-10-28-energy-estimate-design.md`,计划 `docs/superpowers/plans/2026-10-29-energy-estimate.md`(都在仓库根目录下)。

本文的数字全部出自这个目录里的三个文件:`energy.json`(`experiments/energy_estimate.py` 生成:参数、事件汇总、各方案的部件分解、敏感性、嵌入量化扫描)、`gpu.json`(`experiments/energy_gpu.py` 的实测,`energy.json` 的 `gpu` 键是它的拷贝)、`tables.md`(从 `energy.json` 渲染的五张表,第 5 节原样粘贴)。几处用到 `../improve/profile_final.json` 的地方单独标明。结论(第 6 节)、第 7 节以及其余各节推算出来的数字后面,括号里是它在 `energy.json`(GPU 的在 `gpu.json`)里的位置或算式,写法如下;"= a / b"表示这个数是括号里两个量现算的比值;结果一览和各表的读法是对它们的复述。少数不来自这些文件的数字(第 1 节标称电路的准确率、第 2 节 SPICE 测试台的 1 µs 积分窗、第 6 节 GPU 的 66 W 状态底、第 7 节的探测、第 8 节的 512 句重数)标明了来源。

| 写法 | 位置 |
|---|---|
| `M65[T]` | `models.T.memristor.65nm.crossbar_adc`(输出头在 crossbar + ADC;下面接 `.total`、`.by_component.x`、`.by_matrix.y`) |
| `M65h[T]` | `models.T.memristor.65nm.digital_event`(输出头用数字事件驱动,batch 1) |
| `Madv[T]` | `models.T.memristor.advanced.crossbar_adc`(展望列) |
| `Ev65[T,b]` / `Evadv[T,b]` | `models.T.digital_event.65nm.b` / `.advanced.b`(事件驱动数字 SNN,batch b) |
| `Dn65[b]` / `Dnadv[b]` | `dense.65nm.b` / `dense.advanced.b`(稠密 int8,batch b);`Dram` = `dense_dram_b1_65nm` |
| `GPU[m,b]` | `gpu.runs.m.b`(模型 m = A0 或 T2,batch b;`gpu.json` 里同名) |
| `Sw[T,k]` | `models.T.embed_quant_sweep.k.delta_pt`(百分点;`delta` 是小数,别混) |
| `Sens[T]` / `Ev[T]` | `models.T.sensitivity` / `models.T.events_summary` |
| `Par[x]` | `params.columns.65nm.x.value` |

## 结果一览

T2 = T = 2 的脉冲模型,65 nm 主列,输出头放在 crossbar + ADC。

- memristor 98.2 nJ/token(读窗 100 ns 为投影值;按本仓库 SPICE 测试台的 1 µs 读窗是 781 nJ)(T1 56.8、TR2 94.8)。同节点稠密 int8 数字 ASIC:batch 1 22.95 µJ、batch 16 5.547 µJ、batch 64 4.677 µJ。T2 的倍数是 234×(batch 1)、56.5×(16);batch 64 为 47.6×(激活每次乘加走近端 SRAM)到 8.14×(累加器在寄存器、不计激活访存),取决于激活访存口径(量级,口径见结论 1)。
- 这个优势来自存内计算,不是脉冲稀疏:把同一批脉冲放在数字 ASIC 上事件驱动地算,T1 是稠密的 0.625×,T2 反而是稠密的 1.19×(batch 1);事件驱动数字到 memristor 的差(279×,batch 64 为 52.7×)才是大头。
- memristor 的开销在输出头:输出头的 crossbar 导通、栅驱动 / 导线、积分器和 ADC 合计占 T2 的 68.0%。
- 输出头改用数字事件驱动(单 token 流,batch 1)是 17.6 µJ,比稠密 batch 64 还贵 3.76×,优势没了。
- GPU 实测(RTX A4500,8 nm,整卡功率):A0 batch 1 75.9 mJ/token、batch 64 3.35 mJ/token。它被空载功耗和逐 token 调度淹没,反映"一块 GPU 跑这个模型",不是 GPU 的效率上限。
- 嵌入表放进无 DAC 的 crossbar,16 级量化使 token 准确率变化 −19.11 / −7.69 / −11.70 个百分点(T1 / T2 / TR2);只有 T2 在 256 级(每个值两对差分 cell)回到门槛 0.5 个百分点以内,T1 和 TR2 不行(第 7 节)。

## 1. 问题与对象

| 标签 | 模型 | 检查点 | 切片 token 准确率(`embed_quant.acc_before`) | 每 token LIF 脉冲(`Ev[T].spikes_lif`) | 每 token 输出头被驱动的行次(`Ev[T].rows_sum.head`) |
|---|---|---|---|---|---|
| T1 | 基线 C,每 token 最多 1 个脉冲 | `results/conversion-sweep/CL2_l1_lif225_lr3e3_12k` | 0.752 | 1348.8 | 93.32 |
| T2 | T = 2 的 C(种子 0),每 token 最多 2 个脉冲 | `results/improve/T2_c` | 0.825 | 2248.6 | 180.9 |
| TR2 | 三值残差(sat 合并)T = 2 | `results/improve/ternres_sat_T2_c` | 0.811 | 1959.6 | 172.4 |
| A0 尺寸 | 浮点 A0,只用它的尺寸做数字稠密 int8 的成本(不量化、不测精度) | `results/ckpt/a0.pt` | 0.842(`../improve/README.md`) | — | — |

事件统计在 `data/tinystories-slice` 上做:4,003 句、59,491 个有效目标位置(`Ev[T].n_tokens`;减去句数 4,003 是 55,488,即设计文档记的 token 数加每句一个 EOS 目标),所有事件数都是这些位置上的每 token 平均,不是平均发放率假设。A0 尺寸的数字稠密 int8 每 token 775,424 次乘加(`mac_counts.total_macs`,其中 GDN 递推 32,768、各矩阵 742,656),要流的权重 742,672 个(`mac_counts.params_stream`,含输出头,不含嵌入表;嵌入表 412,928 个,每 token 只查 128 个值)。输出头占流式权重的 55.6%。

**能量与准确率的对应。** memristor 各行的能量按软件模型(理想电路)的事件数算,上表并列的是软件准确率(T2 0.825,`embed_quant.acc_before`)。同一器件的标称 16 级电路(编程误差、读噪声、比较器失调)在自由运行下比软件低:T = 2 的三个种子 4.8–6.5 个百分点,T = 1 11.7 个百分点(`../improve/README.md` 的"自由运行时,多脉冲的退化小得多"一条;5 个噪声种子平均,在那里的对照集上,不是本文的切片)。表 2 把 742,672 个流式权重(`mac_counts.params_stream`)按同样的 16 级算电导,而同样的 16 级量化单加在嵌入表上就让 T2 掉 7.69 个百分点(第 7 节)。所以表里的能量配的是理想电路的准确率;在相同准确率下比较的能量—准确率权衡没有算。

## 2. 工艺节点对齐

- **主列统一到 65 nm,两侧同节点。** memristor 侧的 CMOS 周边(选通管、列积分器、比较器、ADC、gain cell、控制)和数字侧的乘加、SRAM 都按 65 nm 取值。数字侧从 Horowitz(ISSCC 2014,45 nm)乘 `node_scale` = 2.0(`Par[node_scale]`)换算。**这个 2.0 是自己估的**,没有外部出处(尺寸比 65/45 ≈ 1.44 乘电压比平方 (1.2/1.0)² = 1.44,约 2.07,取 2.0;`source` 字符串里写明了);memristor 侧周边按 V_dd = 1 V 取,所以对数字侧是偏高端的估计,会让倍数偏向 memristor。数字侧的乘加和 SRAM 各项都与 `node_scale` 成正比,所以它直接线性进倍数:取设计文档范围的下限 1.5,所有数字行和倍数乘 0.75(DRAM 对照行除外:它的主项 `E_dram_bit` 不随 `node_scale` 缩放,该行只动到 0.994×),T2 的 batch 1 / batch 64 倍数是 175× / 35.7×(= 234× / 47.6× × 0.75)。这是算术推论,`node_scale` 本身没有扫。
- **忆阻器件量不随 CMOS 节点缩。** G_on 100 µS、G_off 1 µS、16 级、V_read 0.2 V 两列共用(`params.device`);65 nm 列的 ADM 神经元每脉冲 1.39 pJ(paper Table 7 的 65 nm 配置)。
- **展望列是"各自最优投影,不同节点"。** memristor 取 paper 的 3 nm FinFET 投影(ADM 1 aJ/spike、R_c 242 GΩ、t_read 10 ns),数字取 7 nm 估计值。两侧节点不同,表 3 单独成表,不与表 1 的 65 nm 主列混比,也不是"同节点的未来"。展望列的周边电路和 7 nm 数字值都是估计,`source` 字符串写的是"估计(文献量级)",没有逐项出处。
- **GPU 是 8 nm 的实测,单独一列作参照**,不参与节点对齐,也不能当成同节点比较(整卡功率、浮点、含调度开销)。
- **DRAM。** `E_dram_bit` 取 Horowitz 的 DRAM 2 nJ/64 b 中位,**不随逻辑节点缩放**(由片外接口决定);DRAM 行(权重放在 DRAM,batch 1)只是对照行,不是主列。
- GF180 的 SPICE 测试台只提供器件行为(R_on、IR drop),不贡献能耗数字。
- **读窗 t_read。** 主列的 100 ns 是 paper(SI Note 5)对 ADM 神经元的投影,不是 crossbar 的测量。本仓库仿真过的唯一 crossbar 时序是 SPICE 测试台的 1 µs 积分窗(`step1-spiking-layer/experiments/spice/crossbar.py` 的 `T_INT = 1.0e-6`,全层回路 `block_layer.py` 的积分窗也是它)。按 1 µs,T2 是 781 nJ(表 4,`Sens[T2].t_read["1e-06"]`),比稠密 batch 1 / 64 低 29.4× / 5.99×(= `Dn65[b].total` / `Sens[T2].t_read["1e-06"]`)。

## 3. 评估口径

**事件怎么数**(`experiments/energy_events.py`):对每个模型,在整个切片上用 `record=True` 的前向数出每 token 的事件。脉冲向量取值 c/T(T = 每 token 的脉冲窗口数,T1 = 1,T2、TR2 = 2),每个元素的脉冲数 c = round(|s|·T);一个输入行被驱动 c 次,就是 crossbar 的 c 次读脉冲;SwiGLU 的 W2 每通道收到 c_g·c_u 个事件。同时数 LIF 脉冲数、比较器判决数(每个量化点每通道 2T 次)、gain cell 的检索 / 写入 / 读出行数以及每 token 的 α 均值、状态 S 的均方根、嵌入行(每 token 一行)。`n_lif` 按 d 宽度记(n_blocks·(6d + 2·d_ff) = 2560),d = 128 时与 8 个 LIF 点的真实神经元数一致。

**memristor 各项怎么算**(`experiments/energy_model.py`,对应表 2 的行):

- 突触 / 输出头 / 嵌入 crossbar 导通:V_read²·t_read·Σ_i c_i·G_row,i。c_i 是行 i 每 token 的平均脉冲数,G_row,i 是该行差分对(2 cell / 权重)的总电导,权重按 `crossbar.quantize_weights` 量化到 16 级后逐行加。
- 选通栅驱动 + 行线 / 位线导线:驱动行次 × 每行 cell 数 × (C_gate·V_sel² + C_wire·V_read²)。输出头一行 6452 个 cell,这一项必须单列。
- 未选通 cell 漏电:读窗内未驱动 cell 的 V_read·I_off·t_read。列积分器偏置:列数 × I_int·V_dd·t_read × T 个读窗。
- **读通路的两个假设。** (i) 导通项 V_read²·G·t_read 假设读电流来自一条专用的 0.2 V 读电压轨。如果读电流是从 V_dd = 1 V 经线性驱动器供出的,导通各项要乘 `Par[V_dd]`/`Par[V_read]` = 5:T2 的导通合计(`M65[T2].by_component` 的 `crossbar_blocks` + `crossbar_head` + `embed`)70.1 nJ → 351 nJ,合计 98.2 → 379 nJ(= `M65[T2].total` + 4 × 70.1 nJ)。(ii) 列积分器只按偏置计(列数 × I_int·V_dd·t_read × T);有源虚地积分器吸收信号电流时要从 V_dd 取的信号相关电荷**没有建模**。它有一个推得出的上界:这份电荷不超过读电荷的总量,所以漏掉的项 ≤ `Par[V_dd]`/`Par[V_read]` × 导通项 = 5 × 70.1 nJ = 351 nJ(T2)。差分对的净电流远小于两路电流之和,实际值远低于这个上界;更紧的数要逐 token 记录列电流,没做。即使按上界(98.2 + 351 = 449 nJ),T2 仍比稠密 batch 1 / 64 低 51.1× / 10.4×(= `Dn65[b].total` / 449 nJ);(i)、(ii) 都取最坏是 730 nJ(= 98.2 + 4 × 70.1 + 351),仍低于 t_read = 1 µs 的 781 nJ。
- ADM 神经元:LIF 脉冲数 × E_spike;静态漏电 n_lif·V_gs²/R_c·t_token 单列。
- 比较器:判决数 × E_cmp。gain cell:泄漏 + 检索 + 写入 + 读出,按实测的 k / q 行数、α 均值、S 均方根。
- 输出头 ADC:3226 列各一次 × E_adc。相位时钟与控制:每 token 一个固定量 E_ctrl。
- 输出头有两个变体:**crossbar + ADC**(主变体)和**数字事件驱动**(batch 1,按下面事件驱动数字的公式,只对被驱动的行做加法、取权重)。memristor 芯片是单 token 流,ADM 和 gain cell 的状态按流存放,不批处理,所以 memristor 的行没有 batch 维。

**数字两种参照**(batch 只摊销权重取数,激活读写不摊销):

- **稠密 int8(A0 尺寸)**:每次乘加 E_mul8 + E_add32(`mac`),外加 2 次 8 bit 近端 SRAM 读写(`act`);权重 742,672 个 × 8 bit 从片上大 SRAM 读,按 1/B 摊销(`weight`);嵌入查表 128 × 8 bit(`embed`);GDN 状态 S 每 token 一读一写(`state`)。batch 1 / 16 / 64。
- **事件驱动数字 SNN(同一批脉冲模型)**:每个脉冲一次 16 bit 加法(`add`)、一次 8 bit 权重取数(`weight`,按 1/B 摊销)、2 次近端 SRAM 读写(`act`);LIF 更新 `neurons` = n_lif × (E_mul8 + E_add16);GDN 递推 `core`、`state`、`embed` 与稠密相同(状态与门值是连续量)。有了这一行,优势才能拆成"稀疏 + 三值激活"和"模拟存内计算"两部分。
- 权重 742,672 字节放得下 1 MB 级的片上 SRAM,所以主列权重在片上 SRAM;权重在 DRAM 的只是 batch 1 的对照行。

**GPU 怎么测**(`experiments/energy_gpu.py`,RTX A4500 的 GPU 1,torch 2.14.0+cu126):跑模型自己的逐 token `step` 循环(teacher forcing 前向,递归模型实际推理的形状),A0(浮点)和 T=2 的 C 各测 batch 1 和 batch 64,每格持续 60 s,另用 `nvidia-smi` 在 10 Hz 采样整卡功率(读数阶梯式更新,实际更新更慢;每格 600 个样本)。空载在**进程里还没建 CUDA 上下文**时测,200 个样本平均 21.2 W(`gpu.idle_w`)。`total` = 平均功率 × 用时 ÷ 有效 token;`incremental` = (平均功率 − 空载)× 用时 ÷ 有效 token。有效 token = 目标非 PAD 的位置。

## 4. 参数表

由 `experiments/energy_params.py` 的 `COLUMNS` 生成(值、单位、来源字符串原样;重复很长的出处用缩写,缩写在表后)。"含义"列括号里的范围是设计文档给的取值范围,不是 `energy_params.py` 的内容。`G_on`、`G_off`、`levels`、`V_read` 是 SPICE 测试台的器件常数,两列共用。

| 参数 | 含义(括号内为设计文档给的范围) | 单位 | 65 nm 主列 | 展望列 | 来源(65 nm 主列) | 来源(展望列) |
|---|---|---|---|---|---|---|
| `G_on` | 忆阻高电导态 | S | 100 µS | 100 µS | [D] | 同左(两列共用) |
| `G_off` | 忆阻低电导态 | S | 1 µS | 1 µS | [D] | 同左(两列共用) |
| `levels` | 电导级数 | 级 | 16 | 16 | [D] | 同左(两列共用) |
| `V_read` | 读电压 | V | 0.2 V | 0.2 V | [D] | 同左(两列共用) |
| `t_read` | 读窗(每个输入脉冲一个) | s | 100 ns | 10 ns | paper SI Note 5(投影,非测量):ADM 可在 100 ns 级工作;敏感性 10 ns–1 µs | 同处:10 ns 脉冲刺激 |
| `t_token` | token 周期(定 ADM 静态漏电) | s | 1 ms | 100 ns | t_token = R_c·C_g·ln(1/0.9),R_c 1 GΩ、C_g 10 pF(Table 7) | 同式,R_c 242 GΩ、C_g 4.14 aF(Table 8) |
| `V_sel` | 选通管栅电压(设计文档范围:1 V) | V | 1 V | 0.7 V | [P65] | [PA];3 nm 选通管,展望列(各自最优投影,不同节点) |
| `C_gate` | 选通栅电容 / cell(1–2 fF) | F | 1.5 fF | 100 aF | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `C_wire` | 行线 / 位线导线电容 / cell(0.1–0.2 fF) | F/cell | 150 aF/cell | 50 aF/cell | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `I_off` | 未选通 cell 漏电 | A | 10 pA | 10 pA | [P65] | [PA];未投影,沿用 65 nm 列的值(漏电项本就可忽略) |
| `I_int` | 列积分器偏置电流(1–10 µA) | A | 5 µA | 1 µA | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `V_dd` | 周边电源电压(1 V) | V | 1 V | 0.7 V | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `E_spike` | ADM 神经元每脉冲能量 | J/spike | 1.39 pJ/spike | 1 aJ/spike | [A65] | [A3];1 aJ/spike |
| `R_c` | ADM 膜电阻 | Ω | 1 GΩ | 242 GΩ | [A65] | [A3];242 GΩ |
| `V_gs` | ADM 静态漏电的平均 V_gs | V | 0.15 V | 0.15 V | [P65] | [PA];ADM 器件量,沿用 65 nm 列的值 |
| `E_cmp` | 比较器每次判决(10–100 fJ) | J/判决 | 50 fJ/判决 | 5 fJ/判决 | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `E_adc` | 输出头 ADC 每次转换(1–10 pJ) | J/次 | 5 pJ/次 | 500 fJ/次 | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `C_s` | gain cell 存储电容(50 fF) | F | 50 fF | 10 fF | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `V_s` | gain cell 摆幅(0.5 V) | V | 0.5 V | 0.3 V | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `E_ctrl` | 相位时钟与控制(每 token) | J/token | 100 pJ/token | 10 pJ/token | [P65] | [PA];展望列(各自最优投影,不同节点) |
| `node_scale` | Horowitz 45 → 65 nm 的 CV² 换算系数 | × | 2 | 1 | 自估:45 → 65 nm 的 CV² 换算,尺寸比 65/45 ≈ 1.44 × 电压比平方 (1.2/1.0)² = 1.44,约 2.07,取 2.0;无外部出处。memristor 侧 65 nm 周边按 V_dd = 1.0 V 取,故对数字侧是偏高端的估计 | [PA];7 nm 数字值直接取用,不再换算 |
| `E_mul8` | int8 乘 | J | 400 fJ | 30 fJ | [H];本项 = int8 mult 0.2 pJ × node_scale 2.0 | [PA];7 nm int8 mult,展望列(各自最优投影,不同节点) |
| `E_add8` | int8 加(代码里未使用) | J | 60 fJ | 5 fJ | [H];本项 = int8 add 0.03 pJ × node_scale 2.0 | [PA];7 nm int8 add,展望列(各自最优投影,不同节点) |
| `E_add16` | int16 加(事件驱动数字的脉冲累加) | J | 100 fJ | 8 fJ | [H];本项 = int16 add 0.05 pJ × node_scale 2.0 | [PA];7 nm int16 add,展望列(各自最优投影,不同节点) |
| `E_add32` | int32 加(稠密乘加的累加器) | J | 200 fJ | 15 fJ | [H];本项 = int32 add 0.1 pJ × node_scale 2.0 | [PA];7 nm int32 add,展望列(各自最优投影,不同节点) |
| `E_sram_near_bit` | 近端 SRAM(8 KB 级)每 bit | J/bit | 312.5 fJ/bit | 20 fJ/bit | [H];本项 = 8 KB SRAM 10 pJ/64 b ÷ 64 × node_scale 2.0 | [PA];7 nm 近端 SRAM,展望列(各自最优投影,不同节点) |
| `E_sram_far_bit` | 片上大 SRAM(1 MB 级)每 bit | J/bit | 3.125 pJ/bit | 300 fJ/bit | [H];本项 = 1 MB SRAM 100 pJ/64 b ÷ 64 × node_scale 2.0 | [PA];7 nm 片上大 SRAM,展望列(各自最优投影,不同节点) |
| `E_dram_bit` | DRAM 每 bit(仅对照行) | J/bit | 31.25 pJ/bit | 10 pJ/bit | Horowitz ISSCC 2014 DRAM 1.3–2.6 nJ/64 b 的中位 2 nJ,不随逻辑节点缩放(片外接口决定);仅用于对照行 | [PA];DRAM 每 bit,展望列(各自最优投影,不同节点) |

来源缩写:**[H]** = Horowitz, Computing's energy problem (and what we can do about it), ISSCC 2014, 45 nm: int8 add 0.03 pJ, int8 mult 0.2 pJ, int16 add 0.05 pJ, int32 add 0.1 pJ, 8 KB SRAM 10 pJ/64 b, 1 MB SRAM 100 pJ/64 b, DRAM 1.3–2.6 nJ/64 b;**[D]** = experiments/spice/crossbar.py 测试台器件(HfO₂ 类 1T1R);**[A65]** = Zhao et al. 2025 Nature Electronics, SI Table 7(65 nm 能耗配置),转引自 step1-spiking-layer/results/adm-neuron.md;**[A3]** = 同文 Table 8(3 nm FinFET 投影);**[P65]** = 65 nm 文献量级,估计(见 README 参数表);**[PA]** = 估计(文献量级),见 README。

周边项([P65] 和展望列的 [PA])都是文献量级的估计,没有逐项出处;展望列的 7 nm 数字值同样是估计。`E_add8` 在代码里没有被用到,列出只因它在参数表里。

## 5. 结果表

以下五张表是 `tables.md` 的原文(由 `energy.json` 渲染),每张下面一行是读法。

**表 1 每 token 能量总表**(相对基准 = memristor 65 nm T2、输出头 crossbar+ADC = 98.2 nJ;GPU 行单位为 mJ)

GPU 行是实测:小模型在 GPU 上被空载功耗(约 21 W)与逐 token 调度淹没,'增量'是扣掉深度空载后的值,不是边际计算能量;它反映'GPU 跑这种模型'而非 GPU 的峰值效率。

| 方案 | 模型 | 设定 | 能量/token | 相对 memristor T2(crossbar+ADC) |
|---|---|---|---|---|
| memristor 65 nm | T1 | 输出头 crossbar+ADC | 56.8 nJ | 0.578× |
| memristor 65 nm | T1 | 输出头数字(事件驱动,batch 1) | 9,080 nJ | 92.4× |
| memristor 65 nm | T2 | 输出头 crossbar+ADC | 98.2 nJ | 1.00× |
| memristor 65 nm | T2 | 输出头数字(事件驱动,batch 1) | 17,600 nJ | 179× |
| memristor 65 nm | TR2 | 输出头 crossbar+ADC | 94.8 nJ | 0.965× |
| memristor 65 nm | TR2 | 输出头数字(事件驱动,batch 1) | 16,800 nJ | 171× |
| 数字稠密 int8 65 nm | A0 尺寸 | batch 1,权重在片上 SRAM | 23,000 nJ | 234× |
| 数字稠密 int8 65 nm | A0 尺寸 | batch 16,权重在片上 SRAM | 5,550 nJ | 56.5× |
| 数字稠密 int8 65 nm | A0 尺寸 | batch 64,权重在片上 SRAM | 4,680 nJ | 47.6× |
| 数字稠密 int8 65 nm(对照) | A0 尺寸 | batch 1,权重在 DRAM | 190,000 nJ | 1,930× |
| 数字事件驱动 65 nm | T1 | batch 1,权重在片上 SRAM | 14,300 nJ | 146× |
| 数字事件驱动 65 nm | T1 | batch 16,权重在片上 SRAM | 3,350 nJ | 34.1× |
| 数字事件驱动 65 nm | T1 | batch 64,权重在片上 SRAM | 2,800 nJ | 28.5× |
| 数字事件驱动 65 nm | T2 | batch 1,权重在片上 SRAM | 27,400 nJ | 279× |
| 数字事件驱动 65 nm | T2 | batch 16,权重在片上 SRAM | 6,240 nJ | 63.5× |
| 数字事件驱动 65 nm | T2 | batch 64,权重在片上 SRAM | 5,180 nJ | 52.7× |
| 数字事件驱动 65 nm | TR2 | batch 1,权重在片上 SRAM | 26,600 nJ | 271× |
| 数字事件驱动 65 nm | TR2 | batch 16,权重在片上 SRAM | 6,080 nJ | 61.8× |
| 数字事件驱动 65 nm | TR2 | batch 64,权重在片上 SRAM | 5,050 nJ | 51.4× |
| GPU NVIDIA RTX A4500(实测,8 nm) | A0 | batch 1,总功率 ÷ 吞吐 | 75.9 mJ | 772,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | A0 | batch 1,扣空载 21.2 W | 52.7 mJ | 536,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | A0 | batch 64,总功率 ÷ 吞吐 | 3.35 mJ | 34,100× |
| GPU NVIDIA RTX A4500(实测,8 nm) | A0 | batch 64,扣空载 21.2 W | 2.45 mJ | 25,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | T2 | batch 1,总功率 ÷ 吞吐 | 404 mJ | 4,110,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | T2 | batch 1,扣空载 21.2 W | 291 mJ | 2,960,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | T2 | batch 64,总功率 ÷ 吞吐 | 16.0 mJ | 163,000× |
| GPU NVIDIA RTX A4500(实测,8 nm) | T2 | batch 64,扣空载 21.2 W | 11.7 mJ | 119,000× |

读法:memristor 三行是 56.8–98.2 nJ 量级(输出头在 crossbar + ADC),换成数字事件驱动的输出头后是 9.08–17.6 µJ;稠密 int8 随 batch 摊薄权重取数,22.95 µJ → 4.677 µJ;事件驱动数字在 T1 比稠密省、在 T2 / TR2 不省;GPU 行是 mJ 级,口径见表上方说明和第 3 节。

**表 2 memristor 65 nm 的部件分解**(nJ/token,括号内为占该列合计的百分比)

| 部件 | T1 · 输出头 crossbar+ADC | T1 · 输出头数字(事件驱动,batch 1) | T2 · 输出头 crossbar+ADC | T2 · 输出头数字(事件驱动,batch 1) | TR2 · 输出头 crossbar+ADC | TR2 · 输出头数字(事件驱动,batch 1) |
|---|---|---|---|---|---|---|
| 突触 crossbar 导通(各块) | 13.6 (23.9%) | 13.6 (0.1%) | 24.5 (24.9%) | 24.5 (0.1%) | 23.8 (25.1%) | 23.8 (0.1%) |
| 输出头 crossbar 导通 | 20.7 (36.4%) | 0 (0.0%) | 45.7 (46.5%) | 0 (0.0%) | 43.4 (45.8%) | 0 (0.0%) |
| 嵌入 crossbar 导通 | 0.00703 (0.0%) | 0.00703 (0.0%) | 0.00725 (0.0%) | 0.00725 (0.0%) | 0.00732 (0.0%) | 0.00732 (0.0%) |
| 选通栅驱动 + 行线/位线导线 | 1.41 (2.5%) | 0.506 (0.0%) | 2.72 (2.8%) | 0.958 (0.0%) | 2.64 (2.8%) | 0.969 (0.0%) |
| 列积分器偏置 | 2.84 (5.0%) | 1.22 (0.0%) | 5.67 (5.8%) | 2.45 (0.0%) | 5.67 (6.0%) | 2.45 (0.0%) |
| 未选通 cell 漏电 | 0.000275 (0.0%) | 0.000230 (0.0%) | 0.000184 (0.0%) | 0.000184 (0.0%) | 0.000187 (0.0%) | 0.000187 (0.0%) |
| ADM 神经元发放 | 1.87 (3.3%) | 1.87 (0.0%) | 3.13 (3.2%) | 3.13 (0.0%) | 2.72 (2.9%) | 2.72 (0.0%) |
| ADM 静态漏电 | 0.0576 (0.1%) | 0.0576 (0.0%) | 0.0576 (0.1%) | 0.0576 (0.0%) | 0.0576 (0.1%) | 0.0576 (0.0%) |
| 比较器 | 0.0640 (0.1%) | 0.0640 (0.0%) | 0.128 (0.1%) | 0.128 (0.0%) | 0.128 (0.1%) | 0.128 (0.0%) |
| gain cell 阵列 | 0.0928 (0.2%) | 0.0928 (0.0%) | 0.158 (0.2%) | 0.158 (0.0%) | 0.152 (0.2%) | 0.152 (0.0%) |
| 输出头 ADC | 16.1 (28.4%) | 0 (0.0%) | 16.1 (16.4%) | 0 (0.0%) | 16.1 (17.0%) | 0 (0.0%) |
| 输出头数字(事件驱动) | 0 (0.0%) | 9,060 (99.8%) | 0 (0.0%) | 17,600 (99.8%) | 0 (0.0%) | 16,700 (99.8%) |
| 相位时钟与控制 | 0.100 (0.2%) | 0.100 (0.0%) | 0.100 (0.1%) | 0.100 (0.0%) | 0.100 (0.1%) | 0.100 (0.0%) |
| **合计** | **56.8** | **9,080** | **98.2** | **17,600** | **94.8** | **16,800** |

读法:T2 的 crossbar + ADC 一栏里,输出头的 crossbar 导通(45.7 nJ)加 ADC(16.1 nJ)是最大的两块;换成数字事件驱动的输出头后,`head_digital` 占 99.8%,其余部件(各块 crossbar、ADM、gain cell、比较器)与左栏相同,只有输出头自己的栅 / 导线 / 积分器随之去掉。

**表 3 展望列**(各自最优投影,不同节点:memristor 取 3 nm FinFET 投影,数字取 7 nm;不与表 1 的 65 nm 主列混比;能量单位 nJ/token,相对基准 = memristor 3 nm T2、crossbar+ADC = 8.83 nJ)

| 方案 | 模型 | 设定 | 能量/token (nJ) | 相对 memristor 3 nm T2(crossbar+ADC) |
|---|---|---|---|---|
| memristor 3 nm 投影 | T1 | 输出头 crossbar+ADC | 5.15 | 0.583× |
| memristor 3 nm 投影 | T1 | 输出头数字(事件驱动,batch 1) | 823 | 93.1× |
| memristor 3 nm 投影 | T2 | 输出头 crossbar+ADC | 8.83 | 1.00× |
| memristor 3 nm 投影 | T2 | 输出头数字(事件驱动,batch 1) | 1,590 | 181× |
| memristor 3 nm 投影 | TR2 | 输出头 crossbar+ADC | 8.54 | 0.967× |
| memristor 3 nm 投影 | TR2 | 输出头数字(事件驱动,batch 1) | 1,520 | 172× |
| 数字稠密 int8 7 nm | A0 尺寸 | batch 1 | 2,070 | 234× |
| 数字稠密 int8 7 nm | A0 尺寸 | batch 64 | 314 | 35.5× |
| 数字事件驱动 7 nm | T1 | batch 1 | 1,290 | 147× |
| 数字事件驱动 7 nm | T1 | batch 64 | 186 | 21.1× |
| 数字事件驱动 7 nm | T2 | batch 1 | 2,470 | 280× |
| 数字事件驱动 7 nm | T2 | batch 64 | 345 | 39.0× |
| 数字事件驱动 7 nm | TR2 | batch 1 | 2,410 | 273× |
| 数字事件驱动 7 nm | TR2 | batch 64 | 336 | 38.0× |

读法:展望列 T2 是 8.83 nJ,7 nm 稠密 int8 是 2.068 µJ(batch 1)和 314 nJ(batch 64),倍数 234× / 35.5×;两侧节点不同,别与表 1 横向比较。

**表 4 敏感性**(memristor 65 nm、输出头 crossbar+ADC 的总能量,nJ/token;一次只改一个参数)

| 参数 | 取值 | T1 (nJ) | T2 (nJ) | TR2 (nJ) |
|---|---|---|---|---|
| 基准(主值) | — | 56.8 | 98.2 | 94.8 |
| 读窗 t_read | 10 ns | 23.4 | 30.0 | 29.2 |
| 读窗 t_read | 100 ns | 56.8 | 98.2 | 94.8 |
| 读窗 t_read | 1000 ns | 390 | 781 | 751 |
| 电导缩放(G_on 与 G_off 同乘) | 0.1 × | 26.0 | 35.1 | 34.3 |
| 电导缩放(G_on 与 G_off 同乘) | 1 × | 56.8 | 98.2 | 94.8 |
| ADC 能量 E_adc | 1 pJ | 43.9 | 85.3 | 81.9 |
| ADC 能量 E_adc | 5 pJ | 56.8 | 98.2 | 94.8 |
| ADC 能量 E_adc | 10 pJ | 72.9 | 114 | 111 |
| 列积分器偏置 I_int | 1 µA | 54.5 | 93.7 | 90.3 |
| 列积分器偏置 I_int | 5 µA | 56.8 | 98.2 | 94.8 |
| 列积分器偏置 I_int | 50 µA | 82.3 | 149 | 146 |

读法:t_read 是最大的杠杆(10 ns 到 1 µs,T2 从 30.0 变到 781 nJ,跨 26.0 倍),其次是电导整体缩放(0.357×)和 I_int(1–50 µA 为 0.954×–1.52×;积分器偏置项与 I_int 成正比,50 µA 比主值多 51.1 nJ;50 µA 超出设计文档的 1–10 µA),E_adc 在 1–10 pJ 内只动 0.869×–1.16×。

**表 5 嵌入表量化的 token 准确率变化**(百分点;切片上的 teacher-forcing,只量化嵌入表;global = 整表一个 w_max,per_column = 每个嵌入维度各用自己的 w_max;⚠ = 掉幅超过设计文档的门槛 0.5 个百分点,要在 README 写明并给脉冲注入的退路;量化前:T1 75.19%、T2 82.46%、TR2 81.14%)

| 级数 | T1 global | T1 per_column | T2 global | T2 per_column | TR2 global | TR2 per_column |
|---|---|---|---|---|---|---|
| 16 | -19.11 ⚠ | -11.90 ⚠ | -7.69 ⚠ | -4.19 ⚠ | -11.70 ⚠ | -10.90 ⚠ |
| 32 | -16.25 ⚠ | -8.70 ⚠ | -3.63 ⚠ | -3.64 ⚠ | -2.33 ⚠ | -5.44 ⚠ |
| 64 | -5.73 ⚠ | -7.47 ⚠ | -1.63 ⚠ | -1.80 ⚠ | -8.35 ⚠ | -3.51 ⚠ |
| 128 | -3.98 ⚠ | -6.75 ⚠ | -1.33 ⚠ | -0.93 ⚠ | -6.23 ⚠ | -2.61 ⚠ |
| 256 | -4.29 ⚠ | -5.10 ⚠ | -0.44 | -0.44 | -4.73 ⚠ | -1.86 ⚠ |

- T1:无配置满足 0.5 个百分点的门槛;扫描上限 256 级、global 仍掉 4.29 个百分点,这个配置**不满足**门槛。若仍按该配置(2 对差分 cell),嵌入 crossbar 能量为 0.0141 nJ/token(16 级一对 cell 为 0.00703 nJ);这不是 T1 满足门槛时的嵌入能量
- T2:最小满足 0.5 个百分点的配置 = 256 级、global,每个值 2 对差分 cell,嵌入 crossbar 能量 0.0145 nJ/token(16 级一对 cell 为 0.00725 nJ)
- TR2:无配置满足 0.5 个百分点的门槛;扫描上限 256 级、per_column 仍掉 1.86 个百分点,这个配置**不满足**门槛。若仍按该配置(2 对差分 cell),嵌入 crossbar 能量为 0.0146 nJ/token(16 级一对 cell 为 0.00732 nJ);这不是 TR2 满足门槛时的嵌入能量

读法:16 级一行全部越过 0.5 个百分点的门槛(⚠);只有 T2 在 256 级(global 和 per_column 都是)回到门槛以内,T1、TR2 扫到 256 级仍不满足。这个扫描能说明什么、不能说明什么见第 7 节。

## 6. 结论

1. **量级:T2 的 memristor 比同节点稠密 int8 数字 ASIC 低约 2.4 个数量级(batch 1)和约 1.7 个数量级(batch 64;后者取决于激活访存的口径,见本条末尾)。** T2 的 memristor 是 98.2 nJ/token(`M65[T2].total`;T1 56.8、TR2 94.8),稠密 int8 是 22.95 µJ(batch 1)、5.547 µJ(16)、4.677 µJ(64)(`Dn65[b].total`),T2 的倍数 234× / 56.5× / 47.6×(= `Dn65[b].total` / `M65[T2].total`);若稠密行的 `act` 整个不计(累加器在寄存器里,见下一条),是 194×(batch 1)/ 8.14×(batch 64)(= (`Dn65[b].total` − `Dn65[b].act`) / `M65[T2].total`)。T1 是 404× / 82.3×(batch 1 / 64),TR2 是 242× / 49.3×。展望列(3 nm memristor 对 7 nm 数字,节点不同)T2 是 234×(batch 1)、35.5×(batch 64)(`Dnadv[b].total` / `Madv[T2].total`)。权重放在 DRAM 的对照(batch 1)是 190.1 µJ(`Dram.total`),1,930×,只是对照行。这是量级,不是精确倍数:`node_scale` 取 1.5 使倍数降 25%(第 2 节),`act` 的口径减半使 batch 64 的倍数降 41%、batch 1 降 8.4%(见下一条),周边参数全是估计。batch 64 的"约 1.7 个数量级"(log10 47.6 = 1.68)依赖 `act` 的口径:`act` 整个不计时只剩约 0.9 个数量级(log10 8.14 = 0.911);batch 1 不计 `act` 也还有 2.29 个数量级(log10 194)。

2. **优势来自存内计算,不是脉冲稀疏。** 拆成两步看(batch 1 / batch 64):

   - **稠密 → 事件驱动数字**(同一批脉冲,只对非零行做加法、只取非零行的权重)的差,就是"脉冲稀疏 + 三值激活"的收益。T1:14.3 µJ / 2.80 µJ,为稠密的 0.625× / 0.600×(`Ev65[T1,b].total` / `Dn65[b].total`),省 38% / 40%。T2:27.4 µJ / 5.18 µJ,为稠密的 1.19× / 1.11×,**比稠密还贵**;TR2 同样(1.16× / 1.08×)。原因是 T=2 下加法次数多:事件驱动的加法 T2 901,584 次、TR2 877,526 次、T1 468,969 次(= `Ev65[T,1].add` / `Par[E_add16]`),稠密的矩阵乘加是 742,656 次(`mac_counts.total_macs` − `core`)。表里的事件驱动行是**逐脉冲**计的(c = 2 的行做两次加法、取两次权重、做两次激活读写)。如果每个非零行只做一次权重取数、一次加法、一次激活读写(`add`、`weight`、`act` 三项都除以脉冲数 ÷ 非零数的比值 1.636,c = 2 时权重乘 c 靠移位),T2 的事件驱动行是稠密的 0.73× / 0.70×;只让 `weight` 项除以 1.636、加法和激活仍逐脉冲计,则是 0.81× / 1.08×(batch 1 / 64;= (`add` + `weight`/1.636 + `act` + `neurons` + `core` + `state` + `embed`) / `Dn65[b].total`)。1.636 是 `../improve/profile_final.json` 里 `T2_c` 全部 21 个脉冲点按神经元数加权的平均(脉冲数 ÷ 非零数,512 句的画像,未取整值参与计算),不是逐矩阵的值:输出头占 T2 加法次数的 64.7%,它的比值是 1.79;按矩阵逐个取对应脉冲点的比值(qkv、gates ← `q_in1`,o ← `out`,ffn13 ← `q_in2`,head ← `q_out`,w2 取 1 或 g、u 比值的平均),三项都取一次约为 0.69–0.71× / 0.66–0.67×,仍是粗估。T1 的 c ≤ 1(脉冲数 = 非零数),不受影响。所以"稀疏 + 三值"这一步在这个模型上不是量级收益:T1 省约四成,T2 在逐脉冲与"三项都取一次"两种计数口径、两个 batch 之间是稠密的 0.70×–1.19×(只让 `weight` 取一次的 0.81× / 1.08× 落在其中)。
   - **事件驱动数字 → memristor** 的差是模拟存内计算的收益:T2 是 279× / 52.7×(= `Ev65[T2,b].total` / `M65[T2].total`),T1 253× / 49.4×,TR2 281× / 53.2×。batch 1 时事件驱动数字的 82.4% 是权重取数(`Ev65[T2,1].weight`),memristor 的权重不动;batch 64 时取数摊薄到 6.80%,剩下 87.0% 是每次加法的两次激活读写(`Ev65[T2,64].act`),memristor 里由列积分器电容累加代替。稠密行同理(`Dn65[b].weight` 占 80.9% → 6.20%,`act` 占 16.9% → 82.9%)。
   - 所以 batch 1 的 234× 主要是"权重不用搬",batch 64 的 47.6× 主要是"累加不用过 SRAM"。后一项是口径选择:`act` 按每次乘加 2 次 8 bit 近端 SRAM 读写计,没有扫过;若稠密行的 `act` 只计一半,T2 的倍数降到 214× / 27.9×(batch 1 / 64;= (`Dn65[b].total` − `Dn65[b].act`/2) / `M65[T2].total`),batch 64 对这个口径最敏感。脉动阵列式的 int8 阵列把累加器放在寄存器里,部分和不进 SRAM,所以每次乘加 2 次近端 SRAM 读写的 `act` 是设计文档的计账口径,不是数字 ASIC 非付不可的钱;`act` 整个不计(数字侧最省的一端)时 T2 的倍数是 194× / 8.14×(= (`Dn65[b].total` − `Dn65[b].act`) / `M65[T2].total`)。所以 batch 64 的 47.6× 和"约 1.7 个数量级"取决于这个口径:其余参数不变时,按数字侧激活怎么存,batch 64 的倍数落在 8.14× 到 47.6× 之间。

3. **memristor 的开销在输出头,其次是各块 crossbar 导通;神经元、比较器、gain cell 几乎不要钱。** T2(`M65[T2]`):输出头的 crossbar 导通 45.7 nJ(46.5%)+ 栅驱动 / 导线 1.76 nJ + 积分器 3.23 nJ(`by_matrix.head`)+ ADC 16.1 nJ(16.4%,= 3226 列 × `Par[E_adc]`),合计 66.8 nJ,占总量的 68.0%(T1 69.2%、TR2 68.0%)。输出头占流式权重的 55.6%;去掉 ADC 后它自己(crossbar 导通 + 栅 / 线 + 积分器,`by_matrix.head.total` ÷ 合计)占 memristor 能量的 51.6%,低于权重占比,加上 3226 次 ADC(16.4%)才到 68.0%,也就是 ADC 把它抬到了权重占比之上;词表 3226 让输出头一行有 6452 个 cell,每 token 被驱动 180.9 个行次。各块 crossbar 导通 24.5 nJ(24.9%);crossbar 导通合计(各块 + 输出头 + 嵌入)是 70.1 nJ,占 71.4%。列积分器偏置 5.67 nJ(5.78%)、栅驱动 + 导线 2.72 nJ(2.76%)、ADM 脉冲 3.13 nJ(3.18%)。ADM 漏电(0.0576 nJ)、比较器(0.128 nJ)、gain cell(0.158 nJ)、控制(0.100 nJ)、嵌入 crossbar 和未选通漏电加起来 0.451 nJ(0.459%)。也就是说模拟矩阵读(导通 + 栅 / 线 + 积分器)和输出头 ADC 决定 memristor 的能耗,事件驱动的"神经元侧"开销(ADM、比较器、gain cell)合计只有 3.47 nJ(3.53%)。展望列里输出头也是大头(71.1%,其中 ADC 18.3%;`Madv[T2]`)。

4. **"输出头数字"的混合方案不划算。** 输出头改用数字事件驱动(batch 1)后 T2 是 17.6 µJ(`M65h[T2].total`),其中 `head_digital` 占 99.8%(17.6 µJ),是 crossbar + ADC 变体的 179 倍;T1 9.08 µJ、TR2 16.8 µJ(占比 99.8% / 99.8%)。相对稠密 int8:batch 1 的 22.95 µJ 只省到 0.767×,batch 64 的 4.677 µJ 反而是 3.76×,优势完全没了。原因是 memristor 是单 token 流,没有 batch 可摊:每 token 被驱动 180.9 个行次 × 3226 列 = 583,646 次加法,每次都要从 1 MB 级 SRAM 取一个 8 bit 权重,权重取数摊不下去。输出头必须留在模拟域,ADC 的钱得付。

5. **敏感性(表 4,`Sens[T2]`,一次只改一个参数)。** t_read 是最大的杠杆:10 ns 时 30.0 nJ(0.305×),1 µs 时 781 nJ(7.95×)(T1 23.4 / 390 nJ,TR2 29.2 / 751 nJ),因为 crossbar 导通和列积分器偏置都与 t_read 成正比,这两项合计占 77.2%(主值 100 ns 是 paper SI Note 5 对 ADM 神经元的投影"ADM 可在 100 ns 级工作",不是 crossbar 的测量;本仓库仿真过的唯一 crossbar 时序是 SPICE 测试台的 1 µs 积分窗,见第 2 节)。电导整体缩到 0.1×:35.1 nJ(0.357×),只动导通项。E_adc 从 1 pJ 到 10 pJ:85.3–114 nJ(0.869×–1.16×),是四个里最小的;t_read = 10 ns 时 ADC 占 53.8%,成了最大的一块。列积分器偏置 I_int 取 1 / 5 / 50 µA:T2 是 93.7 / 98.2 / 149 nJ(0.954× / 1.00× / 1.52×;`Sens[T2].I_int`;T1 54.5 / 56.8 / 82.3,TR2 90.3 / 94.8 / 146);积分器偏置项与 I_int 成正比,所以是线性的:50 µA 比主值多 51.1 nJ(= 9 × `M65[T2].by_component.integrator` 5.67 nJ),1 µA 少 4.54 nJ(= 0.8 × 5.67 nJ)。50 µA 已超出设计文档的 1–10 µA,这一档 T2 仍比稠密 batch 1 / 64 低 154× / 31.3×(= `Dn65[b].total` / `Sens[T2].I_int["5e-05"]`)。最不利的角(t_read 1 µs,即 SPICE 测试台的积分窗)T2 仍是 781 nJ,比稠密 batch 1 低 29.4×、比 batch 64 低 5.99×:结论的方向不依赖 t_read。没扫的:V_sel、C_gate、C_wire、E_cmp、E_ctrl、`node_scale`、gain cell 参数,以及参数的组合。

6. **GPU 数字的含义。** 空载 21.2 W(`gpu.idle_w`)。A0(浮点)batch 1:总 75.9 mJ/token(总功率 ÷ 吞吐;`GPU[A0,1].total_j_per_token`),扣空载 52.7 mJ,1.09 ms/token(913 token/s);batch 64:总 3.35 mJ,扣空载 2.45 mJ,0.0426 ms/token。T=2 的脉冲模型 batch 1 总 404 mJ(扣空载 291),batch 64 总 16.0 mJ(扣空载 11.7):它在 GPU 上并不省,是 A0 的 5.3× / 4.8×(batch 1 / 64),因为脉冲模型在 torch 里是稠密模拟的,这不说明脉冲硬件。读这几个数要带着:
   - **模型太小,被空载功耗和逐 token 调度淹没。** A0 batch 1 的平均功率 69.3 W、1.09 ms/token(逐 token 发核函数),A0 batch 64 的平均功率 78.7 W;batch 1 → 64 总能量 A0 降 22.6 倍(T2 降 25.2 倍)。
   - **"incremental"是高于深度空载(21.2 W),不是边际计算能量。** 一次性探测(脚本未保留,现象记在 `energy_gpu.py` 的文档字符串里,不在 `gpu.json` 里)显示,建 CUDA 上下文后 GPU 什么都不跑也会在约 66 W 的高性能态停约 17 s 才回落;探测只测了这段空载下的回落,没测负载下的状态底,所以这是推断:负载下的 69–79 W(四格的平均功率,`GPU[m,b].mean_w`)里很可能有一大块是这个状态底,batch 1 尤其如此。
   - **batch 64 的运行含补齐位置**(算了但不计入 token),所以 batch 64 的 J/token 含补齐浪费。
   - **batch 1 的运行只覆盖了前一部分句子**:A0 前 3,697 句、T2 前 810 句(`GPU[m,1].n_batches`,切片共 4,003 句),两个模型覆盖的句子范围不同,平均句长也就不同。
   - **每格只跑一次**,没有重复,没有跑间方差。
   - 所以 GPU 列反映"一块 GPU 跑这个模型",不是 GPU 的峰值效率。A0 的 GPU 总值是 T2 memristor 的 34,100×(batch 64)、772,000×(batch 1)(= `GPU[A0,b].total_j_per_token` / `M65[T2].total`),这个倍数混了节点(8 nm 对 65 nm)、整卡功率、浮点对脉冲和调度开销,不要读成效率比;能直接比的是同节点的 memristor 对数字 ASIC。

7. **嵌入表放进无 DAC 的 crossbar 要付准确率的代价,只有 T2 检查点(基线 C 的 T = 2)能付得起。** 16 级量化使 token 准确率变化 −19.11 / −7.69 / −11.70 个百分点(T1 / T2 / TR2),T2 在 256 级(两对 cell)回到 0.5 以内,T1、TR2 在表 5 的扫描范围(≤ 256 级)内不行,且这个扫描对它们更像在测检查点的脆弱度(详见第 7 节)。能量上这一项可以忽略(嵌入 crossbar 的导通项只占 T2 合计的 0.0074%;连 128 列的积分器偏置一起算,`by_matrix.embed.total` 是 0.1358 nJ、0.138%),问题是准确率稳健性,不是能耗。

## 7. 嵌入量化的发现

设计里嵌入表放进一个 1T1R crossbar(3226 行 × 128 列,差分对,token 的 one-hot 脉冲只驱动一行,列电流直接灌进 128 个累加器电容,没有 SRAM 也没有 DAC),代价是嵌入量化到有限电导级数。检查方法:只把嵌入表量化(`crossbar.quantize_weights`,量化后在评估结束时还原权重),其余不变,在切片上用 teacher forcing 重测 token 准确率;先做 16 级全局缩放(`embed_quant`),再扫 16 / 32 / 64 / 128 / 256 级 × {global(整表一个 w_max)、per_column(每个嵌入维度各用自己的 w_max)}(`embed_quant_sweep`,表 5)。设计文档的门槛是掉幅 > 0.5 个百分点就要写明并给"脉冲注入"退路。

1. **16 级(一对差分 cell)全部越过门槛。** global:T1 −19.11、T2 −7.69、TR2 −11.70 个百分点(`Sw[T,"global/16"]`),token 准确率 0.7519 → 0.5608(T1)、0.8246 → 0.7476(T2)、0.8114 → 0.6943(TR2)(`embed_quant.acc_before` → `acc_after`)。
2. **扫描(表 5)。** T2 在 256 级进入门槛:global −0.44、per_column −0.44 个百分点(`Sw[T2,"global/256"]`、`Sw[T2,"per_column/256"]`)。256 = 16²,每个值需要两对差分 cell(`embed_levels_needed.cells_per_value` = 2),嵌入 crossbar 能量 0.0145 nJ/token(`embed_levels_needed.embed_energy_nJ_at_levels`),16 级一对 cell 是 0.00725 nJ(`M65[T2].by_component.embed`);按两对 cell 一阶外推(能量 ×2,两对在电路里怎样按 1:16 合并没有设计)。T1 和 TR2 扫到 256 级都不满足(`meets_threshold` 为 false / false):T1 在扫描上限 256 级 global 仍是 −4.29(整个扫描里最好的是 128 级 global 的 −3.98,掉幅不随级数单调下降),TR2 最好是 256 级 per_column 的 −1.86。
3. **这个扫描对 T1、TR2 说不清"要几级"。** 一次性探测(脚本未保留,**结果不在 `energy.json` 里,本文没有重跑**):T1 在 global 下 1024 / 4096 / 65536 级是 −5.57 / −7.74 / −0.31 个百分点,不随级数单调;只加 1e-4·σ 的高斯噪声(σ 为嵌入表的标准差)准确率就变化 −0.64 个百分点(1e-3·σ −3.32,1e-2·σ −5.76)。TR2 在 1024 / 4096 / 65536 级是 −1.74 / +0.10 / −0.31(所以在这次探测里 TR2 从 4096 级、即三对 cell 起进入门槛,单次测量),噪声 1e-4 / 1e-3 / 1e-2·σ 是 −0.09 / −0.33 / −2.65。T2 规整得多:噪声 1e-4 / 1e-3 / 3e-3·σ 的 5 个种子都在 ±0.2 个百分点内,量化在 256 / 1024 / 4096 / 65536 级是 −0.44 / −0.19 / +0.11 / −0.06。所以 T1 和 TR2 的"准确率—级数"曲线测到的更像是**检查点对嵌入扰动的脆弱度**(发放阈值翻转沿递归状态放大),不是电导精度的要求;"256 级内没有满足"不能读成"需要更多级就行"(T1 在 4096 级仍是 −7.74,65536 级的 −0.31 夹在非单调的曲线里,不能当作一条可用的配置曲线)。T2 的 −0.44 离门槛 0.5 只差 0.06,而测量本身的噪声底约 ±0.2 个百分点,这个"通过"是边缘的。
4. **结论。** 嵌入 crossbar 对 T2 检查点可行(256 级、每个值两对 cell,边缘通过);对 T1 和 TR2,在表 5 的扫描范围(≤ 256 级)内不可行。诚实的出路有两条:(a) 用量化后的嵌入做微调,让网络适应这个扰动(没做);(b) 脉冲注入退路——把嵌入量化成整数个单位电荷,用累加器的注入电路打 n 个脉冲。(b) 的能耗按构造应当很小(每个通道只是把整数个单位电荷注入累加器电容),**没有建模**,但可以给一个上界(按构造算的,不在 `energy.json` 里):128 通道 × ≤ 128 个单位电荷脉冲(256 级、有符号,|n| ≤ 128)× C_s·V_s²(借 gain cell 的存储电容与摆幅作一次单位电荷注入的量级:`Par[C_s]` 50 fF × (`Par[V_s]` 0.5 V)² = 12.5 fJ)= 0.205 nJ/token,占 T2 合计 98.2 nJ 的 0.21%;实际的 |n| 平均值远小于 128,真实值更低。本文的结论不依赖它;而且它同样要把嵌入量化,T1 / TR2 的准确率问题在这条路里照样存在。所以这是准确率稳健性问题,不是能耗问题。
5. **对表 1–2 的影响。** 表 1–2 的嵌入项用的是 16 级单对 cell 的能量(`M65[T].by_component.embed`:T1 0.00703、T2 0.00725、TR2 0.00732 nJ)。T2 换成 256 级两对 cell 后合计从 98.23 nJ 变到 98.24 nJ(+0.00725 nJ,占 0.0074%),不改变任何结论(这个外推只放大导通项 `by_component.embed`;若 128 列的积分器偏置 `by_matrix.embed.integrator` = 0.128 nJ 也随两对 cell 翻倍,再多 0.128 nJ、0.13%,仍不改变结论)。16 级时嵌入 crossbar 的导通项只占 T2 合计的 0.0074%,含积分器偏置的整个嵌入 crossbar(`by_matrix.embed.total`)是 0.1358 nJ、0.138%。

## 8. 没做的与已知偏差

**口径上的已知偏差**(都已在上文用到,这里集中列):

- **事件驱动数字行逐脉冲计**(T2 / TR2 的 c = 2 的行取两次权重、加两次、做两次激活读写),T2 的这一行因此偏贵,`add`、`weight`、`act` 三项都只取一次(都除以 1.636)的话是稠密的 0.73×(batch 1)/ 0.70×(batch 64),只让 `weight` 取一次则是 0.81× / 1.08×,见结论 2。T1 不受影响。batch 64 时 T2 / TR2 的 `weight` 项(0.352 / 0.343 µJ,`Ev65[T,64].weight`)比稠密把全部权重取一遍的 0.290 µJ(`Dn65[64].weight`)还大,因为逐脉冲的口径把 c = 2 的行取两次;多出的 0.0621 / 0.0527 µJ 占该行的 1.20% / 1.04%(= (`Ev65[T,64].weight` − `Dn65[64].weight`) / `Ev65[T,64].total`)。
- **`act` 项**(每次乘加或加法 2 次 8 bit 近端 SRAM 读写)是口径选择,没有扫;它占稠密 batch 64 的 82.9%,决定了 batch 64 的倍数(减半则 27.9×,整个不计则 8.14×,见结论 1、2)。脉动阵列把累加器放在寄存器里时接近后一种。
- **`node_scale` = 2.0 是自估**,数字侧各项与它成正比;周边和展望列数值都是估计。
- **事件数取自软件模型**(理想电路),标称电路(编程误差、读噪声、失调)下的发放数没测;`../improve/README.md` 记录标称电路单步的脉冲计数不一致在 T = 1 是 14%、T = 2 是 25%、T = 4 是 35%,总数偏多还是偏少没测。
- `leak_off`(未选通 cell 漏电)的"未驱动 cell 数"在 T=2 下对 qkv / gates / ffn13 被钳到 0(平均每行脉冲数超过 1 时 `max(0, …)` 生效),T2 的这一项合计只有 0.00018 nJ,钳不钳都可以忽略。
- 嵌入 crossbar 的 128 个积分器按 T 个读窗供电(`M65[T2].by_matrix.embed.integrator` = 0.128 nJ),而 one-hot 的那一行每 token 只发一次;多计的是 d × `Par[I_int]` × `Par[V_dd]` × `Par[t_read]` × (T − 1) = 0.0640 nJ(T = 2),占 T2 合计的 0.0652%。
- 门 Wa / Wb 的偏置行每 token 按 1 个脉冲计,不随 T 变;T = 2 时恒为 1.0 的输入应是 2 个脉冲。少计的这一个脉冲最多是 n_blocks × 2·(2·n_heads) = 32 个 cell × (`Par[V_read]`²·`Par[G_on]`·`Par[t_read]` + `Par[C_gate]`·`Par[V_sel]`² + `Par[C_wire]`·`Par[V_read]`²) = 32 × 0.402 pJ = 0.0128 nJ,占 T2 合计的 0.0131%。这一条和上一条都留在代码里没改,免得本文的三位有效数字为了 < 0.1% 的效应整体漂移。
- 稠密行的 `act` 与 `state` 对状态访存有重复计数(`state` 占稠密 batch 1 的 0.18%、batch 64 的 0.88%,小于 1%)。
- `n_lif` 按 d 宽度记,d = 128 时精确(2560)。
- 两侧都没有单列 l2norm、sigmoid / softplus 之类逐元素非线性的能量(`energy_model.py` 里没有这些项)。
- 事件计数与 `../improve/profile_final.json` 交叉核对:profile 的 21 个脉冲点去掉 5 个量化 / 比较器点后,T1 / T2 的 LIF 脉冲总数是 1332.7 / 2228.2(512 句),我们在同一 512 句上重数得到 1332.8 / 2228.2(一次性重数,脚本未保留,不在 `energy.json` 里;profile 一侧的数可用 `../improve/profile_final.json` 复算);`energy.json` 里 4003 句的数是 1348.8 / 2248.6(`Ev[T].spikes_lif`),差别是样本不同。
- **未计的项**(都不在设计文档的部件表里,表 2 没有它们;量级按 T2 估):ADM 亚阈值膜电容充电(未发放的神经元,`E_spike` 只计发放):稳态下积分器补给它们的电荷等于 R_c 漏掉的电荷,从 V_dd 取约 0.384 nJ(= `M65[T2].by_component.adm_leak` × `Par[V_dd]`/`Par[V_gs]`,即 n_lif·V_gs·t_token/R_c × V_dd;这份电荷经积分器供出,也落在第 3 节 (ii) 的上界里);残差累加器的脉冲注入与 128 个累加电容:≤ 0.0128 nJ(每 token 至多 2·n_blocks·d·T = 1024 次注入 × `Par[C_s]`·`Par[V_s]`² = 12.5 fJ,单位注入的量级与第 7 节相同);行驱动器自身的开关:≤ 它驱动的栅与导线负载 2.72 nJ(`M65[T2].by_component.gate_wire`,宽松上界);α/β 的门值模拟转换:≤ 0.0800 nJ(每 token 2·n_heads·n_blocks = 16 个门值 × `Par[E_adc]`,每个按一次 8 bit 转换算);gain cell 写入时的 (v−r)·β 乘法:≤ 1.28 nJ(n_blocks·n_heads·dv = 256 个乘积 × `Par[E_adc]`,同样按每个一次转换的宽松上界);积分器的信号相关电荷(见第 3 节)。前五项按上面的式子每项 ≤ 2.72 nJ(T2 合计的 2.76%),注入与累加电容 ≤ 0.1 nJ;最大的是积分器电荷项,只有上界 351 nJ(第 3 节)。

**没做的:**

- 设计文档的非目标:版图级或 SPICE 级功耗仿真;训练能耗;片外通信(除 DRAM 对照行);真实器件的良率、漂移,以及编程误差和读噪声对能量的影响。
- **int8 精度未验证。** 数字稠密行只计成本,A0 没有量化;GDN 状态和 l2norm 在 int8 下的精度不知道。
- **用量化后的嵌入做微调**(T1 / TR2 的出路)没做;脉冲注入的能耗只给了按构造的上界(第 7 节),没有建模。
- **GPU 没做多次运行统计**:每格一次;GPU 上没测 T1 / TR2,也没测 int8。
- **每个脉冲点各自不同的级数**没支持:`count_events` 遇到混合 levels 会报错,三个检查点都是全模型统一 T。
- 嵌入 256 级时两对 cell 怎样合并(按 1:16 加权)的电路没设计,能量是一阶外推。
- 敏感性只扫了 t_read、电导缩放、E_adc、I_int 四个,且是单参数;其余周边参数、`node_scale`、gain cell 参数和组合没扫。
- 积分器的信号相关电荷与读电压轨的转换效率没有建模,只给了上界(第 3 节)。

## 9. 复现

在 `step2-toy-demo` 下,`--device cuda` 时 `CUDA_VISIBLE_DEVICES` 必须是 1(0 号卡归别的项目,否则脚本退出):

```
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/energy_gpu.py \
    --models A0=results/ckpt/a0.pt,T2=results/improve/T2_c/model.pt --batches 1,64 \
    --seconds 60 --idle-seconds 20 --gpu-index 1 --save results/energy/gpu.json

CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/energy_estimate.py \
    --models T1=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt,T2=results/improve/T2_c/model.pt,TR2=results/improve/ternres_sat_T2_c/model.pt \
    --a0 results/ckpt/a0.pt --out results/energy --gpu-json results/energy/gpu.json --device cuda
```

第一条写 `gpu.json`(约 5 分钟),第二条读它并写 `energy.json` 和 `tables.md`(`--gpu-json` 留空则表 1 不含 GPU 行)。测试在 `tests/test_energy_estimate.py`。

`energy.json` 的 `meta` 记下运行时的 git HEAD 短 SHA(`git_sha`)、日期(`run_date`)、torch 版本(`torch`)和 `--device`(`device`)。本文的数字来自 `meta` 记为 `cf68512`、2026-10-29、torch 2.14.0+cu126、`cuda` 的那次运行。`cf68512` 是运行时的 HEAD:生成脚本的最后一批改动(I_int 扫描、这四个 `meta` 字段、表 5 脚注的措辞、`t_read` 的来源字符串)当时还在工作区里,随后提交在 `cf68512` 之上;与 `cf68512` 本身的输出相比,新的 `energy.json` 只多了这四个 `meta` 字段和 `models.*.sensitivity.I_int`、改了 `t_read` 的来源字符串,其余每个值都相同。
