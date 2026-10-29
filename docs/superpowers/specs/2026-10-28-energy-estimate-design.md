# SNN 电路每 token 能耗估算:设计(2026-10-28,对话中批准的范围)

## 意图

给出 toy demo 这个脉冲 GDN 语言模型在 memristor 电路上每 token 的能量,和同一模型在数字 ASIC 与 GPU 上的每 token 能量放在一张表里,
用来说明 memristor 电路相对数字实现的能耗优势有多大、来自哪里、哪些地方反而是 memristor 的开销。这是量级对比,不是精确功耗仿真。

## 目标 / 非目标 / 反目标

- 目标:三列可比的 J/token(memristor、数字 ASIC、GPU),每列有按部件的分解;memristor 侧按切片上**实测的每 token 事件数**算,
  不用平均发放率假设;所有参数集中在一处、每个参数标来源;给出主要参数的敏感性。
- 非目标:版图级或 SPICE 级功耗仿真;训练能耗;片外通信;真实器件的良率与漂移。
- 反目标(做出来也算失败):只报一个总数不报分解;把输出头的 ADC 或数字侧的 batch 依赖藏起来;两侧工艺节点不一致;
  把 paper 的 3 nm 投影和数字侧的 45 nm 数字放在同一列里比。

## 对象

- 模型:toy demo(d = 128、2 块、4 头、d_k = d_v = 32、d_ff = 256、词表 3226、116 万参数)。
- 脉冲模型三个,都用切片(4003 句、55 488 token)实测事件数:T=1 基线 C(`results/conversion-sweep/CL2_l1_lif225_lr3e3_12k`,0.752)、
  T=2 C(`results/improve/T2_c`,0.825)、三值残差 sat T=2(`results/improve/ternres_sat_T2_c`,0.811)。
- 数字侧用浮点 A0(`results/ckpt/a0.pt`,0.842)的 int8 版本:同结构、同乘加数。
- 事件计数来自模型的 `record=True` 记录:每个权重矩阵被驱动的行(输入脉冲的非零通道;T > 1 时按计数 c 算 c 个脉冲),
  gain cell 的检索/写入/读出行数,各脉冲点的发放数,比较器判决数,DAC 次数。

## 工艺节点对齐(已批准)

- **主对比统一到 65 nm**:memristor 侧全部 CMOS 周边(选通管、列积分器、比较器、DAC/ADC、相位控制)和数字侧的乘加、SRAM 都按 65 nm 取值。
  数字侧从 Horowitz(ISSCC 2014,45 nm)按 CV² 换算到 65 nm,换算系数写明(builder 定,量级 1.5–2);或直接用公开的 65 nm 数字。
- **忆阻器件量不随 CMOS 节点缩**:HfO₂ 1T1R 突触的电导、读电压,ADM 神经元的每脉冲能量,在两列里保持不变。
- **展望列**:memristor 用 paper 的 3 nm FinFET 投影(神经元 ≈ 1 aJ/spike,100 ns 级),数字用 5–7 nm 的公开数字;
  表头明确标"各自最优投影,不同节点"。
- **GPU** 是 8 nm 的实测,单独一列作参照,不参与节点对齐。
- GF180 SPICE 测试台只提供器件行为(R_on、IR drop),不贡献能耗数字。

## memristor 侧:部件与参数(主列 65 nm)

| 部件 | 事件 | 能量模型 | 参数(builder 可在标注范围内定,须标来源) |
|---|---|---|---|
| 1T1R 突触 crossbar(Wq/Wk/Wv 共行、Wo、W1/W3 共行、W2、门 Wa/Wb、输出头) | 每个输入脉冲驱动一整行,差分对 2 cell/权重 | Σ_cells V_read²·G_cell·t_read,G_cell 由该模型训练好的权重量化到 16 级后逐行算 | G_on 100 µS、G_off 1 µS、V_read 0.2 V(测试台器件,定);t_read 100 ns 主值,敏感性 10 ns / 1 µs;未选通 cell 漏电:65 nm 选通管 I_off 量级 × t_token |
| 列积分器(电流传输器 / 膜电容充电) | 每个矩阵每 token 一个读窗,每列一个 | 偏置电流 × V_dd × t_read,只在读窗内供电 | 65 nm 电流传输器偏置 1–10 µA、1 V(文献量级) |
| ADM 神经元(8 个 LIF 点) | 每次发放 | E_spike × 发放数 | 1.39 pJ/spike(paper Table 7,65 nm 配置);静态漏电 R_c 1 GΩ 下可忽略,写明 |
| 块入口 / 输出比较器(Q_in1、Q_in2、Q_out;三值残差版是 Q_emb、M1、M2) | 每通道每 token 2T 次判决(与结果无关) | E_cmp × 判决数 | 65 nm 时钟比较器 10–100 fJ/判决 |
| gain cell 阵列(每块 4 头 × 32×32,共 8192 cell) | 泄漏窗:全部 cell 每 token;检索 / 写入:有 k 脉冲的行 × 32 列;读出:有 q 脉冲的行 × 32 列 | 写入 ½C_sV²;读出 / 检索按读电流 × t_read;泄漏按 (1−α)·C_s·V² | C_s 50 fF、摆幅 0.5 V(65 nm gain-cell 量级);α 取每 token 实际门值 |
| 嵌入 crossbar(3226 行 × 128 列,差分对,16 级;token 的 one-hot 脉冲驱动唯一一行,列电流直接灌进 128 个累加器电容;**没有 SRAM、没有 DAC**) | 每 token 1 行 × 256 cell | 与突触 crossbar 同一公式(导通 + 栅驱动 + 导线) | 同突触器件;三值残差版再过 Q_emb 比较器(或离线折进表) |
| 选通栅驱动与行线 / 位线导线 CV² | 每个驱动行 × 该行 cell 数 | N·C_gate·V_sel² + N·C_wire·V_read² | 65 nm:C_gate 1–2 fF、V_sel 1 V、C_wire 0.1–0.2 fF/cell |
| 输出头的 ADC(头放 crossbar 时) | 每列每 token 1 次,3226 列 | E_adc × 3226 | 65 nm 8 bit SAR ≈ 1–10 pJ/次;**另给一列"输出头用数字算"**(按数字侧公式,batch 同数字侧) |
| 相位时钟与控制 | 每 token | 固定 allowance | 65 nm 小型数字控制,builder 给量级并标明是估计 |

token 周期 t_token 由神经元的膜时间常数和软件的泄漏因子 0.9 定:t_token = R_c·C_g·ln(1/0.9)。65 nm 列 R_c·C_g = 1 GΩ × 10 pF = 10 ms → t_token ≈ 1 ms;
3 nm 列 R_c·C_g = 242 GΩ × 4.14 aF ≈ 1 µs → t_token ≈ 0.1 µs。T = 2 的两个子时隙在同一个 t_token 内(器件弛豫 < 250 ns),静态项不变。
主列的 memristor 结果分三行报:三个脉冲模型各一行;每行再给"输出头在 crossbar + ADC"和"输出头数字(事件驱动,batch 1)"两个数。

### 审稿后定下的细节(我的决定,可改)

1. **未选通 cell 的偏置与选通方式。** 沿用 `crossbar.py` / `block_layer.py` 的做法:位线虚地 0 V,行线只在该行有脉冲时在读窗内驱动到 ±V_read,其余时间 0 V;
   所以读窗外全阵列两端都是 0 V,I_off 漏电按 V_read × I_off × t_read 只对读窗内的未驱动 cell 计(可忽略,但保留为一项)。
   选通栅**按行、随脉冲门控**(只有发脉冲的行在读窗内抬栅),栅驱动能量 = 驱动行数 × 每行 cell 数 × C_gate·V_sel²,C_gate 取 65 nm 最小管 1–2 fF、V_sel 1 V;
   行线 / 位线的导线电容 CV² 也按驱动行计(每 cell 约 0.1–0.2 fF)。输出头一行 6452 个 cell,这一项与导通能量同量级,必须单列。
2. **嵌入(用户定:不用 DAC)。** memristor 侧:嵌入表本身放进一个 1T1R crossbar(3226 行 × 128 列),token 是 one-hot 脉冲、只驱动一行,
   列电流在读窗内直接灌进累加器电容——与 W·x 的 crossbar 读是同一电路,没有 SRAM 也没有 DAC;能耗按突触 crossbar 的公式计(1 行 × 256 cell)。
   三值残差版再过 Q_emb 的 2T 个比较器(每通道每 token 2T 次判决)。代价是嵌入量化到 16 级:**产出里加一项准确率检查**——把三个脉冲模型的嵌入表
   按 `crossbar.quantize_weights` 量化到 16 级后前向跑切片,报 token 准确率的变化;若掉幅 > 0.5 个百分点,在 README 里写明并给"脉冲注入"退路
   (嵌入量化成整数个单位电荷、用累加器的注入电路打 n 个脉冲)的能耗作对照。数字侧仍按 SRAM 查 128 × 8 bit 计,见上节。
3. **"输出头数字"行的含义。** 事件驱动:只对 q_out 非零的行做加法(约百余行 × 3226 列),只取这些行的 8 bit 权重;batch 固定为 1
   (memristor 芯片是单 token 流,ADM 与 gain cell 的状态按流存在,不做批处理)。预计这一行被权重取数支配(µJ 级),会抵消大部分优势——这正是要展示的。
4. **数字参照两种**:稠密 int8 A0 和事件驱动的数字 SNN,见上节。
5. **ADM 能量的覆盖与漏电。** 1.39 pJ/spike 按"神经元内部(器件导通 + C_g 充放电 + 输出级)"计,crossbar 给它的输入电荷另在导通能量里计;
   二者最多重复 ½C_g·V² ≈ 0.2 pJ/spike,保留重复(高估 memristor 侧,对我们要证明的结论是保守的),写明。
   ADM 静态漏电 V_gs²/R_c × t_token 按 V_gs 平均 0.15 V 计(约 20 fJ/神经元/ms,3200 个神经元约 0.07 nJ/token),单列不忽略。

## 数字 ASIC 侧

- 乘加数按模型结构逐层算(投影、W_O、FFN、GDN 状态的检索 / 写入 / 读出、门、输出头),int8 乘加;激活和状态的本地访存按每乘加 2 次 8 bit 近端 SRAM 读写计;
  **权重流量不含嵌入表**:嵌入表(3226 × 128 = 41 万参数)每 token 只查 128 个值,其余约 74 万参数 × 8 bit 在 batch = 1 时每 token 全部从片上 SRAM(1–2 MB 级)读一遍,
  batch = B 时按 1/B 摊销。给 batch 1 / 16 / 64 三档。嵌入查表(128 × 8 bit 的 SRAM 读)两侧都计,用同一个 SRAM 能量。
- **再加一行事件驱动的数字 SNN**:同一个脉冲模型(T=2 C)在数字 ASIC 上跑,只对非零输入行做加法、只取非零行的权重(权重流量 = 驱动行数 × 列数 × 8 bit),
  batch 1 / 16 / 64。有了它,优势才能拆成"脉冲稀疏 + 三值激活"和"模拟存内计算"两部分。
- int8 A0 只计成本,不量化、不测精度(GDN 状态与 l2norm 的 int8 精度未验证,写明)。
- 主列 65 nm:Horowitz 45 nm 数字(int8 mult 0.2 pJ、add 0.03 pJ;8 KB SRAM 10 pJ/64 b;1 MB SRAM 100 pJ/64 b;DRAM 1.3–2.6 nJ/64 b)× 65 nm 换算系数。
  权重放不下片上 SRAM 的情形不在主列(116 万参数放得下),但写明若权重在 DRAM 的数字作对照。
- 展望列 5–7 nm:公开的 int8 乘加与 SRAM 访问能量(builder 选来源,标明)。
- 与 memristor 侧同一张表:总数 + 乘加 / 权重访存 / 激活访存三项分解。

## GPU 侧(实测)

- 设备:GPU 1(RTX A4500,8 nm,200 W);GPU 0 不可用。
- 跑 A0(浮点)的 teacher-forcing 前向(逐 token 的 `step` 循环,就是实际推理的形状),batch 1 和 batch 64 两档,每档持续 ≥ 60 s;
  同时用 `nvidia-smi --query-gpu=power.draw` 以 ≥ 5 Hz 采样;报两个数:总功率 / 吞吐(含空载)和 (功率 − 空载) / 吞吐。
- 顺带跑一次 T=2 C 的脉冲模型做参照(它在 GPU 上并不省)。
- 写明:小模型在 GPU 上被空载功耗与逐 token 调度淹没,这个数字反映的是"GPU 跑这种模型"而不是 GPU 的峰值效率。

## 产出

- `step2-toy-demo/experiments/energy_estimate.py`:参数字典(每项带 `source` 字段)、`count_events(model, tokens)`、`memristor_energy(...)`、
  `digital_energy(...)`、`gpu_measure(...)`、敏感性扫描(t_read、G 缩放、E_adc、batch),输出 `results/energy/energy.json` 与 markdown 表。
- `step2-toy-demo/results/energy/README.md`:参数表(含来源)、总表、分解、敏感性、结论("优势多少倍、来自哪里、开销在哪里"),以及没做的。
- 嵌入 16 级量化的准确率检查(三个脉冲模型,前向),结果进 README。
- 测试:事件计数在微型配置上与手算一致;能量公式对已知输入给出已知输出;数字侧乘加数与按参数量(不含嵌入表)的解析值一致。

## 留给 builder 的

- 65 nm 周边电路各项的具体数值与出处(在上表标注的范围内),Horowitz 45 → 65 nm 的换算系数及依据,展望列的来源。
- 事件计数的实现细节(复用 `toy_demo` 的 record 键;T > 1 的计数;三值残差模型的点名映射)。
- GPU 采样脚本的实现(子进程采样 nvidia-smi 或 `pynvml`,若可用)。
- 表的排版。
