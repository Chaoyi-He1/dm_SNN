# SPICE 测试台原理图

由 `experiments/spice/draw_schematics.py` 生成(schemdraw),每张图对应一个网表生成函数,元件值、节点名与网表一致。
重新生成:`.venv/bin/python experiments/spice/draw_schematics.py`(只画 06:参数 `crossbar`;只画 08a–08d:参数 `block`;只画 07e:参数 `layer`)。
每张图有 SVG 与 PNG 两种格式。中文字体按 Arial Unicode MS / PingFang SC / Hiragino Sans GB / Heiti SC / Noto Sans CJK 的顺序取第一个可用的;
schemdraw 每张图只用一种字体,Noto CJK 缺 ⋮ 与上下标数字,所以 06 与 08 的标注不用上下标。

| 图 | 网表 | 内容 |
|---|---|---|
| `01_gain_cell_single_step` | `gain_cell._netlist` | 单 gain cell 时序判别电路:C_s 存储、R_leak + S_leak(φ1 泄漏窗)、E1 跟随器 + S_sh(φ3 采样)+ C_hold、B_w 受控电流源(φ4 写入) |
| `02_phase_timing_A_B_C` | `gain_cell.schedules` | 四种时序下 φ1 / φ3 / φ4 的相位图及各自的 SPICE 结果 S₁(A 1.250 / B 1.000 / C 0.751 / 常开 0.968) |
| `03_leak_paths_gf180`(含 03a–03d) | `gf180.alpha_leak` | GF180MCU nmos_3p3 上的四种泄漏路径:接地、亚阈区到 V_mid、三极管区到 V_mid、开关 + 电阻(采用) |
| `04_ideal_switch_neuron_placeholder` | `dm_neuron._netlist` | 理想稳定 DM 的占位模型:RC 膜 + 带迟滞的阈值开关 + R_on |
| `05_adm_1m1t1r` | `adm_spice.netlist` | Zhao 2025 的 1M1T1R 神经元:ADM 物理模型、C_g、R_c、M1、R_l、V_dd |
| `05b_adm_state_equations` | `adm_spice.netlist`(B 源部分) | ADM 内部状态方程的实现框图:E → x₁ 积分 → x₂ 积分 → R_d(x) → B_d 电流 → 导通事件 |
| `06_crossbar_1t1r_bench` | `crossbar.bench_netlist` / `crossbar.xbar_lines` | 差分 1T1R crossbar 测试台:每行 ± 两条字线(Vrp_i 与 Ern_i = −V_read·x_i)、每个交叉点忆阻电阻 R_m + GF180 nmos_3p3 选通管(栅接 sel)、位线底的 0 V 电流计 Vs_j、电流复制 F_j 进 10 pF 积分电容 + S_clr。只画行 1、16 与列 1、16 |
| `07a`–`07d` | `gdn_layer.netlist` | GDN 子块闭环的框图:子块数据流、神经元、gain cell、单 token 内的相位时间线 |
| `07e_gdn_layer_schematic`(另切为 `07e_left_projection_output` 与 `07e_right_state_array`) | `gdn_layer.netlist` | 单头 d=16 GDN 子块的整层电路图,横版:左半 = W_q / W_k / W_v 三个电阻 crossbar 共用 16 条字线、每条位线一个神经元、W_O crossbar 与 SN_out;右半 = 16×16 gain cell 阵列(行线、Δv 列线、读线,每 cell 电容 + φ1 泄漏支路 + 写入源 + 读出源)、列底的 φ3s 采样保持、delta 块与 SN_pre。16 行 16 列只画第 1、2、16 行与第 1、2、16 列。 |
| `08a_block_layer_datapath` | `block_layer.netlist` | 完整一层的数据通路:累加器 h_i(C_acc,load 装入 h_in,acc1 注入 y1,acc2 注入 y2)→ Q_in1(2T 个比较器,cmp1 锁存 x1)→ GDN 注意力子块 → acc1 → Q_in2(cmp2 锁存 x2)→ W_1 / W_3 + g、u 神经元 → 锁存 + AND(p = g·u)→ W_2 + y2 神经元 → acc2 |
| `08b_block_layer_schematic` | `block_layer.netlist` | 一层里新增部分的电路图:每通道 DAC + S_ld + C_acc + B_acc、两组比较器与锁存;W_1 / W_3 共用字线 x2_i(列交错画 g_j、u_j)、每列一个神经元、AND 得 p_j;W_2 crossbar(字线 p_j)与 y2 神经元。注意力子块画成方框(电路见 07e)。通道、列只画 1、2、16(前馈 1、2、32) |
| `08c_block_layer_phases` | `block_layer.schedule` | 一个 token 的相位时间线,T = 1(周期 13.3 µs)与 T = 2(15.3 µs):clr/load、cmp1、A(q/k/v)、p1/p3/p4(gain cell)、B(SN_pre)、C(SN_out)、acc1、cmp2、D(g/u)、E(y2)、acc2;每个神经元级 T 个子时隙(采样 s_k、计数 + 复位 a_k) |
| `08d_multilevel_neuron_and_comparators` | `block_layer.neurons` / `block_layer.comparators` | 每 token 最多 T 个脉冲的神经元:B_int 积分、C_n、泄漏窗、比较器 + T 个子时隙采样开关 → f 锁存、B_s 计数到 s(每次 f/T)、B_reset 每次减 f·θ/T;块入口量化器的 2T 个比较器(±jθ/T)+ (Σ 上 − Σ 下)/T + 采样锁存 |
