# SPICE 测试台原理图

由 `experiments/spice/draw_schematics.py` 生成(schemdraw),每张图对应一个网表生成函数,元件值、节点名与网表一致。
重新生成:`.venv/bin/python experiments/spice/draw_schematics.py`。每张图有 SVG 与 PNG 两种格式。

| 图 | 网表 | 内容 |
|---|---|---|
| `01_gain_cell_single_step` | `gain_cell._netlist` | 单 gain cell 时序判别电路:C_s 存储、R_leak + S_leak(φ1 泄漏窗)、E1 跟随器 + S_sh(φ3 采样)+ C_hold、B_w 受控电流源(φ4 写入) |
| `02_phase_timing_A_B_C` | `gain_cell.schedules` | 四种时序下 φ1 / φ3 / φ4 的相位图及各自的 SPICE 结果 S₁(A 1.250 / B 1.000 / C 0.751 / 常开 0.968) |
| `03_leak_paths_gf180`(含 03a–03d) | `gf180.alpha_leak` | GF180MCU nmos_3p3 上的四种泄漏路径:接地、亚阈区到 V_mid、三极管区到 V_mid、开关 + 电阻(采用) |
| `04_ideal_switch_neuron_placeholder` | `dm_neuron._netlist` | 理想稳定 DM 的占位模型:RC 膜 + 带迟滞的阈值开关 + R_on |
| `05_adm_1m1t1r` | `adm_spice.netlist` | Zhao 2025 的 1M1T1R 神经元:ADM 物理模型、C_g、R_c、M1、R_l、V_dd |
| `05b_adm_state_equations` | `adm_spice.netlist`(B 源部分) | ADM 内部状态方程的实现框图:E → x₁ 积分 → x₂ 积分 → R_d(x) → B_d 电流 → 导通事件 |
| `07a`–`07d` | `gdn_layer.netlist` | GDN 子块闭环的框图:子块数据流、神经元、gain cell、单 token 内的相位时间线 |
| `07e_gdn_layer_schematic`(另切为 `07e_left_projection_output` 与 `07e_right_state_array`) | `gdn_layer.netlist` | 单头 d=16 GDN 子块的整层电路图,横版:左半 = W_q / W_k / W_v 三个电阻 crossbar 共用 16 条字线、每条位线一个神经元、W_O crossbar 与 SN_out;右半 = 16×16 gain cell 阵列(行线、Δv 列线、读线,每 cell 电容 + φ1 泄漏支路 + 写入源 + 读出源)、列底的 φ3s 采样保持、delta 块与 SN_pre。16 行 16 列只画第 1、2、16 行与第 1、2、16 列。 |
