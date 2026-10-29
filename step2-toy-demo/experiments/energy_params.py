"""能耗估算的参数表:memristor 电路与数字 ASIC 的全部输入参数集中在这里,每个参数带 value(SI 单位)、unit 与 source。
两列:`65nm` 是主对比(两侧 CMOS 都按 65 nm,忆阻器件与 ADM 神经元不随节点缩),`advanced` 是各自最优投影(memristor 取
3 nm FinFET、数字取 7 nm,节点不同,不与主列混比)。`DEVICE` 是 SPICE 测试台的忆阻器件常数,两列共用。无 I/O,无 torch。"""

NODE_SCALE_65 = 2.0                     # Horowitz 45 nm → 65 nm 的 CV² 换算系数(见下方 node_scale 项的 source)


def _p(value, unit, source):
    return dict(value=value, unit=unit, source=source)


_SRC_DEVICE = "experiments/spice/crossbar.py 测试台器件(HfO₂ 类 1T1R)"
_SRC_ADM_65 = "Zhao et al. 2025 Nature Electronics, SI Table 7(65 nm 能耗配置),转引自 step1-spiking-layer/results/adm-neuron.md"
_SRC_ADM_3 = "同文 Table 8(3 nm FinFET 投影)"
_SRC_PERIPH_65 = "65 nm 文献量级,估计(见 README 参数表)"
_SRC_ADV = "估计(文献量级),见 README"
_HOROWITZ = ("Horowitz, Computing's energy problem (and what we can do about it), ISSCC 2014, 45 nm: int8 add 0.03 pJ, "
             "int8 mult 0.2 pJ, int16 add 0.05 pJ, int32 add 0.1 pJ, 8 KB SRAM 10 pJ/64 b, 1 MB SRAM 100 pJ/64 b, "
             "DRAM 1.3–2.6 nJ/64 b")

# 忆阻器件:G_on、G_off、级数、读电压;两列共用(器件量不随 CMOS 节点缩)。
DEVICE = dict(
    G_on=_p(100e-6, "S", _SRC_DEVICE),
    G_off=_p(1e-6, "S", _SRC_DEVICE),
    levels=_p(16, "级", _SRC_DEVICE),
    V_read=_p(0.2, "V", _SRC_DEVICE),
)


def _device_copy():
    """每列各持一份器件项的拷贝,改一列的值(敏感性扫描)不会串到另一列。"""
    return {k: dict(p) for k, p in DEVICE.items()}


def _digital_65(base, unit, what):
    """Horowitz 45 nm 基准值 × node_scale,已乘好存放;公式写在 source 里。"""
    return _p(base * NODE_SCALE_65, unit, f"{_HOROWITZ};本项 = {what} × node_scale {NODE_SCALE_65}")


_COL_65NM = dict(
    _device_copy(),
    # 读窗与 token 周期
    t_read=_p(100e-9, "s", "paper SI Note 5:ADM 可在 100 ns 级工作;敏感性 10 ns–1 µs"),
    t_token=_p(1e-3, "s", "t_token = R_c·C_g·ln(1/0.9),R_c 1 GΩ、C_g 10 pF(Table 7)"),
    # 1T1R crossbar 的周边
    V_sel=_p(1.0, "V", _SRC_PERIPH_65),
    C_gate=_p(1.5e-15, "F", _SRC_PERIPH_65),
    C_wire=_p(0.15e-15, "F/cell", _SRC_PERIPH_65),
    I_off=_p(10e-12, "A", _SRC_PERIPH_65),
    # 列积分器
    I_int=_p(5e-6, "A", _SRC_PERIPH_65),
    V_dd=_p(1.0, "V", _SRC_PERIPH_65),
    # ADM 神经元
    E_spike=_p(1.39e-12, "J/spike", _SRC_ADM_65),
    R_c=_p(1e9, "Ω", _SRC_ADM_65),
    V_gs=_p(0.15, "V", _SRC_PERIPH_65),
    # 比较器、ADC、gain cell、控制
    E_cmp=_p(50e-15, "J/判决", _SRC_PERIPH_65),
    E_adc=_p(5e-12, "J/次", _SRC_PERIPH_65),
    C_s=_p(50e-15, "F", _SRC_PERIPH_65),
    V_s=_p(0.5, "V", _SRC_PERIPH_65),
    E_ctrl=_p(100e-12, "J/token", _SRC_PERIPH_65),
    # 数字 ASIC:Horowitz 45 nm × node_scale
    node_scale=_p(NODE_SCALE_65, "×", "自估:45 → 65 nm 的 CV² 换算,尺寸比 65/45 ≈ 1.44 × 电压比平方 (1.2/1.0)² = 1.44,约 2.07,取 2.0;无外部出处。memristor 侧 65 nm 周边按 V_dd = 1.0 V 取,故对数字侧是偏高端的估计"),
    E_mul8=_digital_65(0.2e-12, "J", "int8 mult 0.2 pJ"),
    E_add8=_digital_65(0.03e-12, "J", "int8 add 0.03 pJ"),
    E_add16=_digital_65(0.05e-12, "J", "int16 add 0.05 pJ"),
    E_add32=_digital_65(0.1e-12, "J", "int32 add 0.1 pJ"),
    E_sram_near_bit=_digital_65(10e-12 / 64, "J/bit", "8 KB SRAM 10 pJ/64 b ÷ 64"),
    E_sram_far_bit=_digital_65(100e-12 / 64, "J/bit", "1 MB SRAM 100 pJ/64 b ÷ 64"),
    E_dram_bit=_digital_65(2e-9 / 64, "J/bit", "DRAM 取 1.3–2.6 nJ/64 b 的中值 2 nJ/64 b ÷ 64(仅用于权重在 DRAM 的对照)"),
)

_ADV_NOTE = "展望列(各自最优投影,不同节点)"
_COL_ADVANCED = dict(
    _device_copy(),
    t_read=_p(10e-9, "s", "同处:10 ns 脉冲刺激"),
    t_token=_p(0.1e-6, "s", "同式,R_c 242 GΩ、C_g 4.14 aF(Table 8)"),
    V_sel=_p(0.7, "V", f"{_SRC_ADV};3 nm 选通管,{_ADV_NOTE}"),
    C_gate=_p(0.1e-15, "F", f"{_SRC_ADV};{_ADV_NOTE}"),
    C_wire=_p(0.05e-15, "F/cell", f"{_SRC_ADV};{_ADV_NOTE}"),
    I_off=_p(10e-12, "A", f"{_SRC_ADV};未投影,沿用 65 nm 列的值(漏电项本就可忽略)"),
    I_int=_p(1e-6, "A", f"{_SRC_ADV};{_ADV_NOTE}"),
    V_dd=_p(0.7, "V", f"{_SRC_ADV};{_ADV_NOTE}"),
    E_spike=_p(1e-18, "J/spike", f"{_SRC_ADM_3};1 aJ/spike"),
    R_c=_p(242e9, "Ω", f"{_SRC_ADM_3};242 GΩ"),
    V_gs=_p(0.15, "V", f"{_SRC_ADV};ADM 器件量,沿用 65 nm 列的值"),
    E_cmp=_p(5e-15, "J/判决", f"{_SRC_ADV};{_ADV_NOTE}"),
    E_adc=_p(0.5e-12, "J/次", f"{_SRC_ADV};{_ADV_NOTE}"),
    C_s=_p(10e-15, "F", f"{_SRC_ADV};{_ADV_NOTE}"),
    V_s=_p(0.3, "V", f"{_SRC_ADV};{_ADV_NOTE}"),
    E_ctrl=_p(10e-12, "J/token", f"{_SRC_ADV};{_ADV_NOTE}"),
    # 数字 ASIC:7 nm 公开量级,直接取值,不再乘 node_scale
    node_scale=_p(1.0, "×", f"{_SRC_ADV};7 nm 数字值直接取用,不再换算"),
    E_mul8=_p(0.03e-12, "J", f"{_SRC_ADV};7 nm int8 mult,{_ADV_NOTE}"),
    E_add8=_p(0.005e-12, "J", f"{_SRC_ADV};7 nm int8 add,{_ADV_NOTE}"),
    E_add16=_p(0.008e-12, "J", f"{_SRC_ADV};7 nm int16 add,{_ADV_NOTE}"),
    E_add32=_p(0.015e-12, "J", f"{_SRC_ADV};7 nm int32 add,{_ADV_NOTE}"),
    E_sram_near_bit=_p(0.02e-12, "J/bit", f"{_SRC_ADV};7 nm 近端 SRAM,{_ADV_NOTE}"),
    E_sram_far_bit=_p(0.3e-12, "J/bit", f"{_SRC_ADV};7 nm 片上大 SRAM,{_ADV_NOTE}"),
    E_dram_bit=_p(10e-12, "J/bit", f"{_SRC_ADV};DRAM 每 bit,{_ADV_NOTE}"),
)

COLUMNS = {"65nm": _COL_65NM, "advanced": _COL_ADVANCED}


def v(col, name):
    """列 col 的参数 name 的值(SI 单位)。"""
    return COLUMNS[col][name]["value"]
