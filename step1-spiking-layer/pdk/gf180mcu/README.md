# GF180MCU 开源 PDK 的 ngspice 模型(固定提交)

来源:GlobalFoundries / Google 开源 PDK 仓库 `google/globalfoundries-pdk-libs-gf180mcu_fd_pr`,
提交 `9f992d5a9186d1f7820c58f039c484ad35b2edea`(2026-09-14 下载),Apache 2.0(见 LICENSE)。

| 文件 | 内容 | SHA-256 |
|---|---|---|
| `design.ngspice` | 全局开关与工艺角参数 | 8d9721a5bf8f079d3fddbd03339af9a0c84d4feb06db8e06465fbd02c7500508 |
| `sm141064.ngspice` | 器件模型库(MOSFET BSIM4 level 54、电阻、电容、BJT;typical / ss / ff / fs / sf 等) | 73fc67d38747d95ce03f3c2ba5f0a25c98f56a293363a9df4c971a3a28a3dcda |

用法:`.include design.ngspice`,`.lib sm141064.ngspice typical`;MOSFET 为分箱模型,按 `M1 d g s b nmos_3p3 W=1u L=0.28u` 实例化,
3.3 V 器件 `nmos_3p3` / `pmos_3p3`,6 V 器件 `nmos_6p0` / `pmos_6p0` / `nmos_6p0_nat`。最小 L 为 0.28 µm(3.3 V)。

这是 180 nm 3.3 V/6 V 工艺,不是 GF 65 nm。65LPe 模型只能在 NDA 下获取;拿到后换掉本目录的两个文件和 `experiments/spice/gf180.py` 里的模型名即可重跑全部测试台。
