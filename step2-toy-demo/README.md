# Step 2:玩具模型 demo

设计与验收标准见上级目录 `toy-demo-plan.md`。结果页 `results/toy-demo.md`,原始数字 `results/toy-demo.json`,检查点 `results/ckpt/`。

## 运行

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt        # torch + 可编辑安装 ../step1-spiking-layer
.venv/bin/python -m pytest -q                              # 全部离线、秒级
.venv/bin/python -m toy_demo.run --tiny                    # 合成切片上几秒跑通流水线
.venv/bin/python -m toy_demo.run                           # 真实切片:A0 → A1 → B → C → 电路(CPU 数小时)
.venv/bin/python -m toy_demo.run --reuse                   # 复用检查点,只重跑评估与电路
.venv/bin/python -m toy_demo.demo                          # 5 个前缀在各阶段的补全 + 指标表
.venv/bin/python -m toy_demo.demo --prompt "Once upon a"   # 新前缀(需要 requirements-data.txt)
```

重建数据切片(需要网络):`.venv/bin/python -m pip install -r requirements-data.txt && .venv/bin/python -m toy_demo.build_slice`。切片说明见 `data/tinystories-slice/README.md`。

## 模块

`config.py` 配置;`data.py` 切片与批处理;`build_slice.py` 一次性构建;`model.py` A0 浮点模型与 A1/B/C 脉冲模型;`convert.py` 权重转移、阈值标定、leak 切换;`train.py` 训练与统一评估;`circuit.py` 多块电路模型与 teacher forcing 对照;`report.py` 结果页;`run.py` 流水线;`demo.py` demo 命令。

`experiments/` 里是流水线之外的实验:`convert_lab.py` 转换实验台,`multilevel.py` 每 token 多个脉冲的脉冲点,`beam_eval.py` 束搜索,`circuit_levels.py` 任意 T 的电路验收、误差隔离、容限扫描与诊断(`circuit_tables.py` 汇总),其余见各文件开头的说明。

## 本次结果

- **默认配置流水线(2026-09-24,`results/toy-demo-default-run.json`)**:A0 acc 0.842、5/5;A1 0.511、C 0.407,三个种子都失败,补全 0/5;理想电路与 C 逐脉冲一致;标称电路不通过。验收 3 项通过、5 项不通过。
- **转换扫描(2026-09-26,`results/conversion-sweep/README.md`、`diagnostics.md`)**:诊断确认不是代码错误,而是第 4 节的 25% 发放工作点;放宽步数、提高学习率、分位数 0.5、C 训练前 LIF 阈值 ×2.25 后,A1 最好 0.746、C 最好 0.752("C 在 A1 的 1 个百分点内"通过),仍未达到 5/5。主结果页 `results/conversion-v2-fq50-12k/`(A1 0.740 → C 0.752 → 电路),对照页 `results/conversion-v2-fq50/`。
- **每 token 多个脉冲与电路验收(2026-10-15 至 19,`results/improve/README.md`)**:每个脉冲点每 token 最多 2 个脉冲时 C 为 0.825 到 0.830,3 个种子都 5/5;T = 4 为 0.832 到 0.836。理想电路对 T = 2、T = 4 与软件逐位相同;标称电路的逐脉冲判据所有模型都不通过,自由运行时(5 个噪声种子平均)比软件低 T = 1 11.7、T = 2 4.8 到 6.5、T = 4 1.2 到 1.4 个百分点;T = 1 与 T = 2 最大的单项误差来源是比较器失调,T = 4 各项都不到 1 个百分点,电导量化略大于失调。
- `python -m toy_demo.run` 与 `config.py` 未改,仍是计划默认值;采纳配方经 `experiments/convert_lab.py` 与 `experiments/assemble.py` 复现(见扫描 README 的"复现")。
