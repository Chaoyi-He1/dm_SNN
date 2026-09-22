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

## 本次结果

(流水线运行后由 Task 11 填入。)
