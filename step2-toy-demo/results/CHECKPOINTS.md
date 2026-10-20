# 检查点清单

这些是 PyTorch `state_dict`(`torch.save(model.state_dict())`),用 torch 2.14 保存。路径相对于本目录 `step2-toy-demo/results/`。其余检查点(转换扫描的中间运行等)仍不进版本库。

加载:脉冲模型的类型、每 token 最多几个脉冲(levels)、模型尺寸和 leak 都由同目录的 `result.json` 决定,用现成的装载函数即可:

```python
# 在 step2-toy-demo 下
import sys, torch; sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from beam_eval import load_model
from toy_demo.data import load_slice
sl = load_slice('data/tinystories-slice')
model, cfg, info = load_model('results/improve/T2_c/model.pt', sl, torch.device('cpu'))
print(info)            # {'kind': 'spiking', 'leak': 0.9, 'levels': 2, ...}
print(model.generate(sl.ids[0][:5], 40))
```

`ckpt/a0.pt` 没有 `result.json`,按文件名识别为浮点模型。数字含义见 `improve/README.md`。

| 文件 | 模型 | 说明 | 大小 | sha256 |
|---|---|---|---|---|
| `ckpt/a0.pt` | 浮点 A0 | 原宽度 d=128,116 万参数,token 准确率 0.842 | 4.4 MiB | `dababbfe97fa5e92` |
| `improve/W_a0_d256/model.pt` | 浮点 A0,加宽 | d=256,297 万参数,0.840 | 11.3 MiB | `9d848a9d0a5bdb56` |
| `conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt` | 原来的 C | T=1,A1 12k → C 12k,0.752 | 4.4 MiB | `ea2b0710bbb3b1bf` |
| `improve/T2_a1/model.pt` | T=2 A1,种子 0 | T=2,token 准确率 0.832,demo 5/5 | 4.4 MiB | `2425499734833472` |
| `improve/T2_c/model.pt` | T=2 C,种子 0 | T=2,token 准确率 0.825,demo 5/5 | 4.4 MiB | `fc1bbaf5f9a8b296` |
| `improve/T2_a1_s1/model.pt` | T=2 A1,种子 1 | T=2,token 准确率 0.816,demo 5/5 | 4.4 MiB | `62a12d1ae99d2dad` |
| `improve/T2_c_s1/model.pt` | T=2 C,种子 1 | T=2,token 准确率 0.825,demo 5/5 | 4.4 MiB | `7e8170038df44803` |
| `improve/T2_a1_s2/model.pt` | T=2 A1,种子 2 | T=2,token 准确率 0.828,demo 5/5 | 4.4 MiB | `ec6dd5666b53160a` |
| `improve/T2_c_s2/model.pt` | T=2 C,种子 2 | T=2,token 准确率 0.830,demo 5/5 | 4.4 MiB | `2f0f9dc4438b98c9` |
| `improve/T4_a1/model.pt` | T=4 A1,种子 0 | T=4,token 准确率 0.833,demo 5/5 | 4.4 MiB | `90d93b8e62b8fa1d` |
| `improve/T4_c/model.pt` | T=4 C,种子 0 | T=4,token 准确率 0.832,demo 5/5 | 4.4 MiB | `68d59a7cff5e14de` |
| `improve/T4_a1_s1/model.pt` | T=4 A1,种子 1 | T=4,token 准确率 0.833,demo 5/5 | 4.4 MiB | `c619434dca65f217` |
| `improve/T4_c_s1/model.pt` | T=4 C,种子 1 | T=4,token 准确率 0.835,demo 5/5 | 4.4 MiB | `66da46e2061547a3` |
| `improve/T4_a1_s2/model.pt` | T=4 A1,种子 2 | T=4,token 准确率 0.835,demo 5/5 | 4.4 MiB | `8ed7d0cafc70c741` |
| `improve/T4_c_s2/model.pt` | T=4 C,种子 2 | T=4,token 准确率 0.836,demo 5/5 | 4.4 MiB | `85da7f5fc2dc3166` |
| `improve/T8_c/model.pt` | T=8 C | T=8,token 准确率 0.836,demo 5/5 | 4.4 MiB | `460bd782b446943d` |
| `improve/W_c/model.pt` | 加宽 d=256,T=1 C | T=1,token 准确率 0.801,demo 3/5 | 11.3 MiB | `c50e3e5dc9bbc4e0` |
| `improve/WT4_c/model.pt` | 加宽 d=256,T=4 C | T=4,token 准确率 0.836,demo 5/5 | 11.3 MiB | `036aa569684c02d8` |
| `improve/LT_cont12k_fromCL2/model.pt` | 训练更久:在原来的 C 上续训 | T=1,token 准确率 0.757,demo 5/5 | 4.4 MiB | `f25454cd60420bbd` |
| `improve/LT_c24k_fromL1/model.pt` | 训练更久:A1 12k → C 24k | T=1,token 准确率 0.778,demo 5/5 | 4.4 MiB | `a86988bc042227f5` |
| `improve/LT_c24k_from24k/model.pt` | 训练更久:A1 24k → C 24k | T=1,token 准确率 0.764,demo 3/5 | 4.4 MiB | `998ac8db5a5dd33a` |

合计 21 个文件,113.6 MiB。sha256 只列前 16 位。
