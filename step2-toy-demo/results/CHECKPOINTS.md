# 检查点清单

这些是 PyTorch `state_dict`(`torch.save(model.state_dict())`),用 torch 2.14 保存。路径相对于本目录 `step2-toy-demo/results/`。其余检查点(转换扫描的中间运行等)在 2026-11-14 补入,列在文末一节。

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

三值残差检查点(`improve/ternres_*`)走同一个 `beam_eval.load_model`,`info['kind']` 是 `ternres`(另带 `info['merge']`:sat 或 and)。

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
| `improve/ternres_sat_T1_a1/model.pt` | 三值残差 sat T=1 A1,种子 0 | sat 规则,T=1,token 准确率 0.568,demo 1/5 | 4.4 MiB | `c4a6935417516c62` |
| `improve/ternres_sat_T1_c/model.pt` | 三值残差 sat T=1 C,种子 0 | sat 规则,T=1,token 准确率 0.625,demo 0/5 | 4.4 MiB | `01974845635b6fa7` |
| `improve/ternres_and_T1_a1/model.pt` | 三值残差 and T=1 A1,种子 0 | and 规则,T=1,token 准确率 0.451,demo 0/5 | 4.4 MiB | `9607a573317bd4fe` |
| `improve/ternres_and_T1_c/model.pt` | 三值残差 and T=1 C,种子 0 | and 规则,T=1,token 准确率 0.429,demo 0/5 | 4.4 MiB | `47ae1edc57690442` |
| `improve/ternres_sat_T2_a1/model.pt` | 三值残差 sat T=2 A1,种子 0 | sat 规则,T=2,token 准确率 0.804,demo 5/5 | 4.4 MiB | `45cf93006434834d` |
| `improve/ternres_sat_T2_c/model.pt` | 三值残差 sat T=2 C,种子 0 | sat 规则,T=2,token 准确率 0.811,demo 5/5 | 4.4 MiB | `1f2e7bfcff0ecef1` |
| `improve/ternres_and_T2_a1/model.pt` | 三值残差 and T=2 A1,种子 0 | and 规则,T=2,token 准确率 0.788,demo 3/5 | 4.4 MiB | `a988950de9a855b8` |
| `improve/ternres_and_T2_c/model.pt` | 三值残差 and T=2 C,种子 0 | and 规则,T=2,token 准确率 0.769,demo 5/5 | 4.4 MiB | `b346ef2a21c1c1c6` |
| `improve/ternres_sat_T2_s1_a1/model.pt` | 三值残差 sat T=2 A1,种子 1 | sat 规则,T=2,token 准确率 0.818,demo 5/5 | 4.4 MiB | `072460bb73e8154b` |
| `improve/ternres_sat_T2_s1_c/model.pt` | 三值残差 sat T=2 C,种子 1 | sat 规则,T=2,token 准确率 0.814,demo 5/5 | 4.4 MiB | `693b029842aac6bc` |
| `improve/ternres_sat_T2_s2_a1/model.pt` | 三值残差 sat T=2 A1,种子 2 | sat 规则,T=2,token 准确率 0.810,demo 5/5 | 4.4 MiB | `ea19a22f192186e5` |
| `improve/ternres_sat_T2_s2_c/model.pt` | 三值残差 sat T=2 C,种子 2 | sat 规则,T=2,token 准确率 0.805,demo 5/5 | 4.4 MiB | `003991daccd97612` |
| `improve/ternres_sat_T1_s1_a1/model.pt` | 三值残差 sat T=1 A1,种子 1 | sat 规则,T=1,token 准确率 0.566,demo 0/5 | 4.4 MiB | `da50c86891bc3ee7` |
| `improve/ternres_sat_T1_s1_c/model.pt` | 三值残差 sat T=1 C,种子 1 | sat 规则,T=1,token 准确率 0.624,demo 0/5 | 4.4 MiB | `f873f416a8813133` |
| `improve/ternres_sat_T1_s2_a1/model.pt` | 三值残差 sat T=1 A1,种子 2 | sat 规则,T=1,token 准确率 0.566,demo 0/5 | 4.4 MiB | `29d5e66af0fa7945` |
| `improve/ternres_sat_T1_s2_c/model.pt` | 三值残差 sat T=1 C,种子 2 | sat 规则,T=1,token 准确率 0.626,demo 0/5 | 4.4 MiB | `9f492efd3ba56c8f` |

合计 37 个文件,184.4 MiB(前面各方向的 21 个 113.6 MiB,三值残差的 16 个 70.8 MiB)。sha256 只列前 16 位。

## 2026-11-14 补入的其余检查点

按“GitHub 保持完整”的要求,下面这些此前不进版本库的检查点也入库了(转换扫描与诊断的中间运行、各改进方向的 A1、默认配置首跑的 A1 / C)。格式同上;有 `result.json` 的走 `beam_eval.load_model`,没有的按“加载”一列。`conversion-sweep/_smoke/`(冒烟测试)与 `tiny/` 仍不进。

| 文件 | 模型 | 说明 | 加载 | 大小 | sha256 |
|---|---|---|---|---|---|
| `ckpt/a1.pt` | A1(leak 0),T=1,9 月 24 日默认配置首跑 | token 准确率 0.511 | `load_model`(按文件名识别) | 4.4 MiB | `785381c2fa9bd928` |
| `ckpt/c.pt` | C(leak 0.9),T=1,9 月 24 日默认配置首跑 | token 准确率 0.407 | `load_model`(按文件名识别) | 4.4 MiB | `96c87fe36343461f` |
| `conversion-sweep/C1_base4k/model.pt` | C(leak 0.9),T=1 | token 准确率 0.516,demo 0/5 | `load_model` | 4.4 MiB | `67dbf27cc629596a` |
| `conversion-sweep/C2_thr15/model.pt` | C(leak 0.9),T=1 | token 准确率 0.540,demo 0/5 | `load_model` | 4.4 MiB | `9e8c3a6829f366cb` |
| `conversion-sweep/C3_anneal1000/model.pt` | C(leak 0.9),T=1 | token 准确率 0.539,demo 0/5 | `load_model` | 4.4 MiB | `83b7b401ec6fae93` |
| `conversion-sweep/C4_lr3e3/model.pt` | C(leak 0.9),T=1 | token 准确率 0.637,demo 0/5 | `load_model` | 4.4 MiB | `11f941f8850f26a1` |
| `conversion-sweep/C6_lif25/model.pt` | C(leak 0.9),T=1 | token 准确率 0.549,demo 0/5 | `load_model` | 4.4 MiB | `6b1672e0405b7562` |
| `conversion-sweep/C7_lif25_lr3e3/model.pt` | C(leak 0.9),T=1 | token 准确率 0.663,demo 1/5 | `load_model` | 4.4 MiB | `95a1504630520db7` |
| `conversion-sweep/CF1_pk4w_lif225_lr3e3/model.pt` | C(leak 0.9),T=1 | token 准确率 0.649,demo 0/5 | `load_model` | 4.4 MiB | `71bca1dd11fc8d98` |
| `conversion-sweep/CF2_pk4w_lif225_lr5e3_w300/model.pt` | C(leak 0.9),T=1 | token 准确率 0.639,demo 0/5 | `load_model` | 4.4 MiB | `77ba5bf285a0111c` |
| `conversion-sweep/CF3_s1_lif225_lr3e3/model.pt` | C(leak 0.9),T=1 | token 准确率 0.653,demo 0/5 | `load_model` | 4.4 MiB | `852c735fb51b03e1` |
| `conversion-sweep/CF4_pk4w_lif225_lr3e3_anneal1000/model.pt` | C(leak 0.9),T=1 | token 准确率 0.624,demo 0/5 | `load_model` | 4.4 MiB | `491f85a3ee7615f8` |
| `conversion-sweep/CL1_l1_lif225_lr3e3/model.pt` | C(leak 0.9),T=1 | token 准确率 0.682,demo 1/5 | `load_model` | 4.4 MiB | `6b8e011f5beec706` |
| `conversion-sweep/L1_fq50_12k/model.pt` | A1(leak 0),T=1 | token 准确率 0.740,demo 1/5 | `load_model` | 4.4 MiB | `b300001b2d9fba16` |
| `conversion-sweep/L2_fq50_lr5e3_w300_12k/model.pt` | A1(leak 0),T=1 | token 准确率 0.746,demo 1/5 | `load_model` | 4.4 MiB | `c39ab02e1919ed0f` |
| `conversion-sweep/PK3_dense/model.pt` | A1(leak 0),T=1 | token 准确率 0.671,demo 1/5 | `load_model` | 4.4 MiB | `b290a44a40899ee7` |
| `conversion-sweep/PK4_kinds/model.pt` | A1(leak 0),T=1 | token 准确率 0.671,demo 2/5 | `load_model` | 4.4 MiB | `8922b2e634e83fba` |
| `conversion-sweep/PK4_lr5e3_w300/model.pt` | A1(leak 0),T=1 | token 准确率 0.701,demo 0/5 | `load_model` | 4.4 MiB | `4b99b2d78169a340` |
| `conversion-sweep/R1_base6k/model.pt` | A1(leak 0),T=1 | token 准确率 0.659,demo 0/5 | `load_model` | 4.4 MiB | `5049542c6336d35a` |
| `conversion-sweep/R2_fq50/model.pt` | A1(leak 0),T=1 | token 准确率 0.677,demo 0/5 | `load_model` | 4.4 MiB | `6707e1595f4184e7` |
| `conversion-sweep/R3_lr1e3/model.pt` | A1(leak 0),T=1 | token 准确率 0.538,demo 0/5 | `load_model` | 4.4 MiB | `08cfd6ccbe8428b2` |
| `conversion-sweep/R4_kd/model.pt` | A1(leak 0),T=1 | token 准确率 0.637,demo 1/5 | `load_model` | 4.4 MiB | `5e0556c623c023ae` |
| `conversion-sweep/S1_fq50_lr5e3_w300/model.pt` | A1(leak 0),T=1 | token 准确率 0.698,demo 1/5 | `load_model` | 4.4 MiB | `f79d23481e9aa295` |
| `conversion-sweep/S2_fq50_kd03_T1/model.pt` | A1(leak 0),T=1 | token 准确率 0.676,demo 2/5 | `load_model` | 4.4 MiB | `b8625a03f6d8582c` |
| `conversion-sweep/S3_fq50_w300/model.pt` | A1(leak 0),T=1 | token 准确率 0.674,demo 1/5 | `load_model` | 4.4 MiB | `be626b8da2894611` |
| `conversion-sweep/S4_fq40/model.pt` | A1(leak 0),T=1 | token 准确率 0.678,demo 1/5 | `load_model` | 4.4 MiB | `644572fbd60fdbd4` |
| `conversion-sweep/S5_fq60/model.pt` | A1(leak 0),T=1 | token 准确率 0.670,demo 0/5 | `load_model` | 4.4 MiB | `2e1df25637a48efa` |
| `conversion-sweep/S6_fq75_lr5e3/model.pt` | A1(leak 0),T=1 | token 准确率 0.677,demo 1/5 | `load_model` | 4.4 MiB | `f6f5584569a861a6` |
| `conversion-sweep/diag/b-recalibration/base_c/model.pt` | C(leak 0.9),T=1 | token 准确率 0.272,demo 0/5 | `load_model` | 4.4 MiB | `9f1be78cffe7ef7a` |
| `conversion-sweep/diag/b-recalibration/base_c_anneal150/model.pt` | C(leak 0.9),T=1 | token 准确率 0.233,demo 0/5 | `load_model` | 4.4 MiB | `641245becf608e64` |
| `conversion-sweep/diag/b-recalibration/init_lif2.pt` | B 再标定诊断用的 C 初始化,T=1 | 见 `conversion-sweep/README.md` 的 B 再标定 | `SpikingLM` 默认配置 + `load_state_dict` | 4.4 MiB | `7dd27204004f649d` |
| `conversion-sweep/diag/b-recalibration/init_uni15.pt` | B 再标定诊断用的 C 初始化,T=1 | 见 `conversion-sweep/README.md` 的 B 再标定 | `SpikingLM` 默认配置 + `load_state_dict` | 4.4 MiB | `fd9ae00a5b85bc48` |
| `conversion-sweep/diag/b-recalibration/lif2_c/model.pt` | C(leak 0.9),T=1 | token 准确率 0.338,demo 0/5 | `load_model` | 4.4 MiB | `ed1e879009ad0d5a` |
| `conversion-sweep/diag/b-recalibration/lif2_c_anneal150/model.pt` | C(leak 0.9),T=1 | token 准确率 0.287,demo 0/5 | `load_model` | 4.4 MiB | `7728cd362bc9a7a1` |
| `conversion-sweep/diag/b-recalibration/uni15_c/model.pt` | C(leak 0.9),T=1 | token 准确率 0.314,demo 0/5 | `load_model` | 4.4 MiB | `9c91d2c794ddd75e` |
| `conversion-sweep/diag/b-recalibration/uni15_c_anneal150/model.pt` | C(leak 0.9),T=1 | token 准确率 0.273,demo 0/5 | `load_model` | 4.4 MiB | `de79ad29e727fbc0` |
| `conversion-sweep/diag/fq-scan/fq050_plain/model.pt` | A1(leak 0),T=1 | token 准确率 0.348,demo 0/5 | `load_model` | 4.4 MiB | `9c7d32c27a480b38` |
| `conversion-sweep/diag/fq-scan/fq050_refit/model.pt` | A1(leak 0),T=1 | token 准确率 0.283,demo 0/5 | `load_model` | 4.4 MiB | `409d7ffbaaaade77` |
| `conversion-sweep/diag/fq-scan/fq060_plain/model.pt` | A1(leak 0),T=1 | token 准确率 0.340,demo 0/5 | `load_model` | 4.4 MiB | `e7dd46f68b198eed` |
| `conversion-sweep/diag/fq-scan/fq075_plain/model.pt` | A1(leak 0),T=1 | token 准确率 0.330,demo 0/5 | `load_model` | 4.4 MiB | `9e244ea1e84251e4` |
| `conversion-sweep/diag/fq-scan/fq075_refit/model.pt` | A1(leak 0),T=1 | token 准确率 0.245,demo 0/5 | `load_model` | 4.4 MiB | `2fdb88a2af2543d2` |
| `improve/LT_a1_24k/model.pt` | A1(leak 0),T=1 | token 准确率 0.782,demo 2/5 | `load_model` | 4.4 MiB | `cd3013251f0b4b49` |
| `improve/T4mem_a1/model.pt` | A1(leak 0),T=4(只在 mem) | token 准确率 0.776,demo 2/5 | `load_model` | 4.4 MiB | `034a0d9f3e5fdd4c` |
| `improve/T4mem_c/model.pt` | C(leak 0.9),T=4(只在 mem) | token 准确率 0.783,demo 3/5 | `load_model` | 4.4 MiB | `5b6ad27e7b5d1178` |
| `improve/T8_a1/model.pt` | A1(leak 0),T=8 | token 准确率 0.836,demo 5/5 | `load_model` | 4.4 MiB | `cbacfe2027abceff` |
| `improve/WT4_a1/model.pt` | A1(leak 0),T=4,d=256 | token 准确率 0.835,demo 5/5 | `load_model` | 11.3 MiB | `ba8c8f4ff79081aa` |
| `improve/W_a1/model.pt` | A1(leak 0),T=1,d=256 | token 准确率 0.831,demo 5/5 | `load_model` | 11.3 MiB | `24305826c590698c` |

本节 47 个文件,221.7 MiB。

## 二值残差流与对照(2026-11-15)

二值(0/1)残差流实验(`improve/README.md` 的“二值残差流”一节、设计稿 `docs/superpowers/specs/2026-11-14-binary-residual-design.md`)的全部检查点,含三值对照(sat T=4;satc = 三值 + 精确梯度合并)。二值模型走 `beam_eval.load_model`,`info['kind']` 是 `binres`(`merge` 为 sr / src / xor);satc 对照是 `ternres` + `merge='satc'`,装载成 `TernaryClampResidualLM`。队列脚本在每条链结束时已把它们提交进版本库,这里补清单。

| 文件 | 模型 | 说明 | 大小 | sha256 |
|---|---|---|---|---|
| `improve/binres_sr_T1_a1/model.pt` | 二值残差,sr,T=1,A1(leak 0) | token 准确率 0.096,demo 0/5 | 4.4 MiB | `5b52576915eb01f8` |
| `improve/binres_sr_T1_c/model.pt` | 二值残差,sr,T=1,C(leak 0.9) | token 准确率 0.264,demo 0/5 | 4.4 MiB | `54d937579de9795d` |
| `improve/binres_sr_T2_a1/model.pt` | 二值残差,sr,T=2,A1(leak 0) | token 准确率 0.666,demo 0/5 | 4.4 MiB | `0efff81853b2273e` |
| `improve/binres_sr_T2_c/model.pt` | 二值残差,sr,T=2,C(leak 0.9) | token 准确率 0.668,demo 0/5 | 4.4 MiB | `1fb41a78e0070dd7` |
| `improve/binres_sr_T4_a1/model.pt` | 二值残差,sr,T=4,A1(leak 0) | token 准确率 0.827,demo 5/5 | 4.4 MiB | `503f93c9f680b4d9` |
| `improve/binres_sr_T4_c/model.pt` | 二值残差,sr,T=4,C(leak 0.9) | token 准确率 0.816,demo 5/5 | 4.4 MiB | `e7ea28af4e823d81` |
| `improve/binres_sr_T8_a1/model.pt` | 二值残差,sr,T=8,A1(leak 0) | token 准确率 0.839,demo 5/5 | 4.4 MiB | `b6a9085255655968` |
| `improve/binres_sr_T8_c/model.pt` | 二值残差,sr,T=8,C(leak 0.9) | token 准确率 0.828,demo 5/5 | 4.4 MiB | `1c770d45e07fc622` |
| `improve/binres_src_T1_W_a1/model.pt` | 二值残差,src,T=1,d=256,A1(leak 0) | token 准确率 0.679,demo 0/5 | 11.3 MiB | `7c8ad1ca3832eeb1` |
| `improve/binres_src_T1_W_c/model.pt` | 二值残差,src,T=1,d=256,C(leak 0.9) | token 准确率 0.492,demo 0/5 | 11.3 MiB | `d6313a0c9b5fa8b0` |
| `improve/binres_src_T1_a1/model.pt` | 二值残差,src,T=1,A1(leak 0) | token 准确率 0.490,demo 0/5 | 4.4 MiB | `e843c48c20ee61dd` |
| `improve/binres_src_T1_c/model.pt` | 二值残差,src,T=1,C(leak 0.9) | token 准确率 0.500,demo 0/5 | 4.4 MiB | `f374fb9c927e34ac` |
| `improve/binres_src_T2_W_a1/model.pt` | 二值残差,src,T=2,d=256,A1(leak 0) | token 准确率 0.833,demo 5/5 | 11.3 MiB | `31716847bf3a4fda` |
| `improve/binres_src_T2_W_c/model.pt` | 二值残差,src,T=2,d=256,C(leak 0.9) | token 准确率 0.817,demo 5/5 | 11.3 MiB | `0ff68ce22d3d021a` |
| `improve/binres_src_T2_a1/model.pt` | 二值残差,src,T=2,A1(leak 0) | token 准确率 0.715,demo 0/5 | 4.4 MiB | `53227af2410f9fe5` |
| `improve/binres_src_T2_c/model.pt` | 二值残差,src,T=2,C(leak 0.9) | token 准确率 0.713,demo 1/5 | 4.4 MiB | `ce5837c107b9f64a` |
| `improve/binres_src_T2_s1_a1/model.pt` | 二值残差,src,T=2,种子 1,A1(leak 0) | token 准确率 0.717,demo 1/5 | 4.4 MiB | `89c0c87264b01588` |
| `improve/binres_src_T2_s1_c/model.pt` | 二值残差,src,T=2,种子 1,C(leak 0.9) | token 准确率 0.714,demo 2/5 | 4.4 MiB | `96c4fd27659fea24` |
| `improve/binres_src_T2_s2_a1/model.pt` | 二值残差,src,T=2,种子 2,A1(leak 0) | token 准确率 0.720,demo 1/5 | 4.4 MiB | `72f05b65e54ab074` |
| `improve/binres_src_T2_s2_c/model.pt` | 二值残差,src,T=2,种子 2,C(leak 0.9) | token 准确率 0.714,demo 1/5 | 4.4 MiB | `a90cc48ddf28aed5` |
| `improve/binres_src_T4_W_a1/model.pt` | 二值残差,src,T=4,d=256,A1(leak 0) | token 准确率 0.834,demo 5/5 | 11.3 MiB | `48372e9f6569a028` |
| `improve/binres_src_T4_W_c/model.pt` | 二值残差,src,T=4,d=256,C(leak 0.9) | token 准确率 0.828,demo 5/5 | 11.3 MiB | `cf0b371d0a8816b0` |
| `improve/binres_src_T4_a1/model.pt` | 二值残差,src,T=4,A1(leak 0) | token 准确率 0.831,demo 5/5 | 4.4 MiB | `902f780a3068aa8f` |
| `improve/binres_src_T4_c/model.pt` | 二值残差,src,T=4,C(leak 0.9) | token 准确率 0.813,demo 5/5 | 4.4 MiB | `9b7e78b0c2ebd74f` |
| `improve/binres_src_T4_s1_a1/model.pt` | 二值残差,src,T=4,种子 1,A1(leak 0) | token 准确率 0.831,demo 5/5 | 4.4 MiB | `6cfe4cfca57d91e8` |
| `improve/binres_src_T4_s1_c/model.pt` | 二值残差,src,T=4,种子 1,C(leak 0.9) | token 准确率 0.820,demo 5/5 | 4.4 MiB | `82b130d0319a046d` |
| `improve/binres_src_T4_s2_a1/model.pt` | 二值残差,src,T=4,种子 2,A1(leak 0) | token 准确率 0.827,demo 5/5 | 4.4 MiB | `e0b68beaa238964e` |
| `improve/binres_src_T4_s2_c/model.pt` | 二值残差,src,T=4,种子 2,C(leak 0.9) | token 准确率 0.821,demo 5/5 | 4.4 MiB | `f2b27c63e7030012` |
| `improve/binres_src_T8_a1/model.pt` | 二值残差,src,T=8,A1(leak 0) | token 准确率 0.835,demo 5/5 | 4.4 MiB | `9b4c62f5af23d460` |
| `improve/binres_src_T8_c/model.pt` | 二值残差,src,T=8,C(leak 0.9) | token 准确率 0.830,demo 5/5 | 4.4 MiB | `0059f75319a463d6` |
| `improve/binres_src_T8_s1_a1/model.pt` | 二值残差,src,T=8,种子 1,A1(leak 0) | token 准确率 0.838,demo 5/5 | 4.4 MiB | `d277398563a795c6` |
| `improve/binres_src_T8_s1_c/model.pt` | 二值残差,src,T=8,种子 1,C(leak 0.9) | token 准确率 0.835,demo 5/5 | 4.4 MiB | `9e23d36fcb49e37c` |
| `improve/binres_src_T8_s2_a1/model.pt` | 二值残差,src,T=8,种子 2,A1(leak 0) | token 准确率 0.830,demo 5/5 | 4.4 MiB | `7bfd8b095ebb7df5` |
| `improve/binres_src_T8_s2_c/model.pt` | 二值残差,src,T=8,种子 2,C(leak 0.9) | token 准确率 0.835,demo 5/5 | 4.4 MiB | `fccd4b437e5bd9ca` |
| `improve/binres_xor_T1_a1/model.pt` | 二值残差,xor,T=1,A1(leak 0) | token 准确率 0.557,demo 0/5 | 4.4 MiB | `43dab3f2565e3404` |
| `improve/binres_xor_T1_c/model.pt` | 二值残差,xor,T=1,C(leak 0.9) | token 准确率 0.533,demo 0/5 | 4.4 MiB | `12ceac86e657177d` |
| `improve/binres_xor_T2_a1/model.pt` | 二值残差,xor,T=2,A1(leak 0) | token 准确率 0.727,demo 3/5 | 4.4 MiB | `aed0b2ee2320b089` |
| `improve/binres_xor_T2_c/model.pt` | 二值残差,xor,T=2,C(leak 0.9) | token 准确率 0.699,demo 1/5 | 4.4 MiB | `496209498a057d55` |
| `improve/binres_xor_T4_a1/model.pt` | 二值残差,xor,T=4,A1(leak 0) | token 准确率 0.800,demo 1/5 | 4.4 MiB | `af5dbd99743b3953` |
| `improve/binres_xor_T4_c/model.pt` | 二值残差,xor,T=4,C(leak 0.9) | token 准确率 0.814,demo 5/5 | 4.4 MiB | `f29e1e609e404fd0` |
| `improve/binres_xor_T8_a1/model.pt` | 二值残差,xor,T=8,A1(leak 0) | token 准确率 0.812,demo 5/5 | 4.4 MiB | `6c773c6ef9532dd6` |
| `improve/binres_xor_T8_c/model.pt` | 二值残差,xor,T=8,C(leak 0.9) | token 准确率 0.832,demo 5/5 | 4.4 MiB | `be8fc50c4f65e4b4` |
| `improve/ternres_sat_T4_a1/model.pt` | 三值残差,sat,T=4,A1(leak 0) | token 准确率 0.832,demo 5/5 | 4.4 MiB | `7f146987ee921e92` |
| `improve/ternres_sat_T4_c/model.pt` | 三值残差,sat,T=4,C(leak 0.9) | token 准确率 0.826,demo 5/5 | 4.4 MiB | `b9c63d437dcd1109` |
| `improve/ternres_satc_T1_a1/model.pt` | 三值残差,satc,T=1,A1(leak 0) | token 准确率 0.661,demo 0/5 | 4.4 MiB | `a52b52f7d62614b1` |
| `improve/ternres_satc_T1_c/model.pt` | 三值残差,satc,T=1,C(leak 0.9) | token 准确率 0.702,demo 0/5 | 4.4 MiB | `9b37b278b7a669c1` |
| `improve/ternres_satc_T2_a1/model.pt` | 三值残差,satc,T=2,A1(leak 0) | token 准确率 0.815,demo 5/5 | 4.4 MiB | `d0ba7dc96f8c79ba` |
| `improve/ternres_satc_T2_c/model.pt` | 三值残差,satc,T=2,C(leak 0.9) | token 准确率 0.816,demo 5/5 | 4.4 MiB | `4e7106d36d7ce489` |
| `improve/ternres_satc_T4_a1/model.pt` | 三值残差,satc,T=4,A1(leak 0) | token 准确率 0.831,demo 5/5 | 4.4 MiB | `44be7a476568bc64` |
| `improve/ternres_satc_T4_c/model.pt` | 三值残差,satc,T=4,C(leak 0.9) | token 准确率 0.830,demo 5/5 | 4.4 MiB | `b6e6c76dda3f929e` |

本节 50 个文件,262.6 MiB。
