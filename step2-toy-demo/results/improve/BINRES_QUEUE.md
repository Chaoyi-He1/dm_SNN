# 二值(0/1)残差流实验:无人值守队列与交接说明(2026-11-14)

**2026-11-15 11:54 全部跑完**(25 条链,状态文件末尾 `ALL DONE`,每条链的结果与评估都已由队列提交推送)。
结论写在 `README.md` 的"二值残差流(2026-11-15)"一节;下面是当时留给队列的说明,保留作记录。

问题:把残差流从三值(−1 / 0 / +1)改成二值(0 / 1),要付出什么代价才能让模型的性能和浮点 A0 差不多。
设计见 `docs/superpowers/specs/2026-11-14-binary-residual-design.md`。这份说明写在 Claude 会话关闭之前:
训练、评估、汇总和推送都交给服务器上的队列脚本,结论部分(README)等结果出来后再写。

## 现在在跑什么

`step2-toy-demo/run_binres_queue.sh` 在后台循环(由 `start_binres_queue.sh` 启动,脱离终端,只用 GPU 1):

1. 并行的链少于 9 条时,按顺序启动队列里的下一条(每条链 = iscan → A1 → bscan → C,约 4–5 小时;加宽的更久);
2. 每条链的 C 跑完后做整句背对率评估(贪心与 K=4),写 `beam/<名字>.json`;
3. 有新结果或每隔 1 小时,重新生成 `binres_tables.json` / `binres_tables.md`,把 `results/improve` 下的新文件提交并推送到
   分支 `worktree-conversion-sweep` 和 `main`(只快进,失败只记日志)。

| 批次 | 链 | 用途 |
|---|---|---|
| 第一批(2026-11-14 晚启动) | `binres_sr_T1/2/4/8`、`binres_xor_T1/2`、`ternres_sat_T4`、`binres_src_T1/2` | sr / xor 两种合并规则;src = sr 的前向 + 精确梯度;三值 sat T=4 是与二值 T=8 电平数相同的对照 |
| 队列 1 | `binres_src_T4`、`binres_src_T8` | src 的 T 扫描补齐 |
| 队列 2 | `ternres_satc_T2`、`ternres_satc_T1`、`ternres_satc_T4` | 三值残差 + 同样的精确梯度(对照:让"二值对三值"只差残差流的取值) |
| 队列 3 | `binres_src_T1_W`、`binres_src_T2_W`、`binres_src_T4_W` | 加宽到 d=256(连线数与三值 d=128 相同) |
| 队列 4 | `binres_src_T2_s1/_s2`、`binres_src_T4_s1/_s2`、`binres_src_T8_s1/_s2` | 种子 1、2 |
| 队列 5 | `binres_xor_T4`、`binres_xor_T8` | xor 的 T 扫描 |

全部跑完预计在启动后 14–18 小时(第一批约 4.5 小时,之后 16 条分两轮)。

## 怎么看进度、怎么停

```
cd /data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo
cat results/improve/binres_queue_status.txt        # 每条链:queued / running / done(带 C 的 token 准确率)/ STOPPED
tail results/improve/binres_queue.log              # 队列自己的日志(启动、评估、推送)
cat results/improve/binres_tables.md               # 汇总表(每次有新结果重新生成)
tail -2 results/improve/binres_src_T2_chain.log    # 某条链的训练日志
touch results/improve/binres_queue.stop            # 停队列:不再启动新链,收尾后退出;已在跑的链不受影响
./start_binres_queue.sh                            # 队列进程没了(比如机器重启)时重新启动;已完成和已启动的链不会重跑
```

状态文件最后一行出现 `ALL DONE` 就是全部结束。GitHub 上同样能看到:`step2-toy-demo/results/improve/binres_queue_status.txt` 与 `binres_tables.md`。

一条链启动过(有 `<名字>_chain.log`)就不会再启动,即使它中途失败(状态写 STOPPED);要重跑先删掉那个日志再启动队列。
机器重启后正在跑的链会丢,需要删掉它们的 `_chain.log`(以及半成品目录)再启动队列。

## 结果出来以后还要做的(需要人或 Claude)

1. `results/improve/README.md` 新增"二值残差流"一节:按 T 和宽度列准确率 / 整句背对率 / 脉冲数,回答"达到判据(C 的 token 准确率 ≥ 0.832、
   demo 稳定 5/5)最便宜的配置是什么,比实值残差、三值残差各多花多少";sr 对 src 的差是"代理梯度位置"的影响,不算二值本身的代价。
2. 对达到判据或最接近的配置跑 `experiments/energy_estimate.py`(GPU 1,几分钟),估每 token 能耗。
3. `results/CHECKPOINTS.md` 补上新检查点的行。
4. 需要的话更新报告页(GDN 脉冲转换报告)。

## 目前已知的(2026-11-14 晚,A1 训练早期)

- 二值比三值难训得多:1500 步左右时 sr 的 token 准确率 T=1/2/4/8 是 0.30 / 0.44 / 0.59 / 0.70,三值 sat T=4 是 0.78。
- sr 在 T=1 上停滞(2100 步 0.265),xor 在涨(0.400):sr 的代理梯度中心在 r + y = 1,r = 0 的通道恒等通路梯度只有 0.092,
  这正是加 src(精确梯度)的原因。src 当时刚起步,还看不出差别。
- iscan 选 fq 时二值模型的初始准确率接近 0(0.0005–0.016),这个选择基本是任意的。
- 约 40 分钟后的同步数对比(A1 阶段的 token 准确率):

  | 规则 | T=1,2000 步 | T=2,1500 步 |
  |---|---|---|
  | sr(代理梯度) | 0.272,之后继续下降(3700 步 0.202) | 0.443 |
  | src(精确梯度) | 0.376 | 0.462 |
  | xor | 0.397 | 0.420 |

  精确梯度在 T=1 上的作用很明显:sr 不但不涨还在掉,src 和 xor 都在稳步上升。T=2 上三者接近,src 略高。
  同一时刻 sr 的 T=4 是 0.704(2800 步)、T=8 是 0.800(2600 步,demo 3/5),三值 sat T=4 是 0.817(2600 步,demo 5/5)。
  这些都是训练早期的数,最终结果以 `binres_tables.md` 为准。
