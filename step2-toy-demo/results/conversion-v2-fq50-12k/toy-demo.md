# 玩具模型 demo 结果

设备 cuda,torch 2.14.0+cu126,词表 3226,句数 4003。

## 各阶段

| 阶段 | token 准确率 | 困惑度 | 补全正确 | 停止 | 种子 | 用时 s |
|---|---|---|---|---|---|---|
| A0 浮点 | 0.8420 | 1.794 | 5/5 |  |  | 1.4 |
| A1 阈值初始化(未微调) | 0.1229 | 328.496 | 0/5 |  |  | 5.9 |
| A1 可脉冲化 | 0.7404 | 2.631 | 1/5 | max_steps | 0 | 2.6 |
| B 折叠转换(零训练) | 0.1263 | 443.408 | 0/5 |  |  | 5.5 |
| C 微调 | 0.7519 | 2.577 | 1/5 | max_steps | 0 | 2.6 |
| 电路·理想 | 0.7522 | 2.575 | 1/5 |  |  | 24.0 |
| 电路·标称 | 0.6339 | 4.004 | 0/5 |  |  | 36.0 |

## 补全

### A0 浮点

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and watched it spin. | 是 |
| He couldn't wait any |  longer and ran to meet them. |  longer and ran to meet them. | 是 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find him anywhere. | 是 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use a wet cloth to wipe the gum away. | 是 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about how amazing it was. | 是 |

### A1 阈值初始化(未微调)

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  made hugged their whistle. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and okay in the kitty all - someone else put kitty was so fast!", very excited parade he left, Jane was very excitedMom knew with small pieces wanted barked arm bestrolledy voice, | 否 |
| She ran around the whole |  park, but she could not find him anywhere. | , and gentle rabbit in your grew heavier places. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  follow again close others some as if about everything has printing bloomelly said, but Bruce breath then Jimmy around one moment you have home: then home. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  house. have food after after afraid flying after after that made theirrepeat and interrupt theirely ready for his admire it.! he knew. | 否 |

### A1 可脉冲化

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and inside. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and ran to meet them. | 是 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find away. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use the warrior how to find and set it to play with her. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum. | 否 |

### B 折叠转换(零训练)

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  ran around and under the the animals in the little on the brown kayak. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer the the the door... too to to to to to any!! again. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  time that could and was that is that is so and she and said in the and shouted in and shouted and found and found and shouted in up up down on their work on the on the park the | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  the day to the the store and the store and her favorite band steps to her favorite band and and soon and she and soon........ | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  always always looked and and and wrapped towards it and it to help... | 否 |

### C 微调

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and watched as their snack. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and he could see what it. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find him anywhere. | 是 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use every day. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about them. | 否 |

### 电路·理想

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and watched as their snack. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and he could see what it. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find him anywhere. | 是 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use every day. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about them. | 否 |

### 电路·标称

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  flew into the dryer and ran off into the swing. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  more to see the best friends. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, and she had to her new things. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use with all day. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  little and enjoyed their game. | 否 |

## teacher forcing 对照:电路·理想

输出头 argmax 一致率 1.0000,3081 个位置,判定 通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 0 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |

## teacher forcing 对照:电路·标称

输出头 argmax 一致率 0.6248,3081 个位置,判定 不通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 0.745 | 0.746 | 0.752 | 0.750 | 0.0201 | 0.0082 | 0.0041 |
| 0 | ffn | 0.693 | 0.671 | 0.688 | 0.704 | 0.0718 | 0.0082 | 0.0041 |
| 1 | attn | 0.765 | 0.762 | 0.783 | 0.765 | 0.0418 | 0.0085 | 0.0045 |
| 1 | ffn | 0.726 | 0.702 | 0.728 | 0.725 | 0.1085 | 0.0085 | 0.0045 |

## 编程误差扫描

| σ_G | token 准确率 | 困惑度 | 补全正确 |
|---|---|---|---|
| 0.0 | 0.6358 | 3.986 | 0 |
| 0.02 | 0.6339 | 4.004 | 0 |
| 0.05 | 0.6327 | 4.075 | 0 |
| 0.1 | 0.6198 | 4.235 | 0 |
