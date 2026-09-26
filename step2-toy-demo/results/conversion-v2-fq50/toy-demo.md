# 玩具模型 demo 结果

设备 cuda,torch 2.14.0+cu126,词表 3226,句数 4003。

## 各阶段

| 阶段 | token 准确率 | 困惑度 | 补全正确 | 停止 | 种子 | 用时 s |
|---|---|---|---|---|---|---|
| A0 浮点 | 0.8420 | 1.794 | 5/5 |  |  | 1.3 |
| A1 阈值初始化(未微调) | 0.1229 | 328.496 | 0/5 |  |  | 6.2 |
| A1 可脉冲化 | 0.6772 | 3.384 | 0/5 | max_steps | 0 | 2.7 |
| B 折叠转换(零训练) | 0.1119 | 357.973 | 0/5 |  |  | 6.1 |
| C 微调 | 0.6630 | 3.607 | 1/5 | max_steps | 0 | 2.7 |
| 电路·理想 | 0.6633 | 3.603 | 1/5 |  |  | 25.4 |
| 电路·标称 | 0.6129 | 4.403 | 1/5 |  |  | 37.7 |

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
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  wanted to save it with his mom. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but it was always very independent. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use the trap finally the grass very hard to have. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum. | 否 |

### B 折叠转换(零训练)

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  all the the his on the little Lucy. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  as he got he he he listened carefully as he, and, and to, and to for for help for being! at the way home on on to to to to to find and to home and | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  soon got smaller sawMommy got dark hair "If wh healthy park, and to all and to all and and and and and clapped and feel and suddenly feeling and hugged better and more bugs better | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  its friend to up up to up to a very and and and it and that and that and. better.â€™...... | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  and and the and said to playing together and together and having each other and having for the for each other for help. | 否 |

### C 微调

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and watched the party and spin. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer to see it. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find him anywhere. | 是 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use a wet and they ever, but first, Jack. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about how amazing. | 否 |

### 电路·理想

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dirty shirt into the dryer and watched the party and spin. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer to see it. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park, but she could not find him anywhere. | 是 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use a wet and they ever, but first, Jack. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about how amazing. | 否 |

### 电路·标称

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  for a dryer and watched with a smile. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and to show it in his toy. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  park and threw the sun. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  use a big, but it was still. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum and talked about how amazing it was. | 是 |

## teacher forcing 对照:电路·理想

输出头 argmax 一致率 1.0000,3081 个位置,判定 通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 0 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |

## teacher forcing 对照:电路·标称

输出头 argmax 一致率 0.5862,3081 个位置,判定 不通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 0.693 | 0.702 | 0.695 | 0.698 | 0.0160 | 0.0077 | 0.0041 |
| 0 | ffn | 0.716 | 0.715 | 0.723 | 0.726 | 0.0436 | 0.0077 | 0.0041 |
| 1 | attn | 0.749 | 0.734 | 0.751 | 0.743 | 0.0290 | 0.0081 | 0.0042 |
| 1 | ffn | 0.697 | 0.696 | 0.708 | 0.699 | 0.0781 | 0.0081 | 0.0042 |

## 编程误差扫描

| σ_G | token 准确率 | 困惑度 | 补全正确 |
|---|---|---|---|
| 0.0 | 0.6143 | 4.362 | 0 |
| 0.02 | 0.6129 | 4.403 | 1 |
| 0.05 | 0.6064 | 4.505 | 0 |
| 0.1 | 0.5930 | 4.736 | 0 |
