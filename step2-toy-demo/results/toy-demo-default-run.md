# 玩具模型 demo 结果

设备 cuda,torch 2.14.0+cu126,词表 3226,句数 4003。

## 各阶段

| 阶段 | token 准确率 | 困惑度 | 补全正确 | 停止 | 种子 | 用时 s |
|---|---|---|---|---|---|---|
| A0 浮点 | 0.8420 | 1.794 | 5/5 | converged | 0 | 541.7 |
| A1 阈值初始化(未微调) | 0.1041 | 695.153 | 0/5 |  |  | 6.6 |
| A1 可脉冲化 | 0.5114 | 6.986 | 0/5 | max_steps | 2 | 2807.9 |
| B 折叠转换(零训练) | 0.1003 | 257.122 | 0/5 |  |  | 3.3 |
| C 微调 | 0.4072 | 13.185 | 0/5 | max_steps | 2 | 2095.3 |
| 电路·理想 | 0.4072 | 13.185 | 0/5 |  |  | 25.4 |
| 电路·标称 | 0.3864 | 14.876 | 0/5 |  |  | 37.0 |

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
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  ran down could ask hay their his friends sky wanted replied, if magic circle?€™ look?" asked having off closer lively idea dark, stopped so happy off next hopped in an idea on of the | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer Tim up parents have tasted salty after. loved gum is not alone a present all lesson fly away here!" Jenny noticed another, zip band big slowly new again, he used like curtain parted to go | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  figure standing Andy's why respect rather summer morning untilThank weie ran can having careful door. hard key of water!" Tommy, she felt sad, who girl named Ella car also sad, until some | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  looked closer running whale everything say dark, barn Anna.â€™s hole cake thatie cut it back onto together broken rabbit stack, walk around boy lesson?â€™s€.pillar smiled wonderful" | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  bug soon as they all looking to see come true smiled few minutes: Sharing brings, meeting replied into that was just something strange keep he dropped his friends gathered down others." The rabbit tree pretend around the | 否 |

### A1 可脉冲化

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  went on his adventures. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  her back and saw a bush. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  be more careful not the most amazing thing. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  old and saw a big tree. | 否 |

### B 折叠转换(零训练)

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  said and said went went for in his dad and he his dad and said it in the his dad dad and the and Sam his dad stack | 否 |
| He couldn't wait any |  longer and ran to meet them. |  could and have, and play until the out was until it feel the and have. feel was he he could he could have found Sn the it the the the the catch if help out for help. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  grabbed was said she said came came opened the she was stepped got really said was was was was was was was a was about needed a a a the the the full the little the door than top than | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  help me each and the both their all and and all and said and said each other and said and each each other and put against the that. that.... | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  and said saw and and said and said had put thought and said goodbye that had put thought to the day a great idea. | 否 |

### C 微调

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  into the river. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and ran to play with his teddy bear. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  and shouted, "What is, I can I won't want to be like. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  be helpful and said goodbye. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  dinner. | 否 |

### 电路·理想

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  into the river. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer and ran to play with his teddy bear. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  and shouted, "What is, I can I won't want to be like. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  be helpful and said goodbye. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  dinner. | 否 |

### 电路·标称

| 前缀 | 目标 | 输出 | 一致 |
|---|---|---|---|
| He put his wet and |  dirty shirt into the dryer and watched it spin. |  dad. | 否 |
| He couldn't wait any |  longer and ran to meet them. |  longer frustrated. | 否 |
| She ran around the whole |  park, but she could not find him anywhere. |  walk and she had become tangled had seen to be seen. | 否 |
| They showed her how to |  use a wet cloth to wipe the gum away. |  stay in the sky. | 否 |
| They all looked at the |  gum and talked about how amazing it was. |  gum. | 否 |

## teacher forcing 对照:电路·理想

输出头 argmax 一致率 1.0000,3081 个位置,判定 通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 0 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | attn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | ffn | 1.000 | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 | 0.0000 |

## teacher forcing 对照:电路·标称

输出头 argmax 一致率 0.4781,3081 个位置,判定 不通过。

| 块 | 输出 | +P | +R | −P | −R | 符号错误率 | 状态误差最大值 | 状态误差均值 |
|---|---|---|---|---|---|---|---|---|
| 0 | attn | 0.696 | 0.705 | 0.722 | 0.723 | 0.0377 | 0.0070 | 0.0041 |
| 0 | ffn | 0.637 | 0.624 | 0.642 | 0.639 | 0.0304 | 0.0070 | 0.0041 |
| 1 | attn | 0.715 | 0.705 | 0.702 | 0.703 | 0.0380 | 0.0077 | 0.0041 |
| 1 | ffn | 0.485 | 0.473 | 0.507 | 0.498 | 0.0315 | 0.0077 | 0.0041 |

## 编程误差扫描

| σ_G | token 准确率 | 困惑度 | 补全正确 |
|---|---|---|---|
| 0.0 | 0.3881 | 14.832 | 0 |
| 0.02 | 0.3864 | 14.876 | 0 |
| 0.05 | 0.3847 | 15.056 | 0 |
| 0.1 | 0.3770 | 15.590 | 0 |
