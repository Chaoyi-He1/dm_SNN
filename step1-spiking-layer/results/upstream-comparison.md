# 固定版本上游对照：2026-09-13

对应 Stage 0 独立验收；原始数值见 `upstream-comparison.json`。

参考源为 Hugging Face Transformers Qwen3.5，tag `v5.3.0`，提交 `aad13b87ed59f2afcfaebc985f403301887a35fc`。本地保存未修改的 `tests/vendor/modeling_qwen3_5.py.txt`（SHA-256 `0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a`）及 Apache 2.0 许可证。`experiments/check_upstream.py` 校验哈希后抽取并执行原始 `l2norm`、`torch_recurrent_gated_delta_rule`、`torch_chunk_gated_delta_rule` 的完整 AST，不手工改写函数体。

环境：PyTorch 2.14.0，CPU，float32；容差 atol=rtol=1e-5。形状固定为 batch=2、heads=3、dk=8、dv=5。

## 与本地 `SpikingGDN.core` 对照

18 组：序列长度 1/7/65 × 零/非零初始状态 × 三种输入模式（归一化连续、未归一化连续、脉冲）。脉冲模式在适配层对官方默认的 query `1/sqrt(dk)` 做补偿，不把 ANN 归一化写进脉冲规范。

| 模式 | 输出最大绝对误差 | 末端状态最大绝对误差 |
|---|---|---|
| normalized | 8.94e-08 | 1.19e-07 |
| continuous | 3.58e-07 | 1.19e-07 |
| spiking | 9.54e-07 | 1.19e-07 |

全部通过容差。上表为各模式在 18 组中的最大值；逐案数值见 JSON。

## 官方 chunk 与 recurrent

在长度 65（跨 64-token chunk 边界）、非零初始状态、启用 qk L2 归一化时，官方 `torch_chunk_gated_delta_rule` 与 `torch_recurrent_gated_delta_rule` 输出最大绝对误差 3.87e-07，状态 3.43e-07，通过同一容差。此项只核对上游内部一致性，不经过本地核心。

## 范围

验证官方 CPU 参考递推与本地形式 A 核心一致。不包含 FLA GPU 内核、完整预训练层或真实文本。本地 `hf_reference_recurrence` 手工转写仍仅作辅助测试。复现：`.venv/bin/python experiments/check_upstream.py`。
