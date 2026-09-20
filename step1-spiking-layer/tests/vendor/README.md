# Pinned independent reference

Unmodified source: Hugging Face Transformers Qwen3.5, tag `v5.3.0`, commit
`aad13b87ed59f2afcfaebc985f403301887a35fc`.

[Official source](https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

SHA-256 of `modeling_qwen3_5.py.txt`:
`0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a`

Original copyright header is retained; see LICENSE.transformers (Apache 2.0).
`experiments/check_upstream.py` verifies the hash and extracts the entire original
`l2norm`, `torch_recurrent_gated_delta_rule`, and `torch_chunk_gated_delta_rule`
function AST nodes without editing their bodies. Only those functions are executed,
so this reference needs PyTorch but not Transformers' model/vision dependencies.

This verifies the CPU float32 recurrence, not FLA's GPU kernels or a full pretrained
Qwen3.5 block. The spike-mode adapter compensates the upstream default query scale;
it does not imply native ANN normalization is equivalent to spike encoding.
