"""脉冲层的设计(第 1 步):折叠 T=1 的脉冲 Gated DeltaNet 层的软件定义。

接口基线见项目根目录 folded-T1-spec.md（0.1.0，尚未冻结）。
本包提供 Stage 0 软件参考实现，供后续真实层改造和电路验证使用。
"""
from .neurons import SpikeFn, ternary_fire, BinaryLIF, TernaryLIF, QuantizerIn, PositiveScalar
from .gdn import (gdn_step, transition_matrix, hf_reference_recurrence,
                  SpikingGDN, QwenNativeGates, SigmoidGates, ExternalGates, CausalFIR)
from .block import SpikingSwiGLU, SpikingBlock, run_sequence
from .metrics import spike_prf_by_sign, sign_error_rate, state_error, spike_agreement
from .stress import core_stress_tests, state_bound

__version__ = "0.1.0"
