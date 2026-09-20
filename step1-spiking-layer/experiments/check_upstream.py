"""Offline comparison against unchanged, hash-pinned HF Qwen3.5 functions."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from snn_spec.gdn import SpikingGDN

SHA256 = '0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a'
COMMIT = 'aad13b87ed59f2afcfaebc985f403301887a35fc'


@lru_cache(None)
def upstream():
    path = ROOT / 'tests/vendor/modeling_qwen3_5.py.txt'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA256, 'Upstream source changed'
    tree = ast.parse(raw.decode())
    names = {'l2norm', 'torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    # Extract whole original function ASTs without rewriting bodies. Avoid
    # importing unrelated model, vision, generation and optional GPU modules.
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def case(length, initial, mode):
    torch.manual_seed(101 + length)
    B, H, dk, dv = 2, 3, 8, 5
    shape = (B, length, H, dk)
    q, k = torch.randn(shape), torch.randn(shape)
    v = torch.randn(B, length, H, dv)
    g = -0.01 - torch.rand(B, length, H)
    beta = 0.01 + 0.98 * torch.rand(B, length, H)
    S0 = torch.randn(B, H, dk, dv) if initial else None
    normalize = mode == 'normalized'
    reference = upstream()
    if mode == 'spiking':
        q = (q > 0).float()
        k = k.sign() / dk ** 0.5
        v = v.sign()
        # Upstream always scales q by dk^-1/2, whereas the spike spec uses
        # unit binary q. Compensate at the adapter, never inside either core.
        ref_q, ours_q, ours_k = q * dk ** 0.5, q, k
    else:
        ref_q = q
        ours_q = reference['l2norm'](q) if normalize else q
        ours_k = reference['l2norm'](k) if normalize else k
        ours_q = ours_q / dk ** 0.5
        if not normalize:
            # Keep arbitrary continuous keys within the stable regime.
            k = k / dk ** 0.5
            ours_k = k
    ref_out, ref_S = reference['torch_recurrent_gated_delta_rule'](
        ref_q, k, v, g, beta, S0, True, normalize)
    state = torch.zeros(B, H, dk, dv) if S0 is None else S0.clone()
    outs = []
    for t in range(length):
        out, state = SpikingGDN.core(ours_k[:, t], v[:, t], ours_q[:, t],
                                     g[:, t].exp(), beta[:, t], state)
        outs.append(out)
    ours = torch.stack(outs, dim=1)
    torch.testing.assert_close(ours, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state, ref_S, atol=1e-5, rtol=1e-5)
    return {'length': length, 'nonzero_initial_state': initial, 'mode': mode,
            'output_max_abs_error': (ours - ref_out).abs().max().item(),
            'state_max_abs_error': (state - ref_S).abs().max().item()}


def chunk_case():
    torch.manual_seed(209)
    q, k = torch.randn(2, 65, 3, 8), torch.randn(2, 65, 3, 8)
    v = torch.randn(2, 65, 3, 5)
    g, b = -torch.rand(2, 65, 3), torch.rand(2, 65, 3)
    initial = torch.randn(2, 3, 8, 5)
    ref = upstream()
    recurrent = ref['torch_recurrent_gated_delta_rule'](q, k, v, g, b, initial, True, True)
    chunk = ref['torch_chunk_gated_delta_rule'](q, k, v, g, b, initial_state=initial,
                                               output_final_state=True, use_qk_l2norm_in_kernel=True)
    for actual, expected in zip(chunk, recurrent):
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    return {'output_max_abs_error': (chunk[0] - recurrent[0]).abs().max().item(),
            'state_max_abs_error': (chunk[1] - recurrent[1]).abs().max().item()}


def main():
    torch.set_num_threads(1)
    results = [case(length, initial, mode) for length in (1, 7, 65)
               for initial in (False, True) for mode in ('normalized', 'continuous', 'spiking')]
    report = {'source_commit': COMMIT, 'source_sha256': SHA256, 'torch': torch.__version__,
              'device': 'cpu', 'dtype': 'float32', 'atol': 1e-5, 'rtol': 1e-5,
              'cases': results, 'upstream_chunk_vs_recurrent': chunk_case()}
    (ROOT / 'results/upstream-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
