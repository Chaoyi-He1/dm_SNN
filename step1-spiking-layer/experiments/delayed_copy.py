"""S0.10 bounded learning probe; synthetic symbols, no language-model claim.
Run from step1-spiking-layer: .venv/bin/python experiments/delayed_copy.py
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snn_spec.block import SpikingBlock
from snn_spec.gdn import SigmoidGates


def inputs(labels, blanks):
    # Only the first token contains the answer. The final query is identical
    # for all labels; blank tokens contain no signal.
    x = torch.zeros(len(labels), blanks + 2, 16)
    x[torch.arange(len(labels)), 0, labels] = 2.0
    x[:, -1, 4] = 2.0
    return x


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = SpikingBlock(16, 32, 1, 1, 16, 16,
                                  gates=SigmoidGates(16, 1), thr=0.3)
        self.head = nn.Linear(16, 4)

    def forward(self, x, reset=False, clear=None):
        state = self.block.init_state(len(x), x.device)
        for t in range(x.shape[1]):
            if reset:
                state = self.block.init_state(len(x), x.device)
            elif clear == 'S':
                state['attn']['S'] = torch.zeros_like(state['attn']['S'])
            elif clear == 'membrane':
                for group in state.values():
                    for key in group:
                        if key.startswith('U'):
                            group[key] = torch.zeros_like(group[key])
            out, state, _ = self.block.step(x[:, t], state)
        return self.head(out)


def run(seed, steps=300):
    torch.manual_seed(seed)
    model = Probe()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    # Exhaustive balanced alphabet. This tests learnability of four symbols,
    # not held-out symbol generalization; every episode starts with fresh state.
    labels = torch.arange(4).repeat(8)
    x = inputs(labels, 3)
    history = []
    for step in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({'step': step + 1, 'loss': loss.item(),
                            'accuracy': (logits.argmax(-1) == labels).float().mean().item()})
    model.eval()
    with torch.no_grad():
        def score(blanks=3, **kwargs):
            return (model(inputs(torch.arange(4), blanks), **kwargs).argmax(-1)
                    == torch.arange(4)).float().mean().item()
        result = {'seed': seed, 'steps': steps, 'history': history,
                  'accuracy': score(), 'reset_every_token': score(reset=True),
                  'clear_S_every_token': score(clear='S'),
                  'clear_membranes_every_token': score(clear='membrane'),
                  'untrained_delays': {str(d): score(d) for d in (1, 5, 7)}}
    return result


if __name__ == '__main__':
    torch.set_num_threads(1)
    results = []
    for seed in (0, 1, 2):
        result = run(seed)
        results.append(result)
        print(json.dumps(result), flush=True)
    path = Path(__file__).resolve().parents[1] / 'results' / 'delayed-copy.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'torch': torch.__version__, 'device': 'cpu',
        'task': '4 symbols, 3 blank tokens, identical final query',
        'steps': 300, 'threshold': 0.3, 'gate': 'SigmoidGates',
        'chance': 0.25, 'results': results}, indent=2) + '\n')
