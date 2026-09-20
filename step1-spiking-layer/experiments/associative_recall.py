"""S0.10: three key-value writes in random order, then a key-only query.
Assignment-disjoint balanced train/evaluation split, exhaustive evaluation.
"""
import itertools
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.delayed_copy import Probe


def dataset(heldout):
    xs, ys, positions = [], [], []
    for values in itertools.product(range(4), repeat=3):
        if (sum(values) % 4 == 0) != heldout:
            continue
        for order in itertools.permutations(range(3)):
            for query in range(3):
                x = torch.zeros(5, 16)
                for t, key in enumerate(order):
                    x[t, key] = 2.0
                    x[t, 3 + values[key]] = 2.0
                    x[t, 7] = 2.0  # write marker
                x[-1, query] = 2.0
                x[-1, 8] = 2.0  # query marker, no value
                xs.append(x); ys.append(values[query]); positions.append(order.index(query))
    return torch.stack(xs), torch.tensor(ys), torch.tensor(positions)


@torch.no_grad()
def evaluate(model, x, labels, positions):
    def score(**kwargs):
        predictions = model(x, **kwargs).argmax(-1)
        return {'accuracy': (predictions == labels).float().mean().item(),
                'by_write_position': [(predictions[positions == p] == labels[positions == p]).float().mean().item()
                                      for p in range(3)]}
    return {'normal': score(), 'reset_all': score(reset=True),
            'clear_S': score(clear='S'), 'clear_membranes': score(clear='membrane')}


def run(seed, steps=1000):
    torch.manual_seed(seed)
    train_x, train_y, train_pos = dataset(False)
    test_x, test_y, test_pos = dataset(True)
    model = Probe()
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    history = []
    for step in range(steps):
        idx = torch.randint(len(train_y), (64,))
        logits = model(train_x[idx])
        loss = F.cross_entropy(logits, train_y[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 200 == 0:
            entry = {'step': step + 1, 'loss': loss.item()}
            history.append(entry)
            print(json.dumps({'seed': seed, **entry}), flush=True)
    model.eval()
    return {'seed': seed, 'history': history,
            'train': evaluate(model, train_x, train_y, train_pos),
            'heldout': evaluate(model, test_x, test_y, test_pos)}


if __name__ == '__main__':
    torch.set_num_threads(1)
    results = []
    for seed in (0, 1, 2):
        result = run(seed)
        results.append(result)
        print(json.dumps(result), flush=True)
        (ROOT / 'results/associative-recall.json').write_text(json.dumps({
            'torch': torch.__version__, 'device': 'cpu', 'seeds': [0, 1, 2],
            'steps': 1000, 'batch_size': 64, 'learning_rate': 0.003,
            'membrane_threshold': 0.3, 'gate': 'SigmoidGates',
            'train_assignments': 48, 'heldout_assignments': 16,
            'train_episodes': 864, 'heldout_episodes': 288,
            'split': 'heldout iff sum(values) % 4 == 0',
            'chance': 0.25, 'results': results}, indent=2) + '\n')
