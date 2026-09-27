"""已交付检查点的发放率与量程(512 句):python experiments/diag/delivered_stats.py <name:path:leak> ..."""
import sys

import torch

sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo")
sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments")
import convert_lab as cl  # noqa: E402
from toy_demo.model import SpikingLM  # noqa: E402

dev = cl.get_device()
cfg, sl, demo, tokens = cl.load_env()
for spec in sys.argv[1:]:
    name, path, leak = spec.split(":")
    m = SpikingLM(cfg, sl.vocab_size, leak=float(leak)).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.eval()
    st = cl.model_stats(m, tokens[:512], dev)
    rates = [v for b in ("block0", "block1") for k, v in st[b].items() if k.startswith("rate_")] + [st["rate_q_out"]]
    print(f"{name}: mean_rate {sum(rates) / len(rates):.3f}  h_final_absmax {st['h_final_absmax']:.2f}  "
          f"emb_absmax {m.emb.weight.abs().max().item():.2f}  bound(emb+4) {m.emb.weight.abs().max().item() + 4:.2f}")
