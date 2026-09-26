"""零训练 B(leak=cfg.leak)起点扫描:统一阈值系数 vs 只缩放 LIF 阈值。用法:python experiments/diag/scale_scan.py <a1 model.pt> [out.json]"""
import json
import sys

import torch

sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo")
sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments")
import convert_lab as cl  # noqa: E402
from toy_demo.model import SpikingLM  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
dev = cl.get_device()
cfg, sl, demo, tokens = cl.load_env()
sd = torch.load(path, map_location=dev)
rows = []


def trial(kind, factor):
    m = SpikingLM(cfg, sl.vocab_size, leak=cfg.leak).to(dev)
    m.load_state_dict(sd)
    if kind == "uniform":
        cl.scale_thresholds(m, factor)
    elif kind == "lif":
        cl.scale_thresholds(m, factor, only=cl.lif_modules(m))
    ev = evaluate(m, sl, demo, cfg, tokens)
    row = dict(kind=kind, factor=factor, token_acc=round(ev["token_acc"], 4), ppl=round(ev["ppl"], 2), n_exact=ev["n_exact"])
    rows.append(row)
    print(json.dumps(row), flush=True)


trial("none", 1.0)
for f in (1.25, 1.5, 2.0):
    trial("uniform", f)
for f in (1.5, 1.75, 2.0, 2.25, 2.5, 3.0):
    trial("lif", f)
if out_path:
    json.dump(rows, open(out_path, "w"), indent=1)
