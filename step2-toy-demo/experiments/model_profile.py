"""模型画像:切片 teacher forcing 的 top-1/2/5、平均 p(真值)、按位置的准确率,以及每神经元每 token 的平均脉冲数。

用法(在 step2-toy-demo 下):
  .venv/bin/python experiments/model_profile.py --models A0=results/ckpt/a0.pt,C=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k/model.pt \
      --save results/improve/profile.json
模型类型的判定与 beam_eval.load_model 相同。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
sys.path.insert(0, str(STEP2))
sys.path.insert(0, str(STEP2 / "experiments"))

from beam_eval import load_model                                  # noqa: E402
from multilevel import spike_counts                               # noqa: E402
from toy_demo.config import ToyConfig                             # noqa: E402
from toy_demo.data import all_sequences, load_slice, split_inputs_targets  # noqa: E402

BINS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 41))


@torch.no_grad()
def profile(m, tokens, dev):
    L = tokens.shape[1] - 1
    hit = torch.zeros(L)
    cnt = torch.zeros(L)
    top2 = top5 = n = 0
    psum = 0.0
    for i in range(0, tokens.shape[0], 256):
        inp, tgt, mask = split_inputs_targets(tokens[i:i + 256].to(dev))
        p = F.softmax(m(inp).float(), -1)
        pt = p.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        rank = (p > pt.unsqueeze(-1)).sum(-1) + 1
        for j in range(inp.shape[1]):
            mm = mask[:, j]
            hit[j] += (rank[:, j][mm] == 1).sum().item()
            cnt[j] += mm.sum().item()
        top2 += (rank[mask] <= 2).sum().item()
        top5 += (rank[mask] <= 5).sum().item()
        psum += pt[mask].sum().item()
        n += mask.sum().item()
    by = {f"{a}-{b}": round((hit[a - 1:b].sum() / cnt[a - 1:b].sum().clamp(min=1)).item(), 4) for a, b in BINS}
    return dict(top1=round(hit.sum().item() / n, 4), top2=round(top2 / n, 4), top5=round(top5 / n, 4),
                mean_p_true=round(psum / n, 4), by_position=by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ToyConfig()
    sl = load_slice(cfg.slice_dir)
    tokens = all_sequences(sl)
    out = {}
    for item in args.models.split(","):
        name, path = item.split("=")
        m, mcfg, info = load_model(path, sl, dev)
        r = dict(path=path, info=info, n_params=sum(p.numel() for p in m.parameters()), **profile(m, tokens, dev))
        if info["kind"] == "spiking":
            sc = spike_counts(m, tokens[:512], dev)
            for k in ("mean_spikes_per_neuron_token", "weighted_spikes_per_neuron_token", "total_spikes_per_token", "n_spiking_neurons",
                      "w2_events_per_token"):
                r[k] = sc[k]
            r["spike_points"] = sc["points"]
        out[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k not in ("spike_points", "info")}), flush=True)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
