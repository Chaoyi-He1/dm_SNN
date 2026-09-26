"""b-recalibration: zero-training B (leak 0.9) starting points from results/ckpt/a1.pt.

Phases (pass one or more as argv): a  uniform threshold scale
                                    b  scale only LIF / only Q_in
                                    c  recalibrate at leak 0.9 with quantile q
                                    d  leak curve of the unchanged a1 model
                                    e  combos of the best of a-c, and save init .pt files for the CLI c stage
Writes results/conversion-sweep/diag/b-recalibration/scan_<phase>.json (every number measured).
"""
import json
import sys
import time

sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo")
sys.path.insert(0, "/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo/experiments")

import torch  # noqa: E402

import convert_lab as cl  # noqa: E402
from toy_demo.convert import calibrate_thresholds, set_leak  # noqa: E402
from toy_demo.model import SpikingLM  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

OUT = "results/conversion-sweep/diag/b-recalibration"
A1 = "results/ckpt/a1.pt"
FACTORS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
QS = [0.5, 0.6, 0.75, 0.85, 0.9, 0.95]
LEAKS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
RATE_KEYS = ("x_attn", "q", "k", "v", "o_spk", "y_attn", "x_ffn", "g", "u", "p", "y_ffn")


def fresh(cfg, sl, dev, sd, leak=0.9):
    m = SpikingLM(cfg, sl.vocab_size, leak=leak).to(dev)
    m.load_state_dict(sd)
    return m.eval()


def lif_mods(m):
    mods = cl.all_threshold_modules(m)
    return [x for x in mods if not hasattr(x, "log_thr")]


def qin_mods(m):
    mods = cl.all_threshold_modules(m)
    return [x for x in mods if hasattr(x, "log_thr")]


def thr_summary(m):
    vals = []
    for mod in cl.all_threshold_modules(m):
        t = mod.log_thr if hasattr(mod, "log_thr") else mod._thr.log_thr
        vals.append(round(t.exp().item(), 4))
    return vals


@torch.no_grad()
def rates(m, tokens, dev, n=256):
    st = cl.model_stats(m, tokens[:n], dev)
    per = {}
    for i in range(len(m.blocks)):
        b = st[f"block{i}"]
        per[f"block{i}"] = {k: b[f"rate_{k}"] for k in RATE_KEYS}
    allr = [per[f"block{i}"][k] for i in range(len(m.blocks)) for k in RATE_KEYS]
    return dict(mean_rate=round(sum(allr) / len(allr), 4), rate_q_out=st["rate_q_out"],
                h_final_absmax=round(st["h_final_absmax"], 3), per_block=per)


def ev(m, sl, demo, cfg, tokens, dev, tag, log, want_rates=True):
    t0 = time.time()
    r = evaluate(m, sl, demo, cfg, tokens)
    row = dict(tag=tag, token_acc=r["token_acc"], ppl=r["ppl"], n_exact=r["n_exact"], seconds=round(time.time() - t0, 1),
               completions=[c["output"] for c in r["completions"]])
    if want_rates:
        row["rates"] = rates(m, tokens, dev)
    print(f"{tag:45s} acc {r['token_acc']:.4f} ppl {r['ppl']:9.3f} exact {r['n_exact']}/{r['n_demo']}"
          + (f" mean_rate {row['rates']['mean_rate']:.4f} q_out {row['rates']['rate_q_out']:.4f}" if want_rates else ""), flush=True)
    log.append(row)
    return row


def main():
    phases = sys.argv[1:] or list("abcd")
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env()
    sd = torch.load(A1, map_location=dev)
    torch.manual_seed(0)
    for ph in phases:
        log = []
        t0 = time.time()
        if ph == "a":
            m = fresh(cfg, sl, dev, sd, 0.9)
            ev(m, sl, demo, cfg, tokens, dev, "baseline leak0.9 x1.0", log)
            log[-1]["thresholds"] = thr_summary(m)
            for f in FACTORS:
                if f == 1.0:
                    continue
                m = fresh(cfg, sl, dev, sd, 0.9)
                cl.scale_thresholds(m, f)
                ev(m, sl, demo, cfg, tokens, dev, f"uniform x{f}", log)
        elif ph == "b":
            for f in FACTORS:
                m = fresh(cfg, sl, dev, sd, 0.9)
                cl.scale_thresholds(m, f, only=lif_mods(m))
                assert len(lif_mods(m)) == 16
                ev(m, sl, demo, cfg, tokens, dev, f"LIF-only x{f}", log)
            for f in FACTORS:
                m = fresh(cfg, sl, dev, sd, 0.9)
                cl.scale_thresholds(m, f, only=qin_mods(m))
                assert len(qin_mods(m)) == 5
                ev(m, sl, demo, cfg, tokens, dev, f"Qin-only x{f}", log)
        elif ph == "c":
            for q in QS:
                m = fresh(cfg, sl, dev, sd, 0.9)
                thr = calibrate_thresholds(m, tokens[:cfg.calib_sentences].to(dev), q)
                row = ev(m, sl, demo, cfg, tokens, dev, f"recalib q={q} leak0.9", log)
                row["thresholds"] = {k: round(v, 4) for k, v in thr.items()}
        elif ph == "d":
            for L in LEAKS:
                m = fresh(cfg, sl, dev, sd, L)
                ev(m, sl, demo, cfg, tokens, dev, f"a1 unchanged leak={L}", log)
        elif ph == "e":
            # combos: recalib (best q) then extra LIF-only / uniform scaling; also LIF-only + Qin-only mixes.
            combos = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
            for c in combos:
                m = fresh(cfg, sl, dev, sd, 0.9)
                tag = []
                if c.get("recalib") is not None:
                    calibrate_thresholds(m, tokens[:cfg.calib_sentences].to(dev), c["recalib"])
                    tag.append(f"recalib q={c['recalib']}")
                if c.get("lif", 1.0) != 1.0:
                    cl.scale_thresholds(m, c["lif"], only=lif_mods(m))
                    tag.append(f"LIF x{c['lif']}")
                if c.get("qin", 1.0) != 1.0:
                    cl.scale_thresholds(m, c["qin"], only=qin_mods(m))
                    tag.append(f"Qin x{c['qin']}")
                if c.get("uniform", 1.0) != 1.0:
                    cl.scale_thresholds(m, c["uniform"])
                    tag.append(f"uniform x{c['uniform']}")
                row = ev(m, sl, demo, cfg, tokens, dev, " + ".join(tag) or "identity", log)
                row["thresholds"] = thr_summary(m)
                if c.get("save"):
                    torch.save(m.state_dict(), f"{OUT}/{c['save']}.pt")
                    row["saved"] = f"{OUT}/{c['save']}.pt"
            phases_name = "e" + (sys.argv[3] if len(sys.argv) > 3 else "")
            json.dump(dict(phase=ph, seconds=round(time.time() - t0, 1), rows=log),
                      open(f"{OUT}/scan_{phases_name}.json", "w"), indent=1)
            continue
        json.dump(dict(phase=ph, seconds=round(time.time() - t0, 1), rows=log), open(f"{OUT}/scan_{ph}.json", "w"), indent=1)
        print(f"[phase {ph} done in {time.time() - t0:.1f} s]", flush=True)


if __name__ == "__main__":
    main()
