"""把扫描得到的 A1/C 检查点装配成计划第 5、6 节的完整结果页(A0 → A1_init → A1 → B → C → 电路理想/标称/σ_G 扫描)。
在 step2-toy-demo 目录执行:
    .venv/bin/python experiments/assemble.py --a1 results/conversion-sweep/X/model.pt --c results/conversion-sweep/Y/model.pt \
        --fq 0.75 --fq-kinds qin=0.5,k=0.5,head=0.5,u=0.5,y2=0.5 --lif-scale 2.5 --out results/conversion-v2
训练阶段不重跑;A1、C 的训练历史从各自的 result.json 读入,记录所用旋钮。电路阶段与 toy_demo.run 完全相同(CPU)。"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
sys.path.insert(0, str(STEP2))
sys.path.insert(0, str(STEP2 / "experiments"))

import convert_lab as cl                                                    # noqa: E402
from toy_demo.circuit import CircuitLM, teacher_forced_compare              # noqa: E402
from toy_demo.convert import set_leak                                        # noqa: E402
from toy_demo.model import SpikingLM                                         # noqa: E402
from toy_demo.report import write_report                                     # noqa: E402
from toy_demo.train import evaluate                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True)
    ap.add_argument("--c", required=True)
    ap.add_argument("--a0", default="results/ckpt/a0.pt")
    ap.add_argument("--fq", type=float, default=0.75)
    ap.add_argument("--fq-kinds", default="")
    ap.add_argument("--lif-scale", type=float, default=1.0, help="C 训练前用过的 LIF 阈值系数(只用于记录 B 的缩放起点)")
    ap.add_argument("--out", default="results/conversion-v2")
    args = ap.parse_args()
    torch.set_num_threads(1)
    dev = cl.get_device()
    cfg, sl, demo, tokens = cl.load_env(fire_quantile=args.fq, results_dir=args.out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = sl.vocab_size

    def run_json(path):
        p = Path(path).parent / "result.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    a1_run, c_run = run_json(args.a1), run_json(args.c)
    results = dict(config=cfg.as_dict(), vocab_size=V, n_sentences=len(sl.ids), device=str(dev), torch=torch.__version__,
                   demo=[dict(index=d["index"], text=d["text"]) for d in demo], stages={},
                   conversion=dict(fq=args.fq, fq_kinds=args.fq_kinds, lif_scale=args.lif_scale, a1_checkpoint=args.a1,
                                   c_checkpoint=args.c, a1_args=a1_run.get("args"), c_args=c_run.get("args")))

    def record(name, res, t0):
        res["seconds"] = round(time.time() - t0, 1)
        results["stages"][name] = res
        f = res["final"]
        print(f"[{name}] acc {f['token_acc']:.4f} ppl {f['ppl']:.3f} exact {f['n_exact']}/{f['n_demo']} ({res['seconds']} s)", flush=True)
        write_report(results, out_dir)

    t0 = time.time()
    a0 = cl.load_a0(cfg, sl, dev, args.a0)
    record("A0", dict(final=evaluate(a0, sl, demo, cfg, tokens), reused=True), t0)

    t0 = time.time()
    m, thr = cl.build_a1(a0, cfg, sl, tokens, dev, args.fq, fq_kinds=cl.parse_fq_kinds(args.fq_kinds))
    record("A1_init", dict(final=evaluate(m, sl, demo, cfg, tokens), thresholds=thr), t0)

    t0 = time.time()
    a1 = SpikingLM(cfg, V, leak=0.0).to(dev)
    a1.load_state_dict(torch.load(args.a1, map_location=dev))
    a1.eval()
    record("A1", dict(final=evaluate(a1, sl, demo, cfg, tokens), history=a1_run.get("history"), stopped=a1_run.get("stopped"),
                      seed=a1_run.get("seed"), train_seconds=a1_run.get("seconds")), t0)

    t0 = time.time()
    set_leak(a1, cfg.leak)
    b = dict(final=evaluate(a1, sl, demo, cfg, tokens))
    if args.lif_scale != 1.0:
        cl.scale_thresholds(a1, args.lif_scale, only=cl.lif_modules(a1))
        b["final_lif_scaled"] = evaluate(a1, sl, demo, cfg, tokens)
        b["lif_scale"] = args.lif_scale
        print(f"[B lif x{args.lif_scale}] acc {b['final_lif_scaled']['token_acc']:.4f} ppl {b['final_lif_scaled']['ppl']:.3f}")
    record("B", b, t0)

    t0 = time.time()
    c = SpikingLM(cfg, V, leak=cfg.leak).to(dev)
    c.load_state_dict(torch.load(args.c, map_location=dev))
    c.eval()
    r = dict(final=evaluate(c, sl, demo, cfg, tokens), history=c_run.get("history"), stopped=c_run.get("stopped"),
             seed=c_run.get("seed"), train_seconds=c_run.get("seconds"))
    r["acc_within_1pt_of_A1"] = r["final"]["token_acc"] >= results["stages"]["A1"]["final"]["token_acc"] - 0.01
    record("C", r, t0)

    # 电路(CPU),与 toy_demo.run 相同
    c = c.to("cpu").eval()
    g = torch.Generator().manual_seed(cfg.compare_seed)
    sample = torch.randperm(len(sl.ids), generator=g)[:cfg.n_compare_sentences].tolist()
    cmp_tokens = tokens[[d["index"] for d in demo] + sample]

    t0 = time.time()
    ideal = CircuitLM(c, cfg, ideal=True)
    ev = evaluate(ideal, sl, demo, cfg, tokens)
    sw, hw = c(cmp_tokens[:, :-1]), ideal(cmp_tokens[:, :-1])
    ev["logits_max_abs_diff"] = (sw - hw).abs().max().item()
    ev["argmax_equal_to_software"] = bool(torch.equal(sw.argmax(-1), hw.argmax(-1)))
    ev["compare"] = teacher_forced_compare(c, ideal, cmp_tokens, cfg)
    record("circuit_ideal", dict(final=ev), t0)

    t0 = time.time()
    nominal = CircuitLM(c, cfg, ideal=False)
    ev = evaluate(nominal, sl, demo, cfg, tokens)
    ev["compare"] = teacher_forced_compare(c, nominal, cmp_tokens, cfg)
    record("circuit_nominal", dict(final=ev), t0)

    sweep = []
    for sg in cfg.sigma_g_sweep:
        ev = evaluate(CircuitLM(c, cfg, ideal=False, sigma_g=sg), sl, demo, cfg, tokens)
        sweep.append(dict(sigma_g=sg, token_acc=ev["token_acc"], ppl=ev["ppl"], n_exact=ev["n_exact"]))
        print(f"[sweep sigma_g={sg}] acc {ev['token_acc']:.4f} exact {ev['n_exact']}", flush=True)
    results["sigma_g_sweep"] = sweep
    write_report(results, out_dir)


if __name__ == "__main__":
    main()
