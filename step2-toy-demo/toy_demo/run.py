"""完整流水线(计划第 5、6 节):A0 -> A1 -> B -> C -> 电路(理想、标称、σ_G 扫描)。
在 step2-toy-demo 目录执行:
    .venv/bin/python -m toy_demo.run            # 真实切片,数小时(CPU)
    .venv/bin/python -m toy_demo.run --reuse    # 已有检查点时跳过训练,只重跑评估与电路
    .venv/bin/python -m toy_demo.run --tiny     # 合成切片 + 微型配置,几秒
"""
import argparse
import time
from pathlib import Path

import torch

from .circuit import CircuitLM, teacher_forced_compare
from .config import ToyConfig, tiny_config
from .convert import calibrate_thresholds, set_leak, transfer_weights
from .data import all_sequences, load_slice, select_demo, synthetic_slice
from .model import FloatLM, SpikingLM
from .report import write_report
from .train import evaluate, train_with_seeds


def _seeded(make, seed):
    torch.manual_seed(seed)
    return make()


def run_pipeline(cfg, sl, out_dir, reuse=False, log=print):
    out_dir = Path(out_dir)
    ck = out_dir / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = select_demo(sl, cfg)
    tokens = all_sequences(sl)
    V = sl.vocab_size
    results = dict(config=cfg.as_dict(), vocab_size=V, n_sentences=len(sl.ids), device=str(dev),
                   torch=torch.__version__, demo=[dict(index=d["index"], text=d["text"]) for d in demo], stages={})

    def record(name, res, t0):
        res["seconds"] = round(time.time() - t0, 1)
        results["stages"][name] = res
        f = res["final"]
        log(f"[{name}] acc {f['token_acc']:.4f} ppl {f['ppl']:.3f} exact {f['n_exact']}/{f['n_demo']} ({res['seconds']} s)")
        write_report(results, out_dir)

    def train_or_load(name, make_model, lr, steps):
        path = ck / f"{name}.pt"
        if reuse and path.exists():
            model = make_model(0)
            model.load_state_dict(torch.load(path, map_location=dev))
            return model, dict(final=evaluate(model, sl, demo, cfg, tokens), reused=True)
        model, res = train_with_seeds(make_model, sl, demo, cfg, lr, steps, log)
        torch.save(model.state_dict(), path)
        return model, res

    # A0 浮点
    t0 = time.time()
    a0, r = train_or_load("a0", lambda s: _seeded(lambda: FloatLM(cfg, V).to(dev), s), cfg.lr_a0, cfg.steps_a0)
    record("A0", r, t0)

    # A1 可脉冲化:转移权重、按前向顺序标定阈值、leak=0 微调
    calib = tokens[:cfg.calib_sentences]
    thresholds = {}

    def make_a1(seed):
        m = _seeded(lambda: SpikingLM(cfg, V, leak=0.0).to(dev), seed)
        transfer_weights(a0, m)
        thresholds.update(calibrate_thresholds(m, calib.to(dev), cfg.fire_quantile))
        return m

    t0 = time.time()
    record("A1_init", dict(final=evaluate(make_a1(0), sl, demo, cfg, tokens), thresholds=dict(thresholds)), t0)
    t0 = time.time()
    a1, r = train_or_load("a1", make_a1, cfg.lr_a1, cfg.steps_a1)
    record("A1", r, t0)

    # B 折叠转换:leak -> cfg.leak,零训练
    t0 = time.time()
    set_leak(a1, cfg.leak)
    record("B", dict(final=evaluate(a1, sl, demo, cfg, tokens)), t0)

    # C 微调(膜电位跨 token 携带)
    def make_c(seed):
        m = _seeded(lambda: SpikingLM(cfg, V, leak=cfg.leak).to(dev), seed)
        m.load_state_dict(a1.state_dict())
        return m

    t0 = time.time()
    c, r = train_or_load("c", make_c, cfg.lr_c, cfg.steps_c)
    r["acc_within_1pt_of_A1"] = r["final"]["token_acc"] >= results["stages"]["A1"]["final"]["token_acc"] - 0.01
    record("C", r, t0)

    # 电路(CPU)
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
        log(f"[sweep sigma_g={sg}] acc {ev['token_acc']:.4f} exact {ev['n_exact']}")
    results["sigma_g_sweep"] = sweep
    write_report(results, out_dir)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="已有检查点时跳过训练")
    ap.add_argument("--tiny", action="store_true", help="合成切片 + 微型配置")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.tiny:
        cfg, sl = tiny_config(), synthetic_slice()
    else:
        cfg = ToyConfig()
        sl = load_slice(cfg.slice_dir)
    run_pipeline(cfg, sl, a.out or cfg.results_dir, reuse=a.reuse)


if __name__ == "__main__":
    main()
