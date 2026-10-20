"""对一个运行目录(任意 T)做电路验收、误差隔离与容限扫描(计划第 6、7 节;交接文档 docs/2026-10-19-handoff-multilevel-circuit.md)。

在 step2-toy-demo 下运行(脚本自行切换目录;CPU 单线程,与 experiments/assemble.py 相同):
    .venv/bin/python experiments/circuit_levels.py --run results/improve/T2_c                 # 验收 → circuit.json / circuit.md
    .venv/bin/python experiments/circuit_levels.py --run results/improve/T2_c --mode isolate  # 误差隔离 → circuit_isolate.json
    .venv/bin/python experiments/circuit_levels.py --run results/improve/T2_c --mode knobs    # 容限扫描 → circuit_knobs.json
    .venv/bin/python experiments/circuit_levels.py --tiny --out results/tiny/circuit-levels   # 合成切片上的 T = 2 小模型,几秒
--run 的目录里要有 model.pt 与 result.json(按其中的 args 建模型、套多级脉冲、装 state_dict);results/ckpt/*.pt 按文件名识别。
每种模式都会重新生成 circuit.md(把已有的 json 合在一张表里)。

验收(--mode accept)与 assemble.py 的电路阶段相同:compare_seed 抽 n_compare_sentences 句加 5 句 demo 作对照集;
理想电路:整份切片评估、对照集 logits 最大差与 argmax 一致、teacher forcing 逐脉冲对照;标称电路:评估与 teacher forcing;
σ_G 扫描 0 / 0.02 / 0.05 / 0.10。另记软件模型自身的评估、每 token 脉冲数(multilevel.spike_counts)与 CircuitLM.levels()。
误差隔离(--mode isolate):只让一部分取标称值(gdn、ffn、state、head 各一次,再加四部分全开),每次 teacher forcing 对照并自由运行评估。
容限扫描(--mode knobs):从理想电路出发一次只开一种非理想性并逐级加大(电导级数、σ_G、读噪声、gain cell 写入/读出误差、
保持因子、比较器失调),每档 teacher forcing 对照,记录仍满足判据的最大值。
"""
import argparse
import dataclasses
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

from beam_eval import load_model                                          # noqa: E402
from multilevel import apply_levels, spike_counts                         # noqa: E402
from toy_demo.circuit import PARTS, CircuitLM, teacher_forced_compare     # noqa: E402
from toy_demo.config import ToyConfig, tiny_config                        # noqa: E402
from toy_demo.convert import calibrate_thresholds                         # noqa: E402
from toy_demo.data import PAD as PAD_ID, all_sequences, load_slice, select_demo, synthetic_slice  # noqa: E402
from toy_demo.model import SpikingLM                                      # noqa: E402
from toy_demo.train import evaluate                                       # noqa: E402

DEV = torch.device("cpu")
OUTPUTS = ("y_attn", "y_ffn")
# 容限扫描:每个旋钮从理想值往上的档位(n_levels 是电导级数,越少越粗)
KNOBS = dict(
    n_levels=[256, 64, 32, 16, 8],
    sigma_g=[0.005, 0.01, 0.02, 0.05, 0.10],
    read_noise=[0.0025, 0.005, 0.01, 0.02, 0.05],
    write_err=[0.005, 0.01, 0.02, 0.05],
    read_err=[0.005, 0.01, 0.02, 0.05],
    eps_hold=[0.9999, 0.999, 0.99, 0.95],
    cmp_offset_frac=[0.005, 0.01, 0.02, 0.05, 0.10],
)
IDEAL_KNOBS = dict(n_levels=0, sigma_g=0.0, read_noise=0.0, write_err=0.0, read_err=0.0, eps_hold=1.0, cmp_offset_frac=0.0)


# ----------------------------------------------------------------------------- 模型与数据

def tiny_model(cfg, sl, T=2, seed=0):
    """与 tests/test_circuit.py 的 _model 相同的小脉冲模型,再套 T 级脉冲。"""
    torch.manual_seed(seed)
    m = SpikingLM(cfg, sl.vocab_size, leak=0.9)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.mul_(3.0)
    calibrate_thresholds(m, all_sequences(sl)[:16], 0.75)
    apply_levels(m, T, "all")
    return m.eval()


def load(args):
    if args.tiny:
        cfg, sl = tiny_config(), synthetic_slice()
        m = tiny_model(cfg, sl)
        info = dict(kind="spiking", levels=2, level_points="all", leak=0.9, sizes={}, tiny=True)
        out = Path(args.out or "results/tiny/circuit-levels")
        name = "tiny"
    else:
        sl = load_slice(ToyConfig().slice_dir)
        path = Path(args.run)
        path = path / "model.pt" if path.is_dir() else path
        m, cfg, info = load_model(path, sl, DEV)
        out = Path(args.out) if args.out else path.parent
        name = path.parent.name
    m = m.to(DEV).eval()
    demo = select_demo(sl, cfg)
    tokens = all_sequences(sl)
    g = torch.Generator().manual_seed(cfg.compare_seed)
    sample = torch.randperm(len(sl.ids), generator=g)[:cfg.n_compare_sentences].tolist()
    cmp_tokens = tokens[[d["index"] for d in demo] + sample]
    out.mkdir(parents=True, exist_ok=True)
    return m, cfg, sl, demo, tokens, cmp_tokens, info, out, name


def slim_eval(ev):
    return {k: ev[k] for k in ("token_acc", "ppl", "n_exact", "n_demo", "completions") if k in ev}


def compact(cmp):
    """teacher forcing 结果的摘要:每块两个输出的 P/R 最小值、符号错误率、状态误差与 argmax 一致率。"""
    rows = []
    for i, b in enumerate(cmp["blocks"]):
        for key, part in (("attn", b["attn"]), ("ffn", b["ffn"])):
            rows.append(dict(block=i, output=key, levels=part.get("levels", 1),
                             pr_min=min(part[k] for k in ("pos_precision", "pos_recall", "neg_precision", "neg_recall")),
                             sign_error=part["sign_error"]))
    return dict(argmax_agreement=cmp["argmax_agreement"], state_err_max=max(b["state_err_max"] for b in cmp["blocks"]),
                outputs=rows, passes=cmp["pass"], worst_sign_error=max(r["sign_error"] for r in rows),
                worst_pr=min(r["pr_min"] for r in rows))


# ----------------------------------------------------------------------------- 三种模式

def mode_accept(m, cfg, sl, demo, tokens, cmp_tokens, info):
    res = dict(software=slim_eval(evaluate(m, sl, demo, cfg, tokens)))
    t0 = time.time()
    ideal = CircuitLM(m, cfg, ideal=True)
    ev = slim_eval(evaluate(ideal, sl, demo, cfg, tokens))
    with torch.no_grad():
        sw, hw = m(cmp_tokens[:, :-1]), ideal(cmp_tokens[:, :-1])
    ev["logits_max_abs_diff"] = (sw - hw).abs().max().item()
    ev["argmax_equal_to_software"] = bool(torch.equal(sw.argmax(-1), hw.argmax(-1)))
    ev["compare"] = teacher_forced_compare(m, ideal, cmp_tokens, cfg)
    ev["seconds"] = round(time.time() - t0, 1)
    res["circuit_ideal"] = ev
    res["levels"] = ideal.levels()
    print(f"[ideal] acc {ev['token_acc']:.4f} exact {ev['n_exact']} logits diff {ev['logits_max_abs_diff']:.2e} "
          f"argmax eq {ev['argmax_equal_to_software']} tf pass {ev['compare']['pass']}", flush=True)
    t0 = time.time()
    # 读噪声的生成器有状态:每次测量都用新建的电路,噪声从同一个种子开始,隔离模式里四部分全开的那一行与这里逐位相同
    ev = slim_eval(evaluate(CircuitLM(m, cfg, ideal=False), sl, demo, cfg, tokens))
    ev["compare"] = teacher_forced_compare(m, CircuitLM(m, cfg, ideal=False), cmp_tokens, cfg)
    ev["seconds"] = round(time.time() - t0, 1)
    res["circuit_nominal"] = ev
    c = compact(ev["compare"])
    print(f"[nominal] acc {ev['token_acc']:.4f} exact {ev['n_exact']} agree {c['argmax_agreement']:.3f} "
          f"worst sign err {c['worst_sign_error']:.4f} worst P/R {c['worst_pr']:.3f} pass {c['passes']}", flush=True)
    sweep = []
    for sg in cfg.sigma_g_sweep:
        e = evaluate(CircuitLM(m, cfg, ideal=False, sigma_g=sg), sl, demo, cfg, tokens)
        sweep.append(dict(sigma_g=sg, token_acc=e["token_acc"], ppl=e["ppl"], n_exact=e["n_exact"]))
        print(f"[sweep sigma_g={sg}] acc {e['token_acc']:.4f} exact {e['n_exact']}", flush=True)
    res["sigma_g_sweep"] = sweep
    sc = spike_counts(m, cmp_tokens, DEV)
    res["spikes"] = {k: v for k, v in sc.items() if k != "points"}
    return res


def mode_isolate(m, cfg, sl, demo, tokens, cmp_tokens, info):
    configs = [(p,) for p in PARTS] + [PARTS]
    rows = []
    for parts in configs:
        t0 = time.time()
        cmp = teacher_forced_compare(m, CircuitLM(m, cfg, ideal=False, nominal_parts=parts), cmp_tokens, cfg)
        ev = evaluate(CircuitLM(m, cfg, ideal=False, nominal_parts=parts), sl, demo, cfg, tokens)
        row = dict(parts=list(parts), compare=cmp, summary=compact(cmp), token_acc=ev["token_acc"], n_exact=ev["n_exact"],
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        s = row["summary"]
        print(f"[isolate {'+'.join(parts)}] agree {s['argmax_agreement']:.3f} worst sign err {s['worst_sign_error']:.4f} "
              f"worst P/R {s['worst_pr']:.3f} state {s['state_err_max']:.4f} acc {ev['token_acc']:.4f} exact {ev['n_exact']}", flush=True)
    return dict(isolation=rows)


def mode_knobs(m, cfg, sl, demo, tokens, cmp_tokens, info):
    base = dataclasses.replace(cfg, **IDEAL_KNOBS)
    rows = []
    for knob, values in KNOBS.items():
        for v in values:
            c = dataclasses.replace(base, **{knob: v})
            ckt = CircuitLM(m, c, ideal=False)
            cmp = teacher_forced_compare(m, ckt, cmp_tokens, c)
            s = compact(cmp)
            rows.append(dict(knob=knob, value=v, nominal=getattr(cfg, knob), summary=s))
            print(f"[knob {knob}={v}] agree {s['argmax_agreement']:.3f} worst sign err {s['worst_sign_error']:.4f} "
                  f"worst P/R {s['worst_pr']:.3f} state {s['state_err_max']:.4f} pass {s['passes']}", flush=True)
    tol = {}
    for knob, values in KNOBS.items():
        ok = [r["value"] for r in rows if r["knob"] == knob and r["summary"]["passes"]]
        # 档位按"越来越差"排列:容限 = 连续通过的最后一档
        last = None
        for v in values:
            r = next(x for x in rows if x["knob"] == knob and x["value"] == v)
            if not r["summary"]["passes"]:
                break
            last = v
        tol[knob] = dict(tolerance=last, nominal=getattr(cfg, knob), passing=ok)
    return dict(knobs=rows, tolerance=tol)


# ----------------------------------------------------------------------------- 诊断:单步误差、误差随位置的累积、极小扰动

def clone_state(st):
    if torch.is_tensor(st):
        return st.clone()
    if isinstance(st, dict):
        return {k: clone_state(v) for k, v in st.items()}
    if isinstance(st, list):
        return [clone_state(v) for v in st]
    return st


def load_software_state(ckt_st, sw_st):
    """把软件块的状态(膜电位 U、记忆 S)写进行为级仿真器的状态。只适用于 IdealLIFModel(状态只有 U)。
    记忆 S 乘上 gain cell 的保持因子 eps_hold:电路里上一个周期结束时 S 已经乘过一次(end_of_cycle),
    这样单步对照也包含保持损失。"""
    a, f = sw_st["attn"], sw_st["ffn"]
    ca, cf = ckt_st["attn"], ckt_st["ffn"]
    for ck, sk in (("nq", "Uq"), ("nk", "Uk"), ("nv", "Uv"), ("npre", "Upre"), ("nout", "Uout")):
        ca[ck] = dict(U=a[sk].clone())
    for ck, sk in (("ng", "U1"), ("nu", "U3"), ("ny2", "U2")):
        cf[ck] = dict(U=f[sk].clone())
    arr = ca["_arr"]
    arr.reset(a["S"] * arr.eps_hold)
    ca["S"] = arr.S


def _counts(y, T):
    return (y * T).round()


@torch.no_grad()
def one_step_compare(m, ckt, tokens, cfg):
    """单步对照:每个 token 开始前把电路的全部状态与每块的输入换成软件的,只比较这一步的输出,误差不跨 token 累积。
    同一步之内,GDN 子块的误差仍会经 h_mid 传到同一块的 FFN 子块。输出头吃电路最后一块这一步的输出 h_out(其输入已换成软件的),
    比较 argmax;所以输出头一致率反映的是这一步里最后一块与输出头的误差,上一块的误差因为输入重置而看不到。"""
    from snn_spec.metrics import sign_error_rate, spike_prf_by_sign
    n_blocks = len(m.blocks)
    lv = [(int(getattr(b.attn.sn_out, "levels", 1)), int(getattr(b.ffn.sn_2, "levels", 1))) for b in m.blocks]
    acc = [dict(ya_sw=[], ya_hw=[], yf_sw=[], yf_hw=[]) for _ in range(n_blocks)]
    agree = n_pos = 0
    for row in tokens:
        inp = row[row != PAD_ID].unsqueeze(0)[:, :-1]
        st_sw, st_hw = m.init_state(1, DEV), ckt.init_state(1)
        for t in range(inp.shape[1]):
            pre = clone_state(st_sw)
            logits, st_sw, recs = m.step(inp[:, t], st_sw, record=True)
            h_last = None
            for i, sim in enumerate(ckt.sims):
                load_software_state(st_hw[i], pre[i])
                r = recs[i]
                h_last, st_hw[i], h = sim.step(r["h_in"], st_hw[i], ext_gates=(r["alpha"], r["beta"]), record=True)
                acc[i]["ya_sw"].append(r["y_attn"]); acc[i]["ya_hw"].append(h["y_attn"])
                acc[i]["yf_sw"].append(r["y_ffn"]); acc[i]["yf_hw"].append(h["y_ffn"])
            hw_logits = ckt.head_xb(ckt.head_cmp(h_last))
            agree += int((hw_logits.argmax(-1) == logits.argmax(-1)).item())
            n_pos += 1
    out = []
    for i, (a, (Ta, Tf)) in enumerate(zip(acc, lv)):
        for key, sw_k, hw_k, T in (("attn", "ya_sw", "ya_hw", Ta), ("ffn", "yf_sw", "yf_hw", Tf)):
            sw, hw = torch.cat(a[sw_k]), torch.cat(a[hw_k])
            p = spike_prf_by_sign(hw, sw, levels=T)
            out.append(dict(block=i, output=key, levels=T, sign_error=sign_error_rate(hw, sw),
                            pr_min=min(p[k] for k in ("pos_precision", "pos_recall", "neg_precision", "neg_recall")),
                            count_mismatch=(_counts(hw, T) != _counts(sw, T)).float().mean().item()))
    mean = lambda key, part=None: sum(r[key] for r in out if part in (None, r["output"])) / len([r for r in out if part in (None, r["output"])])  # noqa: E731
    return dict(head_argmax_agreement=agree / n_pos, n_positions=n_pos, outputs=out,
                worst_sign_error=max(r["sign_error"] for r in out), worst_pr=min(r["pr_min"] for r in out),
                worst_count_mismatch=max(r["count_mismatch"] for r in out),
                mean_count_mismatch=mean("count_mismatch"),            # 与 position_profile 同口径(四个输出的平均)
                attn_count_mismatch=mean("count_mismatch", "attn"), ffn_count_mismatch=mean("count_mismatch", "ffn"))


POS_BINS = ((1, 3), (4, 6), (7, 10), (11, 15), (16, 25), (26, 41))


@torch.no_grad()
def position_profile(m, ckt, tokens):
    """普通 teacher forcing(状态各自累积)下,每个位置上两块四个输出的脉冲计数不一致的比例,按位置分档。"""
    n_blocks = len(m.blocks)
    lv = [(int(getattr(b.attn.sn_out, "levels", 1)), int(getattr(b.ffn.sn_2, "levels", 1))) for b in m.blocks]
    L = tokens.shape[1]
    mis = torch.zeros(L)
    cnt = torch.zeros(L)
    for row in tokens:
        inp = row[row != PAD_ID].unsqueeze(0)[:, :-1]
        _, sw_recs = m(inp, record=True)
        gates = [(torch.stack([r[i]["alpha"] for r in sw_recs], 1), torch.stack([r[i]["beta"] for r in sw_recs], 1)) for i in range(n_blocks)]
        _, hw_recs = ckt.forward(inp, ext_gates_seq=gates, record=True)
        for t in range(inp.shape[1]):
            vals = []
            for i, (Ta, Tf) in enumerate(lv):
                for key, T in (("y_attn", Ta), ("y_ffn", Tf)):
                    vals.append((_counts(hw_recs[t][i][key], T) != _counts(sw_recs[t][i][key], T)).float().mean().item())
            mis[t] += sum(vals) / len(vals)
            cnt[t] += 1
    return {f"{a}-{b}": round((mis[a - 1:b].sum() / cnt[a - 1:b].sum().clamp(min=1)).item(), 4) for a, b in POS_BINS}


TINY = dict(n_levels=[4096, 1024], sigma_g=[1e-4, 1e-3], read_noise=[1e-4, 1e-3])


def mode_diagnose(m, cfg, sl, demo, tokens, cmp_tokens, info):
    res = dict(one_step=[], tiny_perturbation=[])
    for parts in [PARTS] + [(p,) for p in PARTS]:
        s = one_step_compare(m, CircuitLM(m, cfg, ideal=False, nominal_parts=parts), cmp_tokens, cfg)
        res["one_step"].append(dict(parts=list(parts), **s))
        print(f"[one-step {'+'.join(parts)}] worst sign err {s['worst_sign_error']:.4f} worst P/R {s['worst_pr']:.3f} "
              f"count mismatch {s['worst_count_mismatch']:.4f} head agree {s['head_argmax_agreement']:.3f}", flush=True)
    res["position_profile_nominal"] = position_profile(m, CircuitLM(m, cfg, ideal=False), cmp_tokens)
    print("[position profile, nominal]", res["position_profile_nominal"], flush=True)
    base = dataclasses.replace(cfg, **IDEAL_KNOBS)
    for knob, values in TINY.items():
        for v in values:
            c = dataclasses.replace(base, **{knob: v})
            s = compact(teacher_forced_compare(m, CircuitLM(m, c, ideal=False), cmp_tokens, c))
            one = one_step_compare(m, CircuitLM(m, c, ideal=False), cmp_tokens, c)
            res["tiny_perturbation"].append(dict(knob=knob, value=v, teacher_forced=s, one_step=one))
            print(f"[tiny {knob}={v}] tf worst sign err {s['worst_sign_error']:.4f} agree {s['argmax_agreement']:.3f} | "
                  f"one-step worst sign err {one['worst_sign_error']:.4f} count mismatch {one['worst_count_mismatch']:.5f}", flush=True)
    return res


# ----------------------------------------------------------------------------- 逐项去掉:其余保持标称,每次只关一种非理想性

LEAVE_OUT = [
    ("全标称", {}),
    ("去掉比较器失调", dict(cmp_offset_frac=0.0)),
    ("去掉电导量化", dict(n_levels=0)),
    ("去掉编程误差 σ_G", dict(sigma_g=0.0)),
    ("去掉读噪声", dict(read_noise=0.0)),
    ("去掉 crossbar 的三项误差", dict(n_levels=0, sigma_g=0.0, read_noise=0.0)),
    ("去掉状态阵列误差", dict(write_err=0.0, read_err=0.0, eps_hold=1.0)),
    ("失调改为每个通道独立随机", "random_offset"),
]


def random_offsets(ckt, cfg, seed=1234):
    """把所有比较器共用的失调换成每个通道独立的随机失调(同一通道的 2T 个比较器共用这个值,
    相当于整组阈值一起平移),标准差 = cmp_offset_frac × 阈值,种子固定。"""
    g = torch.Generator().manual_seed(seed)
    for sim in ckt.sims:
        for name in ("cmp1", "cmp2"):
            c = getattr(sim, name)
            c.offset = cfg.cmp_offset_frac * c.thr * torch.randn(sim.block.attn.d, generator=g)
    ckt.head_cmp.offset = cfg.cmp_offset_frac * ckt.head_cmp.thr * torch.randn(ckt.emb.shape[1], generator=g)
    return ckt


def mode_leaveout(m, cfg, sl, demo, tokens, cmp_tokens, info):
    rows = []
    for label, change in LEAVE_OUT:
        t0 = time.time()
        if change == "random_offset":
            make = lambda: random_offsets(CircuitLM(m, cfg, ideal=False), cfg)      # noqa: E731
        else:
            c = dataclasses.replace(cfg, **change)
            make = lambda c=c: CircuitLM(m, c, ideal=False)                         # noqa: E731
        cmp = teacher_forced_compare(m, make(), cmp_tokens, cfg)
        ev = evaluate(make(), sl, demo, cfg, tokens)
        row = dict(label=label, change=change, summary=compact(cmp), token_acc=ev["token_acc"], n_exact=ev["n_exact"],
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        s = row["summary"]
        print(f"[leave-out {label}] agree {s['argmax_agreement']:.3f} worst sign err {s['worst_sign_error']:.4f} worst P/R {s['worst_pr']:.3f} "
              f"acc {ev['token_acc']:.4f} exact {ev['n_exact']}", flush=True)
    return dict(leaveout=rows)


NOISE_SEEDS = (0, 1, 2, 3, 4)


def mode_seeds(m, cfg, sl, demo, tokens, cmp_tokens, info):
    """标称电路换器件噪声的种子(CircuitLM 的 seed:电导编程误差、读噪声、gain cell 误差的抽样),看结果随一次抽样变化多大。"""
    rows = []
    for s in NOISE_SEEDS:
        cmp = compact(teacher_forced_compare(m, CircuitLM(m, cfg, ideal=False, seed=s), cmp_tokens, cfg))
        ev = evaluate(CircuitLM(m, cfg, ideal=False, seed=s), sl, demo, cfg, tokens)
        rows.append(dict(seed=s, summary=cmp, token_acc=ev["token_acc"], ppl=ev["ppl"], n_exact=ev["n_exact"]))
        print(f"[noise seed {s}] agree {cmp['argmax_agreement']:.3f} worst sign err {cmp['worst_sign_error']:.4f} "
              f"acc {ev['token_acc']:.4f} exact {ev['n_exact']}", flush=True)
    accs = [r["token_acc"] for r in rows]
    mean = sum(accs) / len(accs)
    std = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
    return dict(noise_seeds=rows, acc_mean=mean, acc_std=std, acc_min=min(accs), acc_max=max(accs),
                n_exact=[r["n_exact"] for r in rows])


MODES = dict(accept=("circuit.json", mode_accept), isolate=("circuit_isolate.json", mode_isolate), knobs=("circuit_knobs.json", mode_knobs),
             diagnose=("circuit_diagnose.json", mode_diagnose), leaveout=("circuit_leaveout.json", mode_leaveout),
             seeds=("circuit_seeds.json", mode_seeds))


# ----------------------------------------------------------------------------- 报告

def mark(ok):
    return "通过" if ok else "不通过"


def write_md(out, name):
    lines = [f"# 电路验收:{name}", ""]
    acc = json.loads((out / "circuit.json").read_text()) if (out / "circuit.json").exists() else None
    iso = json.loads((out / "circuit_isolate.json").read_text()) if (out / "circuit_isolate.json").exists() else None
    kn = json.loads((out / "circuit_knobs.json").read_text()) if (out / "circuit_knobs.json").exists() else None
    dg = json.loads((out / "circuit_diagnose.json").read_text()) if (out / "circuit_diagnose.json").exists() else None
    lo = json.loads((out / "circuit_leaveout.json").read_text()) if (out / "circuit_leaveout.json").exists() else None
    ns = json.loads((out / "circuit_seeds.json").read_text()) if (out / "circuit_seeds.json").exists() else None
    meta = (acc or iso or kn or dg or lo or ns)["meta"]
    lines += [f"模型:`{meta['source']}`,每 token 最多脉冲数 {meta['levels_summary']};对照集 {meta['n_compare']} 句;"
              f"torch {meta['torch']},{meta['device']},单线程。", ""]
    if acc:
        sw, ci, cn = acc["software"], acc["circuit_ideal"], acc["circuit_nominal"]
        ci_c, cn_c = compact(ci["compare"]), compact(cn["compare"])
        lines += ["## 验收表", "", "| 项 | 判据 | 结果 | 结论 |", "|---|---|---|---|",
                  f"| 软件模型 | 参考 | acc {sw['token_acc']:.4f},{sw['n_exact']}/{sw['n_demo']} | — |",
                  f"| 理想电路:输出与软件相同 | 对照集 logits 差 ≈ 0,逐 token argmax 相同 | logits 最大差 {ci['logits_max_abs_diff']:.2e},"
                  f"argmax 相同 {ci['argmax_equal_to_software']} | {mark(ci['argmax_equal_to_software'])} |",
                  f"| 理想电路:逐脉冲对照 | P/R = 1、符号错误率 0、状态误差 ≈ 0 | P/R 最小 {ci_c['worst_pr']:.3f},符号错误率最大 "
                  f"{ci_c['worst_sign_error']:.4f},状态误差最大 {ci_c['state_err_max']:.2e} | {mark(ci['compare']['pass'] and ci_c['worst_pr'] == 1.0)} |",
                  f"| 理想电路:自由运行 | 与软件相同 | acc {ci['token_acc']:.4f},{ci['n_exact']}/{ci['n_demo']} | "
                  f"{mark(abs(ci['token_acc'] - sw['token_acc']) < 1e-9 and ci['n_exact'] == sw['n_exact'])} |",
                  f"| 标称电路:自由运行 5 句全对 | 5/5 | acc {cn['token_acc']:.4f},{cn['n_exact']}/{cn['n_demo']} | {mark(cn['n_exact'] == cn['n_demo'])} |",
                  f"| 标称电路:argmax 一致率 | ≥ 0.99 | {cn_c['argmax_agreement']:.3f} | {mark(cn_c['argmax_agreement'] >= 0.99)} |",
                  f"| 标称电路:状态误差最大值 | ≤ 0.1 | {cn_c['state_err_max']:.4f} | {mark(cn_c['state_err_max'] <= 0.1)} |"]
        for r in cn_c["outputs"]:
            lines.append(f"| 标称电路:块 {r['block'] + 1} {'注意力' if r['output'] == 'attn' else '前馈'}输出(T = {r['levels']}) | "
                         f"P/R ≥ 0.9,符号错误率 ≤ 2% | P/R 最小 {r['pr_min']:.3f},符号错误率 {100 * r['sign_error']:.2f}% | "
                         f"{mark(r['pr_min'] >= 0.9 and r['sign_error'] <= 0.02)} |")
        lines += ["", "σ_G 扫描(标称电路其余参数不变):", "", "| σ_G | acc | ppl | 5 句 |", "|---|---|---|---|"]
        for r in acc["sigma_g_sweep"]:
            lines.append(f"| {r['sigma_g']} | {r['token_acc']:.4f} | {r['ppl']:.3f} | {r['n_exact']}/5 |")
        sp = acc["spikes"]
        lines += ["", f"每 token 脉冲:整个模型 {sp['total_spikes_per_token']},W2 收到的事件 {sp['w2_events_per_token']}。", ""]
    if iso:
        lines += ["## 误差隔离(只让列出的部分取标称值,其余理想)", "",
                  "| 标称部分 | argmax 一致率 | 最差符号错误率 | 最差 P/R | 状态误差最大 | 自由运行 acc | 5 句 | 判据 |", "|---|---|---|---|---|---|---|---|"]
        for r in iso["isolation"]:
            s = r["summary"]
            lines.append(f"| {' + '.join(r['parts'])} | {s['argmax_agreement']:.3f} | {100 * s['worst_sign_error']:.2f}% | {s['worst_pr']:.3f} | "
                         f"{s['state_err_max']:.4f} | {r['token_acc']:.4f} | {r['n_exact']}/5 | {mark(s['passes'])} |")
        lines.append("")
    if kn:
        lines += ["## 容限(从理想电路出发,一次只开一种非理想性)", "",
                  "| 参数 | 标称值 | 仍满足判据的最差档 | 各档最差符号错误率 |", "|---|---|---|---|"]
        for knob in KNOBS:
            rows = [r for r in kn["knobs"] if r["knob"] == knob]
            t = kn["tolerance"][knob]
            per = ",".join(f"{r['value']}: {100 * r['summary']['worst_sign_error']:.2f}%" for r in rows)
            lines.append(f"| {knob} | {t['nominal']} | {t['tolerance'] if t['tolerance'] is not None else '最轻一档也不过'} | {per} |")
        lines.append("")
    if dg:
        lines += ["## 误差来源诊断", "", "单步对照:每个 token 开始前把电路的膜电位、记忆 S 与每块的输入换成软件的,只比这一步,误差不跨 token 累积"
                  "(同一步里 GDN 的误差仍会传到同一块的 FFN)。脉冲计数不一致取四个输出的平均,与下面按位置的统计同口径。", "",
                  "| 标称部分 | 计数不一致:平均 | 注意力输出 | 前馈输出 | 最差符号错误率 | 最差 P/R | 输出头 argmax 一致 |",
                  "|---|---|---|---|---|---|---|"]
        for r in dg["one_step"]:
            lines.append(f"| {' + '.join(r['parts'])} | {100 * r['mean_count_mismatch']:.2f}% | {100 * r['attn_count_mismatch']:.2f}% | "
                         f"{100 * r['ffn_count_mismatch']:.2f}% | {100 * r['worst_sign_error']:.2f}% | {r['worst_pr']:.3f} | {r['head_argmax_agreement']:.3f} |")
        lines += ["", "全标称电路的普通 teacher forcing 里,四个输出脉冲计数不一致的比例(平均)随句中位置的变化:", "",
                  "| 位置 | " + " | ".join(dg["position_profile_nominal"]) + " |", "|---|" + "---|" * len(dg["position_profile_nominal"]),
                  "| 不一致 | " + " | ".join(f"{100 * v:.2f}%" for v in dg["position_profile_nominal"].values()) + " |", "",
                  "极小扰动(其余理想):", "", "| 参数 | 值 | 逐脉冲对照:最差符号错误率 | argmax 一致率 | 单步:最差符号错误率 | 单步:计数不一致 |",
                  "|---|---|---|---|---|---|"]
        for r in dg["tiny_perturbation"]:
            lines.append(f"| {r['knob']} | {r['value']} | {100 * r['teacher_forced']['worst_sign_error']:.2f}% | {r['teacher_forced']['argmax_agreement']:.3f} | "
                         f"{100 * r['one_step']['worst_sign_error']:.2f}% | {100 * r['one_step']['mean_count_mismatch']:.4f}% |")
        lines.append("")
    if lo:
        lines += ["## 逐项去掉(其余保持标称)", "",
                  "| 配置 | argmax 一致率 | 最差符号错误率 | 最差 P/R | 自由运行 acc | 5 句 | 判据 |", "|---|---|---|---|---|---|---|"]
        for r in lo["leaveout"]:
            s = r["summary"]
            lines.append(f"| {r['label']} | {s['argmax_agreement']:.3f} | {100 * s['worst_sign_error']:.2f}% | {s['worst_pr']:.3f} | "
                         f"{r['token_acc']:.4f} | {r['n_exact']}/5 | {mark(s['passes'])} |")
        lines.append("")
    if ns:
        lines += ["## 标称电路换器件噪声种子", "", "| 噪声种子 | argmax 一致率 | 最差符号错误率 | 自由运行 acc | 5 句 |", "|---|---|---|---|---|"]
        for r in ns["noise_seeds"]:
            s = r["summary"]
            lines.append(f"| {r['seed']} | {s['argmax_agreement']:.3f} | {100 * s['worst_sign_error']:.2f}% | {r['token_acc']:.4f} | {r['n_exact']}/5 |")
        lines += ["", f"自由运行 acc 均值 {ns['acc_mean']:.4f},标准差 {ns['acc_std']:.4f},范围 {ns['acc_min']:.4f} 到 {ns['acc_max']:.4f}。", ""]
    (out / "circuit.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="运行目录(含 model.pt 与 result.json)或检查点路径")
    g.add_argument("--tiny", action="store_true", help="合成切片上的 T = 2 小模型")
    ap.add_argument("--mode", default="accept", choices=list(MODES))
    ap.add_argument("--out", default=None, help="输出目录,默认是运行目录")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    m, cfg, sl, demo, tokens, cmp_tokens, info, out, name = load(args)
    lv = sorted({int(getattr(mod, "levels", 1)) for mod in m.modules() if hasattr(mod, "levels")} or {1})
    meta = dict(source=str(args.run) if args.run else "tiny", info=info, torch=torch.__version__, device=str(DEV),
                n_compare=int(cmp_tokens.shape[0]), levels_summary="/".join(map(str, lv)), cfg=cfg.as_dict(), mode=args.mode)
    t0 = time.time()
    fname, fn = MODES[args.mode]
    res = fn(m, cfg, sl, demo, tokens, cmp_tokens, info)
    meta["seconds"] = round(time.time() - t0, 1)
    res["meta"] = meta
    (out / fname).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    write_md(out, name)
    print(f"wrote {out / fname} and {out / 'circuit.md'} ({meta['seconds']} s)")
    return res


if __name__ == "__main__":
    main()
