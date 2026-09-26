"""training-dynamics diagnostics for the default run (A0 -> A1 -> B -> C).

Subcommands (each short enough for one Bash call):
  curves    (a) parse results/run-default.log, fit acc(step) = a - b*exp(-step/tau) per seed, slopes, LR-vs-progress
  position  (b) teacher-forced token accuracy vs target position for A0 / A1 / C
  demo      (c) greedy completions of the 5 demo prefixes, leading-token match count
  ce        (d) CE vs accuracy consistency (CE on correct/incorrect tokens, entropy, A0-agreement)
  continue  extra: continue A1 (leak 0, a1.pt) for N steps at constant LR to test "LR decay explains flattening"
             usage: continue <lr> <steps> <schedule> <name>
Outputs go to results/conversion-sweep/diag/training-dynamics/.
"""
import ast
import json
import math
import os
import sys

STEP2 = '/data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo'
sys.path.insert(0, STEP2)
sys.path.insert(0, STEP2 + '/experiments')
os.chdir(STEP2)
import convert_lab as cl  # noqa: E402  (chdirs to step2 and puts toy_demo on path)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from toy_demo.data import decode, select_demo, split_inputs_targets  # noqa: E402
from toy_demo.model import SpikingLM  # noqa: E402
from toy_demo.train import evaluate  # noqa: E402

OUT = 'results/conversion-sweep/diag/training-dynamics'
os.makedirs(OUT, exist_ok=True)
LOG = 'results/run-default.log'


def dump(name, obj):
    with open(f'{OUT}/{name}.json', 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------- (a) curves

def parse_log():
    lines = open(LOG, encoding='utf-8').read().splitlines()
    stage_at = {}
    for i, ln in enumerate(lines):
        if ln.startswith('['):
            stage_at.setdefault(ln[1:ln.index(']')], i)

    def seg(lo, hi):
        rows = [ast.literal_eval(ln) for ln in lines[lo + 1:hi] if ln.startswith('{')]
        runs, cur = [], []
        for r in rows:
            if cur and r['step'] <= cur[-1]['step']:
                runs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            runs.append(cur)
        return runs
    a0 = seg(-1, stage_at['A0'])
    a1 = seg(stage_at['A1_init'], stage_at['A1'])
    c = seg(stage_at['B'], stage_at['C'])
    return a0, a1, c


def lr_cos(step, base, T):
    return base * 0.5 * (1 + math.cos(math.pi * min(step / T, 1.0)))


def lr_integral(step, base, T):
    # integral_0^step lr_cos = base*0.5*(s + T/pi * sin(pi s/T))
    s = min(step, T)
    return base * 0.5 * (s + T / math.pi * math.sin(math.pi * s / T))


def fit_sat(x, y):
    """y(x) = a - b*exp(-x/tau); grid over tau, closed-form (a, b) by least squares. Returns dict."""
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    best = None
    xmax = float(x.max())
    for tau in torch.logspace(math.log10(xmax / 50), math.log10(xmax * 20), 800):
        tau = float(tau)
        e = torch.exp(-x / tau)
        A = torch.stack([torch.ones_like(x), -e], 1)
        sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
        res = float(((A @ sol - y) ** 2).sum())
        if best is None or res < best['sse']:
            best = dict(a=float(sol[0]), b=float(sol[1]), tau=tau, sse=res)
    best['rmse'] = math.sqrt(best['sse'] / len(x))
    return best


def slope(rows, key, lo, hi):
    """least-squares slope of key vs step over rows with lo <= step <= hi (per 100 steps)."""
    pts = [(r['step'], r[key]) for r in rows if lo <= r['step'] <= hi]
    xs = torch.tensor([p[0] for p in pts], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in pts], dtype=torch.float64)
    xm, ym = xs.mean(), ys.mean()
    return float(((xs - xm) * (ys - ym)).sum() / ((xs - xm) ** 2).sum()) * 100.0


def steps_to(target, f):
    if f['a'] <= target:
        return None
    return -f['tau'] * math.log((f['a'] - target) / f['b'])


def cmd_curves():
    a0, a1, c = parse_log()
    out = dict(n_a0_runs=len(a0), n_a1_runs=len(a1), n_c_runs=len(c),
               a1_len=[len(r) for r in a1], c_len=[len(r) for r in c])
    print(f'A0 runs {len(a0)} (len {[len(r) for r in a0]}), A1 runs {len(a1)} (len {out["a1_len"]}), C runs {len(c)} (len {out["c_len"]})')
    for stage, runs, base, T in (('A1', a1, 3e-3, 2000), ('C', c, 1e-3, 1500)):
        out[stage] = []
        for si, rows in enumerate(runs):
            steps = [r['step'] for r in rows]
            acc = [r['token_acc'] for r in rows]
            lnppl = [math.log(r['ppl']) for r in rows]
            for r, lp in zip(rows, lnppl):
                r['lr'] = lr_cos(r['step'], base, T)
                r['lr_int'] = lr_integral(r['step'], base, T)
                r['lnppl'] = lp
            f_step = fit_sat(steps, acc)                       # acc vs step
            f_lam = fit_sat([r['lr_int'] for r in rows], acc)  # acc vs cumulative LR
            f_ppl = fit_sat(steps, lnppl)
            mid = slope(rows, 'token_acc', T // 2 - 250, T // 2 + 250)
            last = slope(rows, 'token_acc', T - 500, T)
            first = slope(rows, 'token_acc', 100, 600)
            lmid = slope(rows, 'lnppl', T // 2 - 250, T // 2 + 250)
            llast = slope(rows, 'lnppl', T - 500, T)
            per_lr = []
            for p, q in zip(rows[:-1], rows[1:]):
                dl = q['lr_int'] - p['lr_int']
                da = q['token_acc'] - p['token_acc']
                dp = q['lnppl'] - p['lnppl']
                per_lr.append(dict(step=q['step'], d_acc=da, d_lnppl=dp, d_lr_int=dl, lr=q['lr'],
                                   acc_per_lr=da / dl if dl > 1e-9 else None,
                                   lnppl_per_lr=dp / dl if dl > 1e-9 else None))
            # linear fit acc = c0 + c1 * cumLR (if progress per step were proportional to LR this would be a straight line)
            lam = torch.tensor([r['lr_int'] for r in rows], dtype=torch.float64)
            accT = torch.tensor(acc, dtype=torch.float64)
            A = torch.stack([torch.ones_like(lam), lam], 1)
            c = torch.linalg.lstsq(A, accT[:, None]).solution[:, 0]
            lin_lam = dict(c0=float(c[0]), c1=float(c[1]), rmse=float(((A @ c - accT) ** 2).mean().sqrt()))
            thirds = [w['acc_per_lr'] for w in per_lr if w['acc_per_lr'] is not None]
            k3 = len(thirds) // 3
            per_lr_thirds = [sum(thirds[:k3]) / k3, sum(thirds[k3:2 * k3]) / k3, sum(thirds[2 * k3:3 * k3]) / k3]
            print(f'   linear acc vs cumLR: c0={lin_lam["c0"]:.4f} c1={lin_lam["c1"]:.4f}/unit-LR rmse={lin_lam["rmse"]:.4f}; '
                  f'mean acc-gain per unit cumLR by thirds of run: {[round(v, 3) for v in per_lr_thirds]}')
            rec = dict(seed=si, final_acc=acc[-1], final_ppl=rows[-1]['ppl'], final_lnppl=lnppl[-1],
                       fit_step=f_step, fit_lrint=f_lam, fit_lnppl_step=f_ppl, linear_acc_vs_cumlr=lin_lam,
                       acc_per_cumlr_thirds=per_lr_thirds,
                       slope_acc_first_100_600=first, slope_acc_mid=mid, slope_acc_last500=last,
                       slope_lnppl_mid=lmid, slope_lnppl_last500=llast,
                       steps_to_0842_stepfit=steps_to(0.842, f_step),
                       steps_to_0511_stepfit=steps_to(0.511, f_step),
                       lrint_to_0842=steps_to(0.842, f_lam),
                       per_window=per_lr, rows=rows)
            out[stage].append(rec)
            print(f'[{stage} seed{si}] final acc {acc[-1]:.4f} ppl {rows[-1]["ppl"]:.3f} | fit acc=a-b*exp(-step/tau): '
                  f'a={f_step["a"]:.4f} b={f_step["b"]:.4f} tau={f_step["tau"]:.0f} rmse={f_step["rmse"]:.4f} | '
                  f'fit vs cumLR: a={f_lam["a"]:.4f} tau_lam={f_lam["tau"]:.3f} rmse={f_lam["rmse"]:.4f} | '
                  f'lnppl fit a={f_ppl["a"]:.4f} tau={f_ppl["tau"]:.0f} | '
                  f'slope acc/100: first {first:.4f} mid {mid:.4f} last500 {last:.4f} | '
                  f'lnppl slope/100: mid {lmid:.4f} last {llast:.4f} | steps->0.842 {rec["steps_to_0842_stepfit"]} ->0.511 {rec["steps_to_0511_stepfit"]}')
            print('   window(step, lr, d_acc, d_lnppl, d_lrint, acc_per_lr):')
            for w in per_lr:
                apl = None if w["acc_per_lr"] is None else round(w["acc_per_lr"], 3)
                print(f'     {w["step"]:5d} lr={w["lr"]:.2e} dacc={w["d_acc"]:+.4f} dlnppl={w["d_lnppl"]:+.4f} dLam={w["d_lr_int"]:.4f} acc/Lam={apl}')
    out['A0'] = []
    for si, rows in enumerate(a0):
        steps = [r['step'] for r in rows]
        acc = [r['token_acc'] for r in rows]
        f = fit_sat(steps, acc)
        out['A0'].append(dict(seed=si, final_acc=acc[-1], final_ppl=rows[-1]['ppl'], n=len(rows), last_step=steps[-1], fit_step=f,
                              slope_last500=slope(rows, 'token_acc', steps[-1] - 500, steps[-1]), rows=rows))
        print(f'[A0 seed{si}] steps {steps[-1]} final acc {acc[-1]:.4f} ppl {rows[-1]["ppl"]:.3f} fit a={f["a"]:.4f} b={f["b"]:.4f} tau={f["tau"]:.0f} rmse={f["rmse"]:.4f}')
    dump('curves', out)


# ----------------------------------------------------------------------------- model loading

def load_models(dev, which=('A0', 'A1', 'C')):
    cfg, sl, demo, tokens = cl.load_env()
    models = {}
    if 'A0' in which:
        models['A0'] = cl.load_a0(cfg, sl, dev)
    if 'A1' in which:
        m = SpikingLM(cfg, sl.vocab_size, leak=0.0).to(dev)
        m.load_state_dict(torch.load('results/ckpt/a1.pt', map_location=dev))
        models['A1'] = m.eval()
    if 'C' in which:
        m = SpikingLM(cfg, sl.vocab_size, leak=cfg.leak).to(dev)
        m.load_state_dict(torch.load('results/ckpt/c.pt', map_location=dev))
        models['C'] = m.eval()
    return cfg, sl, demo, tokens, models


# ----------------------------------------------------------------------------- (b) position

@torch.no_grad()
def cmd_position():
    dev = cl.get_device()
    cfg, sl, demo, tokens, models = load_models(dev)
    L = tokens.shape[1] - 1
    out = {}
    correct = {k: torch.zeros(L, dtype=torch.long) for k in models}
    count = torch.zeros(L, dtype=torch.long)
    nll = {k: torch.zeros(L, dtype=torch.float64) for k in models}
    agree = {k: torch.zeros(L, dtype=torch.long) for k in models}
    for i in range(0, tokens.shape[0], 256):
        inputs, targets, mask = split_inputs_targets(tokens[i:i + 256].to(dev))
        count += mask.sum(0).cpu()
        preds = {}
        for k, m in models.items():
            logits = m(inputs)
            preds[k] = logits.argmax(-1)
            correct[k] += ((preds[k] == targets) & mask).sum(0).cpu()
            ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none').view_as(targets)
            nll[k] += (ce * mask).sum(0).double().cpu()
        for k in models:
            agree[k] += ((preds[k] == preds['A0']) & mask).sum(0).cpu()
    pos = list(range(1, L + 1))
    out['n_per_position'] = count.tolist()
    for k in models:
        out[f'acc_{k}'] = (correct[k].double() / count.clamp(min=1)).tolist()
        out[f'ce_{k}'] = (nll[k] / count.clamp(min=1)).tolist()
        out[f'agree_A0_{k}'] = (agree[k].double() / count.clamp(min=1)).tolist()
        out[f'overall_{k}'] = dict(acc=float(correct[k].sum()) / float(count.sum()), ce=float(nll[k].sum()) / float(count.sum()))
    bins = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 35), (36, L)]
    out['bins'] = []
    print('position-binned teacher-forced accuracy (full slice, n tokens per bin):')
    print('  bin       n      A0      A1      C     A1/A0   C/A0   C/A1  agree(A1,A0) agree(C,A0)  CE_A0  CE_A1  CE_C')
    for lo, hi in bins:
        sl_ = slice(lo - 1, hi)
        n = int(count[sl_].sum())
        row = dict(lo=lo, hi=hi, n=n)
        for k in models:
            row[f'acc_{k}'] = float(correct[k][sl_].sum()) / max(1, n)
            row[f'agree_{k}'] = float(agree[k][sl_].sum()) / max(1, n)
            row[f'ce_{k}'] = float(nll[k][sl_].sum()) / max(1, n)
        out['bins'].append(row)
        print(f'  {lo:2d}-{hi:2d}  {n:6d}  {row["acc_A0"]:.4f}  {row["acc_A1"]:.4f}  {row["acc_C"]:.4f}   '
              f'{row["acc_A1"] / row["acc_A0"]:.3f}  {row["acc_C"] / row["acc_A0"]:.3f}  {row["acc_C"] / row["acc_A1"]:.3f}   '
              f'{row["agree_A1"]:.3f}        {row["agree_C"]:.3f}      {row["ce_A0"]:.3f}  {row["ce_A1"]:.3f}  {row["ce_C"]:.3f}')
    print('per-position (pos, n, accA0, accA1, accC, agreeA1, agreeC):')
    for p in pos:
        print(f'  {p:2d} {int(count[p - 1]):5d} {out["acc_A0"][p - 1]:.4f} {out["acc_A1"][p - 1]:.4f} {out["acc_C"][p - 1]:.4f} '
              f'{out["agree_A0_A1"][p - 1]:.4f} {out["agree_A0_C"][p - 1]:.4f}')

    def lin(xs, ys):
        xs = torch.tensor(xs, dtype=torch.float64)
        ys = torch.tensor(ys, dtype=torch.float64)
        return float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    P = 30
    for k in ('A1', 'C'):
        ratio = [out[f'acc_{k}'][p] / out['acc_A0'][p] for p in range(P)]
        out[f'slope_ratio_{k}_per_pos_1_30'] = lin(list(range(1, P + 1)), ratio)
        out[f'slope_acc_{k}_per_pos_1_30'] = lin(list(range(1, P + 1)), out[f'acc_{k}'][:P])
    out['slope_acc_A0_per_pos_1_30'] = lin(list(range(1, P + 1)), out['acc_A0'][:P])
    for k in ('A0', 'A1', 'C'):
        print(f'slope of acc vs position (1..30) {k}: {out[f"slope_acc_{k}_per_pos_1_30"]:+.5f} per position')
    for k in ('A1', 'C'):
        print(f'slope of acc_{k}/acc_A0 vs position (1..30): {out[f"slope_ratio_{k}_per_pos_1_30"]:+.5f} per position')
    for k in models:
        print(f'overall {k}: acc {out[f"overall_{k}"]["acc"]:.4f} ce {out[f"overall_{k}"]["ce"]:.4f} (ppl {math.exp(out[f"overall_{k}"]["ce"]):.3f})')
    dump('position', out)


# ----------------------------------------------------------------------------- (c) demo

@torch.no_grad()
def cmd_demo():
    dev = cl.get_device()
    cfg, sl, demo, tokens, models = load_models(dev)
    demo = select_demo(sl, cfg)
    out = []
    for j, item in enumerate(demo):
        rec = dict(index=item['index'], prefix=decode(sl, item['prefix']), target=decode(sl, item['target']), target_len=len(item['target']))
        print(f'demo {j} [{item["index"]}] prefix="{rec["prefix"]}" target="{rec["target"]}" ({rec["target_len"]} tokens)')
        for k, m in models.items():
            ids = m.generate(item['prefix'], cfg.max_new_tokens)
            lead = 0
            for a, b in zip(ids, item['target']):
                if a != b:
                    break
                lead += 1
            seq = torch.tensor([[0] + list(item['prefix']) + list(item['target'])], device=dev)
            logits = m(seq[:, :-1])
            pred = logits[0].argmax(-1).tolist()
            tf_correct = sum(int(pred[len(item['prefix']) + t] == item['target'][t]) for t in range(len(item['target'])))
            rec[k] = dict(output=decode(sl, ids), n_out=len(ids), leading_match=lead, exact=(ids == list(item['target'])),
                          tf_correct=tf_correct)
            print(f'   {k}: lead {lead}/{rec["target_len"]} tf-correct {tf_correct}/{rec["target_len"]} exact {rec[k]["exact"]} out="{rec[k]["output"]}"')
        out.append(rec)
    tot = sum(r['target_len'] for r in out)
    for k in models:
        print(f'{k}: exact {sum(r[k]["exact"] for r in out)}/5, total leading match {sum(r[k]["leading_match"] for r in out)}/{tot}, '
              f'tf-correct {sum(r[k]["tf_correct"] for r in out)}/{tot}')
    dump('demo', out)


# ----------------------------------------------------------------------------- (d) CE vs acc

@torch.no_grad()
def cmd_ce():
    dev = cl.get_device()
    cfg, sl, demo, tokens, models = load_models(dev)
    st = {k: dict(n=0, correct=0, nll=0.0, nll_correct=0.0, nll_wrong=0.0, ent=0.0, p_target=0.0, top1_p=0.0,
                  rank_hist=[0] * 6, agree_a0=0, ok_and_a0ok=0, ok_and_a0wrong=0, kl_a0=0.0) for k in models}
    n_a0_correct = 0
    for i in range(0, tokens.shape[0], 256):
        inputs, targets, mask = split_inputs_targets(tokens[i:i + 256].to(dev))
        logits = {k: m(inputs)[mask] for k, m in models.items()}
        tg = targets[mask]
        a0_pred = logits['A0'].argmax(-1)
        a0_ok = a0_pred == tg
        n_a0_correct += int(a0_ok.sum())
        lp0 = F.log_softmax(logits['A0'], -1)
        for k, lg in logits.items():
            s = st[k]
            lp = F.log_softmax(lg, -1)
            nll = -lp.gather(1, tg[:, None])[:, 0]
            pred = lg.argmax(-1)
            ok = pred == tg
            s['n'] += int(tg.numel())
            s['correct'] += int(ok.sum())
            s['nll'] += float(nll.sum())
            s['nll_correct'] += float(nll[ok].sum())
            s['nll_wrong'] += float(nll[~ok].sum())
            s['ent'] += float(-(lp.exp() * lp).sum(-1).sum())
            s['p_target'] += float(lp.gather(1, tg[:, None]).exp().sum())
            s['top1_p'] += float(lp.max(-1).values.exp().sum())
            rank = (lp > lp.gather(1, tg[:, None])).sum(-1)     # 0 = top1
            for r in range(5):
                s['rank_hist'][r] += int((rank == r).sum())
            s['rank_hist'][5] += int((rank >= 5).sum())
            s['agree_a0'] += int((pred == a0_pred).sum())
            s['ok_and_a0ok'] += int((ok & a0_ok).sum())
            s['ok_and_a0wrong'] += int((ok & ~a0_ok).sum())
            s['kl_a0'] += float((lp0.exp() * (lp0 - lp)).sum(-1).sum())
    out = {}
    for k, s in st.items():
        n = s['n']
        c = s['correct']
        out[k] = dict(n=n, acc=c / n, ce=s['nll'] / n, ppl=math.exp(s['nll'] / n), ce_on_correct=s['nll_correct'] / max(1, c),
                      ce_on_wrong=s['nll_wrong'] / max(1, n - c), mean_entropy=s['ent'] / n, mean_p_target=s['p_target'] / n,
                      mean_top1_p=s['top1_p'] / n, rank_frac=[v / n for v in s['rank_hist']],
                      agree_with_a0=s['agree_a0'] / n, p_correct_given_a0_correct=s['ok_and_a0ok'] / n_a0_correct,
                      p_correct_given_a0_wrong=s['ok_and_a0wrong'] / (n - n_a0_correct), kl_a0_to_model=s['kl_a0'] / n)
        o = out[k]
        print(f'{k}: n {n} acc {o["acc"]:.4f} CE {o["ce"]:.4f} (ppl {o["ppl"]:.3f}) CE|correct {o["ce_on_correct"]:.4f} CE|wrong {o["ce_on_wrong"]:.4f} '
              f'H(pred) {o["mean_entropy"]:.4f} mean p_target {o["mean_p_target"]:.4f} mean top1 p {o["mean_top1_p"]:.4f} '
              f'rank frac top1..top5,>=5 {[round(v, 4) for v in o["rank_frac"]]} agree(A0) {o["agree_with_a0"]:.4f} '
              f'P(ok|A0 ok) {o["p_correct_given_a0_correct"]:.4f} P(ok|A0 wrong) {o["p_correct_given_a0_wrong"]:.4f} KL(A0||model) {o["kl_a0_to_model"]:.4f}')
    dump('ce', out)


# ----------------------------------------------------------------------------- extra: continue A1 at const LR

def cmd_continue(lr, steps, schedule, name, seed=0):
    torch.set_num_threads(1)
    dev = cl.get_device()
    cfg, sl, demo, tokens, models = load_models(dev, which=('A1',))
    m = models['A1']
    init = evaluate(m, sl, demo, cfg, tokens)
    print(f'[{name}] start acc {init["token_acc"]:.4f} ppl {init["ppl"]:.3f} exact {init["n_exact"]}/5', flush=True)
    hist = f'{OUT}/{name}.history.jsonl'
    if os.path.exists(hist):
        os.remove(hist)
    res = cl.train(m, sl, demo, cfg, tokens, lr=lr, steps=steps, seed=seed, schedule=schedule, warmup=0,
                   eval_every=50, stop=False, hist_path=hist)
    f = res['final']
    print(f'[{name}] lr {lr} {schedule} {steps} steps: acc {init["token_acc"]:.4f} -> {f["token_acc"]:.4f} ppl {init["ppl"]:.3f} -> {f["ppl"]:.3f} exact {f["n_exact"]}/5')
    dump(name, dict(lr=lr, steps=steps, schedule=schedule, init={k: init[k] for k in ('token_acc', 'ppl', 'n_exact')},
                    history=res['history'], final={k: f[k] for k in ('token_acc', 'ppl', 'n_exact')},
                    completions=f['completions']))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'curves':
        cmd_curves()
    elif cmd == 'position':
        cmd_position()
    elif cmd == 'demo':
        cmd_demo()
    elif cmd == 'ce':
        cmd_ce()
    elif cmd == 'continue':
        cmd_continue(float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        raise SystemExit(f'unknown {cmd}')
