"""训练循环与统一评估函数(计划第 5 节)。所有阶段与电路模型都用 evaluate()。"""
import math
import time

import torch
import torch.nn.functional as F

from .data import all_sequences, batches, decode, split_inputs_targets


@torch.no_grad()
def teacher_forced_metrics(model, tokens, chunk=256):
    """切片上的 token 准确率与困惑度。tokens [N, L] 含 BOS/EOS/PAD。"""
    n_correct = n_tok = 0
    nll = 0.0
    for i in range(0, tokens.shape[0], chunk):
        inputs, targets, mask = split_inputs_targets(tokens[i:i + chunk].to(model.device))
        logits = model(inputs)
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        nll += ce[mask].sum().item()
        n_correct += (logits.argmax(-1) == targets)[mask].sum().item()
        n_tok += mask.sum().item()
    return dict(token_acc=n_correct / n_tok, ppl=math.exp(nll / n_tok), n_tokens=n_tok)


@torch.no_grad()
def demo_completions(model, sl, demo, max_new):
    rows = []
    for item in demo:
        out = model.generate(item["prefix"], max_new)
        rows.append(dict(prefix=decode(sl, item["prefix"]), target=decode(sl, item["target"]),
                         output=decode(sl, out), exact=(out == list(item["target"]))))
    return rows


@torch.no_grad()
def evaluate(model, sl, demo, cfg, tokens=None):
    """统一评估:切片 teacher forcing 准确率/困惑度 + demo 前缀的贪心补全。"""
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    tokens = all_sequences(sl) if tokens is None else tokens
    res = teacher_forced_metrics(model, tokens)
    rows = demo_completions(model, sl, demo, cfg.max_new_tokens)
    res.update(completions=rows, n_exact=sum(r["exact"] for r in rows), n_demo=len(rows))
    if was_training:
        model.train()
    return res


def _should_stop(history, cfg):
    """5 句全对,且准确率相对 plateau_steps 步之前的最好值提升不到 plateau_delta。"""
    last = history[-1]
    if last["n_exact"] < cfg.n_demo:
        return False
    older = [h["token_acc"] for h in history if h["step"] <= last["step"] - cfg.plateau_steps]
    return bool(older) and last["token_acc"] - max(older) < cfg.plateau_delta


def train_stage(model, sl, demo, cfg, lr, max_steps, seed, log=print):
    """Adam + 余弦衰减 + 梯度裁剪;每 eval_every 步评估;满足 _should_stop 或到 max_steps 停止。"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    device = model.device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps)
    tokens = all_sequences(sl)
    history, step, t0 = [], 0, time.time()
    model.train()
    while step < max_steps:
        for batch in batches(sl, cfg.batch_size, g):
            inputs, targets, mask = split_inputs_targets(batch.to(device))
            logits = model(inputs)
            loss = F.cross_entropy(logits[mask], targets[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            step += 1
            if step % cfg.eval_every == 0 or step == max_steps:
                ev = evaluate(model, sl, demo, cfg, tokens)
                entry = dict(step=step, loss=loss.item(), token_acc=ev["token_acc"], ppl=ev["ppl"],
                             n_exact=ev["n_exact"], seconds=round(time.time() - t0, 1))
                history.append(entry)
                log(entry)
                if _should_stop(history, cfg):
                    model.eval()
                    return dict(history=history, final=ev, stopped="converged", seed=seed)
            if step >= max_steps:
                break
    model.eval()
    return dict(history=history, final=evaluate(model, sl, demo, cfg, tokens), stopped="max_steps", seed=seed)


def train_with_seeds(make_model, sl, demo, cfg, lr, max_steps, log=print):
    """依次尝试 cfg.seeds,直到某个种子 5 句全对;都失败则返回最后一次。make_model(seed) 返回新模型。"""
    last = None
    for seed in cfg.seeds:
        model = make_model(seed)
        res = train_stage(model, sl, demo, cfg, lr, max_steps, seed, log)
        last = (model, res)
        if res["final"]["n_exact"] == cfg.n_demo:
            return last
    return last
