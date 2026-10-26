"""残差流全程三值的语言模型 TernaryResidualLM,以及它的脉冲点登记、阈值标定和脉冲计数。

与 SpikingLM 的差别:嵌入查表后过一个三值量化器 q_emb,各块是 snn_spec.ternary_residual.TernaryResidualBlock
(残差流 r 本身就是三值,没有 Q_in 和实值累加器 h,两处残差加法换成合并 M1/M2),输出头直接读三值 r,没有 q_out。
权重名称与 FloatLM 一致,transfer_weights 可直接装载(多出的只有各处阈值 log_thr)。
"""
import torch
import torch.nn as nn

from snn_spec.gdn import SigmoidGates
from snn_spec.neurons import QuantizerIn
from snn_spec.ternary_residual import TernaryResidualBlock

from .convert import _set_thr
from .data import split_inputs_targets
from .model import _LMBase


class TernaryResidualLM(_LMBase):
    def __init__(self, cfg, vocab, leak, merge="sat", levels=1):
        super().__init__()
        self.cfg, self.vocab, self.merge, self.levels = cfg, vocab, merge, int(levels)
        rng = (cfg.alpha_min, cfg.alpha_max)
        self.emb = nn.Embedding(vocab, cfg.d)
        self.q_emb = QuantizerIn(1.0, levels=levels)
        self.blocks = nn.ModuleList([
            TernaryResidualBlock(cfg.d, cfg.d_ff, cfg.n_heads, cfg.n_heads, cfg.dk, cfg.dv,
                                 gates=SigmoidGates(cfg.d, cfg.n_heads, rng), alpha_rng=rng,
                                 leak=leak, thr=1.0, learn_thr=True, merge=merge, levels=levels)
            for _ in range(cfg.n_blocks)])
        self.head = nn.Linear(cfg.d, vocab, bias=False)

    def step(self, tok, st, record=False):
        h = self.q_emb(self.emb(tok))
        recs = []
        for i, blk in enumerate(self.blocks):
            h, st[i], rec = blk.step(h, st[i], record=record)
            recs.append(rec)
        logits = self.head(h)
        if record:
            recs.append(dict(h_final=h))
        return logits, st, recs

    def lif_modules(self):
        for blk in self.blocks:
            yield from blk.lif_modules()


def block_points(blk):
    """块内需要按预激活分位数标定的 8 个 LIF 点,按前向顺序:(名字, 模块, 由记录算预激活 [B, C], 是否三值)。
    合并 M1/M2 不在其中:它们的阈值由 merge 方式决定(sat/and 固定,learn 可学)。"""
    a, f = blk.attn, blk.ffn
    return [
        ("q", a.sn_q, lambda r: a.Wq(r["x_attn"]), False),
        ("k", a.sn_k, lambda r: a.Wk(r["x_attn"]), True),
        ("v", a.sn_v, lambda r: a.Wv(r["x_attn"]), True),
        ("pre", a.sn_pre, lambda r: r["o"].reshape(r["o"].shape[0], -1), True),
        ("out", a.sn_out, lambda r: a.Wo(r["o_spk"]), True),
        ("g", f.sn_1, lambda r: f.W1(r["x_ffn"]), False),
        ("u", f.sn_3, lambda r: f.W3(r["x_ffn"]), True),
        ("y2", f.sn_2, lambda r: f.W2(r["p"]), True),
    ]


def named_points(model):
    """全部脉冲点 [(全名, 点名, 模块)]:q_emb,然后每块按 q, k, v, pre, out, m1, g, u, y2, m2。"""
    out = [("q_emb", "q_emb", model.q_emb)]
    for i, blk in enumerate(model.blocks):
        mods = {name: mod for name, mod, _, _ in block_points(blk)}
        mods.update(m1=blk.m1, m2=blk.m2)
        for name in ("q", "k", "v", "pre", "out", "m1", "g", "u", "y2", "m2"):
            out.append((f"blocks.{i}.{name}", name, mods[name]))
    return out


@torch.no_grad()
def calibrate_thresholds(model, tokens, quantile=0.75):
    """tokens [N, L] 含 BOS/EOS/PAD。按前向顺序逐点标定(先 q_emb,再每块的 8 个 LIF),
    阈值 = 有效位置上预激活(三值点取绝对值)的分位数,下限 1e-3。合并 m1/m2 不动。返回 {点名: 阈值}。"""
    inputs, _, mask = split_inputs_targets(tokens)
    out = {}

    def to_thr(z, ternary):
        z = z.abs() if ternary else z
        return max(torch.quantile(z.flatten().float(), quantile).item(), 1e-3)

    def quantile_of(fn, block_index, ternary):
        _, recs = model(inputs, record=True)
        z = torch.stack([fn(recs[t][block_index]) for t in range(inputs.shape[1])], dim=1)[mask]
        return to_thr(z, ternary)

    thr = to_thr(model.emb(inputs)[mask], True)
    _set_thr(model.q_emb, thr)
    out["q_emb"] = thr
    for i, blk in enumerate(model.blocks):
        for name, mod, fn, ternary in block_points(blk):
            thr = quantile_of(fn, i, ternary)
            _set_thr(mod, thr)
            out[f"blocks.{i}.{name}"] = thr
    return out


@torch.no_grad()
def spike_counts(model, tokens, dev):
    """每个脉冲点(含 q_emb、m1、m2)每 token 的平均脉冲数(c = |s| * T)与非零率,以及 SwiGLU 门送进 W2 的事件数。
    字段与 experiments.multilevel.spike_counts 相同。tokens 含 BOS/EOS/PAD。"""
    inputs, _, mask = split_inputs_targets(tokens.to(dev))
    _, recs = model(inputs, record=True)
    keys = {"q_emb": "h_in", "q": "q", "k": "k", "v": "v", "pre": "o_spk", "out": "y_attn", "m1": "h_mid",
            "g": "g", "u": "u", "y2": "y_ffn", "m2": "h_out"}

    def stacked(i, key):
        return torch.stack([recs[t][i][key] for t in range(len(recs))], dim=1)[mask]

    out = {}
    for full, name, mod in named_points(model):
        T = int(getattr(mod, "levels", 1))
        i = 0 if name == "q_emb" else int(full.split(".")[1])
        s = stacked(i, keys[name])
        if name == "k":
            s = s / model.blocks[i].attn.c_k                   # 去掉 1/sqrt(dk)
        s = s.reshape(s.shape[0], -1)
        c = (s.abs() * T).round()
        out[full] = dict(levels=T, neurons=int(c.shape[-1]), spikes_per_token=round(c.mean().item(), 4),
                         nonzero=round((c > 0).float().mean().item(), 4))
    tot = sum(v["spikes_per_token"] for v in out.values()) / len(out)
    n_neurons = sum(v["neurons"] for v in out.values())
    total = sum(v["spikes_per_token"] * v["neurons"] for v in out.values())
    # SwiGLU 门 p = g*u 送进 W2 的事件数:c_g * c_u(T = 1 时就是 p 的非零数)
    w2 = {}
    for i, blk in enumerate(model.blocks):
        Tg, Tu = int(getattr(blk.ffn.sn_1, "levels", 1)), int(getattr(blk.ffn.sn_3, "levels", 1))
        p = stacked(i, "p")
        ev = (p.abs() * Tg * Tu).round()
        w2[f"blocks.{i}"] = dict(events_per_channel_token=round(ev.mean().item(), 4), channels=int(p.shape[-1]))
    return dict(points=out, mean_spikes_per_neuron_token=round(tot, 4),
                weighted_spikes_per_neuron_token=round(total / n_neurons, 4), total_spikes_per_token=round(total, 1),
                n_spiking_neurons=n_neurons, w2_events=w2,
                w2_events_per_token=round(sum(v["events_per_channel_token"] * v["channels"] for v in w2.values()), 1))
