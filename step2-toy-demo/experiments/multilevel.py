"""每 token 多个脉冲(T > 1)的脉冲点:把折叠 T=1 的阈值函数推广成 T 级。

语义(电荷注入式 IF,每 token 内 T 个子步,子步间不泄漏):
    token 开始时 U <- leak * U + I(电流在 token 开始一次注入);
    T 个子步里每次 U >= theta/T 就发一个脉冲并减 theta/T;
    子步结束后脉冲数 c = min(floor(U_0 * T / theta), T),输出幅度 s = c / T ∈ {0, 1/T, ..., 1}。
三值点对正负两支对称处理,s ∈ {-1, ..., -1/T, 0, 1/T, ..., 1}。
所以 theta 仍是满量程(s = 1 的位置),T = 1 时与 snn_spec 原实现逐位相同;软复位 U <- U - theta * s
正好等于减去 c 个 theta/T。下游矩阵乘收到的是 0..T 个脉冲,每个脉冲仍是一次加法(1/T 折进权重)。

前向写成 T 个阶跃之和:s = (1/T) * sum_j H(U - j*theta/T),j = 1..T;每个阶跃用 snn_spec 的 arctan
代理梯度,宽度不变,所以代理梯度的总积分与 T = 1 相同(都是 1)。

用法:apply_levels(model, T, points) 把选中的脉冲点就地换成 T 级版本(参数名不变,state_dict 可互相装载)。
"""
import torch

from snn_spec.neurons import BinaryLIF, QuantizerIn, SpikeFn, TernaryLIF

from toy_demo.convert import _block_points


def _steps(U, thr, T, w):
    """(1/T) * sum_{j=1..T} H(U - j*thr/T)。T == 1 时就是 H(U - thr)。"""
    if T == 1:
        return SpikeFn.apply(U - thr, w)
    j = torch.arange(1, T + 1, device=U.device, dtype=U.dtype) / T
    return SpikeFn.apply(U.unsqueeze(-1) - thr * j, w).mean(-1)


def ml_binary(U, thr, T, w=2.0):
    return _steps(U, thr, T, w)


def ml_ternary(U, thr, T, w=2.0):
    if T == 1:   # 与 snn_spec.neurons.ternary_fire 完全相同的运算顺序
        return SpikeFn.apply(U - thr, w) - SpikeFn.apply(-U - thr, w)
    return _steps(U, thr, T, w) - _steps(-U, thr, T, w)


class MLBinaryLIF(BinaryLIF):
    levels = 1

    def fire(self, U, thr):
        return ml_binary(U, thr, self.levels, self.sg_width)


class MLTernaryLIF(TernaryLIF):
    levels = 1

    def fire(self, U, thr):
        return ml_ternary(U, thr, self.levels, self.sg_width)


class MLQuantizerIn(QuantizerIn):
    levels = 1

    def forward(self, h):
        return ml_ternary(h, self.thr, self.levels, self.sg_width)


_SWAP = {BinaryLIF: MLBinaryLIF, TernaryLIF: MLTernaryLIF, QuantizerIn: MLQuantizerIn}
POINT_NAMES = ("q_in1", "q", "k", "v", "pre", "out", "q_in2", "g", "u", "y2", "q_out")
POINT_ALIASES = {"all": POINT_NAMES, "qin": ("q_in1", "q_in2"), "head": ("q_out",),
                 "attn": ("q", "k", "v", "pre", "out"), "mem": ("q", "k", "v"), "ffn": ("g", "u", "y2")}


def parse_points(spec):
    out = []
    for item in (spec or "all").split(","):
        item = item.strip()
        for name in POINT_ALIASES.get(item, (item,)):
            if name not in POINT_NAMES:
                raise ValueError(f"unknown spike point {name!r}; known: {POINT_NAMES} or aliases {tuple(POINT_ALIASES)}")
            if name not in out:
                out.append(name)
    return out


def named_points(model):
    """[(全名, 点名, 模块)],顺序与 toy_demo.convert._block_points 一致,最后是 q_out。"""
    out = []
    for i, blk in enumerate(model.blocks):
        for name, mod, _, _ in _block_points(blk):
            out.append((f"blocks.{i}.{name}", name, mod))
    out.append(("q_out", "q_out", model.q_out))
    return out


def apply_levels(model, T, points="all"):
    """把 points 里的脉冲点换成 T 级。返回 {全名: 级数}。T == 1 时不改类(结果与原模型逐位相同)。"""
    T = int(T)
    if T < 1:
        raise ValueError("T must be >= 1")
    chosen = set(parse_points(points))
    out = {}
    for full, name, mod in named_points(model):
        if name in chosen and T > 1:
            base = type(mod)
            cls = _SWAP.get(base, base if base in _SWAP.values() else None)
            if cls is None:
                raise TypeError(f"{full}: unexpected module type {base}")
            mod.__class__ = cls
            mod.levels = T
        out[full] = int(getattr(mod, "levels", 1))
    return out


@torch.no_grad()
def spike_counts(model, tokens, dev):
    """每个脉冲点每 token 的平均脉冲数(c = |s| * T)与非零率。tokens 含 BOS/EOS/PAD。"""
    from toy_demo.data import split_inputs_targets
    inputs, _, mask = split_inputs_targets(tokens.to(dev))
    _, recs = model(inputs, record=True)
    keys = {"q_in1": "x_attn", "q": "q", "k": "k", "v": "v", "pre": "o_spk", "out": "y_attn",
            "q_in2": "x_ffn", "g": "g", "u": "u", "y2": "y_ffn"}
    out = {}
    for full, name, mod in named_points(model):
        T = int(getattr(mod, "levels", 1))
        if name == "q_out":
            hf = torch.stack([recs[t][len(model.blocks)]["h_final"] for t in range(len(recs))], dim=1)[mask]
            s = mod(hf)
        else:
            i = int(full.split(".")[1])
            s = torch.stack([recs[t][i][keys[name]] for t in range(len(recs))], dim=1)[mask]
            if name == "k":
                s = s * (model.blocks[i].attn.dk ** 0.5)       # 去掉 1/sqrt(dk)
            s = s.reshape(s.shape[0], -1)
        c = (s.abs() * T).round()
        out[full] = dict(levels=T, spikes_per_token=round(c.mean().item(), 4), nonzero=round((c > 0).float().mean().item(), 4))
    tot = sum(v["spikes_per_token"] for v in out.values()) / len(out)
    return dict(points=out, mean_spikes_per_neuron_token=round(tot, 4))
