"""训练好的脉冲模型每 token 的事件计数:被驱动的 crossbar 行、LIF 脉冲、比较器判决、gain cell 行、嵌入行。
count_events 的输出就是 energy_model.memristor_energy / digital_event_energy 的 events,
model_matrices 的输出是它们的 mats(量化前的权重矩阵,形状 [d_out, d_in],与 events["rows"] 的长度对应)。

计数规则:脉冲向量 s 取值 c/T(T = 每 token 的脉冲窗口数,events["levels"]),每个元素的脉冲数 c = round(|s|·T);
一行被驱动 c 次,就是 crossbar 的 c 次读脉冲。k 记录里已乘 c_k = 1/sqrt(dk),计数前先除掉。
SwiGLU 的 W2 每个通道收到 c_g·c_u 个事件(T > 1 时可以大于 T)。只计有效位置(mask 标出的目标非 PAD 的位置),
所有量都是每 token 的平均;rows 里 gates 的最后一项是偏置行,输入恒为 1.0;head 的输入在 SpikingLM 是 q_out(h_final),
在 TernaryResidualLM 是 h_final 本身。

n_lif = n_blocks·(6·d + 2·d_ff),按 d 宽度记(q、k、v、pre 各 d,out d,g、u 各 d_ff,y2 d);
它等于 8 个 LIF 点的真实神经元数只在 n_heads·dk = d 时成立(真实配置满足,tiny 测试配置不满足)。
cmp_decisions:SpikingLM 是 Q_in1、Q_in2(各块)加 q_out;TernaryResidualLM 是 q_emb 加各块的 m1、m2;
每个比较点 d 个通道、每通道 2T 个比较器(阈值 ±j·thr/T),所以两种模型都是 (n_blocks·2·d + d)·2T。
"""
import toy_demo  # noqa: F401  先导入它才能让 step1 的 snn_spec 可见
import numpy as np
import torch

from toy_demo.data import split_inputs_targets


def model_kind(model):
    """"ternres"(残差流全程三值,有 q_emb)或 "spiking"(SpikingLM)。"""
    return "ternres" if hasattr(model, "q_emb") else "spiking"


def _model_levels(model):
    """全模型共用的脉冲级数 T。各脉冲点级数不一致时无法用一个脉冲窗口表示,直接报错。"""
    if model_kind(model) == "ternres":
        mods = [model.q_emb] + [m for blk in model.blocks for m in (blk.m1, blk.m2)]
    else:
        mods = [model.q_out] + [q for blk in model.blocks for q in (blk.q_in1, blk.q_in2)]
    mods += list(model.lif_modules())
    levels = {int(getattr(m, "levels", 1)) for m in mods}
    if len(levels) != 1:
        raise ValueError(f"各脉冲点的级数不一致 {sorted(levels)},事件计数只支持全模型同一个 T")
    return levels.pop()


def _np(t):
    return t.detach().cpu().double().numpy()


@torch.no_grad()
def count_events(model, tokens, dev=None, batch=64):
    """tokens [N, L](含 BOS/EOS/PAD)。分批前向并累加,返回每 token 平均的 events 字典:
    n_tokens, levels, rows{矩阵名: 每输入行的平均脉冲数}, embed_rows[vocab], spikes_lif, n_lif, cmp_decisions,
    gain{k_rows, q_rows, alpha_mean, s_rms_norm}。dev 默认取 model.device。"""
    dev = model.device if dev is None else dev
    cfg, nb = model.cfg, len(model.blocks)
    T = _model_levels(model)
    spiking = model_kind(model) == "spiking"
    vocab = model.emb.num_embeddings

    def pulses(s):                                     # 每个元素的脉冲数 c = round(|s|·T)
        return (s.abs() * T).round().double()

    n_tokens = 0
    rows = {}
    embed = torch.zeros(vocab, dtype=torch.float64, device=dev)
    spikes = k_rows = q_rows = alpha_sum = s_sq = s_max = 0.0
    alpha_n = s_n = 0

    def add(name, vec):
        rows[name] = vec if name not in rows else rows[name] + vec

    for i0 in range(0, len(tokens), batch):
        inputs, _, mask = split_inputs_targets(tokens[i0:i0 + batch].to(dev))
        _, recs = model(inputs, record=True)
        L = inputs.shape[1]

        def stacked(i, key):                           # [N_valid, ...]:只取有效位置
            return torch.stack([recs[t][i][key] for t in range(L)], dim=1)[mask]

        n_tokens += int(mask.sum())
        embed += torch.bincount(inputs[mask], minlength=vocab).double()
        for i, blk in enumerate(model.blocks):
            c_x, c_o, c_f = pulses(stacked(i, "x_attn")), pulses(stacked(i, "o_spk")), pulses(stacked(i, "x_ffn"))
            c_g, c_u = pulses(stacked(i, "g")), pulses(stacked(i, "u"))
            c_q, c_k = pulses(stacked(i, "q")), pulses(stacked(i, "k") / blk.attn.c_k)
            c_v, c_y1, c_y2 = pulses(stacked(i, "v")), pulses(stacked(i, "y_attn")), pulses(stacked(i, "y_ffn"))
            add(f"blocks.{i}.qkv", c_x.sum(0))
            add(f"blocks.{i}.gates", c_x.sum(0))       # 偏置行在最后补(输入恒为 1.0)
            add(f"blocks.{i}.o", c_o.sum(0))
            add(f"blocks.{i}.ffn13", c_f.sum(0))
            add(f"blocks.{i}.w2", (c_g * c_u).sum(0))
            # 8 个 LIF 点:q, k/c_k, v, o_spk, y_attn, g, u, y_ffn
            spikes += sum(float(c.sum()) for c in (c_q, c_k, c_v, c_o, c_y1, c_g, c_u, c_y2))
            k_rows += float(c_k.sum())                 # 每头 dk 个元素的脉冲数,对头与块求和
            q_rows += float(c_q.sum())
            alpha = stacked(i, "alpha")
            alpha_sum += float(alpha.double().sum()); alpha_n += alpha.numel()
            for t in range(L):                         # S: [n_valid_t, H, dk, dv]
                S = recs[t][i]["S"][mask[:, t]].double()
                s_sq += float((S * S).sum()); s_n += S.numel()
                if S.numel():
                    s_max = max(s_max, float(S.abs().max()))
        h_final = stacked(nb, "h_final")
        add("head", pulses(model.q_out(h_final) if spiking else h_final).sum(0))

    if n_tokens == 0:
        raise ValueError("没有有效 token")
    out_rows = {name: _np(v / n_tokens) for name, v in rows.items()}
    for i in range(nb):                                # gates 的偏置行:每个 token 驱动一次,平均为 1.0
        out_rows[f"blocks.{i}.gates"] = np.append(out_rows[f"blocks.{i}.gates"], 1.0)
    d = cfg.d
    return dict(
        n_tokens=n_tokens, levels=T, rows=out_rows,
        embed_rows=_np(embed / n_tokens),
        spikes_lif=spikes / n_tokens,
        n_lif=nb * (6 * d + 2 * cfg.d_ff),
        cmp_decisions=(nb * 2 * d + d) * 2 * T,
        gain=dict(k_rows=k_rows / n_tokens, q_rows=q_rows / n_tokens, alpha_mean=alpha_sum / alpha_n,
                  s_rms_norm=(s_sq / s_n) ** 0.5 / s_max if s_max > 0 else 0.0))


def model_matrices(model):
    """量化前的权重矩阵(float64 numpy,[d_out, d_in]),键与 count_events 的 rows 一一对应,另有 embed [d, vocab]。
    gates 是 Wa、Wb 竖着拼成 [2H, d] 再右拼偏置列,总是 [2H, d + 1]。"""
    cat = lambda *ts: torch.cat(ts, dim=0)
    mats = {}
    for i, blk in enumerate(model.blocks):
        a, f = blk.attn, blk.ffn
        gw = cat(a.gates.Wa.weight, a.gates.Wb.weight)
        gb = cat(a.gates.Wa.bias, a.gates.Wb.bias).unsqueeze(1)
        mats[f"blocks.{i}.qkv"] = _np(cat(a.Wq.weight, a.Wk.weight, a.Wv.weight))
        mats[f"blocks.{i}.gates"] = _np(torch.cat([gw, gb], dim=1))
        mats[f"blocks.{i}.o"] = _np(a.Wo.weight)
        mats[f"blocks.{i}.ffn13"] = _np(cat(f.W1.weight, f.W3.weight))
        mats[f"blocks.{i}.w2"] = _np(f.W2.weight)
    mats["head"] = _np(model.head.weight)
    mats["embed"] = _np(model.emb.weight.T)
    return mats
