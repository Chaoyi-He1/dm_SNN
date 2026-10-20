"""多块电路模型(计划第 6 节):每块一个 snn_spec.circuit.BehavioralBlock,嵌入查表充当 DAC,
输出头 = 比较器(三值)+ crossbar + 对 bitline 电流取 argmax。两种模式:teacher forcing 对照(软件门值)与自由运行。

每 token 多个脉冲(experiments/multilevel.py 的 levels=T):电路侧不需要额外设置,BehavioralBlock 从软件模块读出
每个脉冲点的 T,神经元变为 T 个子时隙的发放(输出 c/T),比较器变为 2T 个,门 p = g·u 变为脉冲计数之积
(W_2 每通道最多 T² 个事件),输出头前的量化器同样变为 2T 个比较器。teacher forcing 的 precision/recall 按脉冲计数统计。
"""
import torch

from snn_spec.circuit import BehavioralBlock, ComparatorModel, CrossbarModel
from snn_spec.metrics import sign_error_rate, spike_prf_by_sign
from snn_spec.stress import state_bound

from .data import PAD
from .model import greedy_generate


def primitive_kwargs(cfg, ideal, sigma_g=None, seed=0):
    """返回 (crossbar_kw, gain_cell_kw, 比较器失调占阈值的比例)。ideal=True 时全部为理想值。"""
    if ideal:
        return dict(seed=seed), dict(seed=seed), 0.0
    sg = cfg.sigma_g if sigma_g is None else sigma_g
    return (dict(n_levels=cfg.n_levels, sigma_g=sg, read_noise=cfg.read_noise, seed=seed),
            dict(write_err=cfg.write_err, read_err=cfg.read_err, eps_hold=cfg.eps_hold, seed=seed),
            cfg.cmp_offset_frac)


PARTS = ("gdn", "ffn", "state", "head")
PART_MATRICES = {"gdn": ("Wq", "Wk", "Wv", "Wo"), "ffn": ("W1", "W3", "W2")}


class CircuitLM:
    """ideal=True:全部原语取理想值。ideal=False:取 cfg 的标称值;nominal_parts 给出时只有这些部分取标称值,其余理想,
    用于计划第 7 节的误差隔离:
        "gdn"   W_q/W_k/W_v/W_O 四个 crossbar 与块入口比较器 cmp1
        "ffn"   W_1/W_3/W_2 三个 crossbar 与前馈入口比较器 cmp2
        "state" 状态阵列(gain cell 的写入、读出误差与保持因子)
        "head"  输出头的比较器与 crossbar
    各部分的随机种子与全标称时相同,所以只开一部分时它的噪声实现与全标称时那一部分完全一样。"""

    def __init__(self, spiking_lm, cfg, ideal=True, sigma_g=None, seed=0, nominal_parts=None):
        if nominal_parts is not None:
            nominal_parts = tuple(nominal_parts)
            if ideal:
                raise ValueError("nominal_parts 只在 ideal=False 时有意义")
            bad = set(nominal_parts) - set(PARTS)
            if bad:
                raise ValueError(f"未知的部分 {sorted(bad)};可选 {PARTS}")
        self.cfg = cfg
        self.nominal_parts = PARTS if (nominal_parts is None and not ideal) else (nominal_parts or ())
        on = set(self.nominal_parts)
        self.device = spiking_lm.device
        self.emb = spiking_lm.emb.weight.detach()
        self.sims = []
        for i, blk in enumerate(spiking_lm.blocks):
            xb_n, gc_n, frac_n = primitive_kwargs(cfg, ideal, sigma_g, seed + i)
            xb_i, gc_i, _ = primitive_kwargs(cfg, True, sigma_g, seed + i)
            if on >= {"gdn", "ffn"}:
                xb = xb_n
            else:
                xb = {n: (xb_n if p in on else xb_i) for p, names in PART_MATRICES.items() for n in names}
            gc = gc_n if "state" in on else gc_i
            sim = BehavioralBlock(blk, crossbar_kw=xb, gain_cell_kw=gc)
            for name, part in (("cmp1", "gdn"), ("cmp2", "ffn")):   # 失调按各自阈值的比例设定,级数沿用 BehavioralBlock 读出的值
                old = getattr(sim, name)
                frac = frac_n if part in on else 0.0
                setattr(sim, name, ComparatorModel(old.thr, offset=frac * old.thr, levels=old.levels))
            self.sims.append(sim)
        xb, _, frac = primitive_kwargs(cfg, ideal or "head" not in on, sigma_g, seed + len(self.sims))
        thr = spiking_lm.q_out.thr.item()
        self.head_cmp = ComparatorModel(thr, offset=frac * thr, levels=int(getattr(spiking_lm.q_out, "levels", 1)))
        self.head_xb = CrossbarModel(spiking_lm.head.weight, **xb)

    def levels(self):
        """每个脉冲点每 token 最多几个脉冲(从软件模型读出;全 1 即原来的折叠 T=1)。"""
        out = {}
        for i, sim in enumerate(self.sims):
            for name, T in sim.levels.items():
                out[f"blocks.{i}.{name}"] = T
        out["q_out"] = self.head_cmp.levels
        return out

    def init_state(self, B, device=None):
        return [sim.init_state(B, self.device) for sim in self.sims]

    @torch.no_grad()
    def step(self, tok, st, ext_gates=None, record=False):
        """ext_gates: 每块一个 (alpha [B, H], beta [B, H]) 的列表,或 None(门由电路自身脉冲经理想门函数算出)。"""
        h = self.emb[tok]
        recs = []
        for i, sim in enumerate(self.sims):
            ext = None if ext_gates is None else ext_gates[i]
            h, st[i], rec = sim.step(h, st[i], ext_gates=ext, record=record)
            recs.append(rec)
        logits = self.head_xb(self.head_cmp(h))
        if record:
            recs.append(dict(h_final=h))
        return logits, st, recs

    @torch.no_grad()
    def forward(self, tokens, ext_gates_seq=None, record=False):
        """tokens [B, L]。ext_gates_seq: 每块一个 (alpha [B, L, H], beta [B, L, H]) 的列表,或 None。"""
        B, L = tokens.shape
        st = self.init_state(B)
        outs, recs = [], []
        for t in range(L):
            ext = None if ext_gates_seq is None else [(a[:, t], b[:, t]) for a, b in ext_gates_seq]
            logits, st, rec = self.step(tokens[:, t], st, ext_gates=ext, record=record)
            outs.append(logits)
            recs.append(rec)
        logits = torch.stack(outs, dim=1)
        return (logits, recs) if record else logits

    __call__ = forward

    def generate(self, prefix, max_new):
        return greedy_generate(self, prefix, max_new)


@torch.no_grad()
def teacher_forced_compare(spiking_lm, circuit_lm, tokens, cfg):
    """原 S2.4 方法:软件带记录运行;电路吃同样的 token 与软件门值;逐 token 比对两块的输出脉冲、状态与输出头 argmax。
    一句一句跑(B=1),避开 pad。tokens [N, L] 含 BOS/EOS/PAD。"""
    bound = state_bound(cfg.dv, cfg.alpha_max)
    n_blocks = len(spiking_lm.blocks)
    lv = [(int(getattr(b.attn.sn_out, "levels", 1)), int(getattr(b.ffn.sn_2, "levels", 1))) for b in spiking_lm.blocks]
    acc = [dict(ya_sw=[], ya_hw=[], yf_sw=[], yf_hw=[], se=[]) for _ in range(n_blocks)]
    agree = n_pos = 0
    if hasattr(spiking_lm, "eval"):
        spiking_lm.eval()
    for row in tokens:
        inputs = row[row != PAD].unsqueeze(0)[:, :-1].to(spiking_lm.device)
        sw_logits, sw_recs = spiking_lm(inputs, record=True)
        gates = [(torch.stack([r[i]["alpha"] for r in sw_recs], 1), torch.stack([r[i]["beta"] for r in sw_recs], 1))
                 for i in range(n_blocks)]
        hw_logits, hw_recs = circuit_lm.forward(inputs, ext_gates_seq=gates, record=True)
        for i in range(n_blocks):
            for t in range(inputs.shape[1]):
                s, h = sw_recs[t][i], hw_recs[t][i]
                acc[i]["ya_sw"].append(s["y_attn"]); acc[i]["ya_hw"].append(h["y_attn"])
                acc[i]["yf_sw"].append(s["y_ffn"]); acc[i]["yf_hw"].append(h["y_ffn"])
                acc[i]["se"].append(((h["S"] - s["S"]).flatten(2).norm(dim=-1) / bound).max().item())
        agree += (sw_logits.argmax(-1) == hw_logits.argmax(-1)).sum().item()
        n_pos += inputs.shape[1]
    blocks = []
    for a, (Ta, Tf) in zip(acc, lv):
        ya_sw, ya_hw, yf_sw, yf_hw = (torch.cat(a[k]) for k in ("ya_sw", "ya_hw", "yf_sw", "yf_hw"))
        blocks.append(dict(
            attn=dict(**spike_prf_by_sign(ya_hw, ya_sw, levels=Ta), sign_error=sign_error_rate(ya_hw, ya_sw), levels=Ta),
            ffn=dict(**spike_prf_by_sign(yf_hw, yf_sw, levels=Tf), sign_error=sign_error_rate(yf_hw, yf_sw), levels=Tf),
            state_err_max=max(a["se"]), state_err_mean=sum(a["se"]) / len(a["se"])))
    out = dict(argmax_agreement=agree / n_pos, n_positions=n_pos, blocks=blocks,
               levels=circuit_lm.levels() if hasattr(circuit_lm, "levels") else None)
    out["pass"] = passes(out)
    return out


def passes(cmp):
    """计划第 7 节的 teacher forcing 判据。"""
    ok = cmp["argmax_agreement"] >= 0.99
    for b in cmp["blocks"]:
        for part in (b["attn"], b["ffn"]):
            ok &= all(part[k] >= 0.9 for k in ("pos_precision", "pos_recall", "neg_precision", "neg_recall"))
            ok &= part["sign_error"] <= 0.02
        ok &= b["state_err_max"] <= 0.1
    return bool(ok)
