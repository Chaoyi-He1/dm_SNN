"""A0 浮点 GDN 语言模型与 A1/B/C 脉冲语言模型。两者权重名称一致,state_dict 可直接转移(脉冲模型多出阈值参数)。

逐 token 数据流(计划第 4 节):嵌入查表得累加器初值 h(电路上的 DAC)→ 各块 → 输出头 → argmax。
浮点模型:块输入就是 h,q/k 做 L2 归一化,v、o 连续,SwiGLU 用 SiLU,输出头直接读 h。
脉冲模型:snn_spec.SpikingBlock(规范 0.1.3),输出头前多一个三值量化器 q_out。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_spec.block import SpikingBlock
from snn_spec.gdn import SigmoidGates, SpikingGDN
from snn_spec.neurons import QuantizerIn

from .data import BOS, EOS


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


class FloatGDN(nn.Module):
    """连续激活的 GDN 子块;递推核心与脉冲模型共用 SpikingGDN.core(形式 A)。"""

    def __init__(self, d, H, dk, dv, alpha_rng):
        super().__init__()
        self.d, self.H, self.dk, self.dv = d, H, dk, dv
        self.Wq = nn.Linear(d, H * dk, bias=False)
        self.Wk = nn.Linear(d, H * dk, bias=False)
        self.Wv = nn.Linear(d, H * dv, bias=False)
        self.Wo = nn.Linear(H * dv, d, bias=False)
        self.gates = SigmoidGates(d, H, alpha_rng)

    def init_state(self, B, device):
        return dict(S=torch.zeros(B, self.H, self.dk, self.dv, device=device))

    def step(self, x, st, record=False):
        B = x.shape[0]
        alpha, beta = self.gates(x)
        zq, zk, zv = self.Wq(x), self.Wk(x), self.Wv(x)
        q = l2norm(zq.view(B, self.H, self.dk))
        k = l2norm(zk.view(B, self.H, self.dk))
        v = zv.view(B, self.H, self.dv)
        o, st["S"] = SpikingGDN.core(k, v, q, alpha, beta, st["S"])
        o = o.reshape(B, self.H * self.dv)
        y = self.Wo(o)
        rec = dict(zq=zq, zk=zk, zv=zv, o=o, zo=y, alpha=alpha, beta=beta) if record else None
        return y, st, rec


class FloatSwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d, d_ff, bias=False)
        self.W3 = nn.Linear(d, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d, bias=False)

    def step(self, x, record=False):
        z1, z3 = self.W1(x), self.W3(x)
        y = self.W2(F.silu(z1) * z3)
        rec = dict(z1=z1, z3=z3, z2=y) if record else None
        return y, rec


class FloatBlock(nn.Module):
    def __init__(self, d, d_ff, H, dk, dv, alpha_rng):
        super().__init__()
        self.attn = FloatGDN(d, H, dk, dv, alpha_rng)
        self.ffn = FloatSwiGLU(d, d_ff)

    def init_state(self, B, device):
        return dict(attn=self.attn.init_state(B, device))

    def step(self, h, st, record=False):
        y1, st["attn"], r1 = self.attn.step(h, st["attn"], record)
        h_mid = h + y1
        y2, r2 = self.ffn.step(h_mid, record)
        h_out = h_mid + y2
        rec = dict(h_in=h, h_mid=h_mid, h_out=h_out, **r1, **r2) if record else None
        return h_out, st, rec


@torch.no_grad()
def greedy_generate(model, prefix, max_new):
    """model 需有 init_state(B, device)、step(tok, st) -> (logits, st, _)、device。
    prefix 为紧凑 id 列表(不含 BOS);返回生成的 id 列表,遇 EOS 停止(EOS 不含在内)。"""
    dev = model.device
    st = model.init_state(1, dev)
    logits = None
    for tok in [BOS] + list(prefix):
        logits, st, _ = model.step(torch.tensor([tok], device=dev), st)
    out = []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1).item())
        if nxt == EOS:
            break
        out.append(nxt)
        logits, st, _ = model.step(torch.tensor([nxt], device=dev), st)
    return out


class _LMBase(nn.Module):
    """逐 token 执行的语言模型骨架。子类定义 emb、blocks、head 与 head_input()。"""

    @property
    def device(self):
        return self.emb.weight.device

    def init_state(self, B, device):
        return [blk.init_state(B, device) for blk in self.blocks]

    def step(self, tok, st, record=False):
        h = self.emb(tok)
        recs = []
        for i, blk in enumerate(self.blocks):
            h, st[i], rec = blk.step(h, st[i], record=record)
            recs.append(rec)
        logits = self.head(self.head_input(h))
        if record:
            recs.append(dict(h_final=h))
        return logits, st, recs

    def forward(self, tokens, record=False):
        """tokens [B, L] -> logits [B, L, V];record=True 时再返回逐 token 的记录列表。"""
        B, L = tokens.shape
        st = self.init_state(B, tokens.device)
        outs, recs = [], []
        for t in range(L):
            logits, st, rec = self.step(tokens[:, t], st, record=record)
            outs.append(logits)
            recs.append(rec)
        logits = torch.stack(outs, dim=1)
        return (logits, recs) if record else logits

    def generate(self, prefix, max_new):
        return greedy_generate(self, prefix, max_new)


class FloatLM(_LMBase):
    def __init__(self, cfg, vocab):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab
        rng = (cfg.alpha_min, cfg.alpha_max)
        self.emb = nn.Embedding(vocab, cfg.d)
        self.blocks = nn.ModuleList([FloatBlock(cfg.d, cfg.d_ff, cfg.n_heads, cfg.dk, cfg.dv, rng)
                                     for _ in range(cfg.n_blocks)])
        self.head = nn.Linear(cfg.d, vocab, bias=False)

    def head_input(self, h):
        return h


class SpikingLM(_LMBase):
    def __init__(self, cfg, vocab, leak):
        super().__init__()
        self.cfg, self.vocab = cfg, vocab
        rng = (cfg.alpha_min, cfg.alpha_max)
        self.emb = nn.Embedding(vocab, cfg.d)
        self.blocks = nn.ModuleList([
            SpikingBlock(cfg.d, cfg.d_ff, cfg.n_heads, cfg.n_heads, cfg.dk, cfg.dv,
                         gates=SigmoidGates(cfg.d, cfg.n_heads, rng), alpha_rng=rng,
                         leak=leak, thr=1.0, thr_in=1.0, learn_thr=True)
            for _ in range(cfg.n_blocks)])
        self.q_out = QuantizerIn(1.0)
        self.head = nn.Linear(cfg.d, vocab, bias=False)

    def head_input(self, h):
        return self.q_out(h)

    def lif_modules(self):
        for blk in self.blocks:
            yield from (blk.attn.sn_q, blk.attn.sn_k, blk.attn.sn_v, blk.attn.sn_pre, blk.attn.sn_out,
                        blk.ffn.sn_1, blk.ffn.sn_3, blk.ffn.sn_2)
