import torch

from toy_demo.data import all_sequences, EOS
from toy_demo.model import FloatLM, SpikingLM


def test_float_and_spiking_have_same_weight_names_and_shapes(cfg):
    f, s = FloatLM(cfg, 20), SpikingLM(cfg, 20, leak=0.0)
    fs, ss = f.state_dict(), s.state_dict()
    assert set(fs) <= set(ss)
    assert all(fs[k].shape == ss[k].shape for k in fs)
    assert all("log_thr" in k for k in set(ss) - set(fs))
    assert "blocks.0.attn.gates.Wa.weight" in fs and "head.weight" in fs and "emb.weight" in fs


def test_forward_shapes_record_and_generate(cfg, sl):
    tok = all_sequences(sl)[:4]
    for m in (FloatLM(cfg, sl.vocab_size), SpikingLM(cfg, sl.vocab_size, leak=0.9)):
        logits = m(tok[:, :-1])
        assert logits.shape == (4, tok.shape[1] - 1, sl.vocab_size)
        logits2, recs = m(tok[:, :-1], record=True)
        assert torch.equal(logits, logits2) and len(recs) == tok.shape[1] - 1
        assert len(recs[0]) == cfg.n_blocks + 1 and "h_final" in recs[0][-1]
        out = m.generate(sl.ids[0][:3], max_new=6)
        assert len(out) <= 6 and EOS not in out and all(0 <= t < sl.vocab_size for t in out)


def _block0_qkv(m, tokens):
    _, recs = m(tokens, record=True)
    return [(r[0]["q"], r[0]["k"], r[0]["v"]) for r in recs]


def test_leak_zero_neurons_are_stateless_per_token(cfg, sl):
    """块 0 的 q/k/v 只依赖当前 token 的嵌入。leak=0 时序列运行与单 token 运行相同;leak=0.9 时膜电位携带使之不同。"""
    tok = all_sequences(sl)[:2, :-1]
    for leak, expect_same in ((0.0, True), (0.9, False)):
        torch.manual_seed(0)
        m = SpikingLM(cfg, sl.vocab_size, leak=leak)
        with torch.no_grad():
            for lif in m.lif_modules():
                lif._thr.log_thr.fill_(-2.0)          # 阈值 0.135,保证有发放
        seq = _block0_qkv(m, tok)
        same = True
        for t in range(tok.shape[1]):
            alone = _block0_qkv(m, tok[:, t:t + 1])[0]
            same &= all(torch.equal(a, b) for a, b in zip(seq[t], alone))
        assert same == expect_same
