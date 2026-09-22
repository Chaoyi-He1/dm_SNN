import torch

from toy_demo.convert import calibrate_thresholds, set_leak, transfer_weights
from toy_demo.data import all_sequences, split_inputs_targets
from toy_demo.model import FloatLM, SpikingLM


def test_transfer_weights_copies_every_float_weight(cfg, sl):
    torch.manual_seed(0)
    f, s = FloatLM(cfg, sl.vocab_size), SpikingLM(cfg, sl.vocab_size, leak=0.0)
    transfer_weights(f, s)
    sd = s.state_dict()
    for k, v in f.state_dict().items():
        assert torch.equal(sd[k], v), k


def test_set_leak_touches_every_lif(cfg, sl):
    s = SpikingLM(cfg, sl.vocab_size, leak=0.0)
    set_leak(s, 0.9)
    lifs = list(s.lif_modules())
    assert len(lifs) == 8 * cfg.n_blocks and all(m.leak == 0.9 for m in lifs)


def test_calibrate_gives_target_firing_rate(cfg, sl):
    torch.manual_seed(0)
    s = SpikingLM(cfg, sl.vocab_size, leak=0.0)
    tok = all_sequences(sl)
    thr = calibrate_thresholds(s, tok, quantile=0.75)
    assert len(thr) == 10 * cfg.n_blocks + 1 and all(v > 0 for v in thr.values())
    assert abs(s.blocks[0].attn.sn_q.thr.item() - thr["blocks.0.q"]) < 1e-6
    inputs, _, mask = split_inputs_targets(tok)
    _, recs = s(inputs, record=True)
    for i in range(cfg.n_blocks):
        q = torch.stack([r[i]["q"] for r in recs], 1)[mask]        # 二值 q:[n_valid, H, dk]
        u = torch.stack([r[i]["u"] for r in recs], 1)[mask]        # 三值 u:[n_valid, d_ff]
        assert abs((q != 0).float().mean().item() - 0.25) < 0.08
        assert abs((u != 0).float().mean().item() - 0.25) < 0.08
