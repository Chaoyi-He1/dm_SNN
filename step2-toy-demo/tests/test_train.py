import torch

from toy_demo.model import FloatLM
from toy_demo.train import evaluate, train_stage, train_with_seeds


def test_evaluate_reports_fields(cfg, sl, demo):
    torch.manual_seed(0)
    m = FloatLM(cfg, sl.vocab_size)
    ev = evaluate(m, sl, demo, cfg)
    assert 0.0 <= ev["token_acc"] <= 1.0 and ev["ppl"] > 1.0
    assert ev["n_tokens"] == sum(len(s) + 1 for s in sl.ids)
    assert ev["n_demo"] == cfg.n_demo == len(ev["completions"])
    assert {"prefix", "target", "output", "exact"} <= set(ev["completions"][0])
    assert ev["n_exact"] == sum(c["exact"] for c in ev["completions"])


def test_train_stage_lowers_loss(cfg, sl, demo):
    torch.manual_seed(0)
    m = FloatLM(cfg, sl.vocab_size)
    before = evaluate(m, sl, demo, cfg)["ppl"]
    res = train_stage(m, sl, demo, cfg, lr=3e-3, max_steps=20, seed=0, log=lambda *_: None)
    assert res["history"][-1]["step"] == 20 and res["stopped"] == "max_steps"
    assert res["final"]["ppl"] < before


def test_train_with_seeds_returns_last_attempt_when_none_converges(cfg, sl, demo):
    seen = []
    def make(seed):
        seen.append(seed)
        torch.manual_seed(seed)
        return FloatLM(cfg, sl.vocab_size)
    model, res = train_with_seeds(make, sl, demo, cfg, lr=3e-3, max_steps=5, log=lambda *_: None)
    assert seen == list(cfg.seeds) and res["seed"] == cfg.seeds[-1]
