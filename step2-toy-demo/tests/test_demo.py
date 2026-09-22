import pytest

from toy_demo.demo import encode_prompt, load_models, show_demo, show_metrics
from toy_demo.data import select_demo


def test_load_models_and_show_demo(tiny_run, capsys):
    cfg, sl, out, _ = tiny_run
    models = load_models(cfg, sl, out)
    assert list(models) == ["A0", "A1", "B", "C", "circuit"]
    assert all(m.leak == 0.0 for m in models["A1"].lif_modules())
    assert all(m.leak == cfg.leak for m in models["B"].lif_modules())
    show_demo(models, sl, select_demo(sl, cfg), cfg)
    text = capsys.readouterr().out
    assert text.count("前缀:") == cfg.n_demo and "circuit" in text
    show_metrics(out)
    assert "circuit_nominal" in capsys.readouterr().out


def test_encode_prompt_rejects_unknown_tokens(sl, monkeypatch):
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            class R:
                ids = [1000, 1001, 999999]
            return R()

        def decode(self, ids):
            return f"<{ids[0]}>"

    import toy_demo.demo as demo_mod
    monkeypatch.setattr(demo_mod, "_load_tokenizer", lambda meta: FakeTok())
    with pytest.raises(SystemExit) as e:
        encode_prompt("x", sl)
    assert "<999999>" in str(e.value)

    class FakeTok2(FakeTok):
        def encode(self, text, add_special_tokens=False):
            class R:
                ids = [1000, 1001]
            return R()

    monkeypatch.setattr(demo_mod, "_load_tokenizer", lambda meta: FakeTok2())
    assert encode_prompt("x", sl) == [sl.compact_to_qwen.index(1000), sl.compact_to_qwen.index(1001)]
