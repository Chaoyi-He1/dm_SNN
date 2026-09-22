import json

from toy_demo.run import run_pipeline

STAGES = ["A0", "A1_init", "A1", "B", "C", "circuit_ideal", "circuit_nominal"]


def test_pipeline_tiny_end_to_end(tiny_run):
    cfg, sl, out, results = tiny_run
    assert list(results["stages"]) == STAGES
    for s in STAGES:
        f = results["stages"][s]["final"]
        assert {"token_acc", "ppl", "n_exact", "n_demo", "completions"} <= set(f)
    assert results["stages"]["circuit_ideal"]["final"]["compare"]["pass"]
    assert results["stages"]["circuit_ideal"]["final"]["argmax_equal_to_software"]
    assert [r["sigma_g"] for r in results["sigma_g_sweep"]] == list(cfg.sigma_g_sweep)
    assert all((out / "ckpt" / f"{n}.pt").exists() for n in ("a0", "a1", "c"))
    md = (out / "toy-demo.md").read_text(encoding="utf-8")
    assert "A0 浮点" in md and "编程误差扫描" in md
    assert json.loads((out / "toy-demo.json").read_text(encoding="utf-8"))["vocab_size"] == sl.vocab_size


def test_pipeline_reuse_skips_training(tiny_run):
    cfg, sl, out, _ = tiny_run
    again = run_pipeline(cfg, sl, out, reuse=True, log=lambda *_: None)
    assert again["stages"]["A0"]["reused"] and again["stages"]["A1"]["reused"] and again["stages"]["C"]["reused"]
