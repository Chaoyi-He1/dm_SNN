"""ternres_tables:用 tmp_path 里手造的 chain 目录与基线目录检查 read_run / read_baseline / 表格式化 / main。"""
import json

import pytest

from experiments import ternres_tables as tt


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


def hist(n_exact_list, start=100, every=100):
    return [dict(step=start + i * every, token_acc=0.5 + 0.01 * i, ppl=3.0, n_exact=k) for i, k in enumerate(n_exact_list)]


def a1_result(fq=0.75, init=0.12, final=0.74, steps=(100, 200, 300), seconds=600.0, merge="sat", levels=2):
    return dict(args=dict(stage="a1", merge=merge, levels=levels, fq=fq, residual="ternary"),
                init=dict(token_acc=init, ppl=300.0), final=dict(token_acc=final, ppl=2.7, n_exact=3, n_demo=5),
                history=[dict(step=s, n_exact=0, token_acc=final) for s in steps], stopped="converged", seed=0, seconds=seconds)


# 12 条评估:前 2 条是 5/5(在"最后 10 次"之外,不能算),后 10 条里 3 条是 5/5
C_EXACT = [5, 5, 0, 5, 3, 5, 4, 2, 5, 1, 0, 3]


def c_result(stopped="converged", with_lif=True, spikes=True, seconds=300.0):
    r = dict(args=dict(stage="c", lif_scale=2.25, merge="sat", levels=2, fq=0.75),
             b=dict(token_acc=0.10, ppl=400.0),
             final=dict(token_acc=0.7, ppl=2.5, n_exact=3, n_demo=5),
             history=hist(C_EXACT), stopped=stopped, seed=0, seconds=seconds)
    if with_lif:
        r["b_lif_scale"] = dict(token_acc=0.2, ppl=90.0)
    if spikes:
        r["spikes"] = dict(total_spikes_per_token=1234.6, w2_events_per_token=67.8)
    return r


def chain(name="ternres_sat_T2"):
    return dict(name=name, merge="sat", levels=2, fq=0.75, lif_scale=2.25, seed=0,
                iscan=dict(best=dict(fq=0.75, token_acc=0.1111, ppl=5.0, spikes=0.3)))


@pytest.fixture
def full_run(tmp_path):
    name = "ternres_sat_T2"
    write(tmp_path / f"{name}_chain.json", chain(name))
    write(tmp_path / f"{name}_a1" / "result.json", a1_result())
    write(tmp_path / f"{name}_c" / "result.json", c_result())
    return tmp_path, name


def test_read_run_complete(full_run):
    out, name = full_run
    r = tt.read_run(out, name)
    assert r["status"] == "done"
    assert (r["merge"], r["levels"], r["fq"], r["lif_scale"]) == ("sat", 2, 0.75, 2.25)
    assert r["a1_init_acc"] == 0.12                      # 有 a1 result.json 时用它的 init,不用 iscan 的
    assert (r["a1_acc"], r["a1_steps"]) == (0.74, 300)
    assert r["b_acc"] == 0.2                             # b_lif_scale 优先于 b
    assert (r["c_acc"], r["c_ppl"]) == (0.7, 2.5)
    assert r["c_last10_exact"] == 3
    assert (r["c_steps"], r["c_stopped"]) == (1200, "converged")
    assert (r["spikes_per_token"], r["w2_events"]) == (1234.6, 67.8)
    assert (r["seconds_a1"], r["seconds_c"]) == (600.0, 300.0)
    assert r["minutes"] == pytest.approx(15.0)


def test_b_acc_falls_back_to_b_without_lif_scale(tmp_path):
    name = "ternres_sat_T2"
    write(tmp_path / f"{name}_a1" / "result.json", a1_result())
    write(tmp_path / f"{name}_c" / "result.json", c_result(with_lif=False, spikes=False))
    r = tt.read_run(tmp_path, name)
    assert r["b_acc"] == 0.10
    assert r["spikes_per_token"] is None and r["w2_events"] is None
    assert (r["fq"], r["lif_scale"], r["merge"], r["levels"]) == (0.75, 2.25, "sat", 2)   # 没有 chain.json 就退回 result.json 的 args


def test_fmt_row_cells_and_max_steps_suffix(tmp_path):
    name = "ternres_sat_T2"
    write(tmp_path / f"{name}_chain.json", chain(name))
    write(tmp_path / f"{name}_a1" / "result.json", a1_result())
    write(tmp_path / f"{name}_c" / "result.json", c_result(stopped="max_steps"))
    row = tt.fmt_row(tt.read_run(tmp_path, name))
    assert len(row) == len(tt.HEADERS)
    assert row == ["sat", "2", "0.75", "0.120", "0.740 (300)", "0.200", "2.25", "0.700", "2.50", "3", "1200(跑满)", "1235", "68", "15.0"]


def test_run_with_only_chain_and_a1_dir_is_a1_running(tmp_path):
    name = "ternres_and_T1"
    write(tmp_path / f"{name}_chain.json", dict(name=name, merge="and", levels=1, fq=0.5, iscan=dict(best=dict(fq=0.5, token_acc=0.0014))))
    (tmp_path / f"{name}_a1").mkdir()
    write(tmp_path / f"{name}_a1" / "history.jsonl", json.dumps(dict(step=100, n_exact=0)) + "\n")
    r = tt.read_run(tmp_path, name)
    assert r["status"] == "a1 running"
    assert (r["merge"], r["levels"], r["fq"]) == ("and", 1, 0.5)
    assert r["a1_init_acc"] == 0.0014                    # A1 还没结束:用 iscan 在所选 fq 上的 token_acc
    for k in ("lif_scale", "a1_acc", "a1_steps", "b_acc", "c_acc", "c_ppl", "c_last10_exact", "c_steps", "c_stopped",
              "spikes_per_token", "w2_events", "seconds_a1", "seconds_c", "minutes"):
        assert r[k] is None, k
    row = tt.fmt_row(r)
    assert row[:4] == ["and", "1", "0.5", "0.001"] and set(row[4:]) == {tt.DASH}


def test_status_c_running_and_partial_minutes(tmp_path):
    name = "ternres_sat_T2"
    write(tmp_path / f"{name}_chain.json", chain(name))
    write(tmp_path / f"{name}_a1" / "result.json", a1_result(seconds=120.0))
    (tmp_path / f"{name}_c").mkdir()
    r = tt.read_run(tmp_path, name)
    assert r["status"] == "c running"
    assert r["c_acc"] is None and r["seconds_c"] is None
    assert r["minutes"] == pytest.approx(2.0)            # 只有 A1 的部分和


def test_missing_run_and_unreadable_json(tmp_path):
    r = tt.read_run(tmp_path, "ternres_and_T2")
    assert r["status"] == "missing"
    assert all(v is None for k, v in r.items() if k not in ("name", "status", "residual"))
    assert tt.fmt_row(r) == ["ternres_and_T2"] + [tt.DASH] * (len(tt.HEADERS) - 1)
    # 正写到一半的 JSON 当缺失:C 的 result.json 被截断 -> 退回 "c running"(a1 完成、c 目录在)
    name = "ternres_sat_T2"
    write(tmp_path / f"{name}_a1" / "result.json", a1_result())
    write(tmp_path / f"{name}_c" / "result.json", '{"args": {"stage": "c", "lif_')
    assert tt.load_json(tmp_path / f"{name}_c" / "result.json") is None
    assert tt.read_run(tmp_path, name)["status"] == "c running"


def test_read_baseline_uses_args_init_for_a1(tmp_path):
    a1_dir, c_dir = tmp_path / "A1", tmp_path / "C"
    write(a1_dir / "result.json", a1_result(fq=0.5, init=0.123, final=0.74, steps=(12000,), seconds=60.0, merge=None, levels=None))
    c = c_result(spikes=False, seconds=120.0)
    c["args"].update(init=str(a1_dir / "model.pt"), fq=0.9, levels=None, lif_scale=2.25)
    write(c_dir / "result.json", c)
    r = tt.read_baseline("T1", c_dir)
    assert r["status"] == "done" and r["residual"] == "real"
    assert (r["levels"], r["fq"], r["lif_scale"]) == (1, 0.5, 2.25)       # levels 缺省 1;fq 取 A1 的 args.fq
    assert (r["a1_init_acc"], r["a1_acc"], r["a1_steps"]) == (0.123, 0.74, 12000)
    assert (r["c_acc"], r["c_last10_exact"], r["c_steps"]) == (0.7, 3, 1200)
    assert r["spikes_per_token"] is None
    assert r["minutes"] == pytest.approx(3.0)
    assert tt.fmt_row(r)[0] == "实值残差"
    assert tt.fmt_row(tt.read_baseline("T2_s1", c_dir))[0] == "实值残差(T2_s1)"


def test_read_baseline_without_a1_result_or_c(tmp_path):
    c_dir = tmp_path / "C"
    c = c_result(spikes=False)
    c["args"]["init"] = str(tmp_path / "nowhere" / "model.pt")
    write(c_dir / "result.json", c)
    r = tt.read_baseline("T2", c_dir)
    assert r["a1_init_acc"] is None and r["a1_acc"] is None and r["a1_steps"] is None
    assert r["fq"] == 0.75                                # A1 不在时退回 C 的 args.fq
    row = tt.fmt_row(r)
    assert row[3] == tt.DASH and row[4] == tt.DASH and row[7] == "0.700"
    gone = tt.read_baseline("T9", tmp_path / "none")
    assert gone["status"] == "missing" and gone["c_acc"] is None


def test_parse_baselines():
    assert tt.parse_baselines("T1=a/b,T2=c") == [("T1", "a/b"), ("T2", "c")]
    assert tt.parse_baselines("") == []
    with pytest.raises(ValueError):
        tt.parse_baselines("T1")


def test_main_prints_table_and_writes_json(full_run, tmp_path_factory, monkeypatch, capsys):
    out, name = full_run
    monkeypatch.chdir(out)                                # main 会 chdir 到 step2;monkeypatch 负责在测试后还原
    save = tmp_path_factory.mktemp("save") / "t.json"
    tt.main(["--runs", f"{name},ternres_and_T2", "--baselines", "", "--out", str(out), "--save", str(save)])
    text = capsys.readouterr().out
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(lines) == 2 + 2                            # 表头 + 分隔行 + 两个 run
    assert lines[2].split(" | ")[0] == "| sat" and "0.700" in lines[2]
    assert f"{name} done" in text and "ternres_and_T2 missing" in text
    saved = json.loads(save.read_text(encoding="utf-8"))
    assert list(saved) == ["runs", "baselines"] and saved["baselines"] == {}
    assert saved["runs"][name]["c_acc"] == 0.7 and saved["runs"]["ternres_and_T2"]["status"] == "missing"
