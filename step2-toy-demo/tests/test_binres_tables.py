"""binres_tables:电平数、判据、beam 结果索引、对三种残差流记录的补全与表格式化(用 tmp_path 里手造的目录)。"""
import json

from experiments import binres_tables as bt
from experiments.ternres_tables import read_baseline, read_run


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def hist(n_exact_list):
    return [dict(step=100 * (i + 1), token_acc=0.8, ppl=2.0, n_exact=k) for i, k in enumerate(n_exact_list)]


def c_result(acc, n_exact, exact_hist, residual="binary", merge="sr", levels=2, d=128, init="x/model.pt"):
    args = dict(stage="c", merge=merge, levels=levels, fq=0.9, lif_scale=1.5, d=d, init=init)
    if residual != "real":
        args["residual"] = residual
    return dict(args=args, final=dict(token_acc=acc, ppl=2.0, n_exact=n_exact, n_demo=5), history=hist(exact_hist),
                stopped="converged", seconds=600.0, spikes=dict(total_spikes_per_token=2500.0, w2_events_per_token=180.0))


def test_stream_levels_counts_values_per_channel():
    assert bt.stream_levels("binary", 1) == 2 and bt.stream_levels("binary", 4) == 5 and bt.stream_levels("binary", 8) == 9
    assert bt.stream_levels("ternary", 1) == 3 and bt.stream_levels("ternary", 2) == 5 and bt.stream_levels("ternary", 4) == 9
    assert bt.stream_levels("real", 2) is None and bt.stream_levels("binary", None) is None


def test_criterion_needs_accuracy_within_one_point_and_a_stable_demo():
    assert bt.meets_criterion(0.832, 5, 5) is True                    # 正好 1 个百分点、正好 5 次
    assert bt.meets_criterion(0.8319, 5, 10) is False                 # 准确率差一点
    assert bt.meets_criterion(0.840, 4, 10) is False                  # 最终 demo 不是 5/5
    assert bt.meets_criterion(0.840, 5, 4) is False                   # 最后 10 次不够稳
    assert bt.meets_criterion(None, 5, 5) is None and bt.meets_criterion(0.84, None, 5) is None
    assert bt.meets_criterion(0.80, 5, 10, a0_acc=0.805) is True      # 判据跟着 a0_acc 走


def test_beam_index_maps_checkpoint_paths_and_skips_missing_files(tmp_path):
    f = tmp_path / "beam.json"
    write(f, dict(models=dict(a=dict(path="results/improve/x_c/model.pt",
                                     runs=[dict(K=1, alpha=0.0, unique_exact=0.7), dict(K=4, alpha=0.0, unique_exact=0.8),
                                           dict(K=4, alpha=1.0, unique_exact=0.9)]),
                              broken=dict(path=None, runs=[]))))
    idx = bt.beam_index([str(f), str(tmp_path / "missing.json")])
    assert idx == {"results/improve/x_c/model.pt": {(1, 0.0): 0.7, (4, 0.0): 0.8, (4, 1.0): 0.9}}


def test_enrich_and_rows_for_binary_ternary_and_real_runs(tmp_path):
    out = tmp_path / "improve"
    for name, res in (("binres_sr_T2", c_result(0.835, 5, [5] * 10)),
                      ("binres_xor_T1_s1", c_result(0.70, 3, [0] * 10, merge="xor", levels=1)),
                      ("ternres_sat_T2", c_result(0.811, 5, [4, 4, 2, 2, 4, 3, 3, 2, 4, 5], residual="ternary", merge="sat"))):
        write(out / f"{name}_c" / "result.json", res)
        write(out / f"{name}_a1" / "result.json", dict(args=dict(stage="a1", residual=res["args"].get("residual"), merge=res["args"]["merge"],
                                                               levels=res["args"]["levels"], fq=0.9),
                                                    init=dict(token_acc=0.01), final=dict(token_acc=0.8, n_exact=5, n_demo=5),
                                                    history=hist([5]), seconds=600.0))
    write(out / "real_T4" / "result.json", c_result(0.836, 5, [5] * 10, residual="real", merge=None, levels=4, d=256))
    beams = {str(out / "binres_sr_T2_c" / "model.pt"): {(1, 0.0): 0.91, (4, 0.0): 0.95}}
    b = bt.enrich(read_run(out, "binres_sr_T2"), out / "binres_sr_T2_c", beams)
    assert (b["residual"], b["stream_levels"], b["d"], b["c_n_exact"], b["gap_to_a0_pt"]) == ("binary", 3, 128, 5, 0.7)
    assert b["recall_greedy"] == 0.91 and b["recall_k4"] == 0.95 and b["meets"] is True
    x = bt.enrich(read_run(out, "binres_xor_T1_s1"), out / "binres_xor_T1_s1_c", beams)
    assert x["residual"] == "binary" and x["stream_levels"] == 2 and x["meets"] is False and x["recall_greedy"] is None
    t = bt.enrich(read_run(out, "ternres_sat_T2"), out / "ternres_sat_T2_c", beams)
    assert t["residual"] == "ternary" and t["stream_levels"] == 5 and t["c_last10_exact"] == 1 and t["meets"] is False
    r = bt.enrich(read_baseline("T4", out / "real_T4"), out / "real_T4", beams)
    assert r["residual"] == "real" and r["stream_levels"] is None and r["d"] == 256 and r["meets"] is True
    rows = [bt.fmt_row(v) for v in (b, x, t, r)]
    assert all(len(row) == len(bt.HEADERS) for row in rows)
    assert rows[0][:5] == ["二值", "sr", "2", "3", "128"] and rows[0][-1] == "是" and rows[0][8] == "0.7"
    assert rows[1][1] == "xor(种子 1)" and rows[1][9] == "3/5" and rows[1][-1] == "否"
    assert rows[2][0] == "三值" and rows[3][:5] == ["实值", "—", "4", "—", "256"]
    table = bt.render_table([b, x, t, r])
    assert table.count("\n") == 5 and table.splitlines()[1].count("---") == len(bt.HEADERS)


def test_unfinished_chain_is_reported_with_dashes(tmp_path):
    out = tmp_path / "improve"
    write(out / "binres_sr_T8_a1" / "result.json", dict(args=dict(stage="a1", residual="binary", merge="sr", levels=8, fq=0.9),
                                                         init=dict(token_acc=0.002), final=dict(token_acc=0.83, n_exact=5, n_demo=5),
                                                         history=hist([5]), seconds=10.0))
    rec = bt.enrich(read_run(out, "binres_sr_T8"), out / "binres_sr_T8_c", {})
    assert rec["status"] == "c running" and rec["residual"] == "binary" and rec["stream_levels"] == 9 and rec["meets"] is None
    row = bt.fmt_row(rec)
    assert row[0] == "二值" and row[1] == "sr" and row[7] == "—" and row[-1] == "—"
    missing = bt.enrich(read_run(out, "binres_src_T4_s2"), out / "binres_src_T4_s2_c", {})
    assert missing["status"] == "missing"
    assert bt.fmt_row(missing)[:2] == ["二值", "binres_src_T4_s2"]          # 什么都还没有:合并列写链名
