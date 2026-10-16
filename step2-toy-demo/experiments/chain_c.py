"""串联 A1 -> (B 起点扫描) -> C:等 A1 目录出现 result.json,按需扫描 LIF 阈值倍数,选 teacher forcing 准确率最高的倍数,再训练 C。

用法(在 step2-toy-demo 下,后台运行):
  .venv/bin/python experiments/chain_c.py --a1 results/improve/T2_a1 --name T2_c --gpu 1 --steps 12000 --lr 3e-3
  .venv/bin/python experiments/chain_c.py --a1 results/improve/LT_a1_24k --name LT_c24k_from24k --gpu 1 --steps 24000 --lif-scale 2.25
--lif-scale 给定时跳过扫描。C 的其余参数(levels、尺寸、a0)由 convert_lab 从 A1 的 result.json 继承。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
os.chdir(STEP2)
PY = "/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True, help="A1 结果目录(含 model.pt 与 result.json)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lif-scale", type=float, default=None)
    ap.add_argument("--scale-list", default="1,1.5,2,2.25,2.5,3,4")
    ap.add_argument("--out", default="results/improve")
    ap.add_argument("--seed", type=int, default=0, help="C 阶段的种子(数据顺序)")
    args = ap.parse_args()
    a1 = Path(args.a1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    log(f"waiting for {a1}/result.json")
    while not (a1 / "result.json").exists():
        time.sleep(30)
    time.sleep(5)
    log("A1 done: " + json.dumps({k: json.loads((a1 / "result.json").read_text())["final"][k] for k in ("token_acc", "n_exact")}))
    scale = args.lif_scale
    if scale is None:
        save = Path(args.out) / f"{args.name}_bscan.json"
        cmd = [PY, "experiments/convert_lab.py", "bscan", "--init", str(a1 / "model.pt"), "--scale-list", args.scale_list,
               "--save", str(save)]
        log("bscan: " + " ".join(cmd))
        subprocess.run(cmd, env=env, check=True)
        rows = json.loads(save.read_text())["rows"]
        best = max(rows, key=lambda r: r["token_acc"])
        scale = best["lif_scale"]
        log(f"bscan rows {rows}; chosen lif_scale {scale}")
    cmd = [PY, "experiments/convert_lab.py", "c", "--name", args.name, "--out", args.out, "--init", str(a1 / "model.pt"),
           "--lif-scale", str(scale), "--lr", str(args.lr), "--steps", str(args.steps), "--seed", str(args.seed)]
    log("C: " + " ".join(cmd))
    with open(Path(args.out) / f"{args.name}.log", "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    log(f"C exit code {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
