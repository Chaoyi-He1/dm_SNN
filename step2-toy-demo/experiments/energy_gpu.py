"""GPU 实测能耗:在本机 RTX A4500 上用 nvidia-smi 采样功率,测 A0(浮点)与 T=2 C(脉冲)每 token 的焦耳数。
对应设计文档(docs/superpowers/specs/2026-10-28-energy-estimate-design.md)的"GPU 侧(实测)"。

做法
  推理就是模型自己的逐 token step 循环:model(tokens[:, :-1]) 在各位置上跑 step(teacher forcing,这类递归模型真实的推理形状)。
  1. 空载:进程里还没建 CUDA 上下文、GPU 上什么都没有时,采样 --idle-seconds 秒 -> idle_w。
     (实测:建上下文后 GPU 会在 ~66 W 的高性能态停约 17 秒才回落到 ~20 W,所以空载必须在建上下文之前测。)
  2. 每个模型、每个 batch:先预热 3 秒(不采样,让核函数、时钟和功率传感器的滑动平均进入负载状态),
     再启动采样线程、等第一个读数到达后开始计时;循环跑模型,每批之后检查已用时间,≥ --seconds 就停,
     同步 GPU、读钟、立刻停采样。采样窗口与负载窗口重合(只差第一个读数到达的几十毫秒)。
  3. 能量 = 平均功率 × 实际用时,除以有效 token 数(目标非 PAD 的位置;补齐的位置算了但不计)。
     total = mean_w·s/tokens;incremental = (mean_w − idle_w)·s/tokens(扣掉空载);ms_per_token = 1000·s/tokens。
  batch 1:逐句(切片里的句子循环使用);batch > 1:固定种子的 batches(sl, batch, g) 的一个 epoch 循环使用。
  数据都在计时之前搬到 GPU,计时里只有推理。

两个索引别搞混:torch 看到的是 CUDA_VISIBLE_DEVICES 重排后的 cuda:0,nvidia-smi -i 用的是物理编号。
本机两块 A4500,0 号归别的项目,用 1 号:CUDA_VISIBLE_DEVICES=1,--gpu-index 1。脚本会比对两边的 UUID,不一致就退出。

用法(在 step2-toy-demo 下):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/energy_gpu.py \
      --models A0=results/ckpt/a0.pt,T2=results/improve/T2_c/model.pt --batches 1,64 \
      --seconds 60 --idle-seconds 20 --gpu-index 1 --save results/energy/gpu.json
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

STEP2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STEP2))

import toy_demo  # noqa: E402,F401  先导入它才能让 step1 的 snn_spec 可见
import torch  # noqa: E402

from experiments.beam_eval import load_model  # noqa: E402  (导入时会 chdir 到 step2-toy-demo,--save 的相对路径以此为准)
from toy_demo.data import batches, encode_sequence, load_slice, pad_sequences, split_inputs_targets  # noqa: E402

POLL_MS = 100


# ----------------------------------------------------------------------------- 纯函数

def parse_power_lines(lines):
    """nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 的行 -> 瓦。空行与 N/A、[N/A] 之类的非数字行丢掉。"""
    out = []
    for line in lines:
        try:
            out.append(float(line.strip()))
        except ValueError:
            continue
    return out


def joules_per_token(mean_w, idle_w, seconds, tokens):
    """-> dict(total=mean_w·s/tokens, incremental=(mean_w−idle_w)·s/tokens, ms_per_token=1000·s/tokens)。"""
    return dict(total=mean_w * seconds / tokens, incremental=(mean_w - idle_w) * seconds / tokens,
                ms_per_token=1000.0 * seconds / tokens)


# ----------------------------------------------------------------------------- 功率采样

def sample_power(gpu_index, stop_event, out=None):
    """跑 nvidia-smi -i gpu_index ... -lms 100,线程里逐行读功率,直到 stop_event 置位,然后终止子进程。
    读到的瓦特追加进 out(不给就新建);返回这个列表。"""
    out = [] if out is None else out
    cmd = ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", str(POLL_MS)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def reader():
        for line in proc.stdout:
            out.extend(parse_power_lines([line]))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        while not stop_event.wait(0.05):
            if proc.poll() is not None:                     # nvidia-smi 自己退出了:不再有读数
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait()
        th.join(timeout=2.0)
    return out


@contextmanager
def sampling(gpu_index, wait=10.0):
    """进入时启动采样并等第一个读数到达,产出实时增长的列表;退出时停采样(即使工作负载抛异常)。"""
    stop, out = threading.Event(), []
    th = threading.Thread(target=sample_power, args=(gpu_index, stop, out), daemon=True)
    th.start()
    try:
        t_end = time.perf_counter() + wait
        while not out:
            if time.perf_counter() > t_end or not th.is_alive():
                raise RuntimeError(f"nvidia-smi -i {gpu_index} produced no power reading within {wait:.0f} s")
            time.sleep(0.01)
        yield out
    finally:
        stop.set()
        th.join()


# ----------------------------------------------------------------------------- 工作负载与测量

def build_workload(sl, batch, dev):
    """-> [(inputs [B, L-1] 已在 dev 上, 有效 token 数)]。batch 1:每句一项;batch > 1:固定种子的一个 epoch。"""
    if batch == 1:
        toks = [pad_sequences([encode_sequence(ids)]) for ids in sl.ids]
    else:
        toks = list(batches(sl, batch, torch.Generator().manual_seed(0)))
    items = []
    for t in toks:
        inputs, _, mask = split_inputs_targets(t)
        items.append((inputs.to(dev), int(mask.sum())))
    return items


def run_workload(model, items, seconds):
    """循环跑 items(一圈跑完从头再来),每批之后检查用时,≥ seconds 停。返回 (有效 token 数, 批数, 用时 s);用时在 GPU 同步之后读。"""
    n_tok = n_batch = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for inputs, n in itertools.cycle(items):
            model(inputs)
            n_tok += n
            n_batch += 1
            if time.perf_counter() - t0 >= seconds:
                break
    torch.cuda.synchronize()
    return n_tok, n_batch, time.perf_counter() - t0


def measure(model, sl, dev, batch, seconds, gpu_index, idle_w, warmup=3.0):
    """测一个模型在一个 batch 大小下的每 token 能耗。采样窗口 = 负载窗口(预热在采样之前)。
    -> dict(batch, seconds(要求的最短用时), tokens, n_batches, elapsed_s(实际用时,能量按它算), mean_w, idle_w,
            total_j_per_token, incremental_j_per_token, ms_per_token, samples, samples_w)。"""
    model.eval()
    items = build_workload(sl, batch, dev)
    run_workload(model, items, warmup)
    with sampling(gpu_index) as watts:
        n_tok, n_batch, elapsed = run_workload(model, items, seconds)
    watts = list(watts)
    mean_w = sum(watts) / len(watts)
    j = joules_per_token(mean_w, idle_w, elapsed, n_tok)
    return dict(batch=batch, seconds=seconds, tokens=n_tok, n_batches=n_batch, elapsed_s=elapsed, mean_w=mean_w,
                idle_w=idle_w, total_j_per_token=j["total"], incremental_j_per_token=j["incremental"],
                ms_per_token=j["ms_per_token"], samples=len(watts), samples_w=watts)


def measure_idle(gpu_index, seconds):
    with sampling(gpu_index) as watts:
        time.sleep(seconds)
    watts = list(watts)
    return sum(watts) / len(watts), watts


# ----------------------------------------------------------------------------- 设备核对与命令行

def smi_query(gpu_index, field):
    r = subprocess.run(["nvidia-smi", "-i", str(gpu_index), f"--query-gpu={field}", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def require_visible_devices(gpu_index):
    """在碰 CUDA 之前先要求设置 CUDA_VISIBLE_DEVICES,不然 torch 的 cuda:0 会落在 0 号卡(别的项目的)上。"""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit(f"CUDA_VISIBLE_DEVICES is not set: run with CUDA_VISIBLE_DEVICES={gpu_index} so torch's cuda:0 is physical GPU {gpu_index}")


def check_devices(gpu_index):
    """torch 的 cuda:0 必须就是 nvidia-smi 的 gpu_index 号卡(比对 UUID)。这一步会建 CUDA 上下文。返回卡名。"""
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available")
    torch_uuid = str(torch.cuda.get_device_properties(0).uuid)
    smi_uuid = smi_query(gpu_index, "uuid")
    if torch_uuid not in smi_uuid:
        sys.exit(f"torch cuda:0 is {torch_uuid}, but nvidia-smi -i {gpu_index} is {smi_uuid}: wrong CUDA_VISIBLE_DEVICES")
    return smi_query(gpu_index, "name")


def parse_models(spec):
    out = {}
    for item in spec.split(","):
        label, path = item.split("=", 1)
        out[label] = path
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="标签=检查点路径,逗号分隔,如 A0=results/ckpt/a0.pt,T2=results/improve/T2_c/model.pt")
    ap.add_argument("--batches", default="1,64")
    ap.add_argument("--seconds", type=float, default=60.0, help="每个 (模型, batch) 至少跑多久")
    ap.add_argument("--idle-seconds", type=float, default=20.0)
    ap.add_argument("--gpu-index", type=int, default=1, help="nvidia-smi 的物理编号(torch 侧用 CUDA_VISIBLE_DEVICES 指到同一块卡)")
    ap.add_argument("--slice", default="data/tinystories-slice")
    ap.add_argument("--save", default="results/energy/gpu.json")
    a = ap.parse_args()

    require_visible_devices(a.gpu_index)
    idle_w, idle_samples = measure_idle(a.gpu_index, a.idle_seconds)   # 只用 nvidia-smi,此时进程里还没有 CUDA 上下文
    name = check_devices(a.gpu_index)
    print(f"{name} (physical {a.gpu_index}): idle {idle_w:.2f} W over {len(idle_samples)} samples", flush=True)
    dev = torch.device("cuda:0")
    sl = load_slice(a.slice)
    models = {label: load_model(path, sl, dev)[0] for label, path in parse_models(a.models).items()}

    runs = {}
    for label, model in models.items():
        runs[label] = {}
        for b in [int(x) for x in a.batches.split(",")]:
            r = measure(model, sl, dev, b, a.seconds, a.gpu_index, idle_w)
            runs[label][str(b)] = r
            print(f"{label} batch {b}: {r['tokens'] / r['elapsed_s']:.0f} tok/s, {r['mean_w']:.1f} W, "
                  f"{r['total_j_per_token'] * 1e3:.3f} mJ/token total, {r['incremental_j_per_token'] * 1e3:.3f} mJ/token incremental, "
                  f"{r['ms_per_token']:.4f} ms/token ({r['tokens']} tokens, {r['samples']} samples)", flush=True)

    out = dict(gpu=name, gpu_index=a.gpu_index, torch=torch.__version__, idle_w=idle_w, idle_samples_w=idle_samples, runs=runs)
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.save).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {a.save}")


if __name__ == "__main__":
    main()
