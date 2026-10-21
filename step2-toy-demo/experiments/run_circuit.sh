#!/bin/bash
# 后台跑一个电路实验(CPU 单线程):run_circuit.sh <日志名> [circuit_levels.py 参数...]
# 日志写到 results/improve/circuit_logs/<日志名>.log;snn_spec 由 toy_demo 自动取同一代码树的 step1-spiking-layer。
set -u
cd "$(dirname "$0")/.." || exit 1
name=$1; shift
mkdir -p results/improve/circuit_logs
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONUTF8=1 \
  nohup setsid "$PY" experiments/circuit_levels.py "$@" > "results/improve/circuit_logs/$name.log" 2>&1 < /dev/null &
sleep 1
echo "started $name: $*"
