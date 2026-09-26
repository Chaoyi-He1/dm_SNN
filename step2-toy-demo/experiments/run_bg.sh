#!/bin/bash
# 在后台启动一个转换实验:run_bg.sh <gpu> <name> <a1|c> [convert_lab 参数...]
# 日志写到 results/conversion-sweep/<name>.log,结果由 convert_lab 写到 results/conversion-sweep/<name>/。
set -u
cd "$(dirname "$0")/.." || exit 1
gpu=$1; name=$2; stage=$3; shift 3
mkdir -p results/conversion-sweep
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONUTF8=1 nohup setsid "$PY" experiments/convert_lab.py "$stage" --name "$name" "$@" \
  > "results/conversion-sweep/$name.log" 2>&1 < /dev/null &
sleep 1
pid=$(pgrep -f "convert_lab.py $stage --name $name " | head -1)
echo "launched $name on gpu $gpu (pid ${pid:-?}): $stage $*"
