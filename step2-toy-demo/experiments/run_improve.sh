#!/bin/bash
# 改进方向实验的后台启动器:run_improve.sh <gpu> <name> <script.py> <subcommand> [参数...]
# 日志写到 results/improve/<name>.log,结果写到 results/improve/<name>/(脚本参数里自带 --out results/improve)。
set -u
cd "$(dirname "$0")/.." || exit 1
gpu=$1; name=$2; script=$3; stage=$4; shift 4
mkdir -p results/improve
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONUTF8=1 nohup setsid "$PY" "$script" "$stage" --name "$name" --out results/improve "$@" \
  > "results/improve/$name.log" 2>&1 < /dev/null &
sleep 1
pid=$(pgrep -f "$script $stage --name $name " | head -1)
echo "launched $name on gpu $gpu (pid ${pid:-?}): $script $stage $*"
