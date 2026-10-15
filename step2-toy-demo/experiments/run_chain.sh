#!/bin/bash
# 后台启动 chain_c.py:run_chain.sh <name> [chain_c 参数...];日志 results/improve/<name>_chain.log
set -u
cd "$(dirname "$0")/.." || exit 1
name=$1; shift
mkdir -p results/improve
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
PYTHONUNBUFFERED=1 nohup setsid "$PY" experiments/chain_c.py --name "$name" "$@" > "results/improve/${name}_chain.log" 2>&1 < /dev/null &
sleep 1
echo "chain $name started: $*"
