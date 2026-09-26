#!/bin/bash
# 跟踪若干实验日志:watch_sweep.sh <name> [<name>...]
# 只输出阶段行([...])、每 1000 步的评估行和错误,每行前缀为实验名。
cd "$(dirname "$0")/../results/conversion-sweep" || exit 1
files=()
for n in "$@"; do files+=("$n.log"); done
tail -n0 -F "${files[@]}" 2>/dev/null | awk '
  /^==> /   { f = $2; sub(/\.log$/, "", f); next }
  /^\[|Traceback|Error|"step": [0-9]*000,/ { print f ": " $0; fflush() }'
