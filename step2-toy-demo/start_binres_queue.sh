#!/bin/bash
# 在后台启动 run_binres_queue.sh(脱离终端与当前会话,关掉 Claude 或退出登录都不影响)。已经在跑就不重复启动。
cd "$(dirname "$0")" || exit 1
if pgrep -f "run_binres_queue.sh" > /dev/null; then
  echo "run_binres_queue.sh is already running (pid $(pgrep -f run_binres_queue.sh | paste -sd' ' -))"
  exit 0
fi
rm -f results/improve/binres_queue.stop
setsid nohup bash ./run_binres_queue.sh >> results/improve/binres_queue.log 2>&1 < /dev/null &
sleep 2
echo "started run_binres_queue.sh (pid $(pgrep -f run_binres_queue.sh | paste -sd' ' -)); log: results/improve/binres_queue.log"
