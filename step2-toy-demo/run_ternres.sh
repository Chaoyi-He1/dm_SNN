#!/bin/bash
# 三值残差实验:{sat, and} × {T=1, T=2} 四条链,GPU 1 上同一 T 的两条并行,T=1 完成后再跑 T=2。
cd /data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
for T in 1 2; do
  for M in sat and; do
    nohup $PY experiments/ternary_residual.py chain --name ternres_${M}_T${T} --merge $M --levels $T \
      --fq-list 0.5,0.75,0.9 --scale-list 1,1.5,2,2.25,2.5,3,4 --lr 3e-3 --steps 12000 --seed 0 \
      > results/improve/ternres_${M}_T${T}_chain.log 2>&1 &
  done
  wait
done
