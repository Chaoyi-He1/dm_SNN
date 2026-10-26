#!/bin/bash
# 三值残差种子补跑:胜出规则 sat,先 T=2 的种子 1、2 并行,完成后再 T=1 的种子 1、2 并行(GPU 1)。
cd /data/chaoyi_he/dm_snn/.claude/worktrees/conversion-sweep/step2-toy-demo || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
for T in 2 1; do
  for S in 1 2; do
    nohup $PY experiments/ternary_residual.py chain --name ternres_sat_T${T}_s${S} --merge sat --levels $T \
      --fq-list 0.5,0.75,0.9 --scale-list 1,1.5,2,2.25,2.5,3,4 --lr 3e-3 --steps 12000 --seed $S \
      > results/improve/ternres_sat_T${T}_s${S}_chain.log 2>&1 &
  done
  wait
done
