#!/bin/bash
# 二值残差流的能耗估算(与 results/energy 同一套模型与参数,只换检查点)。只用 GPU 1。
#   第一组 results/energy_binres:d=128。倍数以原来的 T2(实值残差,T=2)为分母,和 results/energy 的表可直接对照。
#     T2 / T4 / T8 实值残差;TR2 三值 sat T=2;TC4 三值 satc T=4(精确梯度对照);
#     BR4 二值 src T=4(种子 1);BR8 / BR8b 二值 src T=8(种子 1、2,达到判据的两条);BX8 二值 xor T=8。
#   第二组 results/energy_binres_w256:d=256,稠密数字侧按加宽的 A0 计。T4W 实值残差 T=4;BR4W 二值 src T=4。
cd "$(dirname "$0")" || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
I=results/improve
$PY experiments/energy_estimate.py --device cuda --ref T2 --out results/energy_binres --gpu-json results/energy/gpu.json \
  --models T2=$I/T2_c/model.pt,T4=$I/T4_c/model.pt,T8=$I/T8_c/model.pt,TR2=$I/ternres_sat_T2_c/model.pt,TC4=$I/ternres_satc_T4_c/model.pt,BR4=$I/binres_src_T4_s1_c/model.pt,BR8=$I/binres_src_T8_s1_c/model.pt,BR8b=$I/binres_src_T8_s2_c/model.pt,BX8=$I/binres_xor_T8_c/model.pt \
  > results/energy_binres/run.log 2>&1
$PY experiments/energy_estimate.py --device cuda --ref T4W --out results/energy_binres_w256 --gpu-json "" --a0 $I/W_a0_d256/model.pt \
  --models T4W=$I/WT4_c/model.pt,BR4W=$I/binres_src_T4_W_c/model.pt \
  > results/energy_binres_w256/run.log 2>&1
echo "energy runs finished $(date '+%F %T')"
