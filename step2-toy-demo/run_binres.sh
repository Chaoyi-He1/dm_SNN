#!/bin/bash
# 二值(0/1)残差流实验的启动脚本。只用 GPU 1;每条链 = iscan -> A1 -> bscan -> C,日志写 results/improve/<名字>_chain.log。
#
#   ./run_binres.sh                         第一批(种子 0,d=128)7 条链并行:
#                                           sr × T∈{1,2,4,8},xor × T∈{1,2},三值 sat T=4(与二值 T=8 电平数相同的对照)
#   ./run_binres.sh <merge> <T> [seed] [wide]   只跑一条:merge = sr | src | xor 走二值残差;merge = sat 走三值残差(原驱动);
#                                           merge = satc 是三值残差 + 精确梯度的合并(对照,走 binary_residual.py);
#                                           第 4 个参数写 wide 时用加宽的 d=256(A0 取 results/improve/W_a0_d256)
#   ./run_binres.sh name <merge> <T> [seed] [wide]   只打印这条链的名字,不启动(run_binres_queue.sh 用它)
# 名字:binres_<merge>_T<T>[_W][_s<seed>](三值对照是 ternres_<sat|satc>_T<T>[_W][_s<seed>])。脚本等所有链结束才返回。
cd "$(dirname "$0")" || exit 1
PY=/data/chaoyi_he/dm_snn/step2-toy-demo/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
COMMON="--fq-list 0.5,0.75,0.9 --scale-list 1,1.5,2,2.25,2.5,3,4 --lr 3e-3 --steps 12000"
WIDE="--a0 results/improve/W_a0_d256/model.pt --d 256 --dk 64 --dv 64 --d-ff 512 --heads 4"

name_of() {
  local M=$1 T=$2 S=${3:-0} W=${4:-}
  local name=binres_${M}_T${T}
  if [ "$M" = sat ] || [ "$M" = satc ]; then name=ternres_${M}_T${T}; fi
  if [ "$W" = wide ]; then name=${name}_W; fi
  if [ "$S" != 0 ]; then name=${name}_s${S}; fi
  echo "$name"
}

one() {
  local M=$1 T=$2 S=${3:-0} W=${4:-}
  local script=binary_residual.py extra="" name
  name=$(name_of "$@")
  if [ "$M" = sat ]; then script=ternary_residual.py; fi
  if [ "$W" = wide ]; then extra=$WIDE; fi
  setsid nohup $PY experiments/$script chain --name "$name" --merge "$M" --levels "$T" $COMMON --seed "$S" $extra \
    > "results/improve/${name}_chain.log" 2>&1 &
}

if [ "$1" = name ]; then
  shift
  name_of "$@"
  exit 0
fi
if [ $# -ge 2 ]; then
  one "$@"
else
  for spec in "sr 1" "sr 2" "sr 4" "sr 8" "xor 1" "xor 2" "sat 4"; do
    one $spec
  done
fi
wait
