#!/bin/bash
# 误差来源诊断(单步对照、误差随位置累积、极小扰动),三个模型并行。
set -u
cd "$(dirname "$0")" || exit 1
./run_circuit.sh diagnose_T2_c --run results/improve/T2_c --mode diagnose
./run_circuit.sh diagnose_T4_c --run results/improve/T4_c --mode diagnose
./run_circuit.sh diagnose_T1_CL2 --run results/conversion-sweep/CL2_l1_lif225_lr3e3_12k --out results/improve/circuit_T1_CL2 --mode diagnose
