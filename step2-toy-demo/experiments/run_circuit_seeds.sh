#!/bin/bash
# 标称电路换 5 个器件噪声种子(7 个模型并行)。
set -u
cd "$(dirname "$0")" || exit 1
./run_circuit.sh seeds_T2_c --run results/improve/T2_c --mode seeds
./run_circuit.sh seeds_T2_c_s1 --run results/improve/T2_c_s1 --mode seeds
./run_circuit.sh seeds_T2_c_s2 --run results/improve/T2_c_s2 --mode seeds
./run_circuit.sh seeds_T4_c --run results/improve/T4_c --mode seeds
./run_circuit.sh seeds_T4_c_s1 --run results/improve/T4_c_s1 --mode seeds
./run_circuit.sh seeds_T4_c_s2 --run results/improve/T4_c_s2 --mode seeds
./run_circuit.sh seeds_T1_CL2 --run results/conversion-sweep/CL2_l1_lif225_lr3e3_12k --out results/improve/circuit_T1_CL2 --mode seeds
