#!/bin/bash
# 逐项去掉:其余保持标称,每次只关一种非理想性(7 个模型并行)。
set -u
cd "$(dirname "$0")" || exit 1
./run_circuit.sh leaveout_T2_c --run results/improve/T2_c --mode leaveout
./run_circuit.sh leaveout_T2_c_s1 --run results/improve/T2_c_s1 --mode leaveout
./run_circuit.sh leaveout_T2_c_s2 --run results/improve/T2_c_s2 --mode leaveout
./run_circuit.sh leaveout_T4_c --run results/improve/T4_c --mode leaveout
./run_circuit.sh leaveout_T4_c_s1 --run results/improve/T4_c_s1 --mode leaveout
./run_circuit.sh leaveout_T4_c_s2 --run results/improve/T4_c_s2 --mode leaveout
./run_circuit.sh leaveout_T1_CL2 --run results/conversion-sweep/CL2_l1_lif225_lr3e3_12k --out results/improve/circuit_T1_CL2 --mode leaveout
