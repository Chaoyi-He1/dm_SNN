#!/bin/bash
# 电路验收、误差隔离与容限扫描的全部运行(CPU,每个进程单线程,并行)。
set -u
cd "$(dirname "$0")" || exit 1
CL2=results/conversion-sweep/CL2_l1_lif225_lr3e3_12k
T1OUT=results/improve/circuit_T1_CL2
./run_circuit.sh accept_T2_c --run results/improve/T2_c
./run_circuit.sh accept_T2_c_s1 --run results/improve/T2_c_s1
./run_circuit.sh accept_T2_c_s2 --run results/improve/T2_c_s2
./run_circuit.sh accept_T4_c --run results/improve/T4_c
./run_circuit.sh accept_T4_c_s1 --run results/improve/T4_c_s1
./run_circuit.sh accept_T4_c_s2 --run results/improve/T4_c_s2
./run_circuit.sh accept_T1_CL2 --run "$CL2" --out "$T1OUT"
./run_circuit.sh isolate_T2_c --run results/improve/T2_c --mode isolate
./run_circuit.sh isolate_T4_c --run results/improve/T4_c --mode isolate
./run_circuit.sh isolate_T1_CL2 --run "$CL2" --out "$T1OUT" --mode isolate
./run_circuit.sh knobs_T2_c --run results/improve/T2_c --mode knobs
./run_circuit.sh knobs_T4_c --run results/improve/T4_c --mode knobs
./run_circuit.sh knobs_T1_CL2 --run "$CL2" --out "$T1OUT" --mode knobs
