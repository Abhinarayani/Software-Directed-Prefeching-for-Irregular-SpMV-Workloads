#!/usr/bin/env zsh


# Selecting the partition
#SBATCH -p instruction

# Time limt
#SBATCH -t 0-00:10:00

# Job Name
#SBATCH --job-name=stage1

# Output and error files
#SBATCH -o stage1-%j.out -e stage1-%j.err

# Allocate CPU
#SBATCH -c 1

#Allocate GPU
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

# Compile
nvcc -O3 gpu_spmv_stage1_naive.cu -o spmv_stage1

# Profile with Nsight Compute to see stalls
# We are looking for "Long Scoreboard Stalls" (Waiting for Memory)
# ncu --metrics sm__sass_thread_inst_executed_op_memory_stall_long_scoreboard.sum ./spmv_stage1
ncu --section MemoryWorkloadAnalysis ./spmv_stage1