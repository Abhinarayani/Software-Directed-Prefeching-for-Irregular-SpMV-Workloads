#!/usr/bin/env zsh


# Selecting the partition
#SBATCH -p instruction

# Time limt
#SBATCH -t 0-00:10:00

# Job Name
#SBATCH --job-name=stage3

# Output and error files
#SBATCH -o stage3-%j.out -e stage3-%j.err

# Allocate CPU
#SBATCH -c 1

#Allocate GPU
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/13.0.0

# Compile
nvcc -O3 gpu_spmv_stage3_tiled.cu -o spmv_stage3

# Profile with Nsight Compute to see stalls
# We are looking for "Long Scoreboard Stalls" (Waiting for Memory)
ncu --section MemoryWorkloadAnalysis ./spmv_stage3