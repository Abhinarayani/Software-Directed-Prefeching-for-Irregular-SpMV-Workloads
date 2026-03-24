#!/usr/bin/env zsh

# Selecting the partition
#SBATCH -p instruction
#SBATCH -t 0-00:45:00
#SBATCH --job-name=spmv_large_scale
#SBATCH -o spmv_large_scale.out -e spmv_large_scale.err
#SBATCH --cpus-per-task=8 

cd $SLURM_SUBMIT_DIR

# 1. Load the specific GNU and LIKWID versions found on Euler
module load gnu15/15.1.0
module load likwid/5.4.1

# 2. Basic Compilation (No special Likwid flags needed)
g++ -O3 -fopenmp spmv_large_scale.cpp -o spmv_large_scale

# 3. Set Threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting Large Scale Sweep (No Markers)"
echo "--------------------------------------------------"

for k in 0 1 4 8 16 32 64
do
    echo "Profiling Prefetch with k=$k:"
    # Removed -M 1, kept -f (force) and -g L2CACHE
    likwid-perfctr -f -C 0-7 -g L2CACHE ./spmv_large_scale $k
    echo "--------------------------------------------------"
done