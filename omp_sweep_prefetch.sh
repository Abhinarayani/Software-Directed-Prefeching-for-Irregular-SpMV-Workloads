#!/usr/bin/env zsh

# Selecting the partition
#SBATCH -p instruction

# Time limit
#SBATCH -t 0-00:30:00

# Job Name
#SBATCH --job-name=omp_prefetcher_sweep

# Output and error files
#SBATCH -o omp_prefetcher_sweep.out -e omp_prefetcher_sweep.err

# Allocate CPU
#SBATCH --cpus-per-task=8 

cd $SLURM_SUBMIT_DIR

# 1. LOAD MODULES IN CORRECT ORDER
# You must load the toolchain first as per the cluster help text
module load gnu15/15.1.0
module load likwid/5.4.1

# Set the number of OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compile on the compute node itself to ensure architecture match
g++ -O3 -fopenmp spmv_baseline.cpp -o spmv_baseline
g++ -O3 -fopenmp spmv_prefetch.cpp -o spmv_prefetch

# Check if compilation succeeded before running
if [[ $? -ne 0 ]]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Starting Architectural Sweep: Baseline vs Prefetch"
echo "--------------------------------------------------"

# 2. PROFILE THE BASELINE
echo "Profiling Baseline (No Prefetch):"
# -C 0-7: Pins execution to your 8 reserved cores
# -g L2CACHE: Measures L2 cache misses, hits, and bandwidth
likwid-perfctr -C 0-7 -g L2CACHE ./spmv_baseline

echo "\n--------------------------------------------------"

# 3. SWEEP THROUGH PREFETCH DISTANCES (k = 1, 4, 8, 16, 32, 64)
for k in 1 4 8 16 32 64
do
    echo "Profiling Prefetch with k=$k:"
    # -m provides machine-readable output tables in your .out file
    likwid-perfctr -C 0-7 -g L2CACHE ./spmv_prefetch $k
    echo "--------------------------------------------------"
done