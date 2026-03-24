#!/usr/bin/env zsh

# Selecting the partition
#SBATCH -p instruction

# Time limt
#SBATCH -t 0-00:05:00

# Job Name
#SBATCH --job-name=omp_prefetcher

# Output and error files
#SBATCH -o omp_prefetcher.out -e omp_prefetcher.err

# Allocate CPU
#SBATCH --cpus-per-task=4  # Reserving 4 physical cores - not sure how many will be needed excatly

cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Compilation Command
g++ -O3 -fopenmp spmv_baseline.cpp -o spmv_baseline

# Run Command
./spmv_baseline