#!/bin/bash
#SBATCH --job-name=pond-a100-scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=a100_scaling_%j.out
#SBATCH --error=a100_scaling_%j.err

set -euo pipefail

module load CUDA/12.0

cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SUITES="${SUITES:-capacity grid radius}"
export REPEATS="${REPEATS:-3}"
export TICKS="${TICKS:-96}"
export TICKS_PER_DAY="${TICKS_PER_DAY:-96}"
export LOAD_CUDA_MODULE=0

bash scripts/run_a100_scaling.sh
