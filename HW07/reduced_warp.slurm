#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=reduce_warp.output

set -e

# run from the directory where sbatch was invoked
cd "$SLURM_SUBMIT_DIR"

# load CUDA (adjust module name/version to your cluster if needed)
module purge
module load nvidia/cuda

# build and run
nvcc reduce_warp.cu -o reduce_warp
./reduce_warp 6