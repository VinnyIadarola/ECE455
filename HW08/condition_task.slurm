#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=condition_task.output

cd $SLURM_SUBMIT_DIR

module load gcc

g++ -std=c++20 condition_task.cpp -o condition_task -I path/to/taskflow -pthreads
./condition_task