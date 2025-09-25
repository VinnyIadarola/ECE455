#!/usr/bin/env zsh
#SBATCH --job-name=p2
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:01:00
#SBATCH --output=compile_run.output


# log in the submission directory
cd $SLURM_SUBMIT_DIR


# load gcc for compiling c++ program
module load gcc/13.2.0

# run
g++ main.cpp -o main
./main



