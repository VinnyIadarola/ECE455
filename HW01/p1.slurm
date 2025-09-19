#!/usr/bin/env zsh
#SBATCH --job-name=p1
#SBATCH --partition=instruction
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:01:00
#SBATCH --output=hello.output

cd $SLURM_SUBMIT_DIR


host_name = $(hostname)
echo "Hello from node $host_name"

