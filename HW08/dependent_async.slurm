#!usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=dependent_async.output


cd $SLURM_SUBMIT_DIR
module load gcc
g++ -std=c++20 dependant_async.cpp -o dependent_async -I path/to/taskflow -pthreads
./dependent_async