#!/usr/bin/env zsh
#SBATCH --job-name=p3
#SBATCH --partition=instruction
#SBATCH --time=00:02:00
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --output=multicpu.ouput





# log into the submssion directoy
cd $SLURM_SUBMIT_DIR

# Load gxx for compiling c++ program
module load gcc/13.2.0

# clone
git clone https://github.com/vinnyiadarola/ECE455.git

# run
cd ECE455/HW01
g++ -std=c++17 parallel.cpp -o parallel -pthread
./parallel





