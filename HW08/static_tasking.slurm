#!/user/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpu-per-task=1
#SBATCH --output=static_tasking.output

cd $SLURM_SUBMIT_DIR

# load in c++

module load gcc

g++ -std=c++20 static_tasking.cpp -o static_tasking -I path/to/taskflow -pthread

./static_tasking
