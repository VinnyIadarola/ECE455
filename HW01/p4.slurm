#~/usr/bin/env zsh
#SBATCH --job-name=p4
#SBATCH --partition=instruciton
#SBATCH --time=00:03:00
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --output=gpujob.output

# go into submission directory
cd $SLURM_SUBMIT_DIR

module load nvidia/cuda module

nvcc kernel.cu -o kernel
./kernel