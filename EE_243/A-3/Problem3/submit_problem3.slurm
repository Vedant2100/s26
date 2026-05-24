#!/bin/bash
#SBATCH --job-name=3dgs_p3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=3dgs_%j.log

srun --gres=gpu:1 ./run_problem3.sh
