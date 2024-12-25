#!/bin/bash

#SBATCH --job-name=pytorch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3

source activate ShadowFree

source pipline_java.sh
