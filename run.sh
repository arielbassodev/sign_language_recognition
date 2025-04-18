#!/bin/bash
# Submission script
#SBATCH --time=1-00:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=4000 # megabytes
#SBATCH --partition=gpu
#
#SBATCH --mail-user=ariel.bassomadjoukeng@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb

python ./training/main.py