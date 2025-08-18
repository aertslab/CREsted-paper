#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100_64C_128T_2TB
#SBATCH --time=24:00:00
#SBATCH --mem=100gb
#SBATCH --array=0-5

cd /data/groups/vib.ai/stein.aerts/sdewin/PhD/papers/CREsted_2025/CREsted-paper/extra_analyses/input_size_benchmark/PBMC

INPUT_SIZES=(4228 2114 1057 528 264 132)
INPUT_SIZE=${INPUT_SIZES[$SLURM_ARRAY_TASK_ID]}

echo ${INPUT_SIZE}

./train_given_input_same_shape.py -i ${INPUT_SIZE}
