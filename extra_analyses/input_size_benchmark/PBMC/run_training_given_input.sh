#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100_48C_96T_512GB
#SBATCH --time=12:00:00
#SBATCH --mem=200gb
#SBATCH --array=0-5

cd /data/groups/vib.ai/stein.aerts/sdewin/PhD/papers/CREsted_2025/CREsted-paper/extra_analyses/input_size_benchmark/PBMC

INPUT_SIZES=(4228 2114 1057 528 264 132)
INPUT_SIZE=${INPUT_SIZES[$SLURM_ARRAY_TASK_ID]}

echo ${INPUT_SIZE}

./train_given_input.py -i ${INPUT_SIZE}

