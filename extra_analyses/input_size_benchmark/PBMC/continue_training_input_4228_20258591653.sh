#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100_64C_128T_2TB
#SBATCH --time=24:00:00
#SBATCH --mem=100gb

cd /data/groups/vib.ai/stein.aerts/sdewin/PhD/papers/CREsted_2025/CREsted-paper/extra_analyses/input_size_benchmark/PBMC

./continue_training.py -f input_4228_20258591653
