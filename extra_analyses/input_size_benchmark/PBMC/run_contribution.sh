#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100_48C_96T_512GB
#SBATCH --time=24:00:00
#SBATCH --mem=100gb
#SBATCH --array=0-11

cd /data/groups/vib.ai/stein.aerts/sdewin/PhD/papers/CREsted_2025/CREsted-paper/extra_analyses/input_size_benchmark/PBMC

model_dirs=(input_1057_202585185910
input_132_20258602526
input_2114_20258591653
input_264_202585234212
input_4228_20258591653
input_528_20258521175
input_same_shape_1057_20258516745
input_same_shape_132_20258516744
input_same_shape_2114_20258516745
input_same_shape_264_20258516744
input_same_shape_4228_20258516745
input_same_shape_528_20258516744)

MODEL_DIR=${model_dirs[$SLURM_ARRAY_TASK_ID]}

echo ${MODEL_DIR}

./get_contribution.py -f ${MODEL_DIR}
