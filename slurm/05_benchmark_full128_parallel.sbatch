#!/usr/bin/env bash
#SBATCH --job-name=qmla-full128-parallel
#SBATCH --partition=master
#SBATCH --nodelist=treachery
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd /users/famato/QMLA/code/qmla
export PATH="$HOME/.local/bin:$PATH"
# Slurm assigns one GPU to each task through CUDA_VISIBLE_DEVICES.
# Prepare the full128 cache and environment before submitting this array.
models=(qufex cnn_replacement direct_cnn)
model="${models[$SLURM_ARRAY_TASK_ID]}"
run_root="/data/qmla/famato/runs/full128_array_${SLURM_ARRAY_JOB_ID}"
# Include the array ID in each basename: checkpoint/result directories use it.
run_dir="${run_root}/full128_array_${SLURM_ARRAY_JOB_ID}_${model}"
printf 'Model: %s\nRun directory: %s\n' "$model" "$run_dir"
hostname
nvidia-smi
uv run --no-sync python -u -m scripts.train \
    --config configs/hpc.toml --profile full128 --device cuda \
    --model "$model" --run-dir "$run_dir"
