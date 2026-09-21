#!/usr/bin/env bash
#SBATCH --job-name=qmla-pilot128
#SBATCH --partition=master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nodelist=treachery
#SBATCH --gres=gpu:1

set -euo pipefail
cd /users/famato/QMLA/code/qmla
export PATH="$HOME/.local/bin:$PATH"
# Prepare the environment before submission; jobs never install dependencies.
hostname
nvidia-smi
uv run --no-sync python -u -m scripts.benchmark --config configs/hpc.toml --profile pilot128 --device cuda
