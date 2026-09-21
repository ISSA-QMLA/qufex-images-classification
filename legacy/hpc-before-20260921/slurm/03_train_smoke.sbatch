#!/bin/bash
#
#SBATCH --job-name=qmla-smoke
#SBATCH --partition=master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=treachery
#

set -euo pipefail

hostname

nvidia-smi

cd /users/famato/QMLA/code/qmla

MODEL=${MODEL:-classical}
CONFIG=${CONFIG:-configs/hpc.toml}

/usr/bin/time -v \
uv run --no-sync python -u -m scripts.train \
    --config "$CONFIG" \
    --model "$MODEL" \
    --device cuda
