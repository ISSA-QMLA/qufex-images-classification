#!/usr/bin/env bash
#SBATCH --job-name=qmla-bottleneck
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

hostname
nvidia-smi

uv run --no-sync python -u -m scripts.analyze_bottleneck \
  --run-id "${QMLA_RUN_ID:-12560}" \
  --checkpoint-root /data/qmla/famato/checkpoints \
  --processed-cache /data/qmla/famato/data/processed/gz2-128-c5b8192136704e32 \
  --output-root /data/qmla/famato/results/bottleneck_analysis \
  --device cuda --batch-size 32 --cpu-threads 8

