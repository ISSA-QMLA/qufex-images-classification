#!/bin/bash
#
#SBATCH --job-name=qmla-qufex-cpu
#SBATCH --partition=master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nodelist=treachery
#

set -euo pipefail

hostname

cd /users/famato/QMLA/code/qmla

# /usr/bin/time -v \
uv run --no-sync python -u -m scripts.train \
    --config configs/smoke_cpu.toml \
    --model qufex \
    --device cpu
