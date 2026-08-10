#!/bin/bash
#
#SBATCH --job-name=qmla-preprocess
#SBATCH --partition=master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --nodelist=treachery
#

set -euo pipefail

hostname

cd /users/famato/QMLA/code/qmla

uv run --no-sync python -m scripts.preprocess_data \
    --config configs/hpc.toml