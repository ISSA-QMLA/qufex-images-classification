#!/usr/bin/env bash
#SBATCH --job-name=qmla-download
#SBATCH --partition=master
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --nodelist=treachery

set -euo pipefail
cd /users/famato/QMLA/code/qmla
export PATH="$HOME/.local/bin:$PATH"
# Prepare the environment before submission; jobs never install dependencies.
hostname
uv run --no-sync python -u -m scripts.extract_data --config configs/hpc.toml
