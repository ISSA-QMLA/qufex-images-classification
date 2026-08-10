#!/bin/bash
#
#SBATCH --job-name=qmla-download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --nodelist=treachery
#

set -eou pipefail

hostname

cd /users/famato/QMLA/code/qmla
source .venv/bin/activate

uv run --no-sync python -m scripts.extract_data --config configs/hpc.toml