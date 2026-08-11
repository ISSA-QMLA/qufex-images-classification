# Quantum Machine Learning for Galaxy Morphology

This project compares a classical CNN with a hybrid CNN–QuFeX classifier on three clean Galaxy Zoo 2 morphology classes:

- smooth;
- unbarred spiral;
- barred spiral.

The quantum module follows the eight-qubit, four-parameter QuFeX v1 circuit from Jain and Kalev, [arXiv:2501.13165v1](https://arxiv.org/html/2501.13165v1).

## Architecture

```text
64x64 RGB image
  -> five configurable Conv/BatchNorm/ReLU/Pool encoder blocks
  -> 2x2x16 bottleneck (default)
  -> QuFeX residual (hybrid mode) or identity (classical mode)
  -> configurable CNN -> global average pool -> MLP -> 3 logits
```

The default encoder follows `qcnn_res-small.ipynb`: channel widths `[4, 8, 8, 8, 16]`, two convolutions per block, and five 2x2 pooling operations reduce a `64x64x3` image directly to `2x2x16`. The default 8(1) QuFeX splits the bottleneck into eight adjacent-channel `2x2x2` groups. Each group supplies eight angle-encoded values to the same shared eight-qubit circuit. Even and odd Pauli-Z outputs are reshaped into two `2x2` maps, the eight group outputs are concatenated back to `2x2x16`, and the result is added residually to the encoder bottleneck.

The published circuit families are selected with `quantum.qubits` and `quantum.filters`:

- `qubits = 8`, `filters = 1`: mixes pairs of input maps and returns 16 maps;
- `qubits = 4`, `filters = 1`: processes each input map separately and returns 16 maps;
- `qubits = 4`, `filters = 2`: applies two independent circuits to each map and returns 32 maps. As in `qcnn_res-small4-2.ipynb`, the residual tensor is duplicated to 32 channels before addition.

## Installation

Python 3.11 or newer is required. 
PyTorch is required but absent from `pyproject.toml` because its correct package depends on the current hardware:
[PyTorch installation selector](https://pytorch.org/get-started/locally/). 

To install this project:

```bash
python -m venv .venv
source .venv/bin/activate

# Example only: choose the index/version matching the cluster CUDA stack.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

The project does not require torchvision. 
Image preprocessing uses Pillow and training augmentation uses native Torch tensor operations.

For development tests:

```bash
python -m pip install -e .
python -m pip install pytest
pytest
```

With `uv`, run `uv sync --group dev` first and install the cluster-specific PyTorch wheel afterward with `uv pip install ...`. A later exact `uv sync` may remove packages that are intentionally absent from the lockfile; use `uv sync --inexact` when preserving that custom Torch installation.

## Configuration

All stable settings live in [`configs/default.toml`](configs/default.toml). Copy that file for each experiment and change paths, encoder depth/channels, dense hidden neurons, optimizer settings, or backend without editing Python code.

Relative data/output paths are resolved from `paths.project_root`, which is itself resolved relative to the TOML file. Each run stores both the source TOML and a fully resolved JSON configuration.

Important constraints for source-faithful QuFeX are validated:

- `model.compression_channels = 16`;
- `model.quantum_spatial_size = 2`;
- `quantum.qubits` and `quantum.filters` must be `8/1`, `4/1`, or `4/2`.

CLI values take precedence over TOML only for job-specific settings such as `--device`, `--model`, `--resume`, `--checkpoint`, and `--run-dir`.

## Pipeline

Run every command from an installed environment. `python -m ...` also works consistently from the repository root.

### 1. Download and extract

```bash
python -m scripts.extract_data --config configs/default.toml
```

This downloads the official 3.4 GB Zenodo image archive and mapping plus the Hart et al. debiased catalogue. Downloads are resumable, known Zenodo MD5 checksums are verified, extraction rejects unsafe ZIP paths, and SHA-256 hashes are recorded.

### 2. Preprocess

```bash
python -m scripts.preprocess_data --config configs/default.toml
```

Labels use the Hart/Willett clean flags:

- smooth flag;
- spiral flag and no-bar flag;
- spiral flag and bar flag.

Ambiguous/conflicting rows are dropped. Object IDs are deduplicated before a fixed stratified 70/15/15 split. Images and labels are saved as memory-mapped `.npy` arrays. Normalization statistics are computed from the training split only.

### 3. Train both single-run prototypes

```bash
python -m scripts.train --config configs/default.toml --model classical
python -m scripts.train --config configs/default.toml --model qufex
```

Resume a preempted job:

```bash
python -m scripts.train \
  --config runs/qufex_YYYYMMDD_HHMMSS/resolved_config.toml \
  --model qufex \
  --resume checkpoints/qufex_YYYYMMDD_HHMMSS/latest.pt \
  --run-dir runs/qufex_YYYYMMDD_HHMMSS
```

The primary validation criterion is macro-F1. Training uses class-weighted cross-entropy, AMP for the CNN, float32 for QuFeX, TF32 where available, pinned data loading, non-blocking CUDA copies, and channels-last convolution tensors.

### 4. Test and save results

```bash
python -m scripts.test_model \
  --config runs/qufex_YYYYMMDD_HHMMSS/resolved_config.toml \
  --checkpoint checkpoints/qufex_YYYYMMDD_HHMMSS/best.pt
```

Outputs include `metrics.json`, `classification_report.txt`, `predictions.csv`, `confusion_matrix.png`, resolved/source configuration, and package versions.

## SLURM examples

Cluster module names, partitions, accounts, and wall-time policies differ, so this repository keeps scheduler settings out of fixed `.sbatch` files.

CPU download/preprocessing job:

```bash
sbatch --job-name=gz2-prep --cpus-per-task=16 --mem=64G --time=08:00:00 \
  --wrap='source .venv/bin/activate && python -m scripts.preprocess_data --config configs/default.toml'
```

Single-NVIDIA-GPU training job:

```bash
sbatch --job-name=qufex --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --wrap='source .venv/bin/activate && python -m scripts.train --config configs/default.toml --model qufex'
```

If the eight-qubit simulation is slower on GPU, set both `training.device = "cpu"` and `quantum.backend = "lightning.qubit"` for a CPU comparison. `lightning.gpu` additionally requires the cluster-compatible PennyLane Lightning GPU/cuQuantum installation and usually benefits larger circuits more than this eight-wire model.

## Research limitations

- The clean flags select high-confidence morphological extremes and exclude many ambiguous galaxies.
- A single split and seed are suitable for plumbing and preliminary comparison only.
- The classical bypass is an ablation, not a compute-matched proof that any improvement is specifically quantum.
- Strong conclusions require repeated seeds/splits and stronger classical bottleneck controls.

Please cite the [Galaxy Zoo 2 data release](https://academic.oup.com/mnras/article/435/4/2835/1022913), the [Hart et al. debiasing work](https://academic.oup.com/mnras/article/461/4/3663/2608720), the [Zenodo image release](https://zenodo.org/records/3565489), and the QuFeX paper when using this project.
