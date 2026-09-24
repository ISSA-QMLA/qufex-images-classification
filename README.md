# Galaxy Zoo 2: QuFeX and classical CNN experiments

Train and test three models on clean **smooth**, **unbarred spiral**, and **barred spiral** Galaxy Zoo 2 images. Every supported architecture and experiment preset is configured in **[`configs/experiments.toml`](configs/experiments.toml)**. Python 3.11+; Windows/Linux; CPU or one NVIDIA CUDA GPU.

## Start here

Use **uv** from the repository root on both Windows and Linux. Create a fresh environment on each machine; do not copy `.venv` between machines. The checked-in `.python-version` selects Python 3.11, and `uv.lock` pins the project dependencies.

```bash
uv sync --locked --group dev --inexact
```

This creates `.venv` and installs the project and development tools. `--inexact` preserves an existing machine-specific PyTorch installation. No environment activation is needed for the commands below.

On a fresh environment, install **one** PyTorch build:

```bash
# CPU-only environment
uv pip install torch --torch-backend=cpu

# Alternatively, select a backend from the detected NVIDIA driver
uv pip install torch --torch-backend=auto
```

On a cluster login node without the target GPU/driver, select the backend explicitly according to your site's stack, for example `uv pip install torch --torch-backend=cu128` for a compatible CUDA 12.8 setup. See the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection). Torch remains machine-specific and outside `uv.lock`; no torchvision is required. Each run records its installed Torch version.

The examples use `uv run --no-sync` to execute the already-prepared environment without synchronizing it during a run. After dependency changes, repeat `uv sync --locked --group dev --inexact`; if the lockfile is stale, update it deliberately with `uv lock` first. See [uv's syncing behavior](https://docs.astral.sh/uv/concepts/projects/sync/). An exact `uv sync` can remove manually installed Torch because it is absent from the project dependency list.

Download the official image archive and catalogues once:

```bash
uv run --no-sync python -m scripts.extract_data
```

Then run a real-image smoke experiment:

```bash
uv run --no-sync python -m scripts.train --profile smoke --device cpu
```

This prepares a separate tiny dataset cache, trains the **one model selected by `[run].model`**, and evaluates its best checkpoint. Downloads are explicit: training never starts a download automatically. Existing raw images can be reused. A first run also indexes the image archive and reads the catalogues, so preparation time is separate from training time.

## One TOML controls all experiments

Select defaults at the top of `configs/experiments.toml`:

```toml
[run]
model = "qufex" # qufex | cnn_replacement | direct_cnn
profile = "smoke" # full64 | full128 | smoke | small_learning
```

| Profile | Image size | Train / validation / test images | Epochs |
|---|---:|---|---:|
| `full64` | 64×64 | All clean master-split images | Up to 50 |
| `full128` | 128×128 | All clean master-split images | Up to 50 |
| `smoke` | 32×32 | 96 / 48 / 48 | 1 |
| `small_learning` | 32×32 | 1,200 / 300 / 300 | 3 |

Precedence is **base tables → selected profile → explicit job-level CLI overrides**. Nested tables merge recursively; arrays replace entirely. Zero subset limits select the whole split. Limits larger than a split use all available images. Invalid and unknown settings fail before model allocation.

Architecture tables:

- `[architectures.shared]`: encoder channels/depth, convolutions per block, odd kernel sizes, max/average/no pooling, pooling size, batch/group/no normalization, activation, bottleneck projection, post-extraction convolutions, classifier widths, and dropout for QuFeX and its CNN replacement.
- `[architectures.direct]`: independent direct-CNN encoder; unspecified generic block/head settings inherit from `shared`. It does not inherit the hybrid encoder widths, projection, bottleneck dimensions, or post-layer widths. Direct CNN never uses the quantum bottleneck.
- `[architectures.replacement]`: hidden channels, hidden/output kernel sizes, normalization, hidden/output activation. Each filter's final channel count is inferred from its input group.
- `[quantum]`: `qubits`, `filters`, angle scale, backend, differentiation method, and shots. Supported published circuit families are **8/1, 4/1, 4/2**. Circuit gate connectivity is defined in Python.
- `[profiles.NAME.architectures.shared]` and corresponding `direct`/`replacement` tables: architecture changes specific to a preset, in the same file.

For example, edit these existing smoke-profile tables to change complexity:

```toml
[profiles.smoke.architectures.shared]
encoder_channels = [4, 8, 16]
convolutions_per_block = 1
post_channels = []
classifier_hidden_neurons = [8]

[profiles.smoke.architectures.replacement]
hidden_channels = [4, 4]
```

Connecting dimensions are inferred. Kernels use same-size padding. `projection="auto"` inserts a 1×1 projection when needed, `conv` always inserts it, and `identity` requires matching channels. An adaptive pool produces the required **16×2×2** hybrid interface; the encoder must retain at least 2×2 spatial dimensions. These interface constraints do not apply to the direct CNN. Hidden classifier/post-layer arrays may be empty. Group normalization uses one group per sample.

The quantum layer and its CNN control group adjacent feature-map pairs for 8/1, or individual maps for four-qubit variants. Filters share weights across groups. With two filters, both outputs and residuals interleave per input channel. The direct CNN provides a broader conventional baseline; it is not a strict layer ablation or parameter-matched model.

## Training, comparison, and testing

```bash
# TOML-selected model; prepares its cache if missing, then trains and tests
uv run --no-sync python -m scripts.train
uv run --no-sync python -m scripts.train --profile full128 --model cnn_replacement
uv run --no-sync python -m scripts.train --profile small_learning --model direct_cnn --device cpu

# All three model families, sequentially, for the configured quantum variant
uv run --no-sync python -m scripts.benchmark --profile full64

# Prepare data separately before reserving a GPU
uv run --no-sync python -m scripts.preprocess_data --profile full64

# Evaluate using the checkpoint's saved architecture/configuration
uv run --no-sync python -m scripts.test_model --checkpoint checkpoints/RUN_NAME/best.pt --device cpu
```

`training.seeds = [42]` makes a preliminary single-seed comparison; use e.g. `[42, 43, 44]` for repeated training on the same master split. Each model receives identical split/subset identities, augmentation policy, normalization, optimizer settings, and seed list. Quantum variants are selected explicitly in TOML; benchmarking does not automatically sweep them.

Full profiles stop after eight epochs without validation macro-F1 improvement, with a 50-epoch ceiling. Smaller profiles disable early stopping. The held-out test split is evaluated after selecting the best validation checkpoint. `training.device="auto"` selects available CUDA, otherwise CPU. Explicit CUDA requests fail if unavailable; OOM errors never silently change the experiment. AMP/TF32 acceleration applies on CUDA; quantum execution is outside autocast. Set `training.deterministic=true` for deterministic supported kernels.

The default analytic `default.qubit`/`backprop` combination supports CPU and CUDA. Other PennyLane backends require their own compatible installation and differentiation method; changing the backend alone does not guarantee acceleration. Finite shots cannot use backprop. The software validates known incompatible combinations and otherwise reports backend errors.

### Optional learning-rate decay

Scheduling is disabled by default (`scheduler.name = "none"`), including when older saved configurations omit the scheduler table. To compare experiment 2 against the longer-training experiment, keep its initial learning rate, batch sizes, seeds, data and architecture unchanged. Edit the existing `[training]` entries to `epochs = 100` and `early_stopping_patience = 15`, then edit or add this top-level table in the same configuration file (for example, `configs/hpc.toml` on the HPC):

```toml
[scheduler]
name = "reduce_on_plateau"
monitor = "macro_f1"
factor = 0.5
patience = 4
threshold = 0.001
threshold_mode = "abs"
cooldown = 0
min_lr = 0.000001
```

`ReduceLROnPlateau` steps once after each validation pass and changes the rate for the next training epoch. `macro_f1` monitors **validation** macro-F1 in maximization mode; `validation_loss` selects minimization mode. With the example above, an improvement must exceed 0.001 absolute F1, and five consecutive non-improving epochs trigger a halving of the learning rate. `patience = 0` reduces on the first bad epoch after the initial baseline. The floor is `min_lr`. Neither scheduler thresholds nor LR reductions change the existing best-checkpoint/early-stopping rule, which still counts any strict validation-F1 improvement; a reduction does not reset early-stopping patience. `cooldown` delays counting bad epochs after a reduction. Profiles may override settings with `[profiles.NAME.scheduler]`.

Each history row and epoch log records `learning_rate` (used for that epoch) and `next_learning_rate` (after the scheduler step, even if training then stops). The results notebook plots the rates used alongside the loss/score analysis. Start a **fresh run** when enabling or changing the scheduler: resume requires the same scheduler and training settings, and restores scheduler counters together with the optimizer LR. Older format-2 fixed-rate checkpoints remain resumable with scheduling disabled.

The existing three-task HPC Slurm array can continue to call `scripts.train --model ...` once per task using the edited config: each model gets its own process, optimizer and scheduler. No new preprocessing or scheduler CLI flag is needed. Keep the longer-training time allocation; the notebook can compare the new array directory with experiment 1 using matched models and seeds.

## Data integrity and portability

Objects are deduplicated before the fixed stratified 70/15/15 master split. Small subsets are chosen within each split, retain all classes, and approximate original proportions. Normalization is computed only on the selected training images.

Caches are stored under `processed_dir/gz2-SIZE-SIGNATURE`. Their metadata records configuration, source hashes, master/subset identities, array/manifest hashes, and normalization. Training/evaluation check integrity, dimensions, labels, and split disjointness. This includes reading the processed files for hashing; large caches have an up-front I/O cost. Existing loose 128×128 arrays are preserved and never silently reused as 64×64 or 32×32 data. Failed preprocessing builds remain in isolated `.building-*` directories for inspection. If source data change or a cache is damaged, select a new `processed_dir` to rebuild without overwriting the old cache.

Keep runs and checkpoints on persistent storage. You may place processed caches on local scratch and copy complete cache directories between hosts. All commands accept `--project-root`, `--raw-dir`, `--processed-dir`, `--runs-dir`, `--checkpoints-dir`, and `--results-dir`. Relative data paths in canonical TOML resolve from `project_root`, itself relative to the TOML location. Saved effective configurations use absolute paths. Overriding `--project-root` relocates saved paths beneath the previous project root, including Windows-to-Linux transfers; paths on external volumes need their own override. Pass an absolute new project root when relocating a saved run.

Resume at the last saved epoch boundary:

```bash
uv run --no-sync python -m scripts.train --config runs/RUN_NAME/resolved_config.toml \
  --resume checkpoints/RUN_NAME/latest.pt --run-dir runs/RUN_NAME
```

`latest.pt` stores model, optimizer, optional LR scheduler, AMP scaler, Python/NumPy/Torch RNGs, loader generators, history, and the best model so far. Workers are recreated with reproducible seeds each epoch, and spawned workers reopen memory maps instead of copying arrays. Mid-epoch progress is not saved. CPU/CUDA relocation is supported; exact equality across devices or changed worker counts is not promised. `best.pt` is for evaluation, not resume. Format-1 checkpoints and former configuration files are intentionally unsupported; existing run artifacts remain untouched.

## Notebook and SLURM

Use [`notebooks/04-bottleneck-analysis.ipynb`](notebooks/04-bottleneck-analysis.ipynb) to visualize bottleneck interventions and linear-probe results from the HPC. The notebook reads only small CSV/JSON files and regenerates plots; it does not import Torch or load checkpoints, image arrays, or feature caches. Restart the kernel if the earlier notebook version was used for local processing.

Run [`scripts/analyze_bottleneck.py`](scripts/analyze_bottleneck.py) in an HPC compute allocation using the existing project environment:

```bash
uv run --no-sync python -u -m scripts.analyze_bottleneck \
  --run-id 12560 \
  --checkpoint-root /data/qmla/famato/checkpoints \
  --processed-cache /data/qmla/famato/data/processed/gz2-128-c5b8192136704e32 \
  --output-root /data/qmla/famato/results/bottleneck_analysis \
  --device cuda --batch-size 32 --cpu-threads 4
```

The script processes QUFEX and CNN replacement one at a time, writes features batch by batch to memory-mapped `.npy` caches, then fits linear probes. It defaults to run `12560`; change `--run-id` for another experiment. If `--processed-cache` is omitted, the saved checkpoint configuration supplies the dataset path. CPU execution is available with `--device cpu`. A [SLURM template](scripts/slurm/bottleneck-analysis.sbatch.example) includes editable site/resource settings.

Copy only `interventions.csv`, `intervention_summary.csv`, `probes.csv`, `probe_selection.csv`, and `analysis_settings.json` from `/data/qmla/famato/results/bottleneck_analysis/full128_array_12560/` to local `results/bottleneck_analysis/full128_array_12560/`. The two PNG exports are optional. Leave `features/`, checkpoints, and the dataset on HPC, then run the results notebook locally. No original-network retraining is performed. Verification and custom compatibility checks remain omitted. Fresh extraction is the default; use `--reuse-features` only with complete script-generated caches for unchanged inputs/settings (older notebook `.npz` caches are not reused). Outputs for the selected run are overwritten on rerun.

Use [`notebooks/03-results-analysis.ipynb`](notebooks/03-results-analysis.ipynb) to analyse saved runs without training or loading checkpoints. Edit its experiment-directory mapping to compare Slurm arrays, sequential benchmarks, or individual runs. It plots losses and validation macro-F1, compares scores and stopping epochs, reports per-class test results, and supports paired comparisons across experiments and seeds. Run it on the HPC or point it at locally copied `runs/` and `results/` directories; optional CSV/PNG/PDF exports go to a separate analysis directory. Use validation results for tuning and test results for final reporting.

Use [`notebooks/02-configurable-experiments.ipynb`](notebooks/02-configurable-experiments.ipynb) for the same resolver and run APIs with result plots. Select the uv-created `.venv` Python interpreter as the notebook kernel in your editor; `ipykernel` is installed by `uv sync`. The older exploratory notebook is preserved. Model graph rendering is not required.

Copy/edit the site placeholders in [`scripts/slurm/cpu.sbatch.example`](scripts/slurm/cpu.sbatch.example) and [`scripts/slurm/gpu.sbatch.example`](scripts/slurm/gpu.sbatch.example). They are templates, not measured hardware requirements. Set your account, partition, environment/modules, memory and time allocation. Prepare the uv environment, install the appropriate Torch build, and prepare/download data before the training allocation. Make `uv` available on the job's `PATH`; the templates use `uv run --no-sync` so jobs do not resolve or install dependencies. Each job uses one CPU process or one GPU; requesting multiple GPUs does not accelerate this trainer.

## Outputs and verification

Each run saves effective/source TOML, JSON configuration and CLI overrides, package versions, dataset provenance, per-epoch metrics, and runtime/hardware information. Checkpoints live under `checkpoints_dir/RUN_NAME`; test reports, predictions and confusion matrices live under `results_dir/RUN_NAME`. Comparison rows include all artifact paths. The run root contains `comparison.csv/json` and `summary.csv/json`; single-seed summaries report no estimated standard deviation. Existing completed outputs are protected against accidental reuse. Each invocation saves job metadata, including resume and CLI overrides.

Runtime includes training, validation, and checkpoint writes for the current invocation. Training throughput divides processed training samples by that duration. Evaluation runtime includes report generation. RSS is sampled every 50 ms and includes loader workers (shared pages may be counted multiple times); GPU figures are PyTorch allocator peaks. Treat these as measured process usage, not an estimate of the whole machine or other processes. Resume timing describes the resumed invocation, while history retains all epochs.

```bash
uv sync --locked --group dev --inexact
uv run --no-sync python -m pytest -q
```

Tests cover configuration/profiles, all model variants and gradients, grouped residuals, cache integrity, stratification, deterministic CPU resume including a spawned worker, single-model smoke, paired benchmarks, and CUDA when available. The [real-image pilot report](docs/smoke-pilot.md) records successful CPU/CUDA smoke runs on this laptop. Tiny smoke accuracy is a pipeline check; robust research conclusions need repeated seeds and appropriate controls.

The quantum circuit follows [QuFeX v1](https://arxiv.org/html/2501.13165v1). Data sources: [Galaxy Zoo 2](https://academic.oup.com/mnras/article/435/4/2835/1022913), [Hart et al.](https://academic.oup.com/mnras/article/461/4/3663/2608720), and the [Zenodo image release](https://zenodo.org/records/3565489).
