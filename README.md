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

### Compare best-checkpoint scores on all splits

`scripts.evaluate_experiment` evaluates saved `best.pt` checkpoints for QuFeX and
both CNN benchmarks without retraining. It discovers individual runs recursively,
including separate model folders in a Slurm array. Every split uses evaluation
mode with augmentation disabled; training-set scores describe the best checkpoint,
not predictions collected during training epochs.

Run on an HPC compute node from the repository root:

```bash
uv run --no-sync python -u -m scripts.evaluate_experiment \
  --experiment-dir /data/qmla/famato/runs/full128_array_12560 \
  --checkpoint-root /data/qmla/famato/checkpoints \
  --processed-dir /data/qmla/famato/data/processed \
  --output-dir /data/qmla/famato/results/split_evaluation/full128_array_12560 \
  --device cuda --batch-size 32 --num-workers 0 --cpu-threads 8
```

The default evaluates `train validation test`; use `--splits validation test` to
select fewer splits. Without `--checkpoint-root`, saved checkpoint references are
used. Data paths default to the checkpoint configuration; use `--processed-dir`
or the existing `--project-root` relocation option when needed. The processed
directory is the **parent** of the `gz2-*` cache. Existing nonempty output
directories are rejected; choose a new output directory when rerunning.

Copy only `split_metrics.csv`, `per_class_metrics.csv`, and
`evaluation_settings.json` from that output directory to the matching local
`results/split_evaluation/full128_array_12560/` directory. In
`notebooks/03-results-analysis.ipynb`, set:

```python
SPLIT_EVALUATIONS = {
    'scheduled': RESULTS_ROOT / 'split_evaluation' / 'full128_array_12560',
}
```

Use the same experiment label as in `EXPERIMENTS`. The notebook displays accuracy,
macro precision/recall, a per-class breakdown, and mean/sample standard deviation
across seeds (standard deviation is unavailable for one seed). CSV checkpoint
epochs are zero-based; displayed best epochs are one-based. Split-specific full
reports and predictions stay on the HPC. Saved training/test artifacts are preserved.
A submission template is provided in
`scripts/slurm/evaluate-experiment.sbatch`; edit its site/resource settings and
create `logs/` before submission. Training already produces test reports, but this
command recomputes all requested splits for one consistent evaluation export.

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

## Staged compression study

The independent [`configs/compression_sweep.toml`](configs/compression_sweep.toml) configures
[`scripts/compression_sweep.py`](scripts/compression_sweep.py). It does not inherit settings
from `experiments.toml`. Its defaults are full 64×64 images, seeds **42, 1324, 987654**,
50 epochs with patience 8, and a **0.05 absolute** acceptable drop in mean paired validation
macro-F1 relative to M0. For example, a change from 0.90 to 0.85 is acceptable.

```bash
# Inspect the resolved experiment and resource counts without preparing data or training
uv run --no-sync python -m scripts.compression_sweep --dry-run

# Classical compression curve and direct-CNN reference only
uv run --no-sync python -m scripts.compression_sweep --stage classical --run-dir runs/compression-study

# Continue with profiling and quantum/control training; existing completed jobs are skipped
uv run --no-sync python -m scripts.compression_sweep --stage quantum --run-dir runs/compression-study --resume

# Or run both stages and generate reports in one invocation (test split is not evaluated)
uv run --no-sync python -m scripts.compression_sweep --stage all --run-dir runs/new-compression-study

# Rebuild tables/plots without training or evaluating checkpoints
uv run --no-sync python -m scripts.compression_sweep --stage analyze --run-dir runs/compression-study

# Freeze the completed checkpoint list and explicitly evaluate it on the held-out test split
uv run --no-sync python -m scripts.compression_sweep --stage evaluate --run-dir runs/compression-study --resume
```

`--config`, `--device` and the usual project/data/output path overrides are supported.
`--stage profile --resume --run-dir ...` profiles the selected quantum levels without training
them. Run one process per study directory. Existing studies require `--resume` for execution;
analysis can read an existing study without it. Scientific settings and budgets must match the
original study; changing the device is allowed and each attempt records its hardware.

| Level | C×H×W | Feature values | Forward circuit instances/image |
|---|---:|---:|---:|
| M0 | 64×8×8 | 4096 | 512 |
| M1 | 64×6×6 | 2304 | 288 |
| M2 | 64×4×4 | 1024 | 128 |
| M3 | 64×2×2 | 256 | 32 |
| M4 | 32×2×2 | 128 | 16 |
| M5 | 16×2×2 | 64 | 8 |

M0–M3 vary spatial compression; M4–M5 additionally reduce channels. Each model uses the direct
CNN encoder, an optional channel projection and adaptive average pooling, a residual extraction
module, global pooling and the classifier. The full CNN extraction module mixes all channels;
the patch CNN and quantum module operate on identical nonoverlapping 2×2 patches of adjacent
channel pairs. Shared components have paired initialization, and augmentation uses independent
per-example/per-epoch random streams. The direct CNN reference retains its original architecture.

The quantum circuit is always **8 qubits, 1 filter, 4 shared trainable parameters**, analytic
`default.qubit` with backpropagation. Increasing representation size increases its application
count, not qubit count or trainable circuit capacity. Circuit counts exclude measurement shots
and gradient evaluations and are not counts of batched simulator calls. The default chunk size
of 256 limits inputs to an individual circuit call; autograd can retain state across chunks, so
total peak memory is measured independently. Simulator time is not a forecast of hardware time.

Stage A completes every classical level/seed before selecting the most compressed acceptable
level, including possible recovery after a worse level. Standard deviations describe seed
variability; selection uses the mean paired drop, not a statistical significance test. If every
level passes, M5 is selected and the study reports the observed range without inventing levels.
Stage B visits the selected level, one more compressed level if available, then progressively
larger representations. At each level it profiles a disposable quantum model (5 warm-up and
20 measured optimizer steps), then trains fresh quantum and patch-CNN models for each seed.
Cost estimates do not reject runs based on a hypothetical full 50-epoch duration.

The **4-hour per-job** and **48-hour total-study** budgets persist across invocations. Profiling
has its own per-job cap and counts against the total. Data preparation/integrity checking before
execution is timed separately; downtime and regenerating analysis are excluded. Limits are
checked between batches and can overrun by an in-flight batch and checkpoint finalization.
Quantum time exhaustion or OOM stops expansion. Budget-exhausted and OOM jobs remain incomplete
and are excluded from completed-run comparisons. Resume does not reset an exhausted budget.
Interruptions resume from the last completed epoch (including an initial epoch -1 checkpoint);
partial epochs are replayed, and previously consumed time remains charged. Abrupt process kills
retain accounting through the last saved budget boundary.

`study.json` is the persistent ledger; `selection.json` explains threshold selection. Each run
has its own resolved configuration, package versions, history, runtime and checkpoints.
`resolved_sweep.toml` saves the complete study with absolute paths; pass it as `--config` when
resuming if the original configuration has since been edited.
`analysis/` contains per-run and aggregated CSV/JSON tables, profiling results, paired differences,
and performance/size/time/memory plots. Plots show complete seed sets; incomplete runs and partial
seed sets remain explicitly labelled in the tables. Training-step timing excludes data loading;
run time includes setup, validation and checkpointing. Time to the best checkpoint is recorded
at the end of that epoch's validation, before checkpoint serialization.

The explicit `evaluate` stage freezes checkpoint identities and SHA-256 hashes in
`evaluation_manifest.json`, then writes test predictions and reports under `test/`. Subsequent
evaluation resumes that same list; further exploration in a study with a frozen test list is
rejected. Test scores never influence compression selection or expansion. Existing legacy
configurations, checkpoints and the ordinary three-model benchmark remain supported.

### Running the compression study on Dante

[`scripts/slurm/compression-sweep.sbatch`](scripts/slurm/compression-sweep.sbatch) requests
**one V100, four CPU threads, 32 GB host RAM and 50 hours**. On 2026-09-25, `ssh dante`
reported the `master` partition with unlimited wall time and four 16 GB V100s on `treachery`.
A short allocated CUDA forward/backward check passed with the existing Python environment
(Torch 2.12.0+cu126). These are resource defaults, not a measured memory requirement or a
guarantee that every quantum level will fit. The study retains its 48-hour accumulated budget;
Slurm's extra two hours allow preparation and finalization overhead. Resources remain
overridable through `sbatch` options.

Use one process per study bundle. This trainer does not use multiple GPUs, and a Slurm array
would race on the shared selection and budget. The launcher holds a `flock` lock to prevent
concurrent writers, automatically resumes an existing study using its saved TOML, and saves
checkpoints under the same bundle as the results. It records per-invocation source archives,
package versions, resource allocation and logs. It never installs packages during an allocation.

Transfer the **current implementation**, not just the Slurm files. From the local repository
root, these commands package the study code without the dataset, virtual environment, previous
results, main experiment configuration, or existing notebooks. The separate remote checkout
preserves the existing HPC checkout and can use its installed dependencies:

```powershell
tar -czf compression-study-code.tar.gz --exclude=__pycache__ --exclude='*.pyc' qmla scripts configs/compression_sweep.toml notebooks/05-compression-sweep.ipynb pyproject.toml uv.lock README.md
scp compression-study-code.tar.gz dante:/users/famato/QMLA/code/
ssh dante "mkdir -p /users/famato/QMLA/code/compression-study && tar -xzf /users/famato/QMLA/code/compression-study-code.tar.gz -C /users/famato/QMLA/code/compression-study"
```

On Dante, submit CPU preprocessing first: the existing caches inspected on the HPC were
128×128, whereas this study requires its own 64×64 cache. The preparation script uses the
study's independent TOML and validates/reuses an existing matching cache. Raw data must
already exist; it does not download data.

```bash
cd /users/famato/QMLA/code/compression-study
export QMLA_PYTHON=/users/famato/QMLA/code/qmla/.venv/bin/python
export QMLA_DATA_ROOT=/data/qmla/famato/data
export QMLA_BUNDLE=/data/qmla/famato/results/compression-hpc
prep=$(sbatch --parsable scripts/slurm/compression-prepare.sbatch)
sbatch --dependency=afterok:"${prep%%;*}" scripts/slurm/compression-sweep.sbatch
squeue -u "$USER"
```

`QMLA_STAGE` defaults to `all`, which excludes test evaluation. Other controls are `QMLA_CONFIG`
(new studies only), `QMLA_REPO` (defaults to the submission directory), `QMLA_PYTHON`,
`QMLA_DATA_ROOT` and `QMLA_BUNDLE`. Keep the same bundle and source checkout when resuming;
use a new bundle for another experiment. Do not update source files while a job is running.
An example shorter allocation, or a resume after interruption, is:

```bash
sbatch --time=12:00:00 scripts/slurm/compression-sweep.sbatch
```

The launcher requests a signal five minutes before wall time and forwards it to Python.
The handler sets a flag; the next batch boundary saves interrupted status and accounting,
then generates partial reports. Exit code **75** means resubmit the same command to resume.
Partial epochs replay from the last completed epoch. Unlike a scientific run-time cap, a
scheduler interruption does not mark the run `budget_exhausted` or stop quantum expansion.
`--requeue` permits scheduler-initiated requeues; it does **not** automatically submit a new
job after a time limit. If a batch or finalization exceeds the warning interval, Slurm can
still kill the process; the previous completed checkpoint remains the resume point. See
the [Slurm signal and requeue options](https://slurm.schedmd.com/sbatch.html).

The bundle contains `study/` (state, histories, analysis and any explicit test results),
`checkpoints/`, and `slurm/` (logs and source/environment provenance). Copy it after the job
has stopped for a consistent snapshot. From the local repository root in PowerShell:

```powershell
New-Item -ItemType Directory -Force results | Out-Null
scp -r dante:/data/qmla/famato/results/compression-hpc ./results/
```

Open [`notebooks/05-compression-sweep.ipynb`](notebooks/05-compression-sweep.ipynb) and set
`BUNDLE` if you changed the directory name. It reads the copied state and relative histories,
so it needs no HPC mount, dataset or checkpoint deserialization. It displays partial-run
status, saved threshold decisions, paired comparisons, runtime and memory curves, learning
histories and any frozen test results. Original absolute paths remain provenance only;
do not run the training CLI against the copied HPC configuration to analyze it locally.

Only when exploration is final, explicitly submit test evaluation on the HPC:

```bash
QMLA_STAGE=evaluate sbatch scripts/slurm/compression-sweep.sbatch
```

This uses the same persistent budgets and freezes the completed checkpoint list. If the
48-hour study budget is already exhausted, this command cannot do more evaluation work;
restarting does not grant a new budget. No test results are needed for the local validation
analysis notebook. Inspect `study/study.json` and the job logs: Slurm `COMPLETED` means the
script finished, which can also mean the scientific budget stopped an incomplete study.
