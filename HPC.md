# Dante HPC branch

`hpc` contains the shared code merged from `feat/comply-with-ref`, plus this site's
configuration and Slurm jobs. Keep site changes on `hpc`; bring later shared code
updates in with `git fetch origin` and `git merge origin/feat/comply-with-ref`.

Repository: `/users/famato/QMLA/code/qmla`.
Persistent data, caches, checkpoints, runs and results: `/data/qmla/famato/`.
`configs/hpc.toml` follows the current schema, defaults to `full128`, and retains
the shared production architecture, optimizer, quantum backend and batch sizes.
It uses two loader workers. `pilot128` uses the same architecture and batches on
1,200/300/300 images for two epochs. Production currently uses one seed (42).

## Before submitting jobs

The repository alignment does not install or upgrade the existing environment.
Prepare it with `uv sync --locked --group dev --inexact` and verify the installed
Torch build against the compute-node GPU/driver. Do not copy a Windows environment.
The jobs add `$HOME/.local/bin` to PATH and use `uv run --no-sync`.

Run submissions from the repository root and create the log directory first:

```bash
cd /users/famato/QMLA/code/qmla
mkdir -p logs
```

Check all three full128 models on synthetic images before preprocessing:

```bash
sbatch slurm/00_check_models.sbatch
```

Read `logs/qmla-check-models_JOBID.out` and `.err` after completion. Expect PASS
for qufex, cnn_replacement, and direct_cnn. The job checks finite losses and
gradients with the production training batch size and AMP setting. It does not
read datasets or write checkpoints; resource sizing still requires the real-image
pilot. It requests one GPU, four CPUs, 8 GB RAM, and ten minutes.

The existing raw files remain available. The new preprocessing format requires
versioned caches; old loose arrays are preserved but cannot replace these caches.
Download only if the raw data need completing. Prepare the appropriate cache in a
CPU job before training:

```bash
sbatch --export=ALL,QMLA_PROFILE=pilot128 slurm/02_preprocess_data.sbatch
# After successful pilot preprocessing:
sbatch slurm/03_benchmark_pilot.sbatch
```

Inspect the pilot's runtime/memory and test results before setting production
resources. Both benchmark scripts run QuFeX, cnn_replacement and direct_cnn
sequentially on one GPU, and evaluate the best checkpoints. Requests of 16 GB RAM,
one hour for the pilot and 24 hours for production are provisional, not measured.
The partition and node constraint (`master`, `treachery`) follow the previous jobs.

When ready for the full run:

```bash
sbatch --export=ALL,QMLA_PROFILE=full128 slurm/02_preprocess_data.sbatch
# After successful full preprocessing and adjusting the resource request:
sbatch slurm/04_benchmark_full128.sbatch
```

Full training uses all split images, up to 50 epochs, and patience 8. Set multiple
training seeds in the config for repeated comparisons, and account for all three
models and seeds in the walltime. Resume interruptions one model/seed at a time
using the saved resolved configuration and latest.pt, as described in README.md.

## Run the three models concurrently

After full128 preprocessing, use the array instead of the sequential full job:

```bash
sbatch slurm/05_benchmark_full128_parallel.sbatch
```

Array tasks 0, 1 and 2 run qufex, cnn_replacement and direct_cnn respectively.
Each task requests one GPU, eight CPUs, 16 GB RAM and 24 hours. Up to three tasks
run concurrently when resources are available (three GPUs, 24 CPUs, 48 GB total).
The existing configuration, batch sizes, seeds, datasets and evaluation procedure
are unchanged. A single model still uses one GPU. This does not speed up QuFeX
itself; total completion time is dominated by the slowest model.

Logs are `logs/qmla-full128-parallel_ARRAYID_TASKID.out` and `.err`.
Each model writes its own comparison/summary and run artifacts beneath
`/data/qmla/famato/runs/full128_array_ARRAYID/`. Checkpoint and result names include
the array ID and model to avoid collisions. Unlike scripts.benchmark, the array
does not produce a combined three-model summary; collect the per-model comparison
files after all tasks finish. To resume an interrupted task, use the saved
resolved configuration and latest.pt for that model instead of resubmitting the
array. The 24-hour limit applies separately to each task and is conservative.

## Previous setup

Original files are preserved under `legacy/hpc-before-20260921/`, in Git branch
`backup/hpc-pre-alignment-20260921`, and in the external archive directory
`/users/famato/QMLA/backups/qmla-alignment-20260921/`.
The old `classical` selector is replaced by explicit `cnn_replacement` and
`direct_cnn` choices. Old checkpoints/configurations are not compatible with this
workflow. No datasets, checkpoints or results were changed during alignment.
