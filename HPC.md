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

## Previous setup

Original files are preserved under `legacy/hpc-before-20260921/`, in Git branch
`backup/hpc-pre-alignment-20260921`, and in the external archive directory
`/users/famato/QMLA/backups/qmla-alignment-20260921/`.
The old `classical` selector is replaced by explicit `cnn_replacement` and
`direct_cnn` choices. Old checkpoints/configurations are not compatible with this
workflow. No datasets, checkpoints or results were changed during alignment.
