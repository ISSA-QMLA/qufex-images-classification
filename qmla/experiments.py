"""Shared orchestration for the CLI and notebook."""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qmla.config import AppConfig, MODELS
from qmla.data import GalaxyZooPreprocessor
from qmla.engine import Evaluator, Trainer
from qmla.runtime import process_started_at_utc


def run_experiments(config: AppConfig, *, benchmark: bool = False,
                    run_dir: Path | None = None, resume: Path | None = None) -> list[dict]:
    """Prepare a cache, train selected model(s), then evaluate best checkpoints."""
    invocation_started_at_utc = datetime.now(timezone.utc).isoformat()
    models = MODELS if benchmark else (config.run.model,)
    if resume and (benchmark or len(config.training.seeds) != 1):
        raise ValueError("Resume one model/seed at a time using its resolved configuration")
    # Validate every model before beginning an expensive comparison.
    for model in models:
        config.for_model(model)
    if not config.cache_dir.exists():
        GalaxyZooPreprocessor(config).run()
    tag = "benchmark" if benchmark else config.run.model
    root = run_dir or config.paths.runs_dir / f"{tag}_{config.run.profile}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "comparison.json").exists() and not resume:
        raise RuntimeError("Experiment output already exists; select a new run directory")
    job = {"benchmark": benchmark, "models": list(models), "seeds": list(config.training.seeds),
           "process_started_at_utc": process_started_at_utc(),
           "invocation_started_at_utc": invocation_started_at_utc,
           "run_dir": str(root), "resume": str(resume) if resume else None,
           "cli_overrides": config.overrides}
    # Preserve prior invocation metadata when resuming an existing run.
    job_name = f"job-{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    (root / job_name).write_text(json.dumps(job, indent=2), encoding="utf-8")
    rows = []
    for model in models:
        for seed in config.training.seeds:
            selected = config.for_model(model, seed)
            directory = root if len(models) * len(config.training.seeds) == 1 else root / f"{root.name}_{model}_seed{seed}"
            try:
                trainer = Trainer(selected, directory)
                if resume:
                    trainer.resume(resume)
                best = trainer.run()
                runtime = json.loads((directory / "runtime.json").read_text(encoding="utf-8"))
                # Release optimizer/autograd storage before measuring standalone evaluation.
                del trainer
                gc.collect()
                evaluation_dir = config.paths.results_dir / directory.name
                if resume:
                    evaluation_dir = evaluation_dir / f"resumed_{datetime.now():%Y%m%d_%H%M%S_%f}"
                metrics = Evaluator(selected, best, evaluation_dir).run()
                row = {"model": model, "seed": seed, "profile": config.run.profile,
                       "dataset_id": metrics["dataset_id"], "run_dir": str(directory), "checkpoint": str(best), "evaluation_dir": str(evaluation_dir),
                       **{k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "parameter_count", "quantum_parameter_count")},
                       **{f"training_{k}": runtime[k] for k in ("elapsed_seconds", "samples_per_second", "peak_process_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")},
                       "evaluation_elapsed_seconds": metrics["runtime"]["elapsed_seconds"]}
                rows.append(row)
                pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
                (root / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            except Exception as exc:
                (root / "failure.json").write_text(json.dumps({"model": model, "seed": seed, "error": str(exc)}, indent=2), encoding="utf-8")
                raise
    if len({row["dataset_id"] for row in rows}) != 1:
        raise RuntimeError("Benchmark models did not use identical datasets")
    summaries = []
    for model in models:
        frame = pd.DataFrame([row for row in rows if row["model"] == model])
        item = {"model": model, "seeds": len(frame), "preliminary_single_seed": len(frame) == 1}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "training_elapsed_seconds"):
            item[f"{metric}_mean"] = float(frame[metric].mean())
            item[f"{metric}_std"] = float(frame[metric].std(ddof=1)) if len(frame) > 1 else None
        summaries.append(item)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    print(f"Results: {root}")
    return rows
