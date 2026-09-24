"""Evaluate saved best checkpoints on all splits; export small tables for Jupyter."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from qmla.cli import PATH_OPTIONS, path_overrides
from qmla.config import config_from_dict
from qmla.engine import Evaluator, _torch_load, load_checkpoint_config

SPLITS = ("train", "validation", "test")
SCORES = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy", "weighted_f1")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(experiment_dir: Path, checkpoint_root: Path | None = None):
    """Resolve every run and checkpoint before allocating models or writing output."""
    if not experiment_dir.is_dir():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    runs, seen, names, datasets = [], set(), set(), set()
    for history in sorted(experiment_dir.rglob("history.json")):
        directory = history.parent
        config_path = directory / "resolved_config.json"
        if not config_path.exists() or not read_json(history):
            raise ValueError(f"Incomplete run (configuration or history missing): {directory}")
        raw = read_json(config_path)
        config = config_from_dict(raw, root=directory)
        if len(config.training.seeds) != 1:
            raise ValueError(f"Expected one seed in saved run: {directory}")
        key = (config.run.model, config.training.seeds[0])
        if key in seen or directory.name in names:
            raise ValueError(f"Duplicate model/seed or run folder: {directory}")
        seen.add(key)
        names.add(directory.name)
        if checkpoint_root is not None:
            checkpoint = checkpoint_root / directory.name / "best.pt"
        else:
            comparison = directory / "comparison.json"
            entries = read_json(comparison) if comparison.exists() else []
            saved = next((row.get("checkpoint") for row in entries
                          if PurePosixPath(str(row.get("run_dir", "")).replace("\\", "/")).name == directory.name), None)
            checkpoint = Path(saved) if saved else config.paths.checkpoints_dir / directory.name / "best.pt"
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint missing: {checkpoint}; use --checkpoint-root for relocated artifacts")
        payload = _torch_load(checkpoint, "cpu")
        saved_config = config_from_dict(payload["config"], root=checkpoint.parent)
        if payload.get("kind") != "best":
            raise ValueError(f"Expected a best checkpoint: {checkpoint}")
        if (saved_config.run.model, saved_config.training.seeds) != (key[0], (key[1],)):
            raise ValueError(f"Checkpoint model/seed does not match run: {directory}")
        metadata_path = directory / "dataset_metadata.json"
        if metadata_path.exists() and read_json(metadata_path)["dataset_id"] != payload["dataset_id"]:
            raise ValueError(f"Checkpoint dataset does not match run: {directory}")
        datasets.add(payload["dataset_id"])
        runs.append((directory, checkpoint.resolve()))
        del payload
    if not runs:
        raise ValueError(f"No saved runs found under {experiment_dir}")
    if len(datasets) != 1:
        raise ValueError("Selected checkpoints use different dataset identities")
    return runs


def evaluate_experiment(args):
    splits = tuple(args.splits)
    if len(set(splits)) != len(splits):
        raise ValueError("Duplicate splits requested")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output already exists: {output}; choose a new output directory")
    runs = discover_runs(args.experiment_dir.resolve(), args.checkpoint_root)
    # Validate all resolved configurations before inference begins.
    jobs = []
    for directory, checkpoint in runs:
        config = load_checkpoint_config(checkpoint, device=args.device, paths=path_overrides(args))
        config = replace(config,
            evaluation=replace(config.evaluation, **({"batch_size": args.batch_size} if args.batch_size is not None else {})),
            data=replace(config.data, **({"num_workers": args.num_workers} if args.num_workers is not None else {})),
            training=replace(config.training, **({"cpu_threads": args.cpu_threads} if args.cpu_threads is not None else {})))
        config.validate()
        jobs.append((directory, checkpoint, config))
    output.mkdir(parents=True, exist_ok=True)
    settings = {"experiment": args.experiment_dir.resolve().name, "splits": splits,
                "epoch_indexing": "checkpoint_epoch is zero-based", "train_augmentation": False,
                "runs": [{"run_dir": str(d), "checkpoint": str(c)} for d, c in runs]}
    (output / "evaluation_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    rows, class_rows = [], []
    for directory, checkpoint, config in jobs:
        for split in splits:
            print(f"Evaluating {directory.name}/{split}", flush=True)
            evaluator = Evaluator(config, checkpoint, output / directory.name / split, split=split)
            metrics = evaluator.run()
            identity = {"experiment": settings["experiment"], "run": directory.name,
                        "model": metrics["model_mode"], "seed": metrics["seed"], "split": split,
                        "dataset_id": metrics["dataset_id"], "checkpoint": str(checkpoint),
                        "checkpoint_epoch": metrics["checkpoint_epoch"]}
            rows.append({**identity, **{name: metrics[name] for name in SCORES}})
            class_rows.extend({**identity, "class_name": name, **values}
                              for name, values in metrics["per_class"].items())
            del evaluator
            gc.collect()
    pd.DataFrame(rows).to_csv(output / "split_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output / "per_class_metrics.csv", index=False)
    print(f"Copy split_metrics.csv, per_class_metrics.csv and evaluation_settings.json from {output}")
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N; defaults to checkpoint settings")
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cpu-threads", type=int)
    return parser


def main():
    parser = build_parser()
    try:
        evaluate_experiment(parser.parse_args())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
