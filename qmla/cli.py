"""Shared job-level options; architecture changes belong in TOML."""
import argparse
from pathlib import Path

from qmla.config import MODELS, load_config

PATH_OPTIONS = ("project_root", "raw_dir", "processed_dir", "runs_dir", "checkpoints_dir", "results_dir")


def add_config_arguments(parser: argparse.ArgumentParser, *, checkpoint_defaults: bool = False):
    parser.add_argument("--config", default=None if checkpoint_defaults else "configs/experiments.toml")
    parser.add_argument("--profile")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)


def path_overrides(args) -> dict:
    return {name: getattr(args, name) for name in PATH_OPTIONS if getattr(args, name, None) is not None}


def resolve_config(args):
    return load_config(args.config, profile=args.profile, model=args.model, device=args.device, paths=path_overrides(args))


def resolve_job_path(path: Path | None, config):
    if path is None:
        return None
    return (path if path.is_absolute() else config.paths.project_root / path).resolve()
