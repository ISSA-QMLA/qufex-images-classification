"""Run the staged compression study; test evaluation is always explicit."""
import argparse
import json
import signal
from dataclasses import replace
from pathlib import Path

from qmla.cli import PATH_OPTIONS, path_overrides, resolve_job_path
from qmla.compression_study import StudyRunner, load_study
from qmla.data import GalaxyZooPreprocessor, validate_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compression_sweep.toml")
    parser.add_argument("--stage", choices=("prepare", "classical", "profile", "quantum", "analyze", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-run-hours", type=float, help="Raise the per-run limit when resuming a study")
    parser.add_argument("--extend-total-hours", type=float, help="Raise the accumulated study limit when resuming")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    try:
        extending = args.extend_run_hours is not None or args.extend_total_hours is not None
        if extending and (not args.resume or args.run_dir is None or args.stage in {"prepare", "analyze", "evaluate"}):
            raise ValueError("Budget extension requires --resume, --run-dir, and an exploration stage")
        study = load_study(args.config, device=args.device, paths=path_overrides(args))
        if extending:
            sweep = replace(study.sweep,
                run_hours=args.extend_run_hours if args.extend_run_hours is not None else study.sweep.run_hours,
                total_hours=args.extend_total_hours if args.extend_total_hours is not None else study.sweep.total_hours)
            study = replace(study, sweep=sweep)
            study.validate()
        if args.dry_run:
            print(json.dumps(study.description(), indent=2))
            return
        if args.stage == "prepare":
            if args.run_dir is not None or args.resume:
                raise ValueError("--stage prepare only prepares data; omit --run-dir and --resume")
            if not study.config.cache_dir.exists():
                GalaxyZooPreprocessor(study.config).run()
            metadata = validate_cache(study.config)
            print(f"Prepared cache: {study.config.cache_dir}")
            print(f"Dataset identity: {metadata['dataset_id']}")
            return
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires --run-dir")
        if args.stage in {"analyze", "evaluate", "profile", "quantum"} and args.run_dir is None:
            raise ValueError(f"--stage {args.stage} requires an existing --run-dir")
        directory = resolve_job_path(args.run_dir, study.config)
        runner = StudyRunner(study, directory, resume=args.resume, read_only=args.stage == "analyze",
                             extend_budgets=extending)
        previous = {}
        def request_stop(signum, _frame):
            runner.request_stop(signal.Signals(signum).name)
        try:
            for name in ("SIGUSR1", "SIGTERM"):
                if hasattr(signal, name):
                    sig = getattr(signal, name)
                    previous[sig] = signal.signal(sig, request_stop)
            runner.run(args.stage)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        print(f"Study outputs: {runner.root}")
        if runner.stop_requested:
            raise SystemExit(75)  # Temporary scheduler interruption; resubmit to resume.
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
