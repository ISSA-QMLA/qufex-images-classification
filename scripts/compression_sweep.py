"""Run the staged compression study; test evaluation is always explicit."""
import argparse
import json
import signal
from pathlib import Path

from qmla.cli import PATH_OPTIONS, path_overrides, resolve_job_path
from qmla.compression_study import StudyRunner, load_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compression_sweep.toml")
    parser.add_argument("--stage", choices=("classical", "profile", "quantum", "analyze", "evaluate", "all"), default="all")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    for name in PATH_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    try:
        study = load_study(args.config, device=args.device, paths=path_overrides(args))
        if args.dry_run:
            print(json.dumps(study.description(), indent=2))
            return
        if args.resume and args.run_dir is None:
            raise ValueError("--resume requires --run-dir")
        if args.stage in {"analyze", "evaluate", "profile", "quantum"} and args.run_dir is None:
            raise ValueError(f"--stage {args.stage} requires an existing --run-dir")
        directory = resolve_job_path(args.run_dir, study.config)
        runner = StudyRunner(study, directory, resume=args.resume, read_only=args.stage == "analyze")
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
