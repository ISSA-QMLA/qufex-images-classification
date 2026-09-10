"""Train the TOML-selected model/seed(s) and test each best checkpoint."""
import argparse

from qmla.cli import add_config_arguments, resolve_config, resolve_job_path
from qmla.experiments import run_experiments


def build_parser():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="Format-2 latest.pt; use its resolved_config.toml")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = resolve_config(args)
        run_experiments(config, run_dir=resolve_job_path(args.run_dir, config), resume=resolve_job_path(args.resume, config))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
