"""Train/test all three model families using paired data and training seeds."""
import argparse
from pathlib import Path

from qmla.cli import add_config_arguments, resolve_config, resolve_job_path
from qmla.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    try:
        config = resolve_config(args)
        run_experiments(config, benchmark=True, run_dir=resolve_job_path(args.run_dir, config))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
