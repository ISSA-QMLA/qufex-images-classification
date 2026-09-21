"""Evaluate a versioned checkpoint using its saved architecture by default."""
import argparse
from datetime import datetime
from pathlib import Path

from qmla.cli import add_config_arguments, path_overrides, resolve_config, resolve_job_path
from qmla.engine import Evaluator, load_checkpoint_config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, checkpoint_defaults=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.config:
            config = resolve_config(args)
            checkpoint = resolve_job_path(args.checkpoint, config)
        else:
            if args.profile or args.model:
                parser.error("--profile/--model require --config; otherwise checkpoint settings are used")
            checkpoint = args.checkpoint.resolve()
            config = load_checkpoint_config(checkpoint, device=args.device, paths=path_overrides(args))
        output = resolve_job_path(args.run_dir, config) or config.paths.results_dir / f"{checkpoint.parent.name}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        Evaluator(config, checkpoint, output).run()
        print(f"Results: {output}")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
