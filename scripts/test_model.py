"""Evaluate a trained checkpoint and save reproducible result artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from qmla.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml", help="Training-compatible TOML configuration")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to best.pt or latest.pt")
    parser.add_argument("--device", help="Override training.device")
    parser.add_argument("--run-dir", type=Path, help="Explicit results output directory")
    return parser


def _resolve_override(path: Path, project_root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        checkpoint = _resolve_override(args.checkpoint, config.paths.project_root)
        output_dir = (
            _resolve_override(args.run_dir, config.paths.project_root)
            if args.run_dir
            else config.paths.results_dir
            / f"{checkpoint.parent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        from qmla.engine import Evaluator

        Evaluator(config, checkpoint, output_dir, device_name=args.device).run()
        print(f"Results saved to: {output_dir}")
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
