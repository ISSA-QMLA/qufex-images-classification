"""Train one QuFeX or classical Galaxy Zoo classifier."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from qmla.config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml", help="Path to project TOML configuration")
    parser.add_argument("--model", choices=("qufex", "classical"), help="Override model.mode")
    parser.add_argument("--device", help="Override training.device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--resume", type=Path, help="Resume from a latest.pt checkpoint")
    parser.add_argument("--run-dir", type=Path, help="Explicit experiment output directory")
    return parser


def _resolve_override(path: Path, project_root: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (project_root / path).resolve()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        mode = args.model or config.model.mode
        run_dir = (
            _resolve_override(args.run_dir, config.paths.project_root)
            if args.run_dir
            else config.paths.runs_dir / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        from qmla.engine import Trainer

        trainer = Trainer(config, run_dir, mode=mode, device_name=args.device)
        if args.resume:
            trainer.resume(_resolve_override(args.resume, config.paths.project_root))
        best = trainer.run()
        print(f"Best checkpoint: {best}")
    except (ConfigError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

