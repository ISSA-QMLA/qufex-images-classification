"""Create clean, leakage-free, memory-mapped Galaxy Zoo 2 splits."""

from __future__ import annotations

import argparse

from qmla.config import ConfigError, load_config
from qmla.data import GalaxyZooPreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml", help="Path to project TOML configuration")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        GalaxyZooPreprocessor(config).run()
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

