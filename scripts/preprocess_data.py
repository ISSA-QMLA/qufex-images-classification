"""Create clean, leakage-free, memory-mapped Galaxy Zoo 2 splits."""

from __future__ import annotations

import argparse

from qmla.config import ConfigError
from qmla.data import GalaxyZooPreprocessor
from qmla.cli import add_config_arguments, resolve_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = resolve_config(args)
        GalaxyZooPreprocessor(config).run()
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
