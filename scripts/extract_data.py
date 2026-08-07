"""Download, verify, and extract the official Galaxy Zoo 2 data products."""

from __future__ import annotations

import argparse

from qmla.config import ConfigError, load_config
from qmla.data import GalaxyZooDownloader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.toml", help="Path to project TOML configuration")
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download and verify files without extracting the image ZIP",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        GalaxyZooDownloader(config.paths.raw_dir).run(extract_images=not args.no_extract)
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

