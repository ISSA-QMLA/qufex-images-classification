"""Download and verify the configured dataset's official source files."""

from __future__ import annotations

import argparse

from qmla.config import ConfigError
from qmla.data import GalaxyZooDownloader, download_dermamnist
from qmla.cli import add_config_arguments, resolve_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
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
        config = resolve_config(args)
        if config.data.dataset == "dermamnist":
            if args.no_extract:
                raise ValueError("--no-extract applies only to Galaxy Zoo")
            source = download_dermamnist(config.paths.raw_dir)
            print(f"Verified DermaMNIST archive: {source}")
        else:
            GalaxyZooDownloader(config.paths.raw_dir).run(extract_images=not args.no_extract)
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
