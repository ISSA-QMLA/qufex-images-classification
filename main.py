def main() -> None:
    print(
        "Configure all models and profiles in configs/experiments.toml.\n"
        "  uv run --no-sync python -m scripts.extract_data\n"
        "  uv run --no-sync python -m scripts.preprocess_data\n"
        "  uv run --no-sync python -m scripts.train --model qufex\n"
        "  uv run --no-sync python -m scripts.train --profile smoke --device cpu\n"
        "  uv run --no-sync python -m scripts.benchmark --profile full64\n"
        "  uv run --no-sync python -m scripts.test_model --checkpoint <best.pt>\n"
        "See README.md and configs/experiments.toml for details."
    )


if __name__ == "__main__":
    main()
