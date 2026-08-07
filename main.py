def main() -> None:
    print(
        "QMLA is driven by four commands:\n"
        "  python -m scripts.extract_data\n"
        "  python -m scripts.preprocess_data\n"
        "  python -m scripts.train --model qufex\n"
        "  python -m scripts.test_model --checkpoint <best.pt>\n"
        "See README.md and configs/default.toml for details."
    )


if __name__ == "__main__":
    main()
