from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="machine-specific PyTorch is absent")


def test_one_epoch_resume_and_evaluation_smoke(tmp_path: Path) -> None:
    from qmla.config import load_config
    from qmla.engine import Evaluator, Trainer

    default = Path(__file__).parents[1] / "configs" / "default.toml"
    text = default.read_text(encoding="utf-8")
    replacements = {
        'project_root = ".."': f"project_root = {json.dumps(str(tmp_path))}",
        "image_size = 128": "image_size = 32",
        "num_workers = 8": "num_workers = 0",
        'mode = "qufex"': 'mode = "classical"',
        "encoder_channels = [16, 32, 64, 32]": "encoder_channels = [4, 4, 4, 8]",
        "post_quantum_channels = [16]": "post_quantum_channels = [4]",
        "classifier_hidden_neurons = [32]": "classifier_hidden_neurons = [4]",
        'device = "cuda"': 'device = "cpu"',
        "epochs = 50": "epochs = 1",
        "batch_size = 32": "batch_size = 4",
        "amp = true": "amp = false",
        "batch_size = 64": "batch_size = 4",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    processed = config.paths.processed_dir
    processed.mkdir(parents=True)

    rng = np.random.default_rng(42)
    sizes = {"train": 12, "validation": 6, "test": 6}
    for split, size in sizes.items():
        images = rng.integers(0, 256, size=(size, 32, 32, 3), dtype=np.uint8)
        labels = np.resize(np.arange(3, dtype=np.int64), size)
        np.save(processed / f"{split}_images.npy", images)
        np.save(processed / f"{split}_labels.npy", labels)
        pd.DataFrame(
            {
                "dr7objid": [f"{split}-{index}" for index in range(size)],
                "asset_id": np.arange(size),
                "sample": "synthetic",
                "label": labels,
                "image_path": "synthetic.jpg",
            }
        ).to_csv(processed / f"{split}_manifest.csv", index=False)
    (processed / "dataset_metadata.json").write_text(
        json.dumps(
            {
                "image_size": 32,
                "class_names": ["smooth", "unbarred_spiral", "barred_spiral"],
                "normalization_mean": [0.5, 0.5, 0.5],
                "normalization_std": [0.25, 0.25, 0.25],
            }
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    trainer = Trainer(config, run_dir, mode="classical", device_name="cpu")
    best = trainer.run()
    latest = config.paths.checkpoints_dir / run_dir.name / "latest.pt"
    assert best.is_file() and latest.is_file()

    resumed = Trainer(config, run_dir, mode="classical", device_name="cpu")
    resumed.resume(latest)
    assert resumed.start_epoch == 1

    results = tmp_path / "results"
    metrics = Evaluator(config, best, results, device_name="cpu").run()
    assert "macro_f1" in metrics
    assert (results / "predictions.csv").is_file()
    assert (results / "confusion_matrix.png").is_file()
