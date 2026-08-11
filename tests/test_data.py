from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from qmla.config import load_config
from qmla.data import GalaxyZooPreprocessor, file_digest, safe_extract_zip


def _write_test_config(path: Path, project_root: Path) -> None:
    path.write_text(
        f"""
[paths]
project_root = {json.dumps(str(project_root))}
raw_dir = "raw"
processed_dir = "processed"
runs_dir = "runs"
checkpoints_dir = "checkpoints"
results_dir = "results"

[data]
image_size = 16
train_fraction = 0.70
validation_fraction = 0.15
test_fraction = 0.15
seed = 7
num_workers = 0
preprocessing_batch_size = 8
clean_label_policy = "hart_clean_flags"

[model]
mode = "classical"
encoder_channels = [4, 4, 4, 4]
convolutions_per_block = 1
compression_channels = 16
quantum_spatial_size = 2
post_quantum_channels = [4]
classifier_hidden_neurons = [4]
dropout = 0.0

[quantum]
backend = "default.qubit"
diff_method = "backprop"
shots = 0
qubits = 8
input_angle_scale = 3.141592653589793

[training]
device = "cpu"
epochs = 1
batch_size = 4
optimizer = "adamw"
learning_rate = 0.001
weight_decay = 0.0
early_stopping_patience = 1
amp = false
class_weighting = true
checkpoint_every = 1

[evaluation]
batch_size = 4
save_predictions = true
save_confusion_matrix = true
class_names = ["smooth", "unbarred_spiral", "barred_spiral"]
""",
        encoding="utf-8",
    )


def test_safe_extract_and_digest(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/value.txt", "galaxy")
        bundle.writestr("__MACOSX/nested/._value.txt", "metadata")
        bundle.writestr("nested/.DS_Store", "metadata")
    destination = tmp_path / "output"
    safe_extract_zip(archive, destination)
    assert (destination / "nested" / "value.txt").read_text() == "galaxy"
    assert not (destination / "__MACOSX").exists()
    assert not (destination / "nested" / ".DS_Store").exists()
    assert len(file_digest(archive)) == 64


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "output")


def test_preprocessor_builds_clean_disjoint_splits(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_test_config(config_path, tmp_path)
    config = load_config(config_path)
    images_dir = config.paths.raw_dir / "images"
    images_dir.mkdir(parents=True)

    mapping_rows = []
    catalog_rows = []
    asset_id = 1000
    for label in range(3):
        for item in range(12):
            objid = str(10_000_000 + label * 100 + item)
            mapping_rows.append({"objid": objid, "sample": "original", "asset_id": asset_id})
            flags = {
                "dr7objid": objid,
                "t01_smooth_or_features_a01_smooth_flag": int(label == 0),
                "t03_bar_a06_bar_flag": int(label == 2),
                "t03_bar_a07_no_bar_flag": int(label == 1),
                "t04_spiral_a08_spiral_flag": int(label in {1, 2}),
            }
            catalog_rows.append(flags)
            pixels = np.full((24, 24, 3), 25 + label * 80 + item, dtype=np.uint8)
            Image.fromarray(pixels).save(images_dir / f"{asset_id}.jpg")
            asset_id += 1

    # Ambiguous object must be dropped.
    mapping_rows.append({"objid": "99999999", "sample": "original", "asset_id": asset_id})
    catalog_rows.append(
        {
            "dr7objid": "99999999",
            "t01_smooth_or_features_a01_smooth_flag": 0,
            "t03_bar_a06_bar_flag": 0,
            "t03_bar_a07_no_bar_flag": 0,
            "t04_spiral_a08_spiral_flag": 0,
        }
    )
    Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).save(images_dir / f"{asset_id}.jpg")

    pd.DataFrame(mapping_rows).to_csv(config.paths.raw_dir / "gz2_filename_mapping.csv", index=False)
    pd.DataFrame(catalog_rows).to_csv(
        config.paths.raw_dir / "gz2_hart16.csv.gz", index=False, compression="gzip"
    )

    metadata = GalaxyZooPreprocessor(config).run()
    assert sum(split["samples"] for split in metadata["splits"].values()) == 36
    assert len(metadata["normalization_mean"]) == 3
    assert np.load(config.paths.processed_dir / "train_images.npy", mmap_mode="r").dtype == np.uint8

    ids = {
        split: set(
            pd.read_csv(config.paths.processed_dir / f"{split}_manifest.csv", dtype={"dr7objid": str})[
                "dr7objid"
            ]
        )
        for split in ("train", "validation", "test")
    }
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])
