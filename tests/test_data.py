import json
import pickle
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from qmla.data import GalaxyDataset, GalaxyZooPreprocessor, file_digest, safe_extract_zip, validate_cache


def test_safe_extract(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/value.txt", "galaxy")
        bundle.writestr("__MACOSX/._value", "metadata")
    destination = tmp_path / "extracted"
    safe_extract_zip(archive, destination)
    assert (destination / "nested/value.txt").read_text() == "galaxy"
    assert not (destination / "__MACOSX").exists()
    assert len(file_digest(archive)) == 64
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        safe_extract_zip(archive, destination)


def test_cache_splits_statistics_and_reuse(prepared):
    meta = validate_cache(prepared)
    assert [meta["splits"][s]["samples"] for s in ("train", "validation", "test")] == [12, 6, 6]
    train = np.load(prepared.cache_dir / "train_images.npy").astype(float) / 255
    np.testing.assert_allclose(meta["normalization_mean"], train.mean(axis=(0, 1, 2)))
    np.testing.assert_allclose(meta["normalization_std"], train.std(axis=(0, 1, 2)))
    assert GalaxyZooPreprocessor(prepared).run()["dataset_id"] == meta["dataset_id"]
    original = GalaxyDataset(prepared.cache_dir, "train")
    restored = pickle.loads(pickle.dumps(original))
    assert isinstance(restored.images, np.memmap)
    assert np.array_equal(restored[0][0], original[0][0])


def test_resolution_caches_share_master_and_subset_ids(prepared):
    new = replace(prepared, data=replace(prepared.data, image_size=64))
    first = validate_cache(prepared)
    second = GalaxyZooPreprocessor(new).run()
    assert new.cache_dir != prepared.cache_dir
    assert first["master_ids"] == second["master_ids"]
    assert [first["splits"][s]["ids_sha256"] for s in first["splits"]] == [second["splits"][s]["ids_sha256"] for s in second["splits"]]
    assert np.load(new.cache_dir / "train_images.npy").shape == (12, 64, 64, 3)


def test_corrupted_cache_rejected(prepared):
    np.save(prepared.cache_dir / "train_images.npy", np.zeros((12, 128, 128, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="Cache file changed"):
        validate_cache(prepared)


def test_stale_source_rejected(prepared):
    path = prepared.paths.raw_dir / "gz2_filename_mapping.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="Raw source changed"):
        validate_cache(prepared)


def test_subset_stratification_and_reproducibility(tiny_config):
    frame = pd.DataFrame({"label": [0] * 60 + [1] * 30 + [2] * 10, "id": range(100)})
    processor = GalaxyZooPreprocessor(tiny_config)
    subset = processor._subset(frame, 20)
    assert subset["label"].value_counts().to_dict() == {0: 12, 1: 6, 2: 2}
    assert subset.equals(processor._subset(frame, 20))
    assert set(processor._subset(frame, 3)["label"]) == {0, 1, 2}
