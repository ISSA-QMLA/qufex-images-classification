from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from qmla.config import load_config
from qmla.data import GalaxyZooPreprocessor

CANONICAL = Path(__file__).parents[1] / "configs" / "experiments.toml"


@pytest.fixture
def tiny_config(tmp_path):
    config = load_config(CANONICAL, profile="smoke", device="cpu", paths={"project_root": tmp_path})
    return replace(config, data=replace(config.data, train_limit=12, validation_limit=6, test_limit=6),
                   training=replace(config.training, batch_size=4, cpu_threads=2, deterministic=True),
                   evaluation=replace(config.evaluation, batch_size=4))


@pytest.fixture
def raw_data(tiny_config):
    raw = tiny_config.paths.raw_dir
    images = raw / "images"
    images.mkdir(parents=True)
    mapping, catalog = [], []
    rng = np.random.default_rng(32)
    for label in range(3):
        for index in range(20):
            asset = label * 100 + index
            objid = str(123456789000000000 + asset)
            mapping.append({"objid": objid, "sample": "original", "asset_id": asset})
            catalog.append({"dr7objid": objid,
                "t01_smooth_or_features_a01_smooth_flag": int(label == 0),
                "t03_bar_a06_bar_flag": int(label == 2),
                "t03_bar_a07_no_bar_flag": int(label == 1),
                "t04_spiral_a08_spiral_flag": int(label != 0)})
            Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(images / f"{asset}.jpg")
    pd.DataFrame(mapping).to_csv(raw / "gz2_filename_mapping.csv", index=False)
    pd.DataFrame(catalog).to_csv(raw / "gz2_hart16.csv.gz", index=False, compression="gzip")
    return tiny_config


@pytest.fixture
def prepared(raw_data):
    GalaxyZooPreprocessor(raw_data).run()
    return raw_data
