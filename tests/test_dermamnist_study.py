"""Official-split DermaMNIST loading and compression-study integration."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qmla.compression_study import Level, StudyConfig, StudyRunner, SweepSettings, load_study
from qmla.config import ConfigError, DERMAMNIST_CLASS_NAMES, load_config
from qmla.data import DermaMNISTPreprocessor, GalaxyDataset, RemoteResource, file_digest, validate_cache
from qmla.engine import Evaluator, load_checkpoint_config
from qmla.experiments import run_experiments


def test_dermamnist_configuration_and_galaxy_cache_compatibility(tiny_config):
    study = load_study(Path(__file__).parents[1] / "configs/dermamnist_compression.toml")
    assert study.config.data.dataset == "dermamnist"
    assert study.config.data.image_size == 64
    assert study.config.evaluation.class_names == DERMAMNIST_CLASS_NAMES
    assert list(study.levels) == ["K1", "K0", "M0", "M1", "M2", "M3", "M4", "M5"]
    assert study.config.cache_dir.name.startswith("dermamnist-64-")
    assert tiny_config.cache_dir.name.startswith("gz2-32-")
    assert "dataset" not in tiny_config.data_signature()
    with pytest.raises(ConfigError, match="official"):
        replace(study.config, data=replace(study.config.data, clean_label_policy="hart_clean_flags")).validate()
    with pytest.raises(ConfigError, match="seven"):
        replace(study.config, evaluation=replace(study.config.evaluation, class_names=("a", "b", "c"))).validate()
    # The official split is used regardless of the Galaxy Zoo split fractions.
    replace(study.config, data=replace(study.config.data,
                                       train_fraction=0, validation_fraction=0,
                                       test_fraction=0)).validate()


@pytest.fixture
def synthetic_derma(tmp_path, monkeypatch):
    from qmla import data as module

    original = load_study(Path(__file__).parents[1] / "configs/dermamnist_compression.toml",
                          device="cpu", paths={"project_root": tmp_path})
    config = replace(original.config,
                     training=replace(original.config.training, epochs=2, batch_size=4,
                                      seeds=(42,), deterministic=True, cpu_threads=2),
                     evaluation=replace(original.config.evaluation, batch_size=4))
    config.paths.raw_dir.mkdir(parents=True)
    source = config.paths.raw_dir / "dermamnist_64.npz"
    rng = np.random.default_rng(321)
    content = {}
    for split in ("train", "val", "test"):
        labels = np.tile(np.arange(7, dtype=np.uint8), 2)[:, None]
        images = rng.integers(0, 256, (len(labels), 64, 64, 3), dtype=np.uint8)
        content[f"{split}_images"] = images
        content[f"{split}_labels"] = labels
    np.savez_compressed(source, **content)
    monkeypatch.setattr(module, "DERMAMNIST_RESOURCE", RemoteResource(
        source.name, "https://example.invalid/unused", file_digest(source, "md5")))
    study = StudyConfig(config, SweepSettings(warmup_steps=1, measured_steps=1,
                                              run_hours=1, total_hours=4),
                        {"M0": Level(4, 8), "M1": Level(4, 2)})
    study.validate()
    return study


def test_official_splits_are_preserved_and_cache_is_verified(synthetic_derma, monkeypatch, capsys):
    from scripts import compression_sweep

    config = synthetic_derma.config
    monkeypatch.setattr("sys.argv", ["compression_sweep", "--config",
                        str(Path(__file__).parents[1] / "configs/dermamnist_compression.toml"),
                        "--stage", "prepare", "--project-root", str(config.paths.project_root)])
    compression_sweep.main()
    assert "Prepared cache:" in capsys.readouterr().out
    metadata = DermaMNISTPreprocessor(config).run()
    assert validate_cache(config)["dataset_id"] == metadata["dataset_id"]
    assert metadata["split_policy"] == "MedMNIST official train/val/test"
    assert metadata["class_names"] == list(DERMAMNIST_CLASS_NAMES)
    assert [metadata["splits"][name]["samples"] for name in ("train", "validation", "test")] == [14] * 3
    for name in ("train", "validation", "test"):
        dataset = GalaxyDataset(config.cache_dir, name)
        assert len(dataset) == 14
        assert dataset[0][0].shape == (3, 64, 64)
        assert set(np.asarray(dataset.labels)) == set(range(7))
    assert DermaMNISTPreprocessor(config).run()["dataset_id"] == metadata["dataset_id"]
    labels = config.cache_dir / "test_labels.npy"
    with labels.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="Cache file changed"):
        validate_cache(config)


def test_dermamnist_tiny_study_checkpoint_and_explicit_test(synthetic_derma, monkeypatch):
    from qmla import compression_study as module

    study = synthetic_derma
    runner = StudyRunner(study, study.config.paths.runs_dir / "derma-small")
    seen = []
    original = module.Evaluator

    def tracked(*args, **kwargs):
        seen.append(kwargs.get("split", "test"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "Evaluator", tracked)
    runner.run("classical")
    assert seen == []
    assert runner.state["selection"] is not None
    assert len(runner.state["jobs"]) == 3
    assert runner.train("M1", "compression_qufex", 42)
    quantum = runner.row("M1", "compression_qufex", 42)
    checkpoint = load_checkpoint_config(Path(quantum["checkpoint"]))
    assert checkpoint.data.dataset == "dermamnist"
    assert len(checkpoint.evaluation.class_names) == 7
    assert quantum["status"] == "completed"
    runner.run("evaluate")
    assert seen == ["test"] * 4
    assert len(json.loads((runner.root / "evaluation_manifest.json").read_text())) == 4


def test_dermamnist_benchmark_download_preprocess_and_three_models(synthetic_derma, monkeypatch, capsys):
    from scripts import extract_data, preprocess_data

    source = Path(__file__).parents[1] / "configs/dermamnist_benchmark.toml"
    config = load_config(source, profile="full64", device="cpu",
                         paths={"project_root": synthetic_derma.config.paths.project_root})
    galaxy_full64 = load_config(Path(__file__).parents[1] / "configs/experiments.toml", profile="full64")
    assert config.architectures == galaxy_full64.architectures
    assert config.quantum == galaxy_full64.quantum
    assert config.cache_dir == synthetic_derma.config.cache_dir
    assert config.data.dataset == "dermamnist"
    assert config.evaluation.class_names == DERMAMNIST_CLASS_NAMES
    assert config.training.seeds == (42,)
    for model in ("qufex", "cnn_replacement", "direct_cnn"):
        config.for_model(model)

    common = ["--config", str(source), "--profile", "full64",
              "--project-root", str(config.paths.project_root)]
    monkeypatch.setattr("sys.argv", ["extract_data", *common])
    extract_data.main()
    assert "Verified DermaMNIST archive" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["preprocess_data", *common])
    preprocess_data.main()
    assert validate_cache(config)["dataset_id"]

    fast = replace(config, training=replace(config.training, epochs=1, batch_size=4,
                                            early_stopping_patience=0, cpu_threads=2),
                   evaluation=replace(config.evaluation, batch_size=4))
    rows = run_experiments(fast, benchmark=True, run_dir=config.paths.runs_dir / "benchmark-small")
    assert [row["model"] for row in rows] == ["qufex", "cnn_replacement", "direct_cnn"]
    assert len({row["dataset_id"] for row in rows}) == 1
    assert all(Path(row["checkpoint"]).exists() for row in rows)
    assert all(len(load_checkpoint_config(Path(row["checkpoint"])).evaluation.class_names) == 7
               for row in rows)
    assert all(Path(row["evaluation_dir"]).joinpath("metrics.json").exists() for row in rows)
