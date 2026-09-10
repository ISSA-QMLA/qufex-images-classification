from dataclasses import replace
from pathlib import Path

import pytest

from qmla.config import ConfigError, config_from_dict, load_config, merge_tables, to_toml, relocate_paths

CANONICAL = Path(__file__).parents[1] / "configs" / "experiments.toml"


@pytest.mark.parametrize("profile,size,widths,limit,epochs", [
    ("full64", 64, (4, 8, 8, 8, 16), 0, 50),
    ("full128", 128, (4, 8, 8, 8, 16, 16), 0, 50),
    ("smoke", 32, (4, 4, 8, 16), 96, 1),
    ("small_learning", 32, (4, 4, 8, 16), 1200, 3),
])
def test_profiles(profile, size, widths, limit, epochs):
    config = load_config(CANONICAL, profile=profile)
    assert config.data.image_size == size
    assert config.architectures.shared.encoder_channels == widths
    assert config.data.train_limit == limit and config.training.epochs == epochs
    assert config.paths.project_root == CANONICAL.parents[1]


def test_precedence_and_complete_array_replacement(tmp_path):
    config = load_config(CANONICAL, profile="smoke", model="direct_cnn", device="cpu", paths={"project_root": tmp_path})
    assert config.run.model == "direct_cnn" and config.training.device == "cpu"
    assert config.architecture.encoder_channels == (4, 8)
    assert config.architecture.classifier_hidden_neurons == (8,)
    assert merge_tables({"a": {"b": [1, 2], "c": 4}}, {"a": {"b": [9]}}) == {"a": {"b": [9], "c": 4}}
    assert config.overrides["run"]["profile"] == "smoke"


@pytest.mark.parametrize("text", ["[data]\nimgae_size=32", "[profiles.unused.architectures.direct]\nkernal_size=3", "[model]\nmode='classical'"])
def test_unknown_keys_including_inactive_profiles(tmp_path, text):
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(path)


def test_resolved_roundtrip(tiny_config, tmp_path):
    tiny_config.save_resolved(tmp_path)
    restored = load_config(tmp_path / "resolved_config.toml")
    assert restored.resolved_dict() == tiny_config.resolved_dict()


@pytest.mark.parametrize("patch", [
    {"data": {"test_fraction": 0.2}},
    {"training": {"batch_size": 0}},
    {"training": {"learning_rate": float("nan")}},
    {"architectures": {"shared": {"kernel_size": 2}}},
    {"architectures": {"shared": {"compression_channels": 4}}},
    {"quantum": {"qubits": 6}},
    {"quantum": {"shots": 100}},
])
def test_invalid_configs(tiny_config, patch):
    with pytest.raises(ConfigError):
        config_from_dict(merge_tables(tiny_config.resolved_dict(), patch), root=tiny_config.paths.project_root)


def test_direct_not_constrained_by_quantum_interface(tiny_config):
    raw = merge_tables(tiny_config.resolved_dict(), {"run": {"model": "direct_cnn"},
        "architectures": {"direct": {"compression_channels": 5, "quantum_spatial_size": 7}}, "quantum": {"qubits": 6}})
    config_from_dict(raw, root=tiny_config.paths.project_root)


def test_cache_identity_ignores_model_and_training_seed(tiny_config):
    assert tiny_config.cache_dir == tiny_config.for_model("direct_cnn", 123).cache_dir
    changed = replace(tiny_config, data=replace(tiny_config.data, image_size=64))
    assert changed.cache_dir != tiny_config.cache_dir


def test_architectural_changes_from_toml(tiny_config, tmp_path):
    import torch
    from qmla.model import GalaxyClassifier
    torch.set_num_threads(2)
    for mode in ("qufex", "cnn_replacement", "direct_cnn"):
        raw = tiny_config.for_model(mode).resolved_dict()
        block = "direct" if mode == "direct_cnn" else "shared"
        raw["architectures"][block].update({"encoder_channels": [5, 16], "pooling": "avg", "kernel_size": 5,
            "normalization": "group", "activation": "silu", "post_channels": [], "classifier_hidden_neurons": [7, 5]})
        raw["architectures"]["replacement"]["hidden_channels"] = [3, 5]
        path = tmp_path / f"{mode}.toml"
        path.write_text(to_toml(raw), encoding="utf-8")
        model = GalaxyClassifier(load_config(path))
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 3)


def test_checkpoint_paths_relocate_across_operating_systems(tmp_path):
    saved = {"project_root": "G:\\research\\qmla", "processed_dir": "G:\\research\\qmla\\data\\processed",
             "runs_dir": "G:\\research\\qmla\\runs", "raw_dir": "E:\\datasets\\gz2"}
    relocated = relocate_paths(saved, {"project_root": tmp_path})
    assert Path(relocated["processed_dir"]) == tmp_path / "data/processed"
    assert Path(relocated["runs_dir"]) == tmp_path / "runs"
    assert relocated["raw_dir"] == saved["raw_dir"]
