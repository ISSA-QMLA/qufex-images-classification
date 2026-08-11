from __future__ import annotations

from pathlib import Path

import pytest

from qmla.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.toml"


def test_default_config_resolves_from_config_location() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.paths.project_root == DEFAULT_CONFIG.parents[1].resolve()
    assert config.paths.raw_dir == (DEFAULT_CONFIG.parents[1] / "data" / "raw").resolve()
    assert config.data.image_size == 64
    assert config.model.encoder_channels == (4, 8, 8, 8, 16)
    assert config.model.compression_channels == 16
    assert config.evaluation.class_names[2] == "barred_spiral"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[paths]\nproject_root='.'\n[data]\nimgae_size=64\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="imgae_size"):
        load_config(path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("model", "compression_channels", "4", "compression_channels"),
        ("model", "quantum_spatial_size", "4", "quantum_spatial_size"),
    ],
)
def test_source_faithful_dimensions_are_validated(
    tmp_path: Path, section: str, key: str, value: str, message: str
) -> None:
    original = DEFAULT_CONFIG.read_text(encoding="utf-8")
    defaults = {"compression_channels": "16", "quantum_spatial_size": "2"}
    original = original.replace(f"{key} = {defaults[key]}", f"{key} = {value}")
    path = tmp_path / "bad_dimensions.toml"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(("qubits", "filters"), [(8, 1), (4, 1), (4, 2)])
def test_published_quantum_variants_are_configurable(
    tmp_path: Path, qubits: int, filters: int
) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    text = text.replace("qubits = 8", f"qubits = {qubits}")
    text = text.replace("filters = 1", f"filters = {filters}")
    path = tmp_path / f"variant_{qubits}_{filters}.toml"
    path.write_text(text, encoding="utf-8")

    config = load_config(path)
    assert (config.quantum.qubits, config.quantum.filters) == (qubits, filters)


@pytest.mark.parametrize(("qubits", "filters"), [(8, 2), (4, 3), (6, 1)])
def test_undefined_quantum_variants_are_rejected(
    tmp_path: Path, qubits: int, filters: int
) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8")
    text = text.replace("qubits = 8", f"qubits = {qubits}")
    text = text.replace("filters = 1", f"filters = {filters}")
    path = tmp_path / f"invalid_variant_{qubits}_{filters}.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match="8/1, 4/1, or 4/2"):
        load_config(path)


def test_resolved_toml_round_trip(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.save_resolved(tmp_path)
    reloaded = load_config(tmp_path / "resolved_config.toml")
    assert reloaded.resolved_dict() == config.resolved_dict()
    assert (tmp_path / "resolved_config.json").is_file()
    assert (tmp_path / "source_config.toml").is_file()


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("test_fraction = 0.15", "test_fraction = 0.20")
    path = tmp_path / "bad_split.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(path)
