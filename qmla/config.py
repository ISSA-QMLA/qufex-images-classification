"""Typed TOML configuration with strict validation and portable path handling."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def _unknown_keys(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(unknown)}")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _positive_float(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    if result < 0 if allow_zero else result <= 0:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ConfigError(f"{name} must be {comparator}")
    return result


def _int_tuple(value: Any, name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        suffix = " (an empty list is allowed)" if allow_empty else ""
        raise ConfigError(f"{name} must be a TOML integer array{suffix}")
    return tuple(_positive_int(item, f"{name}[]") for item in value)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ConfigError(f"{name} must be a non-empty TOML string array")
    return tuple(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value.strip()


def _resolved_path(value: Any, root: Path, name: str) -> Path:
    raw = Path(_string(value, name)).expanduser()
    return raw.resolve() if raw.is_absolute() else (root / raw).resolve()


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    runs_dir: Path
    checkpoints_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 64
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 8
    preprocessing_batch_size: int = 256
    clean_label_policy: str = "hart_clean_flags"


@dataclass(frozen=True)
class ModelConfig:
    mode: str = "qufex"
    encoder_channels: tuple[int, ...] = (4, 8, 8, 8, 16)
    convolutions_per_block: int = 2
    compression_channels: int = 16
    quantum_spatial_size: int = 2
    post_quantum_channels: tuple[int, ...] = (16,)
    classifier_hidden_neurons: tuple[int, ...] = (32,)
    dropout: float = 0.20


@dataclass(frozen=True)
class QuantumConfig:
    backend: str = "default.qubit"
    diff_method: str = "backprop"
    shots: int = 0
    qubits: int = 8
    filters: int = 1
    input_angle_scale: float = math.pi


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 32
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 8
    amp: bool = True
    class_weighting: bool = True
    checkpoint_every: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 64
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    class_names: tuple[str, ...] = ("smooth", "unbarred_spiral", "barred_spiral")


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: Path | None = None

    def validate(self) -> None:
        fractions = (self.data.train_fraction, self.data.validation_fraction, self.data.test_fraction)
        if any(not 0 < fraction < 1 for fraction in fractions):
            raise ConfigError("data split fractions must each be between 0 and 1")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ConfigError("data split fractions must sum to 1.0")
        if self.data.clean_label_policy != "hart_clean_flags":
            raise ConfigError("data.clean_label_policy currently supports only 'hart_clean_flags'")
        if self.model.mode not in {"qufex", "classical"}:
            raise ConfigError("model.mode must be 'qufex' or 'classical'")
        if self.model.compression_channels != 16:
            raise ConfigError("model.compression_channels must be 16 for the source-faithful QuFeX mapping")
        if self.model.quantum_spatial_size != 2:
            raise ConfigError("model.quantum_spatial_size must be 2 for the 2x2x2 QuFeX groups")
        if (self.quantum.qubits, self.quantum.filters) not in {(8, 1), (4, 1), (4, 2)}:
            raise ConfigError(
                "quantum qubits/filters must select a published QuFeX variant: "
                "8/1, 4/1, or 4/2"
            )
        if not 0 <= self.model.dropout < 1:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.data.image_size < 2 ** len(self.model.encoder_channels):
            raise ConfigError("data.image_size is too small for the configured encoder pooling depth")
        if len(self.evaluation.class_names) != 3:
            raise ConfigError("evaluation.class_names must contain exactly three names")
        if self.training.optimizer.lower() not in {"adam", "adamw"}:
            raise ConfigError("training.optimizer must be 'adam' or 'adamw'")

    def resolved_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        for key, value in result["paths"].items():
            result["paths"][key] = str(value)
        return result

    def save_resolved(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        resolved = self.resolved_dict()
        destination = directory / "resolved_config.json"
        destination.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
        toml_lines: list[str] = []
        for section, values in resolved.items():
            toml_lines.append(f"[{section}]")
            for key, value in values.items():
                toml_lines.append(f"{key} = {_toml_value(value)}")
            toml_lines.append("")
        (directory / "resolved_config.toml").write_text("\n".join(toml_lines), encoding="utf-8")
        if self.source_path is not None:
            (directory / "source_config.toml").write_bytes(self.source_path.read_bytes())
        return destination


_SECTIONS = {"paths", "data", "model", "quantum", "training", "evaluation"}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot serialize value to TOML: {value!r}")


def load_config(path: str | Path) -> AppConfig:
    """Load and strictly validate a project TOML file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown_sections = sorted(set(raw) - _SECTIONS)
    if unknown_sections:
        raise ConfigError(f"Unknown configuration section(s): {', '.join(unknown_sections)}")

    paths_raw = raw.get("paths", {})
    _unknown_keys("paths", paths_raw, {"project_root", "raw_dir", "processed_dir", "runs_dir", "checkpoints_dir", "results_dir"})
    config_parent = source.parent
    project_value = paths_raw.get("project_root", "..")
    project_root = _resolved_path(project_value, config_parent, "paths.project_root")
    paths = PathsConfig(
        project_root=project_root,
        raw_dir=_resolved_path(paths_raw.get("raw_dir", "data/raw"), project_root, "paths.raw_dir"),
        processed_dir=_resolved_path(paths_raw.get("processed_dir", "data/processed"), project_root, "paths.processed_dir"),
        runs_dir=_resolved_path(paths_raw.get("runs_dir", "runs"), project_root, "paths.runs_dir"),
        checkpoints_dir=_resolved_path(paths_raw.get("checkpoints_dir", "checkpoints"), project_root, "paths.checkpoints_dir"),
        results_dir=_resolved_path(paths_raw.get("results_dir", "results"), project_root, "paths.results_dir"),
    )

    data_raw = raw.get("data", {})
    _unknown_keys("data", data_raw, {"image_size", "train_fraction", "validation_fraction", "test_fraction", "seed", "num_workers", "preprocessing_batch_size", "clean_label_policy"})
    data = DataConfig(
        image_size=_positive_int(data_raw.get("image_size", 64), "data.image_size"),
        train_fraction=_positive_float(data_raw.get("train_fraction", 0.70), "data.train_fraction"),
        validation_fraction=_positive_float(data_raw.get("validation_fraction", 0.15), "data.validation_fraction"),
        test_fraction=_positive_float(data_raw.get("test_fraction", 0.15), "data.test_fraction"),
        seed=_positive_int(data_raw.get("seed", 42), "data.seed", allow_zero=True),
        num_workers=_positive_int(data_raw.get("num_workers", 8), "data.num_workers", allow_zero=True),
        preprocessing_batch_size=_positive_int(data_raw.get("preprocessing_batch_size", 256), "data.preprocessing_batch_size"),
        clean_label_policy=_string(data_raw.get("clean_label_policy", "hart_clean_flags"), "data.clean_label_policy"),
    )

    model_raw = raw.get("model", {})
    _unknown_keys("model", model_raw, {"mode", "encoder_channels", "convolutions_per_block", "compression_channels", "quantum_spatial_size", "post_quantum_channels", "classifier_hidden_neurons", "dropout"})
    model = ModelConfig(
        mode=_string(model_raw.get("mode", "qufex"), "model.mode").lower(),
        encoder_channels=_int_tuple(model_raw.get("encoder_channels", [4, 8, 8, 8, 16]), "model.encoder_channels"),
        convolutions_per_block=_positive_int(model_raw.get("convolutions_per_block", 2), "model.convolutions_per_block"),
        compression_channels=_positive_int(model_raw.get("compression_channels", 16), "model.compression_channels"),
        quantum_spatial_size=_positive_int(model_raw.get("quantum_spatial_size", 2), "model.quantum_spatial_size"),
        post_quantum_channels=_int_tuple(model_raw.get("post_quantum_channels", [16]), "model.post_quantum_channels", allow_empty=True),
        classifier_hidden_neurons=_int_tuple(model_raw.get("classifier_hidden_neurons", [32]), "model.classifier_hidden_neurons", allow_empty=True),
        dropout=_positive_float(model_raw.get("dropout", 0.20), "model.dropout", allow_zero=True),
    )

    quantum_raw = raw.get("quantum", {})
    _unknown_keys("quantum", quantum_raw, {"backend", "diff_method", "shots", "qubits", "filters", "input_angle_scale"})
    quantum = QuantumConfig(
        backend=_string(quantum_raw.get("backend", "default.qubit"), "quantum.backend"),
        diff_method=_string(quantum_raw.get("diff_method", "backprop"), "quantum.diff_method"),
        shots=_positive_int(quantum_raw.get("shots", 0), "quantum.shots", allow_zero=True),
        qubits=_positive_int(quantum_raw.get("qubits", 8), "quantum.qubits"),
        filters=_positive_int(quantum_raw.get("filters", 1), "quantum.filters"),
        input_angle_scale=_positive_float(quantum_raw.get("input_angle_scale", math.pi), "quantum.input_angle_scale"),
    )

    training_raw = raw.get("training", {})
    _unknown_keys("training", training_raw, {"device", "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "early_stopping_patience", "amp", "class_weighting", "checkpoint_every"})
    training = TrainingConfig(
        device=_string(training_raw.get("device", "cuda"), "training.device"),
        epochs=_positive_int(training_raw.get("epochs", 50), "training.epochs"),
        batch_size=_positive_int(training_raw.get("batch_size", 32), "training.batch_size"),
        optimizer=_string(training_raw.get("optimizer", "adamw"), "training.optimizer").lower(),
        learning_rate=_positive_float(training_raw.get("learning_rate", 1e-3), "training.learning_rate"),
        weight_decay=_positive_float(training_raw.get("weight_decay", 1e-4), "training.weight_decay", allow_zero=True),
        early_stopping_patience=_positive_int(training_raw.get("early_stopping_patience", 8), "training.early_stopping_patience"),
        amp=_boolean(training_raw.get("amp", True), "training.amp"),
        class_weighting=_boolean(training_raw.get("class_weighting", True), "training.class_weighting"),
        checkpoint_every=_positive_int(training_raw.get("checkpoint_every", 1), "training.checkpoint_every"),
    )

    evaluation_raw = raw.get("evaluation", {})
    _unknown_keys("evaluation", evaluation_raw, {"batch_size", "save_predictions", "save_confusion_matrix", "class_names"})
    evaluation = EvaluationConfig(
        batch_size=_positive_int(evaluation_raw.get("batch_size", 64), "evaluation.batch_size"),
        save_predictions=_boolean(evaluation_raw.get("save_predictions", True), "evaluation.save_predictions"),
        save_confusion_matrix=_boolean(evaluation_raw.get("save_confusion_matrix", True), "evaluation.save_confusion_matrix"),
        class_names=_string_tuple(evaluation_raw.get("class_names", ["smooth", "unbarred_spiral", "barred_spiral"]), "evaluation.class_names"),
    )

    config = AppConfig(paths=paths, data=data, model=model, quantum=quantum, training=training, evaluation=evaluation, source_path=source)
    config.validate()
    return config
