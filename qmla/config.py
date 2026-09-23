"""One TOML, recursive profiles, and validated experiment settings."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

MODELS = ("qufex", "cnn_replacement", "direct_cnn")
CLASS_NAMES = ("smooth", "unbarred_spiral", "barred_spiral")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    model: str = "qufex"
    profile: str = "full64"


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
    subset_seed: int = 42
    train_limit: int = 0
    validation_limit: int = 0
    test_limit: int = 0
    num_workers: int = 0
    preprocessing_batch_size: int = 256
    clean_label_policy: str = "hart_clean_flags"


@dataclass(frozen=True)
class ArchitectureConfig:
    encoder_channels: tuple[int, ...] = (4, 8, 8, 8, 16)
    convolutions_per_block: int = 2
    kernel_size: int = 3
    pooling: str = "max"
    pool_size: int = 2
    normalization: str = "batch"
    activation: str = "relu"
    compression_channels: int = 16
    quantum_spatial_size: int = 2
    projection: str = "auto"
    post_channels: tuple[int, ...] = (16,)
    post_kernel_size: int = 3
    post_convolutions: int = 1
    classifier_hidden_neurons: tuple[int, ...] = (32,)
    dropout: float = 0.2


@dataclass(frozen=True)
class ReplacementConfig:
    hidden_channels: tuple[int, ...] = (8,)
    kernel_size: int = 3
    output_kernel_size: int = 1
    activation: str = "relu"
    normalization: str = "none"
    output_activation: str = "identity"


@dataclass(frozen=True)
class ArchitecturesConfig:
    shared: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    direct: ArchitectureConfig = field(default_factory=lambda: ArchitectureConfig(
        encoder_channels=(16, 32, 64), convolutions_per_block=1, post_channels=()))
    replacement: ReplacementConfig = field(default_factory=ReplacementConfig)


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
    device: str = "auto"
    epochs: int = 50
    batch_size: int = 8
    optimizer: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 8
    amp: bool = True
    class_weighting: bool = True
    checkpoint_every: int = 1
    seeds: tuple[int, ...] = (42,)
    deterministic: bool = False
    cpu_threads: int = 4


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "none"
    monitor: str = "macro_f1"
    factor: float = 0.5
    patience: int = 4
    threshold: float = 0.001
    threshold_mode: str = "abs"
    cooldown: int = 0
    min_lr: float = 0.000001


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 64
    save_predictions: bool = True
    save_confusion_matrix: bool = True
    class_names: tuple[str, ...] = CLASS_NAMES


def merge_tables(base: dict, override: dict) -> dict:
    """Merge tables recursively; arrays/scalars replace completely."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge_tables(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def relocate_paths(saved: dict, overrides: dict) -> dict:
    """Relocate saved absolute paths beneath the old project, across OSes."""
    result = dict(saved)
    if "project_root" in overrides and "project_root" in saved:
        old_root = str(saved["project_root"])
        path_type = PureWindowsPath if PureWindowsPath(old_root).drive else PurePosixPath
        old = path_type(old_root)
        if old.is_absolute():
            for key, value in saved.items():
                if key != "project_root" and key not in overrides:
                    try:
                        relative = path_type(str(value)).relative_to(old)
                        result[key] = str(Path(str(overrides["project_root"])).joinpath(*relative.parts))
                    except ValueError:
                        pass # external data volumes need their own explicit path override
    result.update({key: str(value) for key, value in overrides.items()})
    return result


SCHEMA = {"run": RunConfig, "data": DataConfig, "quantum": QuantumConfig,
          "training": TrainingConfig, "scheduler": SchedulerConfig, "evaluation": EvaluationConfig,
          "architectures": {"shared": ArchitectureConfig, "direct": ArchitectureConfig, "replacement": ReplacementConfig},
          "paths": {name: None for name in ("project_root", "raw_dir", "processed_dir", "runs_dir", "checkpoints_dir", "results_dir")}}


def _check_keys(raw: dict, schema: dict, prefix: str = "") -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{prefix or 'configuration'} must be a table")
    for key, value in raw.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ConfigError(f"Unknown key: {name}. Legacy configurations are unsupported; use configs/experiments.toml.")
        child = schema[key]
        if isinstance(child, dict):
            _check_keys(value, child, name)
        elif child is not None:
            _check_keys(value, {f.name: None for f in fields(child)}, name)


def _section(cls: type, raw: dict, name: str):
    default = cls()
    values = {}
    for key, value in raw.items():
        expected = getattr(default, key)
        if isinstance(expected, bool):
            valid = isinstance(value, bool)
        elif isinstance(expected, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(expected, float):
            valid = isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
        elif isinstance(expected, str):
            valid = isinstance(value, str) and bool(value.strip())
        else:
            valid = isinstance(value, (list, tuple))
        if not valid:
            raise ConfigError(f"Invalid type/value for {name}.{key}")
        if isinstance(expected, tuple):
            item_type = str if key == "class_names" else int
            if any(not isinstance(item, item_type) or isinstance(item, bool) for item in value):
                raise ConfigError(f"Invalid array for {name}.{key}")
            value = tuple(value)
        values[key] = value
    return cls(**values)


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    architectures: ArchitecturesConfig = field(default_factory=ArchitecturesConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    source_path: Path | None = None
    overrides: dict = field(default_factory=dict)

    @property
    def architecture(self) -> ArchitectureConfig:
        return self.architectures.direct if self.run.model == "direct_cnn" else self.architectures.shared

    def data_signature(self) -> dict:
        return {key: value for key, value in asdict(self.data).items() if key not in {"num_workers", "preprocessing_batch_size"}} | {"class_names": list(self.evaluation.class_names), "format_version": 2}

    @property
    def cache_dir(self) -> Path:
        digest = hashlib.sha256(json.dumps(self.data_signature(), sort_keys=True).encode()).hexdigest()[:16]
        return self.paths.processed_dir / f"gz2-{self.data.image_size}-{digest}"

    def for_model(self, model: str, seed: int | None = None) -> AppConfig:
        result = replace(self, run=replace(self.run, model=model),
                         training=replace(self.training, seeds=(seed,)) if seed is not None else self.training)
        result.validate()
        return result

    def validate(self) -> None:
        if self.run.model not in MODELS:
            raise ConfigError(f"run.model must be one of {MODELS}")
        fractions = (self.data.train_fraction, self.data.validation_fraction, self.data.test_fraction)
        if any(not 0 < x < 1 for x in fractions) or not math.isclose(sum(fractions), 1, abs_tol=1e-8):
            raise ConfigError("Data split fractions must be positive and sum to one")
        if self.data.clean_label_policy != "hart_clean_flags" or self.evaluation.class_names != CLASS_NAMES:
            raise ConfigError("Only the three ordered Hart clean-flag classes are supported")
        for section in (self.data, self.training, self.scheduler, self.evaluation, self.quantum):
            for name, value in asdict(section).items():
                if isinstance(value, int) and not isinstance(value, bool) and value < 0:
                    raise ConfigError(f"{name} must be nonnegative")
        for name, value in (("image_size", self.data.image_size), ("preprocessing_batch_size", self.data.preprocessing_batch_size),
                            ("epochs", self.training.epochs), ("batch_size", self.training.batch_size),
                            ("evaluation.batch_size", self.evaluation.batch_size), ("checkpoint_every", self.training.checkpoint_every),
                            ("cpu_threads", self.training.cpu_threads)):
            if value < 1:
                raise ConfigError(f"{name} must be positive")
        if any(0 < getattr(self.data, f"{s}_limit") < 3 for s in ("train", "validation", "test")):
            raise ConfigError("Each subset limit must be zero (all samples) or at least three")
        if not self.training.seeds or len(set(self.training.seeds)) != len(self.training.seeds) or any(s < 0 for s in self.training.seeds):
            raise ConfigError("training.seeds must contain distinct nonnegative integers")
        if self.training.optimizer not in {"adam", "adamw"} or self.training.learning_rate <= 0 or self.training.weight_decay < 0:
            raise ConfigError("Invalid optimizer, learning_rate, or weight_decay")
        scheduler = self.scheduler
        if scheduler.name not in {"none", "reduce_on_plateau"}:
            raise ConfigError("scheduler.name must be none or reduce_on_plateau")
        if scheduler.monitor not in {"macro_f1", "validation_loss"}:
            raise ConfigError("scheduler.monitor must be macro_f1 or validation_loss")
        if not 0 < scheduler.factor < 1:
            raise ConfigError("scheduler.factor must be between zero and one (exclusive)")
        if scheduler.threshold_mode not in {"abs", "rel"}:
            raise ConfigError("scheduler.threshold_mode must be abs or rel")
        if any(not math.isfinite(value) or value < 0 for value in (scheduler.threshold, scheduler.min_lr)):
            raise ConfigError("scheduler.threshold and scheduler.min_lr must be finite and nonnegative")
        if scheduler.name != "none" and scheduler.min_lr > self.training.learning_rate:
            raise ConfigError("scheduler.min_lr cannot exceed training.learning_rate")
        device = self.training.device
        if device not in {"auto", "cpu", "cuda"} and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ConfigError("training.device must be auto, cpu, cuda, or cuda:N")
        activations = {"relu", "gelu", "silu", "tanh", "identity"}
        for label, arch in (("shared", self.architectures.shared), ("direct", self.architectures.direct)):
            if not arch.encoder_channels or any(c <= 0 for c in (*arch.encoder_channels, *arch.post_channels, *arch.classifier_hidden_neurons)):
                raise ConfigError(f"architectures.{label}: widths must be positive")
            if arch.normalization not in {"batch", "group", "none"} or arch.activation not in activations:
                raise ConfigError(f"architectures.{label}: unsupported normalization/activation")
            if arch.pooling not in {"max", "avg", "none"} or arch.projection not in {"auto", "conv", "identity"}:
                raise ConfigError(f"architectures.{label}: unsupported pooling/projection")
            if not 0 <= arch.dropout < 1 or min(arch.pool_size, arch.convolutions_per_block, arch.post_convolutions) < 1:
                raise ConfigError(f"architectures.{label}: invalid dropout/pooling/repetitions")
            if any(k < 1 or k % 2 == 0 for k in (arch.kernel_size, arch.post_kernel_size)):
                raise ConfigError("Convolution kernels must be positive odd integers")
        arch = self.architecture
        spatial = self.data.image_size // (arch.pool_size ** len(arch.encoder_channels) if arch.pooling != "none" else 1)
        if spatial < 1:
            raise ConfigError("image_size is too small for the encoder pooling depth")
        rep = self.architectures.replacement
        if any(c < 1 for c in rep.hidden_channels) or any(k < 1 or k % 2 == 0 for k in (rep.kernel_size, rep.output_kernel_size)):
            raise ConfigError("Invalid replacement channels or kernels")
        if rep.activation not in activations or rep.output_activation not in activations or rep.normalization not in {"batch", "group", "none"}:
            raise ConfigError("Invalid replacement activation/normalization")
        if self.run.model != "direct_cnn":
            if arch.compression_channels != 16 or arch.quantum_spatial_size != 2 or spatial < 2:
                raise ConfigError("QuFeX/replacement require compression_channels=16, quantum_spatial_size=2 and encoder spatial size >=2")
            if arch.projection == "identity" and arch.encoder_channels[-1] != 16:
                raise ConfigError("Identity projection requires 16 encoder output channels")
            if (self.quantum.qubits, self.quantum.filters) not in {(8, 1), (4, 1), (4, 2)}:
                raise ConfigError("Supported qubits/filters are 8/1, 4/1, and 4/2")
        if self.run.model == "qufex":
            if self.quantum.input_angle_scale <= 0:
                raise ConfigError("quantum.input_angle_scale must be positive")
            if self.quantum.diff_method == "backprop" and (self.quantum.shots or self.quantum.backend != "default.qubit"):
                raise ConfigError("backprop requires analytic default.qubit; choose compatible differentiation for other backends")

    def resolved_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path")
        result.pop("overrides")
        result["paths"] = {k: str(v) for k, v in result["paths"].items()}
        return result

    def save_resolved(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        resolved = self.resolved_dict()
        source_bytes = self.source_path.read_bytes() if self.source_path and self.source_path.exists() else None
        destination = directory / "resolved_config.json"
        destination.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
        (directory / "resolved_config.toml").write_text(to_toml(resolved), encoding="utf-8")
        (directory / "cli_overrides.json").write_text(json.dumps(self.overrides, indent=2), encoding="utf-8")
        if source_bytes is not None:
            (directory / "source_config.toml").write_bytes(source_bytes)
        return destination


def to_toml(raw: dict) -> str:
    lines = []
    def visit(table: dict, prefix: str):
        if prefix:
            lines.append(f"[{prefix}]")
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                visit(value, f"{prefix}.{key}" if prefix else key)
    visit(raw, "")
    return "\n".join(lines)


def config_from_dict(raw: dict, *, root: Path, source: Path | None = None, overrides: dict | None = None) -> AppConfig:
    _check_keys(raw, SCHEMA)
    path_values = raw.get("paths", {})
    def resolve(value: str, base: Path) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError("Paths must be non-empty strings")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else base / path).resolve()
    project = resolve(path_values.get("project_root", ".."), root)
    paths = PathsConfig(project_root=project, **{name: resolve(path_values.get(name, default), project) for name, default in
        (("raw_dir", "data/raw"), ("processed_dir", "data/processed"), ("runs_dir", "runs"), ("checkpoints_dir", "checkpoints"), ("results_dir", "results"))})
    architectures = raw.get("architectures", {})
    shared = architectures.get("shared", {})
    inherited = {k: v for k, v in shared.items() if k not in {"encoder_channels", "compression_channels", "quantum_spatial_size", "projection", "post_channels"}}
    direct = merge_tables(asdict(ArchitecturesConfig().direct), inherited)
    direct = merge_tables(direct, architectures.get("direct", {}))
    config = AppConfig(paths=paths,
        run=_section(RunConfig, raw.get("run", {}), "run"),
        data=_section(DataConfig, raw.get("data", {}), "data"),
        architectures=ArchitecturesConfig(_section(ArchitectureConfig, shared, "architectures.shared"), _section(ArchitectureConfig, direct, "architectures.direct"), _section(ReplacementConfig, architectures.get("replacement", {}), "architectures.replacement")),
        quantum=_section(QuantumConfig, raw.get("quantum", {}), "quantum"),
        training=_section(TrainingConfig, raw.get("training", {}), "training"),
        scheduler=_section(SchedulerConfig, raw.get("scheduler", {}), "scheduler"),
        evaluation=_section(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
        source_path=source, overrides=overrides or {})
    config.validate()
    return config


def load_config(path: str | Path = "configs/experiments.toml", *, profile: str | None = None,
                model: str | None = None, device: str | None = None, paths: dict | None = None) -> AppConfig:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    profiles = raw.pop("profiles", {})
    _check_keys(raw, SCHEMA)
    if not isinstance(profiles, dict):
        raise ConfigError("profiles must be a table")
    for name, values in profiles.items():
        _check_keys(values, {k: v for k, v in SCHEMA.items() if k != "run"}, f"profiles.{name}")
    selected = profile or raw.get("run", {}).get("profile", "full64")
    if (profiles or profile) and selected not in profiles:
        raise ConfigError(f"Unknown profile {selected!r}; choices: {list(profiles)}")
    raw = merge_tables(raw, profiles.get(selected, {}))
    raw = merge_tables(raw, {"run": {"profile": selected}})
    overrides: dict = {}
    if profile is not None:
        overrides["run"] = {"profile": profile}
    if model is not None:
        overrides.setdefault("run", {})["model"] = model
    if device is not None:
        overrides["training"] = {"device": device}
    if paths:
        overrides["paths"] = {k: str(v) for k, v in paths.items()}
        raw["paths"] = relocate_paths(raw.get("paths", {}), paths)
    return config_from_dict(merge_tables(raw, overrides), root=source.parent, source=source, overrides=overrides)
