"""Validation-only compression sweeps, persistent budgets and paired reporting."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import time
import tomllib
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qmla.config import AppConfig, ConfigError, _check_keys, _section, config_from_dict, relocate_paths, to_toml
from qmla.data import GalaxyZooPreprocessor, validate_cache
from qmla.engine import Evaluator, Trainer, load_checkpoint_config
from qmla.runtime import RuntimeMonitor
from qmla.utils import require_torch


@dataclass(frozen=True)
class Level:
    channels: int = 64
    spatial_size: int = 8

    @property
    def values(self):
        return self.channels * self.spatial_size**2


@dataclass(frozen=True)
class SweepSettings:
    max_macro_f1_drop: float = 0.05
    run_hours: float = 4.0
    total_hours: float = 48.0
    warmup_steps: int = 5
    measured_steps: int = 20
    circuit_chunk_size: int = 256
    include_more_compressed: bool = True


@dataclass(frozen=True)
class StudyConfig:
    config: AppConfig
    sweep: SweepSettings
    levels: dict[str, Level]

    def for_run(self, level: str, variant: str, seed: int) -> AppConfig:
        config = self.config
        if variant == "direct_cnn":
            return config.for_model(variant, seed)
        shape = self.levels[level]
        arch = replace(config.architectures.direct, compression_channels=shape.channels,
                       quantum_spatial_size=shape.spatial_size, circuit_chunk_size=self.sweep.circuit_chunk_size)
        config = replace(config, architectures=replace(config.architectures, shared=arch))
        return config.for_model(variant, seed)

    def validate(self):
        self.config.validate()
        if not self.config.training.paired_randomness:
            raise ConfigError("Sweep requires training.paired_randomness=true")
        if self.config.architectures.direct.post_channels:
            raise ConfigError("Sweep requires direct.post_channels=[]")
        if not self.levels:
            raise ConfigError("Sweep requires at least one level")
        s = self.sweep
        if not 0 <= s.max_macro_f1_drop <= 1 or min(s.run_hours, s.total_hours) <= 0:
            raise ConfigError("Invalid degradation threshold or time budget")
        if s.warmup_steps < 0 or min(s.measured_steps, s.circuit_chunk_size) < 1:
            raise ConfigError("Invalid profiling step counts or circuit chunk size")
        previous = math.inf
        for name, level in self.levels.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
                raise ConfigError("Level names must be safe alphanumeric identifiers")
            if level.values >= previous:
                raise ConfigError("Levels must have strictly decreasing feature counts")
            previous = level.values
            self.for_run(name, "compression_qufex", self.config.training.seeds[0]).validate()

    def description(self):
        return {"experiment": self.config.resolved_dict(), "sweep": asdict(self.sweep),
                "levels": {name: {**asdict(level), "values": level.values,
                                  "circuit_instances_per_image": level.values // 8}
                           for name, level in self.levels.items()},
                "selection_metric": "mean paired best-validation macro-F1 drop versus first level",
                "test_evaluation": "explicit evaluate stage only"}


def load_study(path, *, device=None, paths=None) -> StudyConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    sweep_raw = raw.pop("sweep", {})
    levels_raw = sweep_raw.pop("levels", {})
    _check_keys({"sweep": sweep_raw}, {"sweep": SweepSettings})
    if not isinstance(levels_raw, dict):
        raise ConfigError("sweep.levels must be a table")
    for name, values in levels_raw.items():
        _check_keys({name: values}, {name: Level})
    overrides = {}
    if device:
        raw.setdefault("training", {})["device"] = device
        overrides["training"] = {"device": device}
    if paths:
        raw["paths"] = relocate_paths(raw.get("paths", {}), paths)
        overrides["paths"] = {k: str(v) for k, v in paths.items()}
    config = config_from_dict(raw, root=source.parent, source=source, overrides=overrides)
    result = StudyConfig(config, _section(SweepSettings, sweep_raw, "sweep"),
                         {k: _section(Level, v, f"sweep.levels.{k}") for k, v in levels_raw.items()})
    result.validate()
    return result


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def choose_level(study: StudyConfig, rows: list[dict]) -> dict | None:
    """All classical seeds/levels must be complete; do not assume monotonicity."""
    scores = {(r["level"], r["seed"]): r["macro_f1"] for r in rows
              if r["variant"] == "compression_cnn" and r["status"] == "completed"}
    names = list(study.levels)
    seeds = study.config.training.seeds
    if any((name, seed) not in scores for name in names for seed in seeds):
        return None
    reference = [scores[names[0], seed] for seed in seeds]
    decisions = []
    selected = names[0]
    for name in names:
        values = [scores[name, seed] for seed in seeds]
        delta = np.asarray(reference) - values
        mean_drop = float(np.mean(delta))
        acceptable = mean_drop <= study.sweep.max_macro_f1_drop or math.isclose(
            mean_drop, study.sweep.max_macro_f1_drop, rel_tol=0, abs_tol=1e-12)
        decisions.append({"level": name, "scores": values, "mean_macro_f1": float(np.mean(values)),
                          "mean_drop": mean_drop, "drop_std": float(np.std(delta, ddof=1)) if len(seeds) > 1 else None,
                          "acceptable": acceptable})
        if acceptable:
            selected = name
    return {"selected_level": selected, "reference_level": names[0], "reference_scores": reference,
            "seeds": list(seeds), "threshold": study.sweep.max_macro_f1_drop, "decisions": decisions}


def quantum_order(study: StudyConfig, selected: str) -> list[str]:
    names = list(study.levels)
    index = names.index(selected)
    extra = names[index + 1:index + 2] if study.sweep.include_more_compressed else []
    return [selected, *extra, *reversed(names[:index])]


class BudgetExhausted(RuntimeError):
    pass


class StudyInterrupted(KeyboardInterrupt):
    """Scheduler interruption: keep jobs resumable without consuming their budget."""


def is_oom(exc):
    return isinstance(exc, (MemoryError, require_torch().OutOfMemoryError)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


class StudyRunner:
    """Single-process runner. State is committed at every budget checkpoint."""

    def __init__(self, study: StudyConfig, root: Path | None = None, *, resume=False, read_only=False):
        study.validate()
        self.study = study
        self.root = (root or study.config.paths.runs_dir / f"compression_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}").resolve()
        self.state_path = self.root / "study.json"
        self.active = None
        self.last_tick = None
        self.stop_requested = None
        description = study.description()
        # Device relocation is allowed and recorded per attempt; scientific settings are fixed.
        fingerprint_data = json.loads(json.dumps(description))
        fingerprint_data["experiment"]["training"].pop("device")
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
        if self.state_path.exists():
            if not resume and not read_only:
                raise ValueError("Study exists; pass --resume with --run-dir")
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state["fingerprint"] != fingerprint:
                raise ValueError("Study configuration differs; use its original configuration or start a new study")
            if not read_only:
                for group in ("jobs", "profiles", "evaluations"):
                    for row in self.state[group].values():
                        if row["status"] == "running":
                            row.update(status="interrupted", stopping_reason="previous invocation interrupted")
        else:
            if resume or read_only:
                raise ValueError("No existing study.json in --run-dir")
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError("New study requires an empty output directory")
            self.state = {"version": 1, "id": f"compression_{uuid.uuid4().hex[:12]}", "fingerprint": fingerprint,
                          "configuration": description, "elapsed_seconds": 0.0, "preparation_seconds": 0.0,
                          "jobs": {}, "profiles": {}, "evaluations": {}, "selection": None,
                          "expansion_stopped": None, "dataset_id": None, "evaluation_manifest": None}
            self.root.mkdir(parents=True, exist_ok=True)
            study.config.save_resolved(self.root)
            snapshot = study.config.resolved_dict() | {"sweep": asdict(study.sweep) | {
                "levels": {name: asdict(level) for name, level in study.levels.items()}}}
            (self.root / "resolved_sweep.toml").write_text(to_toml(snapshot), encoding="utf-8")
        if not read_only:
            self.save()

    def save(self):
        atomic_json(self.state_path, self.state)

    def request_stop(self, reason):
        # Signal handlers only set a flag; never interrupt checkpoint/JSON writes.
        self.stop_requested = str(reason)

    def tick(self, *, check=True):
        now = time.perf_counter()
        if self.last_tick is not None:
            delta = now - self.last_tick
            self.state["elapsed_seconds"] += delta
            if self.active is not None:
                self.active["elapsed_seconds"] += delta
        self.last_tick = now
        self.save()
        if check:
            if self.stop_requested:
                raise StudyInterrupted(self.stop_requested)
            if self.state["elapsed_seconds"] >= self.study.sweep.total_hours * 3600:
                raise BudgetExhausted("study_time_limit")
            if self.active is not None and self.active["elapsed_seconds"] >= self.study.sweep.run_hours * 3600:
                raise BudgetExhausted("run_time_limit")

    def control(self, event, trainer):
        # Commit a fully validated epoch even if its final batch just consumed the budget.
        self.tick(check=event != "epoch_metrics")
        if event == "epoch_metrics":
            trainer.history[-1]["cumulative_elapsed_seconds"] = self.active["elapsed_seconds"]

    def prepare(self):
        started = time.perf_counter()
        config = self.study.config
        if not config.cache_dir.exists():
            GalaxyZooPreprocessor(config).run()
        metadata = validate_cache(config)
        if self.state["dataset_id"] not in (None, metadata["dataset_id"]):
            raise RuntimeError("Study dataset identity changed")
        self.state["dataset_id"] = metadata["dataset_id"]
        self.state["preparation_seconds"] += time.perf_counter() - started
        self.save()

    def row(self, level, variant, seed):
        key = f"{level}_{variant}_seed{seed}"
        if key not in self.state["jobs"]:
            shape = self.study.levels.get(level)
            arch = self.study.config.architectures.direct
            reference_spatial = self.study.config.data.image_size // (
                arch.pool_size ** len(arch.encoder_channels) if arch.pooling != "none" else 1)
            actual_shape = shape or Level(arch.encoder_channels[-1], reference_spatial)
            self.state["jobs"][key] = {"key": key, "level": level, "variant": variant, "seed": seed,
                "status": "pending", "stopping_reason": None, "elapsed_seconds": 0.0, "attempts": [],
                "channels": actual_shape.channels, "spatial_size": actual_shape.spatial_size,
                "height": actual_shape.spatial_size, "width": actual_shape.spatial_size,
                "feature_values": actual_shape.values,
                "compression_axis": ("spatial" if shape.channels == next(iter(self.study.levels.values())).channels else "channel") if shape else "reference",
                "circuit_instances_per_image": shape.values // 8 if variant == "compression_qufex" else 0,
                "qubits": 8 if variant == "compression_qufex" else 0,
                "filters": 1 if variant == "compression_qufex" else 0,
                "circuit_chunk_size": self.study.sweep.circuit_chunk_size,
                "backend": self.study.config.quantum.backend if variant == "compression_qufex" else None,
                "shots": 0, "diff_method": "backprop" if variant == "compression_qufex" else None,
                "run_dir": str(self.root / "jobs" / f"{self.state['id']}_{key}")}
        return self.state["jobs"][key]

    def _begin(self, row):
        self.tick()
        self.active = row
        self.tick()
        row.update(status="running", stopping_reason=None)
        self.save()

    def _finish(self, row, exc=None):
        self.tick(check=False)
        if exc is not None:
            status = "budget_exhausted" if isinstance(exc, BudgetExhausted) else "oom" if is_oom(exc) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            row.update(status=status, stopping_reason=str(exc) or type(exc).__name__)
        self.active = None
        self.save()
        gc.collect()
        torch = require_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, level, variant, seed):
        row = self.row(level, variant, seed)
        if row["status"] in {"completed", "budget_exhausted", "oom"}:
            return row["status"] == "completed"
        trainer = None
        exc = None
        attempt = {}
        try:
            self._begin(row)
            config = self.study.for_run(level, variant, seed)
            trainer = Trainer(config, Path(row["run_dir"]), control=self.control)
            latest = trainer.checkpoint_dir / "latest.pt"
            if latest.exists():
                trainer.resume(latest)
            best = trainer.run()
            row.update(status="completed", stopping_reason="early_stopping" if trainer.bad_epochs >= config.training.early_stopping_patience > 0 else "epochs_completed",
                       checkpoint=str(best), macro_f1=trainer.best_macro_f1, best_epoch=trainer.best_epoch,
                       epochs_completed=len(trainer.history), dataset_id=trainer.metadata["dataset_id"],
                       time_to_best_seconds=trainer.history[trainer.best_epoch].get("cumulative_elapsed_seconds"))
        except BaseException as error:
            exc = error
        finally:
            runtime = Path(row["run_dir"]) / "runtime.json"
            if trainer is not None and runtime.exists():
                attempt = json.loads(runtime.read_text(encoding="utf-8"))
                row["parameter_count"] = attempt["parameter_count"]
                row["quantum_parameter_count"] = attempt["quantum_parameter_count"]
                row["attempts"].append(attempt)
            trainer = None
            self._finish(row, exc)
        if exc is not None and not isinstance(exc, (BudgetExhausted, StudyInterrupted)) and not is_oom(exc):
            atomic_json(self.root / "failure.json", {"job": row["key"], "error": str(exc)})
            raise exc
        if isinstance(exc, StudyInterrupted):
            raise exc
        return row["status"] == "completed"

    def classical(self):
        for level in self.study.levels:
            for seed in self.study.config.training.seeds:
                self.train(level, "compression_cnn", seed)
                self.tick()
        for seed in self.study.config.training.seeds:
            self.train("reference", "direct_cnn", seed)
            self.tick()
        if self.select() is None:
            self.state["last_stop"] = "classical_incomplete"
            self.save()

    def select(self):
        if self.state["selection"] is None:
            selection = choose_level(self.study, list(self.state["jobs"].values()))
            if selection is not None:
                self.state["selection"] = selection
                atomic_json(self.root / "selection.json", selection)
                self.save()
        return self.state["selection"]

    def profile(self, level):
        row = self.state["profiles"].setdefault(level, {"level": level, "status": "pending", "elapsed_seconds": 0.0})
        if row["status"] in {"completed", "budget_exhausted", "oom"}:
            return row["status"] == "completed"
        trainer = None
        monitor = None
        exc = None
        try:
            self._begin(row)
            config = self.study.for_run(level, "compression_qufex", self.study.config.training.seeds[0])
            trainer = Trainer(config, self.root / "profiles" / f"{self.state['id']}_{level}_profile")
            config.save_resolved(trainer.run_dir)
            iterator = iter(trainer.train_loader)
            times = []
            with RuntimeMonitor(trainer.device) as monitor:
                for step in range(self.study.sweep.warmup_steps + self.study.sweep.measured_steps):
                    self.tick()
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(trainer.train_loader)
                        batch = next(iterator)
                    if trainer.device.type == "cuda":
                        trainer.torch.cuda.synchronize(trainer.device)
                    started = time.perf_counter()
                    trainer._train_step(batch)
                    if trainer.device.type == "cuda":
                        trainer.torch.cuda.synchronize(trainer.device)
                    if step >= self.study.sweep.warmup_steps:
                        times.append(time.perf_counter() - started)
                    self.tick()
            row.update(status="completed", step_seconds_mean=float(np.mean(times)), step_seconds_std=float(np.std(times)),
                       warmup_steps=self.study.sweep.warmup_steps, measured_steps=len(times),
                       estimated_training_epoch_seconds=float(np.mean(times) * len(trainer.train_loader)),
                       estimated_max_training_seconds=float(np.mean(times) * len(trainer.train_loader) * config.training.epochs),
                       estimate_scope="training steps only; excludes validation/checkpointing/early stopping",
                       runtime=monitor.results(0))
        except BaseException as error:
            exc = error
        finally:
            if monitor is not None and hasattr(monitor, "elapsed"):
                row["runtime"] = monitor.results(0)
            trainer = None
            self._finish(row, exc)
        if exc is not None and not isinstance(exc, (BudgetExhausted, StudyInterrupted)) and not is_oom(exc):
            atomic_json(self.root / "failure.json", {"profile": level, "error": str(exc)})
            raise exc
        if isinstance(exc, StudyInterrupted):
            raise exc
        return row["status"] == "completed"

    def quantum(self, *, profiles_only=False):
        selection = self.select()
        if selection is None:
            raise ValueError("Quantum stages require every classical level/seed to complete")
        for level in quantum_order(self.study, selection["selected_level"]):
            if self.state["expansion_stopped"] and self.state["expansion_stopped"]["level"] == level:
                return
            if not self.profile(level):
                self.state["expansion_stopped"] = {"level": level, "reason": self.state["profiles"][level]["status"]}
                self.save()
                break
            if profiles_only:
                continue
            for seed in self.study.config.training.seeds:
                if not self.train(level, "compression_qufex", seed):
                    self.state["expansion_stopped"] = {"level": level, "seed": seed, "reason": self.row(level, "compression_qufex", seed)["status"]}
                    self.save()
                    return
                self.train(level, "compression_patch_cnn", seed)
                self.tick()

    def evaluate(self):
        if self.state["evaluation_manifest"] is None:
            manifest = [{"key": row["key"], "checkpoint": row["checkpoint"],
                         "sha256": file_hash(Path(row["checkpoint"]))}
                        for row in self.state["jobs"].values() if row["status"] == "completed"]
            if not manifest:
                raise ValueError("No completed checkpoints to evaluate")
            self.state["evaluation_manifest"] = manifest
            atomic_json(self.root / "evaluation_manifest.json", manifest)
            self.save()
        for item in self.state["evaluation_manifest"]:
            row = self.state["evaluations"].setdefault(item["key"], {"status": "pending", "elapsed_seconds": 0.0})
            if row["status"] == "completed":
                continue
            exc = None
            try:
                self._begin(row)
                checkpoint = Path(item["checkpoint"])
                if file_hash(checkpoint) != item["sha256"]:
                    raise RuntimeError("Frozen evaluation checkpoint has changed")
                output = self.root / "test" / item["key"]
                marker = output / "metrics.json"
                if marker.exists():
                    metrics = json.loads(marker.read_text(encoding="utf-8"))
                else:
                    config = load_checkpoint_config(checkpoint, device=self.study.config.training.device)
                    metrics = Evaluator(config, checkpoint, output, split="test", control=self.tick).run()
                row.update(status="completed", metrics=metrics)
            except BaseException as error:
                exc = error
            finally:
                self._finish(row, exc)
            if exc is not None:
                raise exc

    def run(self, stage):
        if stage == "analyze":
            analyze(self.study, self.state, self.root)
            return
        if stage not in {"classical", "profile", "quantum", "all", "evaluate"}:
            raise ValueError(f"Unknown stage {stage}")
        if self.state["evaluation_manifest"] is not None and stage != "evaluate":
            raise ValueError("Test checkpoint list is frozen; start a new study for further exploration")
        self.state.pop("last_stop", None)
        self.prepare()
        self.last_tick = time.perf_counter()
        try:
            self.tick()
            if stage in {"all", "classical"}:
                self.classical()
            if stage in {"all", "profile", "quantum"}:
                if stage != "all" or self.select() is not None:
                    self.quantum(profiles_only=stage == "profile")
            if stage == "evaluate":
                self.evaluate()
        except BudgetExhausted as exc:
            self.state["last_stop"] = str(exc)
        except StudyInterrupted as exc:
            self.state["last_stop"] = f"scheduler_interruption: {exc}"
        finally:
            self.tick(check=False)
            self.last_tick = None
            analyze(self.study, self.state, self.root)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def analyze(study, state, root):
    """Regenerate portable tables and figures, including incomplete-run statuses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in state["jobs"].values():
        row = {k: v for k, v in job.items() if k != "attempts"}
        attempts = job["attempts"]
        for metric in ("peak_process_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            row[metric] = max((a.get(metric, 0) for a in attempts), default=0)
        steps = sum(a.get("training_step_count", 0) for a in attempts)
        row["training_step_seconds_mean"] = sum(a.get("training_step_seconds_total", 0) for a in attempts) / steps if steps else None
        row["hardware"] = {k: attempts[-1].get(k) for k in ("device", "gpu", "cpu", "platform")} if attempts else None
        rows.append(row)
    atomic_json(output / "runs.json", rows)
    pd.DataFrame(rows).to_csv(output / "runs.csv", index=False)
    summaries, differences = [], []
    complete = [r for r in rows if r["status"] == "completed"]
    for level, variant in sorted({(r["level"], r["variant"]) for r in complete}):
        group = [r for r in complete if (r["level"], r["variant"]) == (level, variant)]
        item = {"level": level, "variant": variant, "seeds_completed": len(group),
                "all_seeds_completed": len(group) == len(study.config.training.seeds),
                "feature_values": group[0]["feature_values"], "compression_axis": group[0]["compression_axis"]}
        for metric in ("macro_f1", "elapsed_seconds", "peak_process_rss_bytes", "peak_gpu_allocated_bytes"):
            values = [r[metric] for r in group]
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summaries.append(item)
    for name in study.levels:
        for control in ("compression_cnn", "compression_patch_cnn"):
            paired = []
            for seed in study.config.training.seeds:
                q = next((r for r in complete if (r["level"], r["variant"], r["seed"]) == (name, "compression_qufex", seed)), None)
                c = next((r for r in complete if (r["level"], r["variant"], r["seed"]) == (name, control, seed)), None)
                if q and c:
                    paired.append({"seed": seed, "delta": q["macro_f1"] - c["macro_f1"]})
            if paired:
                values = [p["delta"] for p in paired]
                differences.append({"level": name, "control": control, "feature_values": study.levels[name].values,
                                    "pairs": paired, "pairs_completed": len(paired),
                                    "delta_mean": float(np.mean(values)),
                                    "delta_std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    for name, values in (("summary", summaries), ("paired_differences", differences), ("profiles", list(state["profiles"].values()))):
        atomic_json(output / f"{name}.json", values)
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)
    for filename, xkey, ykey, xlabel, ylabel in (
        ("f1_vs_size", "feature_values", "macro_f1_mean", "Feature values (C × H × W)", "Validation macro-F1"),
        ("f1_vs_time", "elapsed_seconds_mean", "macro_f1_mean", "Mean training run time (s)", "Validation macro-F1"),
        ("memory_vs_size", "feature_values", "peak_process_rss_bytes_mean", "Feature values (C × H × W)", "Peak process RSS (bytes)"),
        ("gpu_memory_vs_size", "feature_values", "peak_gpu_allocated_bytes_mean", "Feature values (C × H × W)", "Peak GPU allocation (bytes)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in sorted({s["variant"] for s in summaries}):
            group = sorted([s for s in summaries if s["variant"] == variant and s[xkey] is not None and s["all_seeds_completed"]], key=lambda s: s[xkey])
            if group:
                ax.errorbar([s[xkey] for s in group], [s[ykey] for s in group],
                            yerr=[s.get(ykey.replace("_mean", "_std")) or 0 for s in group], marker="o", label=variant)
        if filename == "f1_vs_size" and state["selection"]:
            ax.axhline(float(np.mean(state["selection"]["reference_scores"])) - study.sweep.max_macro_f1_drop,
                       linestyle="--", color="gray", label="classical acceptance threshold")
            reference = next((s for s in summaries if s["variant"] == "direct_cnn" and s["all_seeds_completed"]), None)
            if reference:
                ax.axhline(reference["macro_f1_mean"], linestyle=":", color="black", label="original direct CNN")
        if xkey == "feature_values":
            for item in summaries:
                if item["compression_axis"] == "channel" and item["all_seeds_completed"]:
                    ax.annotate(item["level"] + " (channels)", (item[xkey], item[ykey]),
                                xytext=(3, 5), textcoords="offset points", fontsize=7)
        ax.set(xlabel=xlabel, ylabel=ylabel, title="Completed seed sets; incomplete runs remain in tables")
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{filename}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for control in ("compression_cnn", "compression_patch_cnn"):
        group = sorted([d for d in differences if d["control"] == control and d["pairs_completed"] == len(study.config.training.seeds)], key=lambda d: d["feature_values"])
        if group:
            ax.errorbar([d["feature_values"] for d in group], [d["delta_mean"] for d in group],
                        yerr=[d["delta_std"] or 0 for d in group], marker="o", label=f"quantum − {control}")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set(xlabel="Feature values (C × H × W)", ylabel="Paired validation macro-F1 difference")
    if differences:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "paired_differences.png", dpi=160)
    plt.close(fig)
