"""Training and evaluation engines for Galaxy Zoo classifiers."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from qmla.config import AppConfig, config_from_dict, relocate_paths
from qmla.data import GalaxyDataset, validate_cache
from qmla.runtime import RuntimeMonitor
from qmla.utils import (
    configure_accelerator,
    require_torch,
    resolve_device,
    seed_everything,
    write_package_versions,
)


def _torch_load(path: Path, map_location: Any) -> dict[str, Any]:
    torch = require_torch()
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("format_version") != 2:
        raise RuntimeError("Incompatible legacy checkpoint. New experiments require checkpoint format 2; existing artifacts were preserved.")
    return checkpoint


def load_checkpoint_config(path: Path, *, device: str | None = None, paths: dict | None = None) -> AppConfig:
    checkpoint = _torch_load(path, "cpu")
    raw = checkpoint["config"]
    if device:
        raw["training"]["device"] = device
    if paths:
        raw["paths"] = relocate_paths(raw["paths"], paths)
    return config_from_dict(raw, root=path.resolve().parent)


def _check_checkpoint(checkpoint: dict, config: AppConfig, dataset_id: str, *, resume: bool = False) -> None:
    saved = config_from_dict(checkpoint["config"], root=config.paths.project_root)
    if saved.run.model != config.run.model or saved.architecture != config.architecture:
        raise RuntimeError("Checkpoint architecture/model differs from the selected configuration")
    if config.run.model != "direct_cnn" and saved.quantum != config.quantum:
        raise RuntimeError("Checkpoint quantum/grouping settings differ from the selected configuration")
    if config.run.model == "cnn_replacement" and saved.architectures.replacement != config.architectures.replacement:
        raise RuntimeError("Checkpoint CNN replacement settings differ")
    if checkpoint["dataset_id"] != dataset_id:
        raise RuntimeError("Checkpoint dataset identity differs from the selected cache")
    if resume:
        # Device, loader workers, output paths and maximum epochs may change on relocation.
        ignored = {"device", "epochs", "cpu_threads", "checkpoint_every"}
        current = config.resolved_dict()["training"]
        previous = saved.resolved_dict()["training"]
        if any(current[k] != previous[k] for k in current if k not in ignored):
            raise RuntimeError("Resume requires the original optimizer, batch size, seed and stopping settings")
        if saved.scheduler != config.scheduler:
            raise RuntimeError("Resume requires the original scheduler settings; start a new run to change them")


def seed_worker(_worker_id: int) -> None:
    seed = require_torch().initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=np.arange(len(class_names)), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def _make_grad_scaler(enabled: bool) -> Any:
    torch = require_torch()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    """Single-GPU/CPU trainer with resumable, self-describing checkpoints."""

    def __init__(
        self,
        config: AppConfig,
        run_dir: Path,
        *,
        mode: str | None = None,
        device_name: str | None = None,
    ) -> None:
        torch = require_torch()
        from torch.utils.data import DataLoader

        from qmla.model import GalaxyClassifier

        self.torch = torch
        config = config.for_model(mode or config.run.model)
        if device_name:
            config = replace(config, training=replace(config.training, device=device_name))
        config.validate()
        self.config = config
        self.mode = config.run.model
        if len(config.training.seeds) != 1:
            raise ValueError("Trainer requires one seed; use run_experiments for a seed list")
        self.metadata = validate_cache(config)
        self.run_dir = run_dir.resolve()
        self.checkpoint_dir = (config.paths.checkpoints_dir / self.run_dir.name).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device_name or config.training.device)
        configure_accelerator(self.device, deterministic=config.training.deterministic, cpu_threads=config.training.cpu_threads)
        seed_everything(config.training.seeds[0])

        self.model = GalaxyClassifier(config, mode=self.mode).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        train_dataset = GalaxyDataset(config.cache_dir, "train", augment=True)
        validation_dataset = GalaxyDataset(config.cache_dir, "validation", augment=False)
        self.loader_generator = torch.Generator().manual_seed(config.training.seeds[0])
        self.validation_generator = torch.Generator().manual_seed(config.training.seeds[0] + 1)
        loader_kwargs = {
            "num_workers": config.data.num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": False, # epoch boundaries reconstruct seeded workers on resume
            "worker_init_fn": seed_worker,
        }
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            generator=self.loader_generator,
            **loader_kwargs,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            generator=self.validation_generator,
            **loader_kwargs,
        )

        weights = None
        if config.training.class_weighting:
            counts = np.bincount(np.asarray(train_dataset.labels), minlength=len(config.evaluation.class_names))
            if np.any(counts == 0):
                raise RuntimeError(f"Training data contains an empty class: counts={counts.tolist()}")
            values = len(train_dataset) / (len(counts) * counts.astype(np.float64))
            weights = torch.tensor(values, dtype=torch.float32, device=self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer_class = torch.optim.AdamW if config.training.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = None
        if config.scheduler.name == "reduce_on_plateau":
            scheduler = config.scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if scheduler.monitor == "macro_f1" else "min",
                factor=scheduler.factor,
                patience=scheduler.patience,
                threshold=scheduler.threshold,
                threshold_mode=scheduler.threshold_mode,
                cooldown=scheduler.cooldown,
                min_lr=scheduler.min_lr,
            )
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp_enabled)
        self.start_epoch = 0
        self.best_macro_f1 = -math.inf
        self.bad_epochs = 0
        self.history: list[dict[str, Any]] = []
        self.best_model_state = None
        self.best_epoch = -1
        self.resumed = False

    def resume(self, checkpoint_path: Path) -> None:
        checkpoint = _torch_load(checkpoint_path, "cpu")
        _check_checkpoint(checkpoint, self.config, self.metadata["dataset_id"], resume=True)
        if checkpoint.get("kind") != "latest":
            raise RuntimeError("Resume from latest.pt; best.pt is an evaluation checkpoint")
        scheduler_state = checkpoint.get("scheduler_state")
        if self.scheduler is not None and scheduler_state is None:
            raise RuntimeError("Checkpoint is missing scheduler state; cannot resume the learning-rate schedule")
        if self.scheduler is None and scheduler_state is not None:
            raise RuntimeError("Checkpoint has scheduler state but scheduling is disabled")
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        self.history = list(checkpoint.get("history", []))
        self.best_model_state = checkpoint["best_model_state"]
        self.best_epoch = checkpoint["best_epoch"]
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        self.torch.set_rng_state(checkpoint["rng"]["torch"])
        cuda_states = checkpoint["rng"].get("cuda", [])
        if self.device.type == "cuda":
            for index, state in enumerate(cuda_states[:self.torch.cuda.device_count()]):
                self.torch.cuda.set_rng_state(state, device=index)
        self.loader_generator.set_state(checkpoint["loader_rng"])
        self.validation_generator.set_state(checkpoint["validation_rng"])
        self.resumed = True
        print(f"Resuming from epoch {self.start_epoch}: {checkpoint_path}")

    def _atomic_checkpoint(self, path: Path, epoch: int) -> None:
        payload = {
            "format_version": 2,
            "kind": path.stem,
            "epoch": epoch,
            "model_mode": self.mode,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": self.config.resolved_dict(),
            "run_dir": str(self.run_dir),
            "dataset_id": self.metadata["dataset_id"],
            "best_model_state": self.best_model_state,
            "best_epoch": self.best_epoch,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": self.torch.get_rng_state(),
                    "cuda": self.torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []},
            "loader_rng": self.loader_generator.get_state(),
            "validation_rng": self.validation_generator.get_state(),
        }
        if path.stem == "best":
            payload["model_state"] = self.best_model_state
            payload["epoch"] = self.best_epoch
        temporary = path.with_suffix(path.suffix + ".tmp")
        self.torch.save(payload, temporary)
        os.replace(temporary, path)

    def _batch_to_device(self, batch: tuple[Any, Any]) -> tuple[Any, Any]:
        inputs, targets = batch
        inputs = inputs.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            inputs = inputs.contiguous(memory_format=self.torch.channels_last)
        targets = targets.to(self.device, non_blocking=True)
        return inputs, targets

    def _train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        samples = 0
        for batch in self.train_loader:
            inputs, targets = self._batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self.torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
            if not self.torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; no automatic device/model changes were made")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += float(loss.detach()) * len(targets)
            samples += len(targets)
        return running_loss / max(samples, 1)

    def _validate(self) -> tuple[float, dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        samples = 0
        targets_all: list[np.ndarray] = []
        predictions_all: list[np.ndarray] = []
        with self.torch.inference_mode():
            for batch in self.validation_loader:
                inputs, targets = self._batch_to_device(batch)
                with self.torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)
                predictions = logits.argmax(dim=1)
                running_loss += float(loss) * len(targets)
                samples += len(targets)
                targets_all.append(targets.cpu().numpy())
                predictions_all.append(predictions.cpu().numpy())
        metrics = classification_metrics(
            np.concatenate(targets_all),
            np.concatenate(predictions_all),
            self.config.evaluation.class_names,
        )
        return running_loss / max(samples, 1), metrics

    def run(self) -> Path:
        if ((self.run_dir / "history.json").exists() or (self.checkpoint_dir / "latest.pt").exists()) and not self.resumed:
            raise RuntimeError("Run directory already contains training history; use resume or a new directory")
        self.config.save_resolved(self.run_dir)
        write_package_versions(self.run_dir)
        (self.run_dir / "dataset_metadata.json").write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        initial_epochs = len(self.history)
        with RuntimeMonitor(self.device) as monitor:
            best = self._run_epochs()
        stats = monitor.results((len(self.history) - initial_epochs) * len(self.train_loader.dataset))
        stats.update({"parameter_count": sum(p.numel() for p in self.model.parameters()),
                      "quantum_parameter_count": self.model.quantum_parameter_count,
                      "dataset_id": self.metadata["dataset_id"], "model": self.mode,
                      "seed": self.config.training.seeds[0], "epochs_completed": len(self.history),
                      "throughput_scope": "training samples divided by training plus validation/checkpoint time for this invocation"})
        (self.run_dir / "runtime.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return best

    def _run_epochs(self) -> Path:
        best_path = self.checkpoint_dir / "best.pt"
        latest_path = self.checkpoint_dir / "latest.pt"
        last_epoch = self.start_epoch - 1
        patience = self.config.training.early_stopping_patience
        if self.resumed and self.best_model_state is not None:
            self._atomic_checkpoint(best_path, self.best_epoch)
        for epoch in range(self.start_epoch, self.config.training.epochs):
            if patience and self.bad_epochs >= patience:
                break
            epoch_started = time.perf_counter()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            train_loss = self._train_epoch()
            validation_loss, metrics = self._validate()
            score = float(metrics["macro_f1"])
            if self.scheduler is not None:
                monitored = score if self.config.scheduler.monitor == "macro_f1" else validation_loss
                if not math.isfinite(monitored):
                    raise RuntimeError("Non-finite validation metric; cannot step the learning-rate scheduler")
                # The new rate applies to the NEXT epoch. Save this state with the
                # updated optimizer so a resumed run makes the same future reductions.
                self.scheduler.step(monitored)
            next_learning_rate = float(self.optimizer.param_groups[0]["lr"])
            improved = score > self.best_macro_f1
            if improved:
                self.best_macro_f1 = score
                self.bad_epochs = 0
                self.best_epoch = epoch
                self.best_model_state = {k: value.detach().cpu().clone() for k, value in self.model.state_dict().items()}
            else:
                self.bad_epochs += 1
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "next_learning_rate": next_learning_rate,
                "epoch_seconds": time.perf_counter() - epoch_started,
                **{key: value for key, value in metrics.items() if key != "per_class"},
            }
            self.history.append(record)
            last_epoch = epoch
            print(
                f"epoch={epoch + 1}/{self.config.training.epochs} "
                f"train_loss={train_loss:.5f} val_loss={validation_loss:.5f} macro_f1={score:.5f} "
                f"lr={learning_rate:.6g} next_lr={next_learning_rate:.6g}"
            )
            if (epoch + 1) % self.config.training.checkpoint_every == 0:
                self._atomic_checkpoint(latest_path, epoch)
            if improved:
                self._atomic_checkpoint(best_path, epoch)
            self._save_history()
            if patience and self.bad_epochs >= patience:
                print(f"Early stopping after {self.bad_epochs} epochs without macro-F1 improvement")
                break
        if last_epoch >= 0:
            self._atomic_checkpoint(latest_path, last_epoch)
        if not best_path.exists():
            raise RuntimeError("No best checkpoint is available")
        return best_path

    def _save_history(self) -> None:
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)


class Evaluator:
    """Restore one checkpoint and write artifacts for an unaugmented split."""

    def __init__(
        self,
        config: AppConfig,
        checkpoint_path: Path,
        output_dir: Path,
        *,
        device_name: str | None = None,
        split: str = "test",
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown evaluation split: {split}")
        self.split = split
        torch = require_torch()
        from torch.utils.data import DataLoader

        from qmla.model import GalaxyClassifier

        self.torch = torch
        self.config = config
        config.validate()
        self.metadata = validate_cache(config)
        self.checkpoint_path = checkpoint_path.resolve()
        self.output_dir = output_dir.resolve()
        if (self.output_dir / "metrics.json").exists():
            raise RuntimeError("Evaluation output already exists; choose a new output directory")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device_name or config.training.device)
        configure_accelerator(self.device, deterministic=config.training.deterministic, cpu_threads=config.training.cpu_threads)
        checkpoint = _torch_load(self.checkpoint_path, "cpu")
        _check_checkpoint(checkpoint, config, self.metadata["dataset_id"])
        self.checkpoint_epoch = int(checkpoint["epoch"])
        self.seed = int(checkpoint["config"]["training"]["seeds"][0])
        self.mode = config.run.model
        self.model = GalaxyClassifier(config, mode=self.mode).to(self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match this TOML configuration. Use the resolved_config.toml "
                "saved beside the training run."
            ) from exc
        self.model.eval()
        self.dataset = GalaxyDataset(config.cache_dir, split, augment=False)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=config.data.num_workers > 0,
        )
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        config.save_resolved(self.output_dir)
        write_package_versions(self.output_dir)

    def run(self) -> dict[str, Any]:
        with RuntimeMonitor(self.device) as monitor:
            metrics = self._evaluate()
        metrics["runtime"] = monitor.results(len(self.dataset))
        metrics["parameter_count"] = sum(p.numel() for p in self.model.parameters())
        metrics["quantum_parameter_count"] = self.model.quantum_parameter_count
        metrics["dataset_id"] = self.metadata["dataset_id"]
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def _evaluate(self) -> dict[str, Any]:
        target_batches: list[np.ndarray] = []
        prediction_batches: list[np.ndarray] = []
        probability_batches: list[np.ndarray] = []
        with self.torch.inference_mode():
            for inputs, targets in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                if self.device.type == "cuda":
                    inputs = inputs.contiguous(memory_format=self.torch.channels_last)
                with self.torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    logits = self.model(inputs)
                probabilities = self.torch.softmax(logits.float(), dim=1)
                target_batches.append(targets.numpy())
                prediction_batches.append(probabilities.argmax(dim=1).cpu().numpy())
                probability_batches.append(probabilities.cpu().numpy())

        targets = np.concatenate(target_batches)
        predictions = np.concatenate(prediction_batches)
        probabilities = np.concatenate(probability_batches)
        names = self.config.evaluation.class_names
        metrics = classification_metrics(targets, predictions, names)
        metrics["checkpoint"] = str(self.checkpoint_path)
        metrics["model_mode"] = self.mode
        metrics["split"] = self.split
        metrics["seed"] = self.seed
        metrics["checkpoint_epoch"] = self.checkpoint_epoch
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report = classification_report(
            targets,
            predictions,
            labels=np.arange(len(names)),
            target_names=names,
            zero_division=0,
        )
        (self.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        if self.config.evaluation.save_predictions:
            manifest = pd.read_csv(self.config.cache_dir / f"{self.split}_manifest.csv", dtype={"dr7objid": str})
            manifest["target"] = targets
            manifest["prediction"] = predictions
            for index, name in enumerate(names):
                manifest[f"probability_{name}"] = probabilities[:, index]
            manifest.to_csv(self.output_dir / "predictions.csv", index=False)

        if self.config.evaluation.save_confusion_matrix:
            matrix = confusion_matrix(targets, predictions, labels=np.arange(len(names)))
            figure, axis = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(matrix, display_labels=names).plot(
                ax=axis, cmap="Blues", colorbar=False, xticks_rotation=25
            )
            figure.tight_layout()
            figure.savefig(self.output_dir / "confusion_matrix.png", dpi=180)
            plt.close(figure)
        print(json.dumps(metrics, indent=2))
        return metrics
