"""Training and evaluation engines for Galaxy Zoo classifiers."""

from __future__ import annotations

import json
import math
import os
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

from qmla.config import AppConfig
from qmla.data import GalaxyDataset
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
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


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
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
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
        self.config = config
        self.mode = mode or config.model.mode
        self.run_dir = run_dir.resolve()
        self.checkpoint_dir = (config.paths.checkpoints_dir / self.run_dir.name).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device_name or config.training.device)
        configure_accelerator(self.device)
        seed_everything(config.data.seed)

        self.model = GalaxyClassifier(config, mode=self.mode).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        train_dataset = GalaxyDataset(config.paths.processed_dir, "train", augment=True)
        validation_dataset = GalaxyDataset(config.paths.processed_dir, "validation", augment=False)
        loader_kwargs = {
            "num_workers": config.data.num_workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": config.data.num_workers > 0,
        }
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.evaluation.batch_size,
            shuffle=False,
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
        self.amp_enabled = config.training.amp and self.device.type == "cuda"
        self.scaler = _make_grad_scaler(self.amp_enabled)
        self.start_epoch = 0
        self.best_macro_f1 = -math.inf
        self.bad_epochs = 0
        self.history: list[dict[str, Any]] = []

        self.config.save_resolved(self.run_dir)
        write_package_versions(self.run_dir)

    def resume(self, checkpoint_path: Path) -> None:
        checkpoint = _torch_load(checkpoint_path, self.device)
        if checkpoint.get("model_mode") != self.mode:
            raise RuntimeError(
                f"Checkpoint mode {checkpoint.get('model_mode')!r} does not match requested mode {self.mode!r}"
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_macro_f1 = float(checkpoint.get("best_macro_f1", -math.inf))
        self.bad_epochs = int(checkpoint.get("bad_epochs", 0))
        self.history = list(checkpoint.get("history", []))
        print(f"Resuming from epoch {self.start_epoch}: {checkpoint_path}")

    def _atomic_checkpoint(self, path: Path, epoch: int) -> None:
        payload = {
            "format_version": 1,
            "epoch": epoch,
            "model_mode": self.mode,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "bad_epochs": self.bad_epochs,
            "history": self.history,
            "config": self.config.resolved_dict(),
            "run_dir": str(self.run_dir),
        }
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
        best_path = self.checkpoint_dir / "best.pt"
        latest_path = self.checkpoint_dir / "latest.pt"
        for epoch in range(self.start_epoch, self.config.training.epochs):
            train_loss = self._train_epoch()
            validation_loss, metrics = self._validate()
            score = float(metrics["macro_f1"])
            improved = score > self.best_macro_f1
            if improved:
                self.best_macro_f1 = score
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{key: value for key, value in metrics.items() if key != "per_class"},
            }
            self.history.append(record)
            print(
                f"epoch={epoch + 1}/{self.config.training.epochs} "
                f"train_loss={train_loss:.5f} val_loss={validation_loss:.5f} macro_f1={score:.5f}"
            )
            if (epoch + 1) % self.config.training.checkpoint_every == 0:
                self._atomic_checkpoint(latest_path, epoch)
            if improved:
                self._atomic_checkpoint(best_path, epoch)
            self._save_history()
            if self.bad_epochs >= self.config.training.early_stopping_patience:
                print(f"Early stopping after {self.bad_epochs} epochs without macro-F1 improvement")
                break
        if not latest_path.exists():
            self._atomic_checkpoint(latest_path, max(self.start_epoch, 0))
        return best_path

    def _save_history(self) -> None:
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)


class Evaluator:
    """Restore one checkpoint and write complete test-set artifacts."""

    def __init__(
        self,
        config: AppConfig,
        checkpoint_path: Path,
        output_dir: Path,
        *,
        device_name: str | None = None,
    ) -> None:
        torch = require_torch()
        from torch.utils.data import DataLoader

        from qmla.model import GalaxyClassifier

        self.torch = torch
        self.config = config
        self.checkpoint_path = checkpoint_path.resolve()
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(device_name or config.training.device)
        configure_accelerator(self.device)
        checkpoint = _torch_load(self.checkpoint_path, self.device)
        self.mode = checkpoint.get("model_mode", config.model.mode)
        self.model = GalaxyClassifier(config, mode=self.mode).to(self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match this TOML configuration. Use the source_config.toml "
                "saved beside the training run."
            ) from exc
        self.model.eval()
        self.dataset = GalaxyDataset(config.paths.processed_dir, "test", augment=False)
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
            manifest = pd.read_csv(self.config.paths.processed_dir / "test_manifest.csv")
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
