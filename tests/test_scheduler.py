"""Scheduler configuration, epoch timing, and checkpoint continuation regressions."""
import json
from dataclasses import replace

import pytest
import torch

from qmla.config import ConfigError, config_from_dict, load_config, merge_tables
from qmla.engine import Trainer


@pytest.mark.parametrize("patch", [
    {"name": "unsupported"},
    {"monitor": "test_accuracy"},
    {"factor": 0},
    {"factor": 1},
    {"factor": float("nan")},
    {"patience": -1},
    {"patience": True},
    {"threshold": -0.01},
    {"threshold_mode": "unsupported"},
    {"cooldown": -1},
    {"min_lr": -0.01},
    {"min_lr": float("inf")},
    {"name": "reduce_on_plateau", "min_lr": 1.0},
])
def test_invalid_scheduler_settings(tiny_config, patch):
    raw = merge_tables(tiny_config.resolved_dict(), {"scheduler": patch})
    with pytest.raises(ConfigError):
        config_from_dict(raw, root=tiny_config.paths.project_root)


def test_scheduler_profile_and_saved_configuration(tiny_config, tmp_path):
    path = tmp_path / "scheduler.toml"
    path.write_text('''[run]
profile = "decay"
[scheduler]
factor = 0.25
[profiles.decay.scheduler]
name = "reduce_on_plateau"
monitor = "validation_loss"
patience = 2
''', encoding="utf-8")
    config = load_config(path)
    assert config.scheduler.name == "reduce_on_plateau"
    assert config.scheduler.factor == 0.25
    assert config.scheduler.patience == 2
    config.save_resolved(tmp_path / "resolved")
    restored = load_config(tmp_path / "resolved/resolved_config.toml")
    assert restored.scheduler == config.scheduler
    changed = replace(tiny_config, scheduler=config.scheduler)
    assert changed.cache_dir == tiny_config.cache_dir


@pytest.mark.parametrize("monitor", ["macro_f1", "validation_loss"])
def test_reductions_apply_next_epoch_and_respect_minimum(prepared, monkeypatch, monitor):
    config = prepared.for_model("direct_cnn")
    config = replace(config,
        training=replace(config.training, epochs=5, learning_rate=0.001, early_stopping_patience=0),
        scheduler=replace(config.scheduler, name="reduce_on_plateau", monitor=monitor,
                          patience=0, factor=0.5, min_lr=0.00025))
    trainer = Trainer(config, config.paths.runs_dir / f"floor-{monitor}")
    assert trainer.scheduler.mode == ("max" if monitor == "macro_f1" else "min")
    # Fixed validation produces a known plateau, independent of model learning.
    monkeypatch.setattr(trainer, "_validate", lambda: (1.0, {"macro_f1": 0.5}))
    trainer.run()
    assert [row["learning_rate"] for row in trainer.history] == pytest.approx(
        [0.001, 0.001, 0.0005, 0.00025, 0.00025])
    assert [row["next_learning_rate"] for row in trainer.history] == pytest.approx(
        [0.001, 0.0005, 0.00025, 0.00025, 0.00025])
    saved = json.loads((trainer.run_dir / "history.json").read_text())
    assert saved[-1]["next_learning_rate"] == pytest.approx(0.00025)


def test_resume_preserves_pending_plateau_and_optimizer_trajectory(prepared, monkeypatch):
    config = prepared.for_model("direct_cnn")
    config = replace(config, training=replace(config.training, epochs=4, early_stopping_patience=0),
                     scheduler=replace(config.scheduler, name="reduce_on_plateau", patience=1))
    monkeypatch.setattr(Trainer, "_validate", lambda self: (1.0, {"macro_f1": 0.5}))
    full = Trainer(config, config.paths.runs_dir / "schedule-full")
    full.run()
    first_config = replace(config, training=replace(config.training, epochs=2))
    first = Trainer(first_config, config.paths.runs_dir / "schedule-first")
    first.run()
    assert first.scheduler.num_bad_epochs == 1  # Reduction is due after the next bad epoch.
    resumed = Trainer(config, config.paths.runs_dir / "schedule-resumed")
    resumed.resume(first.checkpoint_dir / "latest.pt")
    resumed.run()
    assert resumed.scheduler.state_dict() == full.scheduler.state_dict()
    assert resumed.optimizer.param_groups[0]["lr"] == full.optimizer.param_groups[0]["lr"]
    for key, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key]), key
    for uninterrupted, continued in zip(full.history, resumed.history):
        assert {k: v for k, v in uninterrupted.items() if k != "epoch_seconds"} == {
            k: v for k, v in continued.items() if k != "epoch_seconds"}


def test_old_fixed_rate_checkpoint_remains_resumable(prepared):
    config = prepared.for_model("direct_cnn")
    trainer = Trainer(config, config.paths.runs_dir / "old-fixed-rate")
    trainer.run()
    path = trainer.checkpoint_dir / "latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("scheduler_state")
    checkpoint["config"].pop("scheduler")
    torch.save(checkpoint, path)
    resumed = Trainer(config, config.paths.runs_dir / "old-fixed-rate-resumed")
    resumed.resume(path)
    assert resumed.scheduler is None
    assert resumed.start_epoch == 1
    changed = replace(config, scheduler=replace(config.scheduler, name="reduce_on_plateau"))
    with pytest.raises(RuntimeError, match="original scheduler settings"):
        Trainer(changed, config.paths.runs_dir / "cannot-enable-on-resume").resume(path)


def test_enabled_scheduler_requires_checkpoint_state(prepared):
    config = prepared.for_model("direct_cnn")
    config = replace(config, scheduler=replace(config.scheduler, name="reduce_on_plateau"))
    trainer = Trainer(config, config.paths.runs_dir / "missing-scheduler")
    trainer.run()
    path = trainer.checkpoint_dir / "latest.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.pop("scheduler_state")
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="missing scheduler state"):
        Trainer(config, config.paths.runs_dir / "missing-scheduler-resumed").resume(path)


def test_lr_reduction_does_not_reset_early_stopping(prepared, monkeypatch):
    config = prepared.for_model("direct_cnn")
    config = replace(config, training=replace(config.training, epochs=10, early_stopping_patience=2),
                     scheduler=replace(config.scheduler, name="reduce_on_plateau", patience=0))
    trainer = Trainer(config, config.paths.runs_dir / "scheduler-early-stop")
    monkeypatch.setattr(trainer, "_validate", lambda: (1.0, {"macro_f1": 0.5}))
    trainer.run()
    assert len(trainer.history) == 3
    assert trainer.bad_epochs == 2
    assert trainer.history[-1]["next_learning_rate"] < trainer.history[-1]["learning_rate"]
