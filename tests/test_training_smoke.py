import json
from dataclasses import replace

import pytest
import torch

from qmla.engine import Evaluator, Trainer, load_checkpoint_config
from qmla.experiments import run_experiments


@pytest.mark.parametrize("mode", ["qufex", "cnn_replacement", "direct_cnn"])
def test_training_resume_evaluation(prepared, mode):
    config = prepared.for_model(mode)
    directory = config.paths.runs_dir / mode
    trainer = Trainer(config, directory)
    best = trainer.run()
    latest = trainer.checkpoint_dir / "latest.pt"
    evaluation_config = load_checkpoint_config(best, device="cpu")
    metrics = Evaluator(evaluation_config, best, directory / "evaluation").run()
    assert metrics["dataset_id"] == trainer.metadata["dataset_id"]
    assert metrics["runtime"]["peak_process_rss_bytes"] > 0
    assert (directory / "evaluation/predictions.csv").exists()
    resumed = Trainer(config, directory)
    resumed.resume(latest)
    assert resumed.start_epoch == 1
    assert resumed.run().exists()


@pytest.mark.parametrize("workers", [0, 1])
def test_resume_matches_uninterrupted_cpu(prepared, workers):
    config = prepared.for_model("direct_cnn")
    config = replace(config, data=replace(config.data, num_workers=workers), training=replace(config.training, epochs=2))
    full = Trainer(config, config.paths.runs_dir / "full")
    full.run()
    first_config = replace(config, training=replace(config.training, epochs=1))
    first = Trainer(first_config, config.paths.runs_dir / "first")
    first.run()
    resumed = Trainer(config, config.paths.runs_dir / "resumed")
    resumed.resume(first.checkpoint_dir / "latest.pt")
    resumed.run()
    for key, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key]), key
    for a, b in zip(full.history, resumed.history):
        assert {k: v for k, v in a.items() if k != "epoch_seconds"} == {k: v for k, v in b.items() if k != "epoch_seconds"}


def test_smoke_selection_and_benchmark_paired_data(prepared):
    config = prepared.for_model("cnn_replacement")
    single = run_experiments(config, run_dir=config.paths.runs_dir / "single")
    assert [row["model"] for row in single] == ["cnn_replacement"]
    comparison = run_experiments(config, benchmark=True, run_dir=config.paths.runs_dir / "comparison")
    assert len(comparison) == 3 and len({row["dataset_id"] for row in comparison}) == 1
    assert len({row["seed"] for row in comparison}) == 1
    summary = json.loads((config.paths.runs_dir / "comparison/summary.json").read_text())
    assert all(row["preliminary_single_seed"] and row["macro_f1_std"] is None for row in summary)
    assert all(row["evaluation_dir"].startswith(str(config.paths.results_dir)) for row in comparison)
    with pytest.raises(RuntimeError, match="already exists"):
        run_experiments(config, run_dir=config.paths.runs_dir / "single")


def test_seed_list_and_nondefault_checkpoint_interval(prepared):
    config = prepared.for_model("direct_cnn")
    config = replace(config, training=replace(config.training, seeds=(42, 43), checkpoint_every=3))
    rows = run_experiments(config, run_dir=config.paths.runs_dir / "seeds")
    assert [r["seed"] for r in rows] == [42, 43]
    assert len({r["dataset_id"] for r in rows}) == 1
    for row in rows:
        from pathlib import Path
        assert (Path(row["checkpoint"]).parent / "latest.pt").exists()


def test_legacy_and_architecture_mismatch_rejected(prepared):
    legacy = prepared.paths.project_root / "old.pt"
    torch.save({"format_version": 1}, legacy)
    with pytest.raises(RuntimeError, match="legacy"):
        load_checkpoint_config(legacy)
    config = prepared.for_model("direct_cnn")
    trainer = Trainer(config, config.paths.runs_dir / "original")
    best = trainer.run()
    changed = replace(config, architectures=replace(config.architectures,
        direct=replace(config.architectures.direct, activation="tanh")))
    with pytest.raises(RuntimeError, match="architecture"):
        Evaluator(changed, best, config.paths.results_dir)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_training_and_cpu_checkpoint_evaluation(prepared):
    config = replace(prepared, training=replace(prepared.training, device="cuda", deterministic=False))
    trainer = Trainer(config, config.paths.runs_dir / "cuda")
    best = trainer.run()
    cpu = load_checkpoint_config(best, device="cpu")
    metrics = Evaluator(cpu, best, config.paths.results_dir).run()
    assert metrics["runtime"]["device"] == "cpu"
