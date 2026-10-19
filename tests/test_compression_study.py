import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qmla.config import COMPRESSION_MODELS, MODELS, ConfigError
from qmla.compression_study import (
    BudgetExhausted, Level, StudyConfig, StudyRunner, SweepSettings, choose_level,
    load_study, quantum_order,
)
from qmla.data import GalaxyDataset
from qmla.engine import Trainer, load_checkpoint_config
from qmla.model import GalaxyClassifier, PatchCNN, PatchQuFeXLayer, QuFeXLayer, pack_patches, unpack_patches


def study_for(config, *, seeds=(42,)):
    config = replace(config.for_model("direct_cnn"),
                     training=replace(config.training, seeds=seeds, paired_randomness=True, deterministic=True))
    return StudyConfig(config, SweepSettings(warmup_steps=1, measured_steps=2),
                       {"M0": Level(4, 4), "M1": Level(4, 2)})


def test_prepare_stage_creates_and_reuses_cache_without_a_study(raw_data, monkeypatch):
    from scripts import compression_sweep as cli
    study = study_for(raw_data)
    monkeypatch.setattr(cli, "load_study", lambda *_a, **_k: study)
    monkeypatch.setattr(cli, "StudyRunner", lambda *_a, **_k: pytest.fail("Preparation must not start a study"))
    monkeypatch.setattr("sys.argv", ["compression_sweep", "--stage", "prepare"])
    cli.main()
    assert study.config.cache_dir.exists()
    monkeypatch.setattr(cli, "GalaxyZooPreprocessor", lambda *_a: pytest.fail("Existing cache should be reused"))
    cli.main()


def test_standalone_defaults_and_validation(tiny_config):
    study = load_study(Path(__file__).parents[1] / "configs/compression_sweep.toml")
    assert study.config.training.seeds == (42, 1324, 987654)
    assert study.config.data.image_size == 64
    assert study.sweep.max_macro_f1_drop == 0.05
    assert list(study.levels) == ["K1", "K0", "M0", "M1", "M2", "M3", "M4", "M5"]
    assert [l.values // 8 for l in study.levels.values()] == [8192, 2048, 512, 288, 128, 32, 16, 8]
    assert MODELS == ("qufex", "cnn_replacement", "direct_cnn")
    for levels in ({"M0": Level(3, 4)}, {"M0": Level(4, 3)},
                   {"M0": Level(4, 100)}, {"M0": Level(4, 2), "M1": Level(4, 4)}):
        with pytest.raises(ConfigError):
            replace(study_for(tiny_config), levels=levels).validate()
    with pytest.raises(ConfigError, match="eight-qubit"):
        bad = replace(study.config, quantum=replace(study.config.quantum, qubits=4))
        replace(study, config=bad).validate()


def test_selection_threshold_nonmonotonic_and_missing_seeds(tiny_config):
    study = replace(study_for(tiny_config, seeds=(42, 43)),
                    levels={"M0": Level(8, 4), "M1": Level(4, 4), "M2": Level(4, 2)})
    rows = [{"level": level, "seed": seed, "variant": "compression_cnn", "status": "completed", "macro_f1": score}
            for level, score in (("M0", 0.90), ("M1", 0.70), ("M2", 0.85)) for seed in (42, 43)]
    result = choose_level(study, rows)
    assert result["selected_level"] == "M2"  # equality, and recovery after a failed level
    assert choose_level(study, rows[:-1]) is None
    rows[-1]["status"] = "budget_exhausted"
    assert choose_level(study, rows) is None
    assert quantum_order(study, "M1") == ["M1", "M2", "M0"]
    assert quantum_order(study, "M2") == ["M2", "M1", "M0"]


def test_larger_levels_use_fewer_pools_and_preserve_old_architectures():
    study = load_study(Path(__file__).parents[1] / "configs/compression_sweep.toml", device="cpu")
    torch.set_num_threads(2)
    for name, size, depth in (("K1", 32, 1), ("K0", 16, 2), ("M0", 8, 3)):
        config = study.for_run(name, "compression_cnn", 42)
        model = GalaxyClassifier(config).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            encoded = model.encoder(x)
            assert encoded.shape == (2, 64, size, size)
            assert model.compression(encoded).shape == encoded.shape
            assert model(x).shape == (2, 3)
        assert sum(isinstance(m, torch.nn.MaxPool2d) for m in model.encoder.modules()) == depth
        assert sum(isinstance(m, torch.nn.Conv2d) for m in model.encoder.modules()) == 3
        paired = GalaxyClassifier(study.for_run(name, "compression_qufex", 42))
        for module in ("encoder", "compression", "classifier"):
            for key, value in getattr(model, module).state_dict().items():
                assert torch.equal(value, getattr(paired, module).state_dict()[key])
    reference = GalaxyClassifier(study.for_run("reference", "direct_cnn", 42))
    assert reference.encoder(x).shape == (2, 64, 8, 8)
    assert study.for_run("M0", "compression_cnn", 42).architecture.pool_blocks == -1
    with pytest.raises(ConfigError, match="upsample"):
        replace(study, levels={"K0": Level(64, 16)}).validate()
    for invalid in (-2, 4, 1.5, True):
        with pytest.raises(ConfigError, match="pool_blocks"):
            replace(study, levels={"K0": Level(64, 16, invalid)}).validate()


def test_prepend_levels_reuses_runs_and_reselects_against_k1(prepared):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "prepend")
    runner.run("classical")
    previous_selection = runner.state["selection"]
    old_jobs = json.loads(json.dumps(runner.state["jobs"]))
    elapsed = runner.state["elapsed_seconds"]
    extended = replace(study, levels={"K1": Level(4, 16, 1), "K0": Level(4, 8, 2), **study.levels})
    with pytest.raises(ValueError, match="configuration differs"):
        StudyRunner(extended, runner.root, resume=True)
    resumed = StudyRunner(extended, runner.root, resume=True, extend_levels=True)
    assert resumed.state["jobs"] == old_jobs
    assert resumed.state["elapsed_seconds"] == elapsed
    assert resumed.state["selection"] is None
    assert json.loads((runner.root / "selection.json").read_text()) is None
    assert resumed.state["level_extensions"][0]["previous_selection"] == previous_selection
    with pytest.raises(ValueError, match="every classical"):
        resumed.quantum()
    resumed.run("classical")
    assert resumed.state["selection"]["reference_level"] == "K1"
    for key, record in old_jobs.items():
        assert resumed.state["jobs"][key] == record
    assert len(resumed.state["jobs"]) == len(old_jobs) + 2
    assert quantum_order(extended, "M1") == ["M1", "M0", "K0", "K1"]
    config = load_checkpoint_config(Path(resumed.row("K1", "compression_cnn", 42)["checkpoint"]))
    assert config.architecture.pool_blocks == 1
    assert config.architecture.quantum_spatial_size == 16
    assert load_study(runner.root / "resolved_sweep.toml").description() == extended.description()
    again = StudyRunner(extended, runner.root, resume=True, extend_levels=True)
    assert len(again.state["level_extensions"]) == 1
    assert again.state["selection"] == resumed.state["selection"]


def test_level_extension_rejects_changes_and_frozen_studies(prepared):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "invalid-prepend")
    original = json.loads(runner.state_path.read_text())["configuration"]
    extended = replace(study, levels={"K0": Level(4, 8, 2), **study.levels})
    for levels in ({"K0": Level(4, 8, 2), "M0": Level(4, 4)},
                   {"K0": Level(4, 8, 2), "M0": Level(4, 4, 2), "M1": Level(4, 2)}):
        with pytest.raises(ValueError, match="only prepend"):
            StudyRunner(replace(study, levels=levels), runner.root, resume=True, extend_levels=True)
    changed = replace(extended, sweep=replace(study.sweep, max_macro_f1_drop=0.1))
    with pytest.raises(ValueError, match="Only time budgets"):
        StudyRunner(changed, runner.root, resume=True, extend_levels=True)
    changed_budget = replace(extended, sweep=replace(study.sweep, total_hours=72))
    with pytest.raises(ValueError, match="explicit budget extension"):
        StudyRunner(changed_budget, runner.root, resume=True, extend_levels=True)
    assert json.loads(runner.state_path.read_text())["configuration"] == original
    runner.state["evaluation_manifest"] = []
    runner.save()
    with pytest.raises(ValueError, match="frozen"):
        StudyRunner(extended, runner.root, resume=True, extend_levels=True)


def test_legacy_study_fingerprint_ignores_default_pooling_field(prepared):
    import hashlib
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "old-fingerprint")
    description = runner.state["configuration"]
    for arch in description["experiment"]["architectures"].values():
        arch.pop("pool_blocks", None)
    description["experiment"]["data"].pop("dataset", None)
    for level in description["levels"].values():
        level.pop("pool_blocks")
    payload = json.loads(json.dumps(description))
    payload["experiment"]["training"].pop("device")
    runner.state["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    runner.save()
    assert StudyRunner(study, runner.root, resume=True).state["id"] == runner.state["id"]


def test_combined_level_and_budget_extension_keeps_expansion_stop_without_reopening(tiny_config):
    study = study_for(tiny_config)
    runner = StudyRunner(study, tiny_config.paths.runs_dir / "combined-extension")
    runner.state["elapsed_seconds"] = 30
    runner.state["selection"] = {"selected_level": "M1", "reference_scores": [0.8]}
    runner.state["expansion_stopped"] = {"level": "M1", "reason": "oom"}
    runner.save()
    extended = replace(study, levels={"K0": Level(4, 8, 2), **study.levels},
                       sweep=replace(study.sweep, run_hours=24, total_hours=144))
    resumed = StudyRunner(extended, runner.root, resume=True, extend_levels=True, extend_budgets=True)
    assert resumed.state["elapsed_seconds"] == 30
    assert resumed.state["expansion_stopped"] == runner.state["expansion_stopped"]
    assert resumed.state["selection"] is None
    assert len(resumed.state["level_extensions"]) == len(resumed.state["budget_extensions"]) == 1


def test_cli_level_extension_requires_classical_resume(prepared, monkeypatch):
    from scripts import compression_sweep as cli
    study = study_for(prepared)
    monkeypatch.setattr(cli, "load_study", lambda *_a, **_k: study)
    seen = {}

    class FakeRunner:
        def __init__(self, chosen, root, **options):
            seen.update(options)
            self.root, self.stop_requested = root, None

        def run(self, stage):
            seen["stage"] = stage

        def request_stop(self, reason):
            self.stop_requested = reason

    monkeypatch.setattr(cli, "StudyRunner", FakeRunner)
    base = ["compression_sweep", "--run-dir", str(prepared.paths.runs_dir / "prior"), "--extend-levels"]
    for extra in (["--stage", "classical"], ["--stage", "quantum", "--resume"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit, match="2"):
            cli.main()
    monkeypatch.setattr("sys.argv", base + ["--stage", "classical", "--resume"])
    cli.main()
    assert seen["stage"] == "classical"
    assert seen["resume"] and seen["extend_levels"]


def test_patch_order_and_legacy_quantum_equivalence():
    torch.set_num_threads(2)
    x = torch.arange(2 * 6 * 4 * 6.).reshape(2, 6, 4, 6)
    assert torch.equal(unpack_patches(pack_patches(x), x.shape), x)
    legacy = QuFeXLayer()
    layer = PatchQuFeXLayer(chunk_size=3)
    layer.load_state_dict(legacy.state_dict())
    x1 = torch.randn(2, 16, 2, 2, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_()
    old, new = legacy(x1), layer(x2)
    assert torch.allclose(old, new, atol=1e-6)
    weights = torch.randn_like(old)
    (old * weights).sum().backward()
    (new * weights).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(legacy.theta.grad, layer.theta.grad, atol=1e-5)
    assert layer.quantum_parameter_count == 4


def test_chunking_and_patch_locality():
    torch.set_num_threads(2)
    a, b = PatchQuFeXLayer(chunk_size=1), PatchQuFeXLayer(chunk_size=256)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a(x).sum().backward()
    b(y).sum().backward()
    assert torch.allclose(x.grad, y.grad, atol=1e-5)
    assert torch.allclose(a.theta.grad, b.theta.grad, atol=1e-5)
    for layer in (a, PatchCNN()):
        first = layer(x.detach())
        changed = x.detach().clone()
        changed[:, :2, :2, :2] += 1
        delta = layer(changed) - first
        delta[:, :2, :2, :2] = 0
        assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)


@pytest.mark.parametrize("mode", COMPRESSION_MODELS)
def test_compression_model_gradients_and_shared_initialization(tiny_config, mode):
    torch.set_num_threads(2)
    study = study_for(tiny_config)
    baseline = GalaxyClassifier(study.for_run("M0", "compression_cnn", 42))
    model = GalaxyClassifier(study.for_run("M0", mode, 42))
    for name in ("encoder", "compression", "classifier"):
        for key, tensor in getattr(baseline, name).state_dict().items():
            assert torch.equal(tensor, getattr(model, name).state_dict()[key])
    model(torch.randn(2, 3, 32, 32)).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert next(model.encoder.parameters()).grad.abs().sum() > 0


def test_paired_augmentation_independent_of_global_rng(prepared):
    a = GalaxyDataset(prepared.cache_dir, "train", augment=True, augmentation_seed=42)
    b = GalaxyDataset(prepared.cache_dir, "train", augment=True, augmentation_seed=42)
    a.epoch = b.epoch = 3
    image = a[0][0]
    torch.rand(100)
    assert torch.equal(image, b[0][0])
    assert any(not torch.equal(a[i][0], GalaxyDataset(prepared.cache_dir, "train", augment=True,
                                                      augmentation_seed=43)[i][0]) for i in range(5))


def test_partial_epoch_resume_matches_uninterrupted(prepared):
    study = study_for(replace(prepared, training=replace(prepared.training, epochs=2)))
    config = study.for_run("M1", "compression_cnn", 42)
    full = Trainer(config, config.paths.runs_dir / "full_compression", control=lambda *_: None)
    full.run()
    batches = []

    def interrupt(event, trainer):
        if event == "batch" and len(trainer.history) == 1:
            batches.append(event)
            if len(batches) == 3:
                raise BudgetExhausted("interrupted after one complete epoch and one extra batch")

    first = Trainer(config, config.paths.runs_dir / "partial_compression", control=interrupt)
    with pytest.raises(BudgetExhausted):
        first.run()
    resumed = Trainer(config, config.paths.runs_dir / "partial_compression", control=lambda *_: None)
    resumed.resume(first.checkpoint_dir / "latest.pt")
    assert resumed.start_epoch == 1
    resumed.run()
    for key, tensor in full.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[key]), key
    restored = load_checkpoint_config(resumed.checkpoint_dir / "best.pt")
    assert restored.architecture == config.architecture


def test_budget_accounting_resume_and_partial_analysis(prepared):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "budget")
    row = runner.row("M0", "compression_cnn", 42)
    runner.active = row
    runner.last_tick = __import__("time").perf_counter() - 2
    runner.tick(check=False)
    assert runner.state["elapsed_seconds"] >= 2 and row["elapsed_seconds"] >= 2
    row["elapsed_seconds"] = study.sweep.run_hours * 3600
    with pytest.raises(BudgetExhausted, match="run_time_limit"):
        runner.tick()
    runner.save()
    restored = StudyRunner(study, runner.root, resume=True)
    assert restored.state["jobs"][row["key"]]["elapsed_seconds"] >= 14400
    restored.run("analyze")
    assert (runner.root / "analysis/runs.json").exists()
    restored.state["elapsed_seconds"] = study.sweep.total_hours * 3600
    with pytest.raises(BudgetExhausted, match="study_time_limit"):
        restored.tick()


def test_end_to_end_sweep_and_explicit_frozen_evaluation(prepared, monkeypatch):
    from qmla import compression_study as module
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "study")
    assert load_study(runner.root / "resolved_sweep.toml").description() == study.description()
    evaluated = []
    evaluator = module.Evaluator

    def tracked(*args, **kwargs):
        evaluated.append(kwargs.get("split", "test"))
        return evaluator(*args, **kwargs)

    monkeypatch.setattr(module, "Evaluator", tracked)
    runner.run("all")
    assert evaluated == []
    assert runner.state["selection"] is not None
    assert all(r["status"] == "completed" for r in runner.state["jobs"].values())
    assert runner.state["profiles"]
    count = len(runner.state["jobs"])
    times = {k: r["elapsed_seconds"] for k, r in runner.state["jobs"].items()}
    resumed = StudyRunner(study, runner.root, resume=True)
    resumed.run("all")
    assert len(resumed.state["jobs"]) == count
    assert times == {k: r["elapsed_seconds"] for k, r in resumed.state["jobs"].items()}
    resumed.run("evaluate")
    assert evaluated == ["test"] * count
    manifest = json.loads((runner.root / "evaluation_manifest.json").read_text())
    assert len(manifest) == count
    resumed.run("evaluate")
    assert len(evaluated) == count
    with pytest.raises(ValueError, match="frozen"):
        resumed.run("all")


def test_quantum_oom_stops_expansion(prepared, monkeypatch):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "oom")
    runner.state["selection"] = {"selected_level": "M1"}
    visited = []

    def oom_profile(level):
        visited.append(level)
        runner.state["profiles"][level] = {"status": "oom", "elapsed_seconds": 0.0}
        return False

    monkeypatch.setattr(runner, "profile", oom_profile)
    runner.quantum()
    assert visited == ["M1"]
    assert runner.state["expansion_stopped"]["reason"] == "oom"
    runner.quantum()
    assert visited == ["M1"]


def test_profiling_stop_still_allows_training_feasible_prefix(prepared, monkeypatch):
    runner = StudyRunner(study_for(prepared), prepared.paths.runs_dir / "profile-prefix")
    runner.state["selection"] = {"selected_level": "M1"}
    runner.state["profiles"] = {"M1": {"status": "completed"}, "M0": {"status": "oom"}}
    runner.state["expansion_stopped"] = {"level": "M0", "reason": "oom"}
    trained = []
    monkeypatch.setattr(runner, "train", lambda level, variant, seed: trained.append((level, variant)) or True)
    runner.quantum()
    assert trained == [("M1", "compression_qufex"), ("M1", "compression_patch_cnn")]


def test_capped_job_is_never_restarted_and_has_no_selection(prepared, monkeypatch):
    from qmla import compression_study as module
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "capped")
    row = runner.row("M0", "compression_cnn", 42)
    row["elapsed_seconds"] = study.sweep.run_hours * 3600
    monkeypatch.setattr(module, "Trainer", lambda *_a, **_k: pytest.fail("Should not allocate a capped model"))
    assert not runner.train("M0", "compression_cnn", 42)
    assert row["status"] == "budget_exhausted"
    assert runner.select() is None
    restored = StudyRunner(study, runner.root, resume=True)
    assert not restored.train("M0", "compression_cnn", 42)


def test_extend_budgets_resumes_capped_quantum_without_retraining_completed_jobs(prepared):
    study = study_for(prepared)
    study = replace(study, config=replace(study.config,
        training=replace(study.config.training, epochs=2)))
    runner = StudyRunner(study, prepared.paths.runs_dir / "extend-quantum")
    classical = runner.row("M1", "compression_cnn", 42)
    classical.update(status="completed", macro_f1=0.8)
    runner.state["selection"] = {"selected_level": "M1"}
    original_control = runner.control

    def cap_after_first_epoch(event, trainer):
        if event == "epoch_complete" and len(trainer.history) == 1:
            runner.active["elapsed_seconds"] = study.sweep.run_hours * 3600
            runner.state["elapsed_seconds"] = study.sweep.run_hours * 3600
        original_control(event, trainer)

    runner.control = cap_after_first_epoch
    assert not runner.train("M1", "compression_qufex", 42)
    capped = runner.row("M1", "compression_qufex", 42)
    assert capped["status"] == "budget_exhausted"
    history = Path(capped["run_dir"]) / "history.json"
    assert len(json.loads(history.read_text())) == 1
    runner.state["expansion_stopped"] = {"level": "M1", "seed": 42, "reason": "budget_exhausted"}
    runner.save()

    extended = replace(study, sweep=replace(study.sweep, run_hours=6, total_hours=72))
    with pytest.raises(ValueError, match="configuration differs"):
        StudyRunner(extended, runner.root, resume=True)
    changed_metric = replace(extended, sweep=replace(extended.sweep, max_macro_f1_drop=0.1))
    with pytest.raises(ValueError, match="Only time budgets"):
        StudyRunner(changed_metric, runner.root, resume=True, extend_budgets=True)
    lowered = replace(study, sweep=replace(study.sweep, run_hours=3, total_hours=72))
    with pytest.raises(ValueError, match="only increase"):
        StudyRunner(lowered, runner.root, resume=True, extend_budgets=True)

    resumed = StudyRunner(extended, runner.root, resume=True, extend_budgets=True)
    assert resumed.state["expansion_stopped"] is None
    assert resumed.state["jobs"][classical["key"]]["status"] == "completed"
    assert resumed.state["jobs"][capped["key"]]["status"] == "interrupted"
    assert resumed.state["selection"] == {"selected_level": "M1"}
    assert len(resumed.state["budget_extensions"]) == 1
    assert load_study(runner.root / "resolved_sweep.toml").sweep == extended.sweep
    assert resumed.train("M1", "compression_qufex", 42)
    completed = resumed.state["jobs"][capped["key"]]
    assert completed["elapsed_seconds"] >= 4 * 3600
    assert completed["epochs_completed"] == 2
    assert len(json.loads(history.read_text())) == 2
    assert len(completed["attempts"]) == 2
    assert resumed.train("M1", "compression_cnn", 42)
    assert resumed.state["jobs"][classical["key"]]["elapsed_seconds"] == 0
    again = StudyRunner(extended, runner.root, resume=True, extend_budgets=True)
    assert len(again.state["budget_extensions"]) == 1


def test_cli_budget_extension_requires_resume_and_runs_quantum_only(prepared, monkeypatch):
    from scripts import compression_sweep as cli
    study = study_for(prepared)
    monkeypatch.setattr(cli, "load_study", lambda *_a, **_k: study)
    seen = {}

    class FakeRunner:
        def __init__(self, chosen, root, **options):
            seen.update(run_hours=chosen.sweep.run_hours, total_hours=chosen.sweep.total_hours,
                        root=root, options=options)
            self.root = root
            self.stop_requested = None

        def request_stop(self, reason):
            self.stop_requested = reason

        def run(self, stage):
            seen["stage"] = stage

    monkeypatch.setattr(cli, "StudyRunner", FakeRunner)
    arguments = ["compression_sweep", "--stage", "quantum", "--run-dir", str(prepared.paths.runs_dir / "prior"),
                 "--extend-run-hours", "24", "--extend-total-hours", "144"]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    monkeypatch.setattr("sys.argv", arguments + ["--resume"])
    cli.main()
    assert seen["stage"] == "quantum"
    assert (seen["run_hours"], seen["total_hours"]) == (24, 144)
    assert seen["options"]["extend_budgets"] is True
    assert seen["options"]["resume"] is True


@pytest.mark.parametrize("stage", ["classical", "quantum"])
def test_scheduler_stop_is_resumable_and_does_not_stop_expansion(prepared, monkeypatch, stage):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / f"scheduler-{stage}")
    if stage == "quantum":
        runner.state["selection"] = {"selected_level": "M1", "reference_scores": [0.8]}
    original = Trainer._train_step

    def interrupted_step(trainer, batch):
        result = original(trainer, batch)
        runner.request_stop("SIGUSR1")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Trainer, "_train_step", interrupted_step)
        runner.run(stage)
    assert runner.state["last_stop"] == "scheduler_interruption: SIGUSR1"
    assert runner.state["expansion_stopped"] is None
    assert not (runner.root / "failure.json").exists()
    group = "profiles" if stage == "quantum" else "jobs"
    row = next(iter(runner.state[group].values()))
    assert row["status"] == "interrupted" and row["elapsed_seconds"] > 0
    elapsed = runner.state["elapsed_seconds"]
    restored = StudyRunner(study, runner.root, resume=True)
    assert restored.stop_requested is None
    restored.run(stage)
    assert restored.state["elapsed_seconds"] > elapsed
    assert all(r["status"] == "completed" for r in restored.state[group].values())


def test_classical_failure_blocks_quantum_and_config_changes_block_resume(prepared):
    study = study_for(prepared)
    runner = StudyRunner(study, prepared.paths.runs_dir / "missing")
    with pytest.raises(ValueError, match="every classical"):
        runner.quantum()
    changed = replace(study, sweep=replace(study.sweep, max_macro_f1_drop=0.1))
    with pytest.raises(ValueError, match="configuration differs"):
        StudyRunner(changed, runner.root, resume=True)


def test_old_format2_compression_defaults_remain_loadable(prepared):
    trainer = Trainer(prepared.for_model("direct_cnn"), prepared.paths.runs_dir / "legacy-fields")
    best = trainer.run()
    checkpoint = torch.load(best, weights_only=False)
    checkpoint["config"]["training"].pop("paired_randomness")
    for name in ("shared", "direct"):
        checkpoint["config"]["architectures"][name].pop("circuit_chunk_size")
        checkpoint["config"]["architectures"][name].pop("pool_blocks")
    torch.save(checkpoint, best)
    restored = load_checkpoint_config(best)
    model = GalaxyClassifier(restored)
    model.load_state_dict(checkpoint["model_state"])
    assert restored.architecture.circuit_chunk_size == 256
    assert not restored.training.paired_randomness


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", COMPRESSION_MODELS)
def test_compression_cuda(prepared, mode):
    study = study_for(prepared)
    config = study.for_run("M1", mode, 42)
    config = replace(config, training=replace(config.training, device="cuda", deterministic=False))
    trainer = Trainer(config, config.paths.runs_dir / f"cuda_{mode}", control=lambda *_: None)
    best = trainer.run()
    assert best.exists()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.model.parameters())
