import json
import shutil

import pandas as pd
import pytest

from qmla.engine import Trainer
from scripts.evaluate_experiment import build_parser, discover_runs, evaluate_experiment


def test_all_splits_and_portable_notebook_tables(prepared):
    config = prepared.for_model("direct_cnn")
    root = config.paths.runs_dir / "array"
    directory = root / "array_direct_cnn"
    trainer = Trainer(config, directory)
    trainer.run()
    output = config.paths.results_dir / "split_evaluation"
    args = build_parser().parse_args([
        "--experiment-dir", str(root), "--checkpoint-root", str(config.paths.checkpoints_dir),
        "--output-dir", str(output), "--device", "cpu", "--batch-size", "3", "--num-workers", "0"])
    rows = evaluate_experiment(args)
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    for split in ("train", "validation", "test"):
        path = output / directory.name / split
        predictions = pd.read_csv(path / "predictions.csv", dtype={"dr7objid": str})
        manifest = pd.read_csv(config.cache_dir / f"{split}_manifest.csv", dtype={"dr7objid": str})
        assert predictions.dr7objid.tolist() == manifest.dr7objid.tolist()
        assert predictions.target.tolist() == manifest.label.tolist()
        metrics = json.loads((path / "metrics.json").read_text())
        assert metrics["split"] == split
        assert metrics["checkpoint_epoch"] == 0
        assert metrics["seed"] == config.training.seeds[0]
    with pytest.raises(ValueError, match="already exists"):
        evaluate_experiment(args)

    # Execute the new notebook cell with only tables and pandas; no inference APIs.
    from pathlib import Path
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/03-results-analysis.ipynb").read_text(encoding="utf-8"))
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if ''.join(cell['source']).startswith('split_frames, class_frames'))
    import warnings
    namespace = dict(pd=pd, Path=Path, warnings=warnings, display=lambda value: None,
                     SPLIT_EVALUATIONS={"example": output}, SHOW_TEST_RESULTS=True,
                     runs=pd.DataFrame([dict(experiment="example", model="direct_cnn", seed=42,
                                             dataset_id=rows[0]["dataset_id"], best_epoch=1)]))
    exec(source, namespace)
    assert len(namespace["split_scores"]) == 3
    assert len(namespace["split_per_class"]) == 9
    assert namespace["split_summary"]["std"].isna().all()
    namespace["SPLIT_EVALUATIONS"] = {}
    exec(source, namespace)
    assert len(namespace["missing_splits"]) == 3

    # Individual and nested roots both resolve; duplicates fail before inference.
    assert len(discover_runs(directory, config.paths.checkpoints_dir)) == 1
    duplicate = root / "duplicate"
    shutil.copytree(directory, duplicate)
    with pytest.raises(ValueError, match="Duplicate"):
        discover_runs(root, config.paths.checkpoints_dir)
    with pytest.raises(ValueError, match="Checkpoint missing"):
        discover_runs(directory, output / "missing")
