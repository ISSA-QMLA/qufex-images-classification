"""The copied-results notebook must execute without HPC paths or a dataset."""
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("has_results", [False, True])
def test_portable_notebook(tmp_path, monkeypatch, has_results):
    import matplotlib.pyplot as plt

    root = tmp_path / "bundle" / "study"
    root.mkdir(parents=True)
    jobs = {}
    if has_results:
        for variant, score in (("compression_cnn", 0.8), ("compression_qufex", 0.85),
                               ("compression_patch_cnn", 0.79)):
            key = f"M0_{variant}_seed42"
            jobs[key] = dict(key=key, level="M0", variant=variant, seed=42,
                             status="completed", feature_values=64, compression_axis="spatial",
                             macro_f1=score, elapsed_seconds=12, time_to_best_seconds=10,
                             attempts=[], run_dir="/nonexistent/hpc/jobs/" + key)
        jobs["partial"] = dict(key="partial", level="M1", variant="compression_qufex", seed=42,
                               status="interrupted", feature_values=32, compression_axis="channel",
                               elapsed_seconds=3, attempts=[])
    state = dict(id="portable", configuration={"experiment": {"training": {"seeds": [42]}},
                 "levels": {}}, jobs=jobs, elapsed_seconds=40, preparation_seconds=1,
                 selection=None, profiles={}, evaluations={}, evaluation_manifest=None)
    (root / "study.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setenv("QMLA_RESULTS_DIR", str(root.parent))
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    source = Path(__file__).parents[1] / "notebooks/05-compression-sweep.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(source), "exec"), namespace)
    assert len(namespace["complete"]) == (3 if has_results else 0)
    assert len(namespace["pairs"]) == (2 if has_results else 0)
