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
    assert "bottleneck_configurations" in namespace
    assert set(namespace["extractor_configurations"].model) == {
        "compression_cnn", "compression_patch_cnn", "compression_qufex", "direct_cnn"}


@pytest.mark.parametrize("include_larger", [False, True])
def test_saved_bottleneck_configurations_include_planned_variants(include_larger):
    import pandas as pd

    levels = {
        "M0": dict(channels=64, spatial_size=8),
        "M4": dict(channels=32, spatial_size=2),
        "M5": dict(channels=16, spatial_size=2),
    }
    if include_larger:
        levels = {"K1": dict(channels=64, spatial_size=32, pool_blocks=1),
                  "K0": dict(channels=64, spatial_size=16, pool_blocks=2), **levels}
    experiment = {"architectures": {"direct": {"encoder_channels": [16, 32, 64],
                   "pooling": "max", "pool_size": 2}}, "data": {"image_size": 64},
                  "quantum": {"qubits": 8, "filters": 1}}
    state = {"configuration": {"experiment": experiment, "levels": levels,
             "sweep": {"circuit_chunk_size": 64}}, "jobs": {
        "done": dict(level="M0", variant="compression_cnn", seed=42, status="completed")}}
    source = Path(__file__).parents[1] / "notebooks/05-compression-sweep.ipynb"
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == "bottleneck-configurations")
    displayed = []
    namespace = dict(state=state, levels=levels, seeds=[42, 1324, 987654], pd=pd,
                     display=displayed.append)
    exec(compile("".join(cell["source"]), str(source), "exec"), namespace)
    table = namespace["bottleneck_configurations"].set_index(["level", "model"])
    assert len(table) == 3 * len(levels) + 1
    assert len(displayed) == 2
    assert table.loc[("M0", "compression_cnn"), "completed_seeds"] == "1/3"
    assert table.loc[("M0", "compression_qufex"), "completed_seeds"] == "0/3"
    assert table.loc[("M0", "compression_cnn"), "extractor_parameters"] == 41088
    assert table.loc[("M0", "compression_patch_cnn"), "extractor_parameters"] == 170
    assert table.loc[("M0", "compression_qufex"), "circuit_instances_per_image"] == 512
    assert table.loc[("M0", "compression_qufex"), "circuit_chunk_size"] == 64
    assert table.loc[("M4", "compression_cnn"), "channel_projection"] == "1×1 conv 64→32"
    assert table.loc[("reference", "direct_cnn"), "bottleneck_shape"] == "None (encoder → GAP)"
    assert table.loc[("reference", "direct_cnn"), "encoder_shape"] == "64×8×8"
    if include_larger:
        for level, spatial_size, depth in (("K1", 32, 1), ("K0", 16, 2)):
            row = table.loc[(level, "compression_qufex")]
            assert row.bottleneck_shape == row.encoder_shape == f"64×{spatial_size}×{spatial_size}"
            assert row.encoder_pool_blocks == depth
            assert row.circuit_instances_per_image == 64 * spatial_size**2 // 8
            assert row.qubits == 8 and row.filters == 1 and row.extractor_parameters == 4
