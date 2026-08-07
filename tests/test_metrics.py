from __future__ import annotations

import numpy as np

from qmla.engine import classification_metrics


def test_classification_metrics_contains_required_outputs() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(targets, predictions, ("smooth", "unbarred", "barred"))
    assert set(("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")) <= set(metrics)
    assert set(metrics["per_class"]) == {"smooth", "unbarred", "barred"}
