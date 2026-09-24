from __future__ import annotations

import numpy as np
import pytest

from qmla.engine import classification_metrics


def test_classification_metrics_contains_required_outputs() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 2, 0])
    metrics = classification_metrics(targets, predictions, ("smooth", "unbarred", "barred"))
    assert set(("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")) <= set(metrics)
    assert set(metrics["per_class"]) == {"smooth", "unbarred", "barred"}


def test_macro_metrics_include_class_without_predictions():
    metrics = classification_metrics(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), ("a", "b", "c"))
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_precision"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert metrics["macro_recall"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert metrics["per_class"]["c"]["precision"] == 0
