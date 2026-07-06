import math

import numpy as np
import pytest
from bielik_hallu.analysis.auroc import auroc, auroc_per_layer


def test_perfect_separation_is_one():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_direction_invariant():
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_per_layer_keys():
    metric_by_layer = {0: np.array([0.1, 0.9]), 1: np.array([0.5, 0.5])}
    labels = np.array([0, 1])
    out = auroc_per_layer(metric_by_layer, labels)
    assert set(out.keys()) == {0, 1}
    assert out[0] == pytest.approx(1.0)


def test_auroc_single_class_returns_nan():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 0, 0])
    result = auroc(scores, labels)
    assert math.isnan(result)
