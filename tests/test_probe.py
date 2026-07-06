import math

import numpy as np
from bielik_hallu.analysis.probe import probe_auroc_per_layer


def test_separable_layer_high_auroc():
    rng = np.random.default_rng(0)
    n = 40
    labels = np.array([0] * n + [1] * n)
    # layer 0: separable (shifted mean), layer 1: pure noise
    sep = np.concatenate([rng.normal(-2, 1, (n, 5)), rng.normal(2, 1, (n, 5))])
    noise = rng.normal(0, 1, (2 * n, 5))
    out = probe_auroc_per_layer({0: sep, 1: noise}, labels, n_splits=5)
    assert out[0] > 0.9
    assert out[1] < 0.75


def test_probe_degenerate_labels_returns_nan():
    rng = np.random.default_rng(0)
    n = 40
    # only 1 positive sample among many negatives -> can't cross-validate
    labels = np.array([0] * (n - 1) + [1] * 1)
    X = rng.normal(0, 1, (n, 5))
    out = probe_auroc_per_layer({0: X}, labels, n_splits=5)
    assert math.isnan(out[0])
