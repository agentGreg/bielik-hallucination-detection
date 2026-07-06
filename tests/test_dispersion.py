import numpy as np
import pytest
from bielik_hallu.metrics.dispersion import ipr, shannon_entropy, normalize_outliers


def test_ipr_one_hot_is_one():
    a = np.array([0.0, 5.0, 0.0, 0.0])
    assert ipr(a) == pytest.approx(1.0)


def test_ipr_uniform_is_inverse_n():
    a = np.ones(4)
    assert ipr(a) == pytest.approx(0.25)


def test_entropy_uniform_is_log_n():
    a = np.ones(8)
    assert shannon_entropy(a) == pytest.approx(np.log(8))


def test_entropy_one_hot_is_zero():
    a = np.array([0.0, 3.0, 0.0])
    assert shannon_entropy(a) == pytest.approx(0.0)


def test_normalize_outliers_caps_large_value():
    a = np.array([1.0, 1.0, 1.0, 100.0])
    out = normalize_outliers(a, quantile=0.75)
    assert out.max() < 100.0
    assert out.max() == pytest.approx(1.0)
