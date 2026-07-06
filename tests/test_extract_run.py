import numpy as np
from bielik_hallu.extract.run import metrics_for_activation


def test_metrics_for_activation_keys():
    vec = np.array([0.0, 3.0, 0.0, 0.0])
    m = metrics_for_activation(vec)
    assert set(m.keys()) == {"ipr", "entropy"}
    assert m["ipr"] > 0.9  # concentrated -> high IPR
