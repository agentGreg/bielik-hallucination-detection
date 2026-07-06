import numpy as np
from bielik_hallu.analysis.contrasts import condition_contrast_labels


def test_selects_only_positive_negative_conditions():
    conditions = ["KNOWN", "FABRICATED", "UNKNOWN_REAL", "FABRICATED"]
    mask, labels = condition_contrast_labels(conditions, positive="FABRICATED", negative="KNOWN")
    assert list(np.where(mask)[0]) == [0, 1, 3]
    assert list(labels) == [0, 1, 1]


def test_unknown_real_excluded():
    conditions = ["KNOWN", "FABRICATED", "UNKNOWN_REAL", "FABRICATED"]
    mask, _ = condition_contrast_labels(conditions, positive="FABRICATED", negative="KNOWN")
    assert mask[2] == False
    assert mask.sum() == 3
