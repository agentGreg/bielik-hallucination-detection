import numpy as np
import pytest
from bielik_hallu.metrics.baseline import first_token_logprob_entropy


def test_uniform_logits_max_entropy():
    logits = np.zeros(4)
    assert first_token_logprob_entropy(logits) == pytest.approx(np.log(4))


def test_peaked_logits_low_entropy():
    logits = np.array([100.0, 0.0, 0.0, 0.0])
    assert first_token_logprob_entropy(logits) == pytest.approx(0.0, abs=1e-6)
