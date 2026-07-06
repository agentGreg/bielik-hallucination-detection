import pytest

from bielik_hallu.dataset.build import call_with_retry


def test_call_with_retry_succeeds_after_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return 42

    result = call_with_retry(fn, tries=3, base_delay=1.0, sleep=lambda *_: None)
    assert result == 42
    assert calls["n"] == 3


def test_call_with_retry_reraises():
    def fn():
        raise ValueError("always fails")

    with pytest.raises(ValueError):
        call_with_retry(fn, tries=3, base_delay=1.0, sleep=lambda *_: None)
