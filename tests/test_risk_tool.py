"""Pure-CPU tests for the hallucination-risk tool (no model load, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from bielik_hallu.risk.probe import (
    BAND_LOW_MAX,
    BAND_MEDIUM_MAX,
    RiskProbe,
    brier_score,
    load_probe,
    reliability_bins,
    risk_band,
    save_probe,
    sigmoid,
)


def _make_probe(hidden: int = 8, layer: int = 3) -> RiskProbe:
    rng = np.random.default_rng(0)
    return RiskProbe(
        weights=rng.normal(0, 1, hidden),
        bias=0.5,
        scaler_mean=rng.normal(0, 1, hidden),
        scaler_std=np.abs(rng.normal(1, 0.1, hidden)) + 0.1,
        calibrator_a=1.3,
        calibrator_b=-0.2,
        layer=layer,
        metadata={"model_slug": "test", "cv_auroc": 0.99},
    )


# --- serialization round-trip ------------------------------------------------
def test_probe_roundtrip_preserves_everything(tmp_path):
    probe = _make_probe()
    x = np.random.default_rng(1).normal(0, 1, probe.weights.shape[0])
    p_before = float(probe.predict_proba(x))
    raw_before = float(probe.raw_score(x))

    out = save_probe(probe, tmp_path / "risk_probe.npz")
    assert out.exists()
    loaded = load_probe(out)

    assert loaded.layer == probe.layer
    assert loaded.metadata == probe.metadata
    np.testing.assert_allclose(loaded.weights, probe.weights)
    np.testing.assert_allclose(loaded.scaler_mean, probe.scaler_mean)
    np.testing.assert_allclose(loaded.scaler_std, probe.scaler_std)
    assert loaded.bias == pytest.approx(probe.bias)
    assert loaded.calibrator_a == pytest.approx(probe.calibrator_a)
    assert loaded.calibrator_b == pytest.approx(probe.calibrator_b)
    # forward path identical after round-trip
    assert float(loaded.predict_proba(x)) == pytest.approx(p_before)
    assert float(loaded.raw_score(x)) == pytest.approx(raw_before)


def test_probe_loads_without_pickle(tmp_path):
    """The npz must load with allow_pickle=False (metadata via JSON, not object)."""
    out = save_probe(_make_probe(), tmp_path / "risk_probe.npz")
    with np.load(out, allow_pickle=False) as d:
        assert "weights" in d.files
        assert "metadata" in d.files


def test_save_returns_npz_path(tmp_path):
    out = save_probe(_make_probe(), tmp_path / "myprobe")
    assert out.suffix == ".npz"
    assert out.exists()


# --- sigmoid + calibration math ----------------------------------------------
def test_sigmoid_stable_and_bounded():
    xs = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    ys = sigmoid(xs)
    assert np.all(ys >= 0.0) and np.all(ys <= 1.0)
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert not np.any(np.isnan(ys))


def test_calibration_is_monotonic_in_raw_logit():
    """Higher raw probe logit -> higher (or equal) calibrated probability."""
    probe = _make_probe()
    # Sweep the standardized input along +weights so raw_logit increases.
    base = probe.scaler_mean.copy()
    direction = probe.weights * probe.scaler_std
    probs = []
    logits = []
    for t in np.linspace(-3, 3, 25):
        x = base + t * direction
        logits.append(float(probe.raw_logit(x)))
        probs.append(float(probe.predict_proba(x)))
    logits = np.array(logits)
    probs = np.array(probs)
    order = np.argsort(logits)
    sorted_probs = probs[order]
    assert np.all(np.diff(sorted_probs) >= -1e-9)  # non-decreasing


def test_negative_calibrator_a_flips_direction():
    probe = _make_probe()
    probe.calibrator_a = -1.0
    base = probe.scaler_mean.copy()
    direction = probe.weights * probe.scaler_std
    lo = float(probe.predict_proba(base - 3 * direction))
    hi = float(probe.predict_proba(base + 3 * direction))
    # with negative A, higher logit -> lower calibrated prob
    assert hi < lo


def test_brier_score_perfect_is_zero():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y.astype(float), y) == pytest.approx(0.0)


def test_brier_score_worst_is_one():
    y = np.array([0, 1])
    p = np.array([1.0, 0.0])
    assert brier_score(p, y) == pytest.approx(1.0)


def test_reliability_bins_partition_counts():
    p = np.array([0.05, 0.15, 0.45, 0.85, 0.95])
    y = np.array([0, 0, 1, 1, 1])
    bins = reliability_bins(p, y, n_bins=5)
    assert sum(b["count"] for b in bins) == len(p)
    for b in bins:
        assert 0.0 <= b["empirical_rate"] <= 1.0
        assert 0.0 <= b["mean_pred"] <= 1.0


# --- band thresholds ---------------------------------------------------------
def test_band_thresholds():
    assert risk_band(0.0) == "LOW"
    assert risk_band(BAND_LOW_MAX - 1e-9) == "LOW"
    assert risk_band(BAND_LOW_MAX) == "MEDIUM"
    assert risk_band(BAND_MEDIUM_MAX - 1e-9) == "MEDIUM"
    assert risk_band(BAND_MEDIUM_MAX) == "HIGH"
    assert risk_band(1.0) == "HIGH"


def test_band_ordering_over_range():
    seen = [risk_band(p) for p in np.linspace(0, 1, 11)]
    # bands only ever escalate LOW -> MEDIUM -> HIGH as p rises
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    ranks = [rank[b] for b in seen]
    assert ranks == sorted(ranks)


# --- CLI arg parsing (no model load) ----------------------------------------
def test_cli_defaults_to_45b():
    import hallucination_risk as hr

    assert hr.DEFAULT_MODEL == "speakleash/Bielik-4.5B-v3.0-Instruct"


def test_probe_path_for_uses_slug():
    import hallucination_risk as hr

    p = hr.probe_path_for("speakleash/Bielik-1.5B-v3.0-Instruct")
    assert p.name == "risk_probe.npz"
    assert p.parent.name == "Bielik-1.5B-v3.0-Instruct"


def test_cli_missing_probe_errors(monkeypatch, tmp_path):
    import hallucination_risk as hr

    # point probe lookup at an empty dir so the probe is "missing"
    monkeypatch.setattr(hr, "RESULTS_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        hr.main(["--model", "speakleash/Does-Not-Exist", "Kim jest X?"])


def test_render_prompt_wraps_question():
    import hallucination_risk as hr

    class FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            return "<s>[INST] " + messages[0]["content"] + " [/INST]"

    out = hr._render_prompt(FakeTok(), "Kim jest Robert Lewandowski?")
    assert "Kim jest Robert Lewandowski?" in out


def test_disclaimer_mentions_calibration_scope():
    import hallucination_risk as hr

    assert "Kim jest X?" in hr.DISCLAIMER
    assert "does NOT fact-check" in hr.DISCLAIMER


def test_normalize_appends_calibration_suffix():
    import hallucination_risk as hr

    out = hr.normalize_question("Czym jest Kraków?")
    assert out.endswith("Odpowiedz jednym zdaniem.")
    assert "Czym jest Kraków?" in out


def test_normalize_no_double_suffix():
    import hallucination_risk as hr

    q = "Kim jest Robert Lewandowski? Odpowiedz jednym zdaniem."
    assert hr.normalize_question(q) == q
    # count stays at one occurrence
    assert hr.normalize_question(q).count("Odpowiedz jednym zdaniem.") == 1


def test_normalize_adds_question_mark_when_missing():
    import hallucination_risk as hr

    out = hr.normalize_question("Napisz haiku o jesieni")
    assert "Odpowiedz jednym zdaniem." in out


# --- trained artifacts sanity (skips if not yet trained) ---------------------
@pytest.mark.parametrize("slug", [
    "Bielik-1.5B-v3.0-Instruct",
    "Bielik-4.5B-v3.0-Instruct",
    "Bielik-Minitron-7B-v3.0-Instruct",
    "Bielik-11B-v3.0-Instruct",
])
def test_trained_probe_is_wellformed(slug):
    path = _ROOT / "results" / slug / "risk_probe.npz"
    if not path.exists():
        pytest.skip(f"probe not trained: {path}")
    probe = load_probe(path)
    assert probe.weights.ndim == 1
    assert probe.scaler_mean.shape == probe.weights.shape
    assert probe.scaler_std.shape == probe.weights.shape
    assert not np.any(probe.scaler_std == 0.0)
    assert probe.layer >= 0
    assert 0.5 <= probe.metadata["cv_auroc"] <= 1.0
    # forward path returns a valid probability
    x = np.zeros(probe.weights.shape[0])
    p = float(probe.predict_proba(x))
    assert 0.0 <= p <= 1.0
