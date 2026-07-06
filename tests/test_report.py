import math

import pandas as pd

from bielik_hallu.analysis.report import (build_extended_report,
                                          extended_auroc_curve, go_no_go_summary)
from bielik_hallu.extract.extended import EXTENDED_METRICS


def _synthetic_extended_signals() -> pd.DataFrame:
    """Two layers x two points, KNOWN separable from FABRICATED on effective_rank."""
    rows = []
    for point in ("prompt", "entity"):
        for layer in (0, 1):
            for i, cond in enumerate(["KNOWN"] * 3 + ["FABRICATED"] * 3):
                er = 1.0 + i if cond == "KNOWN" else 100.0 + i
                rows.append({
                    "entity": f"{cond}_{point}_{layer}_{i}",
                    "condition": cond, "layer": layer, "point": point,
                    "attn_entropy_mean": 0.5, "attn_entropy_norm_mean": 0.5,
                    "logitlens_entropy": 2.0, "effective_rank": er,
                })
    return pd.DataFrame(rows)


def test_extended_auroc_curve_perfect_separation():
    sig = _synthetic_extended_signals()
    curve = extended_auroc_curve(sig, "effective_rank", "prompt")
    assert set(curve) == {0, 1}
    assert all(v == 1.0 for v in curve.values())


def test_build_extended_report_structure():
    sig = _synthetic_extended_signals()
    report = build_extended_report(sig, "test-model")
    assert report["model_id"] == "test-model"
    assert report["contrast"] == "KNOWN(0) vs FABRICATED(1)"
    assert set(report["metrics"]) == set(EXTENDED_METRICS)
    er = report["metrics"]["effective_rank"]["prompt"]
    assert er["best_auroc"] == 1.0
    assert er["best_layer"] in (0, 1)
    for metric in EXTENDED_METRICS:
        for point in ("prompt", "entity"):
            assert set(report["metrics"][metric][point]) == {
                "best_auroc", "best_layer", "per_layer"}



def test_go_when_probe_high_global_high():
    s = go_no_go_summary(
        probe_auroc={5: 0.82, 6: 0.9},
        global_auroc={"ipr": {5: 0.78, 6: 0.8}},
        threshold=0.75,
    )
    assert s["verdict"] == "GO"
    assert s["signal_kind"] == "volumetric"
    assert s["probe_best_layer"] == 6


def test_go_directional_when_probe_high_global_low():
    s = go_no_go_summary(
        probe_auroc={5: 0.88},
        global_auroc={"ipr": {5: 0.55}},
        threshold=0.75,
    )
    assert s["verdict"] == "GO"
    assert s["signal_kind"] == "directional"


def test_nogo_when_probe_flat():
    s = go_no_go_summary(
        probe_auroc={5: 0.58},
        global_auroc={"ipr": {5: 0.54}},
        threshold=0.75,
    )
    assert s["verdict"] == "NO-GO"
    assert s["signal_kind"] == "none"


def test_go_no_go_ignores_nan_layers():
    s = go_no_go_summary(
        probe_auroc={5: float("nan"), 6: 0.82},
        global_auroc={"ipr": {5: 0.78, 6: 0.8}},
        threshold=0.75,
    )
    assert s["probe_best_layer"] == 6
    assert s["verdict"] == "GO"


def test_go_no_go_all_nan():
    s = go_no_go_summary(
        probe_auroc={5: float("nan"), 6: float("nan")},
        global_auroc={"ipr": {5: 0.78, 6: 0.8}},
        threshold=0.75,
    )
    assert s["verdict"] == "NO-GO"
    assert s["signal_kind"] == "none"
    assert s["probe_best_layer"] is None
    assert math.isnan(s["probe_max"])
