"""Run E2: extended metrics extraction + per-layer AUROC report.

Usage: BIELIK_MODEL_ID=<hf-model-id> uv run python run_extended.py

Reads data/<slug>/labeled.parquet (must already exist — run run_mvp.py first),
writes results/<slug>/extended_signals.parquet and
results/<slug>/extended_report.json.

Primary contrast: KNOWN(0) vs FABRICATED(1); prompt point is primary, entity
point is reported for comparison (same convention as run_mvp.py). AUROC values
are separability scores — analysis.auroc.auroc already returns max(a, 1-a)
because a metric's direction may flip across layers and models.
"""
import json

import pandas as pd

from bielik_hallu import config
from bielik_hallu.analysis.report import build_extended_report, extended_auroc_curve
from bielik_hallu.extract.extended import EXTENDED_METRICS, extract_extended_signals

# Re-exported for backward compatibility with any external callers.
PRIMARY_POSITIVE = "FABRICATED"
PRIMARY_NEGATIVE = "KNOWN"
_auroc_curve = extended_auroc_curve


def main():
    labeled_path = config.DATA_DIR / "labeled.parquet"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"{labeled_path} not found — run run_mvp.py for this model first.")

    signals_path = extract_extended_signals(labeled_path)
    sig = pd.read_parquet(signals_path)

    report = build_extended_report(sig, config.MODEL_ID)

    report_path = config.RESULTS_DIR / "extended_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== E2 extended metrics: {config.MODEL_ID} ===")
    print(f"contrast: {report['contrast']} | primary point: prompt")
    for metric in EXTENDED_METRICS:
        for point in ("prompt", "entity"):
            r = report["metrics"][metric][point]
            if r["best_layer"] is None:
                print(f"{metric:24s} @ {point:6s}: no data")
            else:
                print(f"{metric:24s} @ {point:6s}: best AUROC {r['best_auroc']:.3f} "
                      f"@ layer {r['best_layer']}")
    print(f"signals: {signals_path}")
    print(f"report:  {report_path}")


if __name__ == "__main__":
    main()
