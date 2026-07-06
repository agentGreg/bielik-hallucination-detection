"""Train the calibrated hallucination-risk probe from EXISTING artifacts.

For one model slug, this pools the last-prompt-token residual hidden states from
the athletes set (results/<slug>/hidden_states.npz, 126 rows) and the three extra
domains (results/<slug>/domains/<domain>/hidden_states.npz, 126 each) into a
504-row training set. Labels collapse the three conditions to a binary
knowledge-absence-risk target:

    KNOWN         -> 0   (model knew the entity; answers were correct)
    UNKNOWN_REAL  -> 1   (real entity the model didn't know; ~100% confabulated)
    FABRICATED    -> 1   (invented entity; ~100% confabulated)

Rationale: behaviorally, UNKNOWN_REAL and FABRICATED answers were essentially
always confabulated, so the tool predicts *absence of knowledge about the topic*,
not lie-vs-truth. This is the honest reading of the paper's evidence.

Pipeline (no new model extraction — reads npz only):
  1. Pool prompt-point features across athletes + 3 domains.
  2. Select the best residual layer by 5-fold stratified CV AUROC on the pooled,
     3-class-collapsed task (recomputed here; NOT the per-domain K-vs-F layers).
  3. Standardize features (save mean/std).
  4. Calibrate: take out-of-fold CV logistic decision scores at the chosen layer,
     fit a Platt/sigmoid calibrator on them (sigmoid chosen for n=504).
  5. Fit the final probe on ALL data at the chosen layer; save probe + scaler +
     calibrator + layer + metadata to results/<slug>/risk_probe.npz.

Usage:
  uv run python scripts/train_risk_probe.py --slug Bielik-4.5B-v3.0-Instruct
  uv run python scripts/train_risk_probe.py --all         # every slug with artifacts

Seeds: numpy default_rng(0), sklearn CV random_state=0 (reproducible).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Make ``src`` importable when run as a plain script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from bielik_hallu.risk.probe import (  # noqa: E402
    RiskProbe,
    brier_score,
    reliability_bins,
    sigmoid,
)

RESULTS_ROOT = _ROOT / "results"
DOMAINS = ("cities", "writers", "musicians")  # athletes come from the top-level npz
SEED = 0
N_SPLITS = 5


def _collapse_labels(conditions: np.ndarray) -> np.ndarray:
    """KNOWN -> 0, everything else (UNKNOWN_REAL, FABRICATED) -> 1."""
    return np.array([0 if c == "KNOWN" else 1 for c in conditions], dtype=np.int64)


def _load_source(npz_path: Path) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return ({layer: prompt_features}, collapsed_labels) for one npz file."""
    with np.load(npz_path, allow_pickle=False) as d:
        conditions = d["conditions"]
        layers = sorted(
            int(k.split("_")[-1]) for k in d.files if k.startswith("prompt_layer_")
        )
        by_layer = {layer: d[f"prompt_layer_{layer}"].astype(np.float64) for layer in layers}
    return by_layer, _collapse_labels(conditions)


def load_pooled(slug: str) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """Pool prompt-point features + collapsed labels across athletes + 3 domains."""
    base = RESULTS_ROOT / slug
    sources: list[Path] = [base / "hidden_states.npz"]  # athletes
    for dom in DOMAINS:
        sources.append(base / "domains" / dom / "hidden_states.npz")

    per_layer_chunks: dict[int, list[np.ndarray]] = {}
    label_chunks: list[np.ndarray] = []
    used: list[str] = []
    common_layers: set[int] | None = None

    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"missing artifact: {src}")
        by_layer, labels = _load_source(src)
        layers = set(by_layer)
        common_layers = layers if common_layers is None else (common_layers & layers)
        for layer, feats in by_layer.items():
            per_layer_chunks.setdefault(layer, []).append(feats)
        label_chunks.append(labels)
        used.append(str(src.relative_to(_ROOT)))

    assert common_layers, "no common residual layers across sources"
    pooled = {
        layer: np.concatenate(per_layer_chunks[layer], axis=0)
        for layer in sorted(common_layers)
    }
    y = np.concatenate(label_chunks, axis=0)
    return pooled, y, used


def _cv_oof_scores(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold decision scores (log-odds) and calibrated probabilities.

    Uses a fresh StandardScaler fit *inside* each CV fold via a pipeline so the
    reported AUROC has no train/test leakage. Returns (oof_logit, oof_proba).
    """
    from sklearn.pipeline import make_pipeline

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    oof_logit = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
    oof_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return oof_logit, oof_proba


def select_best_layer(pooled: dict[int, np.ndarray], y: np.ndarray) -> tuple[int, dict[int, float]]:
    """Pick the layer with the highest 5-fold CV AUROC on the pooled task."""
    auroc_by_layer: dict[int, float] = {}
    for layer in sorted(pooled):
        _, oof_proba = _cv_oof_scores(pooled[layer], y)
        a = roc_auc_score(y, oof_proba)
        auroc_by_layer[layer] = float(max(a, 1.0 - a))
    best = max(auroc_by_layer, key=auroc_by_layer.get)
    return best, auroc_by_layer


def _fit_platt(oof_logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a 1-D logistic (Platt scaling) mapping raw logit -> calibrated prob.

    p = sigmoid(A * logit + B). Fit with a LogisticRegression on the single
    out-of-fold logit feature; A is the coefficient, B the intercept.
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(oof_logit.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def train_slug(slug: str, verbose: bool = True) -> tuple[RiskProbe, dict]:
    pooled, y, sources = load_pooled(slug)
    n_layers_incl_emb = max(pooled) + 1  # residual axis is 0..N

    best_layer, auroc_by_layer = select_best_layer(pooled, y)
    Xb = pooled[best_layer]

    # Out-of-fold scores at the chosen layer, for honest calibration + CV AUROC.
    oof_logit, oof_proba = _cv_oof_scores(Xb, y)
    a = roc_auc_score(y, oof_proba)
    cv_auroc = float(max(a, 1.0 - a))

    # 95% bootstrap CI for the CV AUROC (resample the OOF probabilities).
    rng = np.random.default_rng(SEED)
    boot = []
    n = len(y)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        ba = roc_auc_score(y[idx], oof_proba[idx])
        boot.append(max(ba, 1.0 - ba))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # Platt calibrator on the OOF logits.
    cal_a, cal_b = _fit_platt(oof_logit, y)
    # Calibrated probabilities from the OOF logits (for reporting calibration).
    cal_proba_oof = sigmoid(cal_a * oof_logit + cal_b)
    brier = brier_score(cal_proba_oof, y)
    bins = reliability_bins(cal_proba_oof, y, n_bins=5)

    # Final probe: fit scaler + logistic on ALL data at the chosen layer.
    scaler = StandardScaler().fit(Xb)
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(Xb), y)
    # Guard against zero-variance features producing inf during standardization.
    std = scaler.scale_.copy()
    std[std == 0.0] = 1.0

    rel_depth = best_layer / (n_layers_incl_emb - 1)
    today = _dt.date.today().isoformat()

    metadata = {
        "model_slug": slug,
        "task": "KNOWN(0) vs UNKNOWN_REAL/FABRICATED(1), prompt point (last prompt token)",
        "measurement_point": "prompt",
        "n_samples": int(len(y)),
        "n_known": int((y == 0).sum()),
        "n_risk": int((y == 1).sum()),
        "sources": sources,
        "layer_absolute": int(best_layer),
        "n_residual_layers_incl_embedding": int(n_layers_incl_emb),
        "layer_relative_depth": round(float(rel_depth), 4),
        "cv_auroc": round(cv_auroc, 4),
        "cv_auroc_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "brier_score": round(brier, 4),
        "reliability_bins": bins,
        "calibrator": "platt_sigmoid",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "trained_date": today,
        "auroc_by_layer": {str(k): round(v, 4) for k, v in auroc_by_layer.items()},
    }

    probe = RiskProbe(
        weights=lr.coef_[0].astype(np.float64),
        bias=float(lr.intercept_[0]),
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_std=std.astype(np.float64),
        calibrator_a=cal_a,
        calibrator_b=cal_b,
        layer=int(best_layer),
        metadata=metadata,
    )

    out_path = RESULTS_ROOT / slug / "risk_probe.npz"
    probe.save(out_path)

    if verbose:
        print(f"\n=== {slug} ===")
        print(f"  pooled samples : {len(y)} (KNOWN={metadata['n_known']}, RISK={metadata['n_risk']})")
        print(f"  chosen layer   : L{best_layer} (rel. depth {rel_depth:.2f}, "
              f"of {n_layers_incl_emb} residual layers incl. embedding)")
        print(f"  pooled CV AUROC: {cv_auroc:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Brier score    : {brier:.4f}")
        print("  reliability (5 bins, calibrated OOF):")
        for b in bins:
            print(f"    [{b['bin'][0]:.2f},{b['bin'][1]:.2f}) n={b['count']:>3}  "
                  f"pred={b['mean_pred']:.3f}  emp={b['empirical_rate']:.3f}")
        print(f"  saved -> {out_path.relative_to(_ROOT)}")

    return probe, metadata


def discover_slugs() -> list[str]:
    slugs = []
    for p in sorted(RESULTS_ROOT.iterdir()):
        if p.is_dir() and (p / "hidden_states.npz").exists():
            slugs.append(p.name)
    return slugs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slug", help="model slug, e.g. Bielik-4.5B-v3.0-Instruct")
    ap.add_argument("--all", action="store_true", help="train for every slug with artifacts")
    args = ap.parse_args(argv)

    if args.all:
        slugs = discover_slugs()
    elif args.slug:
        slugs = [args.slug]
    else:
        ap.error("pass --slug <SLUG> or --all")
        return 2

    summary = []
    for slug in slugs:
        _, meta = train_slug(slug)
        summary.append((slug, meta["cv_auroc"], meta["brier_score"],
                        meta["layer_absolute"], meta["layer_relative_depth"]))

    print("\n=== summary ===")
    print(f"{'slug':<36} {'CV AUROC':>9} {'Brier':>7} {'layer':>6} {'rel.depth':>9}")
    for slug, auroc, brier, layer, rel in summary:
        print(f"{slug:<36} {auroc:>9.4f} {brier:>7.4f} {layer:>6} {rel:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
