"""Pure-CPU tests for the Gradio demo (app.py).

These exercise the pure helpers only — no model load, no GPU, no network. The
Gradio app object is built (which imports gradio) but never launched. If gradio
is not installed (core install without the `demo` extra) the whole module skips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# app.py imports gradio at module top; skip the file entirely if it's absent.
gr = pytest.importorskip("gradio")
import app  # noqa: E402


# --- static config -----------------------------------------------------------
def test_model_is_pinned_to_15b():
    assert app.MODEL_ID == "speakleash/Bielik-1.5B-v3.0-Instruct"
    assert app.MODEL_SLUG == "Bielik-1.5B-v3.0-Instruct"


def test_probe_path_is_relative_to_app_and_exists():
    assert app.PROBE_PATH.name == "risk_probe.npz"
    assert app.PROBE_PATH.parent.name == "Bielik-1.5B-v3.0-Instruct"
    # Resolved relative to app.py -> must sit under the repo, not the cwd.
    assert app.PROBE_PATH.is_relative_to(_ROOT)
    assert app.PROBE_PATH.exists()  # the bundled probe ships with the repo


def test_band_meta_covers_all_bands():
    for band in ("LOW", "MEDIUM", "HIGH"):
        label, color = app.BAND_META[band]
        assert color.startswith("#")
        assert label  # non-empty bilingual label


def test_examples_include_famous_fabricated_offtemplate():
    flat = [e[0] for e in app.EXAMPLES]
    assert any("Lewandowski" in q for q in flat)  # famous
    assert any("Płatkowieński" in q for q in flat)  # fabricated
    assert any("haiku" in q.lower() for q in flat)  # off-template
    # every example is a single-string row
    assert all(isinstance(row, list) and len(row) == 1 for row in app.EXAMPLES)


# --- gauge HTML --------------------------------------------------------------
@pytest.mark.parametrize(
    "p,band,expected_pct",
    [(0.0, "LOW", "0.0%"), (0.5, "MEDIUM", "50.0%"), (1.0, "HIGH", "100.0%")],
)
def test_gauge_html_shows_percentage_and_band(p, band, expected_pct):
    html = app._gauge_html(p, band)
    assert expected_pct in html
    _, color = app.BAND_META[band]
    assert color in html


def test_gauge_html_clamps_bar_width():
    # p_risk outside [0,1] must not produce a bar wider than 100% or negative.
    over = app._gauge_html(2.0, "HIGH")
    assert "width:100.0%" in over
    under = app._gauge_html(-1.0, "LOW")
    assert "width:0.0%" in under


# --- hf token ----------------------------------------------------------------
def test_hf_token_reads_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert app._hf_token() is None
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    assert app._hf_token() == "hf_abc"


def test_hf_token_falls_back_to_hub_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_xyz")
    assert app._hf_token() == "hf_xyz"


# --- analyze() control flow (no model load) ---------------------------------
def test_analyze_empty_question_short_circuits(monkeypatch):
    # Guard: if analyze tried to load a model on empty input, this would hang/fail.
    def _boom():
        raise AssertionError("ensure_loaded must not run for empty input")

    monkeypatch.setattr(app, "ensure_loaded", _boom)
    answer, gauge, status = app.analyze("   ")
    assert answer == "" and gauge == ""
    assert "Enter a question" in status or "Wpisz pytanie" in status


def test_analyze_surfaces_runtime_error_in_ui(monkeypatch):
    def _fail():
        raise RuntimeError("gated model needs HF_TOKEN")

    monkeypatch.setattr(app, "ensure_loaded", _fail)
    answer, gauge, status = app.analyze("Kim jest Robert Lewandowski?")
    assert answer == "" and gauge == ""
    assert "Error" in status
    assert "HF_TOKEN" in status


def test_analyze_renders_result_without_loading_model(monkeypatch):
    # Stub ensure_loaded + score_and_answer so no model/network is touched.
    monkeypatch.setattr(app, "ensure_loaded", lambda: None)
    app._RUNTIME["device"] = "cpu"
    fake = {
        "question": "Kim jest X?",
        "answer": "Testowa odpowiedź.",
        "p_risk": 0.87,
        "band": "HIGH",
        "raw_score": 0.91,
        "layer": 21,
        "model": app.MODEL_SLUG,
    }
    monkeypatch.setattr(app, "score_and_answer", lambda *a, **k: fake)
    answer, gauge, status = app.analyze("Kim jest X?")
    assert "Testowa odpowiedź." in answer
    assert "87.0%" in gauge
    assert "L21" in status


# --- ensure_loaded missing-probe guard --------------------------------------
def test_ensure_loaded_errors_on_missing_probe(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "PROBE_PATH", tmp_path / "nope.npz")
    app._RUNTIME["model"] = None
    app._RUNTIME["probe"] = None
    with pytest.raises(RuntimeError, match="probe not found"):
        app.ensure_loaded()


# --- the Blocks app builds (imports gradio, does not launch) -----------------
def test_build_demo_returns_blocks():
    demo = app.build_demo()
    assert isinstance(demo, gr.Blocks)
