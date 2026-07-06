"""Gradio web demo of the Bielik hallucination-risk tool (for Hugging Face Spaces).

Given a Polish entity question ("Kim jest ...?" / "Czym jest ...?") this returns
Bielik's answer plus a calibrated P(hallucination-from-ignorance): a linear probe
reads the model's last-prompt-token activations to judge how familiar it is with
the topic. The probe never sees the generated answer, so the risk estimate is
invariant to decoding settings.

Model: speakleash/Bielik-1.5B-v3.0-Instruct (CPU-viable, ~3 GB). Device is
auto-detected (cuda > mps > cpu); on CPU the model loads in float32 (bf16 matmul
is slow/poorly supported on CPU) at the cost of ~2x memory. The bundled probe is
results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz, resolved relative to this file
so it works in-repo and when copied into a Space repo alongside a results/ folder.

Run locally:  uv sync --extra demo && uv run python app.py   (or: just app)
The gated Bielik license requires HF access; set HF_TOKEN (Space secret / env) if
the model is not already cached with authorized credentials.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gradio as gr

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from bielik_hallu.risk.inference import (  # noqa: E402
    DISCLAIMER,
    load_runtime,
    score_and_answer,
    select_device,
)
from bielik_hallu.risk.probe import load_probe  # noqa: E402

MODEL_ID = "speakleash/Bielik-1.5B-v3.0-Instruct"
MODEL_SLUG = MODEL_ID.split("/")[-1]
PROBE_PATH = _ROOT / "results" / MODEL_SLUG / "risk_probe.npz"

# Bilingual (Polish-first) band labels + colors for the risk gauge.
BAND_META = {
    "LOW": ("NISKIE ryzyko / LOW risk", "#1a9850"),
    "MEDIUM": ("ŚREDNIE ryzyko / MEDIUM risk", "#f9a825"),
    "HIGH": ("WYSOKIE ryzyko / HIGH risk", "#d73027"),
}

EXAMPLES = [
    # (label shown to user, question). One famous, one fabricated, one off-template.
    ["Kim jest Robert Lewandowski?"],
    ["Kim jest Zdzisław Płatkowieński?"],  # fabricated entity
    ["Czym jest Kraków?"],
    ["Napisz haiku o jesieni."],  # off-template (no entity)
]

HOW_IT_WORKS = """\
### Jak to działa / How it works

**PL.** Model dostaje pytanie ("Kim jest X?" / "Czym jest X?"). Robimy **jeden przebieg**
modelu i odczytujemy jego wewnętrzny stan przy **ostatnim tokenie pytania** (zanim
padnie odpowiedź). Wytrenowany liniowy *probe* zamienia ten stan na **skalibrowane
prawdopodobieństwo, że model nie zna tematu** i konfabuluje. Odpowiedź jest generowana
dopiero potem — probe **nigdy jej nie widzi**, więc wynik ryzyka nie zależy od sposobu
generowania.

**EN.** We run the model **once**, read the residual-stream hidden state at the **last
prompt token**, and a linear probe maps it to a **calibrated P(hallucination-from-ignorance)**.
The answer is generated afterwards from the same prompt, so the probe never sees it.

**Uczciwe zastrzeżenie / Honest disclaimer**

- To mierzy **znajomość encji/tematu**, a nie weryfikuje faktów. Wynik LOW znaczy "model
  zna temat", a nie "odpowiedź jest w każdym szczególe poprawna". / This measures
  **familiarity, not fact-checking**. LOW means "the model is familiar", not "every detail is correct".
- Kalibrowane na **jednozdaniowych polskich pytaniach o encje** ("Kim jest X?" / "Czym jest X?").
  / Calibrated on **one-sentence Polish entity questions**.
- **Eksperymentalne** dla innych typów pytań (rozumowanie, twórczość, pytania nie o encje) —
  takie prompty mogą dawać mylące wyniki. / **Experimental off-template** (reasoning, creative
  writing, non-entity questions) — such prompts may score misleadingly.

Progi / Bands: **LOW** < 25% · **MEDIUM** 25–60% · **HIGH** > 60%.
"""

# Lazily-initialized runtime, shared across requests after the first (cold start).
_RUNTIME: dict = {"model": None, "tokenizer": None, "device": None, "probe": None}


def _hf_token() -> str | None:
    """Read a Hugging Face token from the environment (Space secret) if present."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def ensure_loaded() -> None:
    """Lazy-load the probe + model on first use; cache for subsequent requests.

    Raises RuntimeError with a user-facing message if the gated model cannot be
    downloaded (missing/insufficient HF access).
    """
    if _RUNTIME["model"] is not None:
        return
    if _RUNTIME["probe"] is None:
        if not PROBE_PATH.exists():
            raise RuntimeError(
                f"Bundled probe not found at {PROBE_PATH}. Copy "
                f"results/{MODEL_SLUG}/risk_probe.npz next to app.py."
            )
        _RUNTIME["probe"] = load_probe(PROBE_PATH)
    try:
        model, tokenizer, device = load_runtime(MODEL_ID, token=_hf_token())
    except Exception as exc:  # noqa: BLE001 — surface a clean message to the UI
        raise RuntimeError(
            "Could not load the gated Bielik model "
            f"'{MODEL_ID}'. The model is gated: accept its license on Hugging Face "
            "and set the HF_TOKEN secret (Space Settings -> Secrets) to a token with "
            f"access.\n\nUnderlying error: {type(exc).__name__}: {exc}"
        ) from exc
    _RUNTIME.update(model=model, tokenizer=tokenizer, device=device)


def _gauge_html(p_risk: float, band: str) -> str:
    """Simple self-contained HTML gauge: percentage + colored band bar."""
    label, color = BAND_META.get(band, (band, "#888"))
    pct = p_risk * 100.0
    width = max(0.0, min(100.0, pct))
    return f"""
<div style="font-family: system-ui, sans-serif;">
  <div style="display:flex; align-items:baseline; gap:.5rem;">
    <span style="font-size:2rem; font-weight:700; color:{color};">{pct:.1f}%</span>
    <span style="font-size:1rem; font-weight:600; color:{color};">{label}</span>
  </div>
  <div style="margin-top:.4rem; background:#e9e9e9; border-radius:6px; height:14px; overflow:hidden;">
    <div style="width:{width:.1f}%; height:100%; background:{color};"></div>
  </div>
  <div style="margin-top:.3rem; font-size:.8rem; color:#666;">
    P(halucynacja z niewiedzy) / P(hallucination-from-ignorance)
  </div>
</div>
""".strip()


def analyze(question: str, progress=gr.Progress()):
    """Gradio callback: answer + risk gauge for one question.

    Returns (answer_markdown, gauge_html, status_markdown).
    """
    q = (question or "").strip()
    if not q:
        return (
            "",
            "",
            "Wpisz pytanie. / Enter a question.",
        )
    try:
        progress(0.1, desc="Ładowanie modelu / Loading model (cold start ~10-60 s)...")
        ensure_loaded()
        progress(0.5, desc="Liczenie ryzyka + odpowiedź / Scoring + generating...")
        res = score_and_answer(
            _RUNTIME["model"],
            _RUNTIME["tokenizer"],
            _RUNTIME["device"],
            _RUNTIME["probe"],
            q,
        )
    except RuntimeError as exc:
        return "", "", f"**Błąd / Error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        return "", "", f"**Błąd / Error:** {type(exc).__name__}: {exc}"

    answer_md = f"**Odpowiedź modelu / Model answer:**\n\n{res['answer']}"
    gauge = _gauge_html(res["p_risk"], res["band"])
    status = (
        f"Layer L{res['layer']} · {MODEL_SLUG} · device: {_RUNTIME['device']} · "
        f"raw {res['raw_score'] * 100:.1f}%"
    )
    return answer_md, gauge, status


def build_demo() -> gr.Blocks:
    """Construct (but do not launch) the Gradio Blocks app."""
    with gr.Blocks(title="Bielik — wykrywacz ryzyka halucynacji") as demo:
        gr.Markdown(
            "# Bielik — wykrywacz ryzyka halucynacji\n"
            "### Hallucination-risk detector for Polish entity questions\n"
            "Zadaj pytanie o osobę lub rzecz. Model odpowie, a *probe* oszacuje "
            "**ryzyko, że model tematu nie zna** i konfabuluje.\n"
            "*Ask about a person or thing; a probe estimates the risk the model is confabulating.*"
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Pytanie / Question",
                    placeholder="Kim jest ...? / Czym jest ...?",
                    lines=1,
                )
                submit = gr.Button("Sprawdź / Analyze", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=question, label="Przykłady / Examples")
            with gr.Column(scale=2):
                gauge = gr.HTML(label="Ryzyko / Risk")
        answer = gr.Markdown()
        status = gr.Markdown()
        with gr.Accordion("Jak to działa? / How it works (+ disclaimer)", open=False):
            gr.Markdown(HOW_IT_WORKS)

        submit.click(analyze, inputs=question, outputs=[answer, gauge, status])
        question.submit(analyze, inputs=question, outputs=[answer, gauge, status])
    return demo


def main() -> None:
    device = select_device()
    print(f"[app] device auto-detected: {device}", file=sys.stderr)
    if _hf_token():
        print("[app] HF_TOKEN found in environment.", file=sys.stderr)
    demo = build_demo()
    demo.queue()  # serialize requests; cold start loads the model once
    demo.launch(server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
