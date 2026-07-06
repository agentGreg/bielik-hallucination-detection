# Hallucination-risk tool

An end-user CLI that, given a Polish question, returns (a) Bielik's answer and
(b) a **calibrated probability that the answer is a hallucination-from-ignorance**
— i.e. that the model does not actually know the entity/topic and is confabulating.

- Training: `scripts/train_risk_probe.py`
- Inference: `scripts/hallucination_risk.py`
- Probe math + serialization: `src/bielik_hallu/risk/probe.py`
- Trained probes: `results/<slug>/risk_probe.npz`
- Tests: `tests/test_risk_tool.py`

## How it works

The paper shows that Bielik's internal activations at the **last prompt token**
(after the question, before generation) carry a strong signal for whether the model
knows the entity being asked about. In particular, a linear probe over the residual-stream
hidden state separates KNOWN entities from UNKNOWN_REAL (real but unfamiliar) and
FABRICATED (invented) entities with very high AUROC, and this separability sits at a
stable relative depth (~0.6–0.8) across model sizes. The tool puts that
finding into practice: it trains one logistic probe per model on the pooled prompt-point hidden
states from four entity domains (athletes, cities, writers, musicians), collapsing the
three conditions into a binary **knowledge-absence-risk** target (KNOWN = 0;
UNKNOWN_REAL and FABRICATED = 1). The collapse is honest to the evidence: behaviorally,
both UNKNOWN_REAL and FABRICATED answers were essentially always confabulated, so the
probe predicts *absence of knowledge about the topic*, not lie-vs-truth.

At inference the tool runs a **single forward pass** with `output_hidden_states`,
grabs the last-prompt-token hidden state at the probe's chosen layer, standardizes it
(saved per-feature mean/std), applies the linear probe, and passes the raw log-odds
through a Platt/sigmoid calibrator fitted on 5-fold out-of-fold scores. The result is
a calibrated `P(hallucination-risk)`. The model's answer is generated afterwards by
continuing from the **same** prompt — so the probe never sees the answer, and the risk
estimate is invariant to generation settings (greedy vs sampling).

## Trained probes (per model)

Pooled 504 samples (168 KNOWN, 336 risk) per model, prompt point, 5-fold CV, seed 0.

| Model | Layer (abs / rel. depth) | Pooled CV AUROC (95% CI) | Brier |
|---|---|---|---|
| Bielik-1.5B-v3.0-Instruct | L21 / 0.66 | 0.9983 [0.9963, 0.9996] | 0.0155 |
| Bielik-4.5B-v3.0-Instruct | L46 / 0.77 | 0.9999 [0.9995, 1.0000] | 0.0044 |
| Bielik-Minitron-7B-v3.0-Instruct | L25 / 0.62 | 0.9973 [0.9946, 0.9992] | 0.0183 |
| Bielik-11B-v3.0-Instruct | L30 / 0.60 | 0.9995 [0.9985, 1.0000] | 0.0058 |

The chosen relative depths (0.60–0.77) fall inside the paper's cross-model probe band.
Note the pooled KNOWN-vs-{UNKNOWN_REAL, FABRICATED} task is *easier* than per-domain
KNOWN-vs-FABRICATED alone (UNKNOWN_REAL is highly separable from KNOWN at the prompt
point), which is why AUROC is near-ceiling here. Full per-layer AUROC and reliability
bins are stored in each probe's `metadata`.

## Install & run

We have used an Apple-Silicon Mac (MPS) - using on other hardware might require some adaptations, `uv`, and Hugging Face access to the **gated** Bielik v3.0 models (request access on the model page; the local HF cache token must
have it — no `HF_TOKEN` injection is needed if the cache is already authorized).

```bash
# one-shot: answer + risk (defaults to the 4.5B model)
uv run python scripts/hallucination_risk.py "Kim jest Robert Lewandowski?"

# pick a model
uv run python scripts/hallucination_risk.py \
  --model speakleash/Bielik-1.5B-v3.0-Instruct "Czym jest Kraków?"

# interactive REPL (no question argument)
uv run python scripts/hallucination_risk.py

# machine-readable output (for wrapping in a web UI)
uv run python scripts/hallucination_risk.py --json "Kim jest Jan Kowalski?"

# show the raw uncalibrated score + layer
uv run python scripts/hallucination_risk.py --verbose "Kim jest X?"
```

Flags: `--sample` (temperature decoding; default is deterministic greedy),
`--temperature` (default 0.7), `--json`, `--verbose`.

Risk bands: **LOW** `< 25%` · **MEDIUM** `25–60%` · **HIGH** `> 60%`.

Retrain from existing artifacts (no new extraction needed):

```bash
uv run python scripts/train_risk_probe.py --slug Bielik-4.5B-v3.0-Instruct
uv run python scripts/train_risk_probe.py --all
```

### Prompt normalization

The probe is calibrated on prompts ending with `Odpowiedz jednym zdaniem.` (the exact
training template was `Kim/Czym jest {entity}? Odpowiedz jednym zdaniem.`). The CLI
appends that suffix automatically unless your question already asks for a one-sentence
answer, so real usage stays on the calibrated distribution. This matters: the bare
`Czym jest Kraków?` (no suffix) is off-distribution and scores spuriously ~100%, while
`Czym jest Kraków? Odpowiedz jednym zdaniem.` correctly scores ~0%.

## Example output (Bielik-4.5B, captured)

Smoke test on the default 4.5B model, greedy decoding:

| Kind | Question | P(risk) | Band | Answer (snippet) |
|---|---|---|---|---|
| famous | Kim jest Robert Lewandowski? | 0.0% | LOW | "...polski piłkarz, uznawany za jednego z najlepszych..." |
| famous | Czym jest Kraków? | 0.0% | LOW | "...historyczne miasto w południowej Polsce..." |
| famous | Kim jest Fryderyk Chopin? | 0.0% | LOW | "...polski kompozytor i pianista..." |
| fabricated | Kim jest Zdzisław Płatkowieński? | 100.0% | HIGH | "...polski aktor teatralny..." (confabulated) |
| fabricated | Czym jest Wąchobrzeźno? | 100.0% | HIGH | "...fikcyjna miejscowość z powieści..." (confabulated) |
| fabricated | Kim jest Mirosław Karczewijski? | 100.0% | HIGH | "...polski aktor teatralny..." (confabulated) |
| off-template | Jaka jest stolica Australii? | 0.0% | LOW | "Stolicą Australii jest **Canberra**." |
| off-template | Napisz haiku o jesieni. | 100.0% | HIGH | (a haiku) |

Invariance check: `P(risk)` is byte-identical for greedy vs sampled decoding, confirming
the probe reads prompt-side activations only.

The two off-template rows document behavior outside the calibration shape (no
expectation): a factual non-entity question ("stolica Australii") scores LOW because the
model is familiar with the topic, while a creative-writing instruction ("napisz haiku")
scores HIGH — there is no entity for the probe to be "familiar" with, so it reads as
knowledge-absent. Interpret non-entity prompts with care.

## Limitations

- The estimate measures the model's **familiarity with the entity/topic** from its
  internal activations. It is **calibrated on one-sentence Polish entity questions**
  (`Kim jest X?` / `Czym jest X?`).
- It does **NOT verify facts** when the model *does* know the topic. A LOW score means
  "the model is familiar", not "the answer is correct in every detail".
- It is **experimental for other question shapes** (multi-hop, reasoning, creative
  writing, non-entity factual questions). Off-template prompts may score misleadingly.
- The `cities` domain uses a *Czym jest X?* template; the other three use *Kim jest X?*.
  Both are covered by the pooled training set.

## Web demo

A Gradio Blocks app (`app.py` at the repo root) wraps the tool for the browser and for
[Hugging Face Spaces](https://huggingface.co/spaces). It is pinned to the CPU-viable
**Bielik-1.5B-v3.0-Instruct** and ships with that model's bundled probe
(`results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz`, resolved relative to `app.py`).

- **UI (bilingual, Polish-first):** question box (`Kim jest ...? / Czym jest ...?`), an
  answer panel, a colored risk gauge (percentage + LOW/MEDIUM/HIGH band), a collapsible
  "how it works" panel with the honest disclaimer, and clickable examples (famous,
  fabricated, off-template).
- **Shared logic:** the app and the CLI both import
  `src/bielik_hallu/risk/inference.py` (`normalize_question`, `render_prompt`,
  `load_runtime`, `score_and_answer`, `DISCLAIMER`), so there is a single inference path.
- **Device / dtype:** auto-detected `cuda > mps > cpu`; on CPU the model loads in
  **float32** (bf16 matmul is slow/poorly supported on CPU) at ~2x memory. Local Macs use
  MPS + bf16 unchanged.
- **Cold start:** the model is lazily loaded on the first request and cached; `gr.queue()`
  serializes requests. On free CPU the first answer takes ~10–60 s.
- **Gated model:** `HF_TOKEN` is read from the environment (a Space secret). If the gated
  model can't be downloaded, the UI shows a clear error instead of crashing.

Run locally:

```bash
uv sync --extra demo          # installs gradio (kept out of core deps)
uv run python app.py          # or: just app
```

Deploy to a Space: see `spaces/README-space.md` (HF front-matter, `cpu-basic` hardware
note, `HF_TOKEN` secret) and its copy list — push `app.py`,
`requirements-spaces.txt` renamed to `requirements.txt`, `src/bielik_hallu/risk/`, and
`results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz`.

The CLI's `hallucination_risk.py --json` (emitting
`{question, answer, p_risk, band, raw_score, layer, model}`) remains available as a
scripting/integration surface.
