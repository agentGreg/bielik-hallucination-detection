---
title: Bielik Hallucination Risk
emoji: 🦅
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ryzyko halucynacji dla polskich pytań o encje (Bielik-1.5B)
---

# Bielik — wykrywacz ryzyka halucynacji / Hallucination-risk detector

Zadaj polskie pytanie o encję ("Kim jest X?" / "Czym jest X?"). Model **Bielik-1.5B-v3.0-Instruct**
odpowie, a liniowy *probe* czytający jego wewnętrzne aktywacje przy ostatnim tokenie pytania
oszacuje **skalibrowane ryzyko, że model tematu nie zna** i konfabuluje.

Ask a Polish entity question; a linear probe over the model's last-prompt-token activations
returns a calibrated **P(hallucination-from-ignorance)**. The probe never sees the generated
answer, so the risk estimate is invariant to decoding settings.

## Hardware

- **Suggested:** `cpu-basic` (the free tier). The 1.5B model was chosen to be CPU-viable
  (~3 GB, loaded in float32 on CPU).
- **Latency:** the first request triggers a cold start (model download + load); subsequent
  answers take roughly **~10–60 s on free CPU**. The model is loaded once and cached.
- Auto device selection is `cuda > mps > cpu`, so upgrading the Space to a GPU tier works
  with no code change.

## Secrets

The Bielik v3.0 models are **gated**. Before the Space can download the model:

1. Accept the model license at https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct
2. In **Space Settings → Variables and secrets**, add a secret named **`HF_TOKEN`** whose
   value is a Hugging Face token with access to that model.

If `HF_TOKEN` is missing or lacks access, the app surfaces a clear error in the UI instead
of crashing.

## What to copy into the Space repo

From this project's root, push the following to the Space repo:

| Source (this repo) | Destination (Space repo) |
|---|---|
| `app.py` | `app.py` |
| `requirements-spaces.txt` | `requirements.txt` *(renamed)* |
| `src/bielik_hallu/risk/` | `src/bielik_hallu/risk/` |
| `results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz` | `results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz` |
| this file | `README.md` *(the YAML front-matter above is required by Spaces)* |

`app.py` inserts `src/` on `sys.path` and resolves the probe path relative to itself, so the
layout above is all that is needed — no packaging/install step. The `src/bielik_hallu/risk/`
folder must contain `__init__.py`, `probe.py`, and `inference.py` (its only dependencies are
`numpy`, `torch`, `transformers`, and `gradio`, all pinned in `requirements.txt`).

Minimal copy (shell, run from repo root):

```bash
mkdir -p SPACE/src/bielik_hallu SPACE/results/Bielik-1.5B-v3.0-Instruct
cp app.py SPACE/app.py
cp requirements-spaces.txt SPACE/requirements.txt
cp spaces/README-space.md SPACE/README.md
touch SPACE/src/bielik_hallu/__init__.py
cp -r src/bielik_hallu/risk SPACE/src/bielik_hallu/risk
cp results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz \
   SPACE/results/Bielik-1.5B-v3.0-Instruct/risk_probe.npz
```

## Honest disclaimer

This measures the model's **familiarity with the entity/topic**, not factual correctness.
A LOW score means "the model is familiar", not "the answer is correct in every detail". It is
**calibrated on one-sentence Polish entity questions** and is **experimental off-template**
(reasoning, creative writing, non-entity questions) — those prompts may score misleadingly.
See the "How it works" panel in the app for details.
