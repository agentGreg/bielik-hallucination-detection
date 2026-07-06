"""Hallucination-risk tool: answer a Polish question + estimate ignorance risk.

Given a Polish question (typically an entity question like "Kim jest X?" or
"Czym jest X?"), this:

  1. renders the model's chat prompt,
  2. runs ONE forward pass with output_hidden_states, captures the
     last-prompt-token residual hidden state at the probe's trained layer,
  3. standardizes it and computes a calibrated P(hallucination-from-ignorance),
  4. generates the model's answer (greedy by default; --sample for temperature),
  5. prints the answer, the risk %, a qualitative band, and an honest disclaimer.

The probe NEVER sees the generated answer — it reads prompt-side activations
only, so the risk estimate is invariant to generation settings.

Usage:
  uv run python scripts/hallucination_risk.py "Kim jest Robert Lewandowski?"
  uv run python scripts/hallucination_risk.py --model speakleash/Bielik-1.5B-v3.0-Instruct "Czym jest Kraków?"
  uv run python scripts/hallucination_risk.py            # interactive REPL
  uv run python scripts/hallucination_risk.py --json "Kim jest Jan Kowalski?"

Defaults to the 4.5B model (best size/quality trade-off). Requires HF access to
the gated Bielik v3.0 models and an MPS-capable Mac (Apple Silicon).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from bielik_hallu.risk.probe import load_probe, risk_band  # noqa: E402,F401
from bielik_hallu.risk.inference import (  # noqa: E402
    DISCLAIMER,
    TEMPLATE_SUFFIX,
    normalize_question,
    render_prompt,
    score_and_answer,
)
from bielik_hallu.risk.inference import load_runtime as _load_runtime  # noqa: E402

DEFAULT_MODEL = "speakleash/Bielik-4.5B-v3.0-Instruct"
RESULTS_ROOT = _ROOT / "results"

# Back-compat alias: earlier code / tests referenced the private name.
_render_prompt = render_prompt


def probe_path_for(model_id: str) -> Path:
    slug = model_id.split("/")[-1]
    return RESULTS_ROOT / slug / "risk_probe.npz"


def load_runtime(model_id: str):
    """Load model + tokenizer with auto device/dtype selection (cuda > mps > cpu)."""
    return _load_runtime(model_id)


def _print_human(res: dict, verbose: bool) -> None:
    print()
    print(f"Q: {res['question']}")
    print(f"A: {res['answer']}")
    print()
    print(f"Hallucination-risk: {res['p_risk'] * 100:.1f}%  [{res['band']}]")
    if verbose:
        print(f"  (raw uncalibrated probe score: {res['raw_score'] * 100:.1f}%; "
              f"layer L{res['layer']}; model {res['model']})")
    print()
    print(DISCLAIMER)


def run_once(model, tokenizer, device, probe, question, args) -> dict:
    return score_and_answer(
        model, tokenizer, device, probe, question,
        sample=args.sample, temperature=args.temperature,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", help="Polish question; omit for REPL mode")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    ap.add_argument("--sample", action="store_true",
                    help="sample the answer (temperature) instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=0.7, help="sampling temperature (with --sample)")
    ap.add_argument("--json", dest="as_json", action="store_true", help="machine-readable JSON output")
    ap.add_argument("--verbose", action="store_true", help="print raw probe score + layer")
    args = ap.parse_args(argv)

    probe_path = probe_path_for(args.model)
    if not probe_path.exists():
        ap.error(
            f"no trained probe for {args.model} at {probe_path}. "
            f"Run: uv run python scripts/train_risk_probe.py --slug {args.model.split('/')[-1]}"
        )
        return 2
    probe = load_probe(probe_path)

    print(f"Loading {args.model} ...", file=sys.stderr)
    model, tokenizer, device = load_runtime(args.model)
    print(f"Loaded on {device}. Probe layer L{probe.layer} "
          f"(CV AUROC {probe.metadata.get('cv_auroc')}).", file=sys.stderr)

    if args.question:
        res = run_once(model, tokenizer, device, probe, args.question, args)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            _print_human(res, args.verbose)
        return 0

    # Interactive REPL.
    print("Interactive mode. Type a Polish question, or 'exit' to quit.", file=sys.stderr)
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        res = run_once(model, tokenizer, device, probe, q, args)
        if args.as_json:
            print(json.dumps(res, ensure_ascii=False))
        else:
            _print_human(res, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
