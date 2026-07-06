# Bielik hallucination-detection — common tasks
# Usage: `just <recipe>` (https://github.com/casey/just)

# Default model for the risk tool (override per invocation, see `risk`)
default_model := "speakleash/Bielik-4.5B-v3.0-Instruct"

# List available recipes
default:
    @just --list

# Ask a question through the hallucination-risk tool.
# Usage:
#   just risk "Kim jest Robert Lewandowski?"
#   just risk "Czym jest Wąchobrzeźno?" speakleash/Bielik-11B-v3.0-Instruct
risk question model=default_model:
    uv run python scripts/hallucination_risk.py --model {{model}} "{{question}}"

# Interactive REPL mode of the risk tool.
# Usage: just risk-repl [model]
risk-repl model=default_model:
    uv run python scripts/hallucination_risk.py --model {{model}}

# NOTE: needs regenerated extraction artifacts (hidden_states.npz / signals.parquet),
# not shipped in the public release — see the README Reproduction section.
# Retrain all four risk probes from extraction artifacts
train-probes:
    uv run python scripts/train_risk_probe.py --all

# Run the Gradio demo locally (Bielik-1.5B)
app:
    uv run python app.py

# Run the test suite
test:
    uv run pytest -q
