"""Run the full MVP: dataset -> extraction -> analysis -> go/no-go report."""
import numpy as np
import pandas as pd

from bielik_hallu import config
from bielik_hallu.dataset.build import build_dataset
from bielik_hallu.extract.run import extract_signals
from bielik_hallu.analysis.auroc import auroc_per_layer
from bielik_hallu.analysis.contrasts import condition_contrast_labels
from bielik_hallu.analysis.probe import probe_auroc_per_layer
from bielik_hallu.analysis.plots import plot_auroc_vs_depth
from bielik_hallu.analysis.report import go_no_go_summary

# NOTE on layer-index caveat: the two metric families live in different index
# spaces. `signals.parquet` "layer" values (used for ipr/entropy) are
# MLP-layer indices 0..N-1, captured via act_fn hooks on each transformer
# block's MLP. `hidden_states.npz` "*_layer_<k>" keys (used for the probe)
# are residual-stream indices 0..N, where layer_0 is the embedding output
# (i.e. one more point than there are transformer blocks). They are not the
# same axis: on the overlaid AUROC-vs-depth plot the probe curve is shifted
# by one and has one extra point relative to the ipr/entropy curves. This is
# expected and does not affect the GO/NO-GO verdict, since each metric is
# evaluated independently over its own native set of layers.

PRIMARY_POSITIVE = "FABRICATED"
PRIMARY_NEGATIVE = "KNOWN"


def _global_auroc_by_point(sig: pd.DataFrame, point: str) -> dict:
    """AUROC per layer for each global metric, restricted to the primary
    KNOWN-vs-FABRICATED contrast, at the given measurement point."""
    subset = sig[(sig["point"] == point) & (sig["condition"].isin([PRIMARY_POSITIVE, PRIMARY_NEGATIVE]))]
    global_auroc = {}
    for metric in ("ipr", "entropy"):
        metric_by_layer = {int(layer): grp[metric].to_numpy()
                           for layer, grp in subset.groupby("layer")}
        labels_by_layer = {int(layer): (grp["condition"] == PRIMARY_POSITIVE).astype(int).to_numpy()
                           for layer, grp in subset.groupby("layer")}
        any_layer = next(iter(labels_by_layer))
        global_auroc[metric] = auroc_per_layer(metric_by_layer, labels_by_layer[any_layer])
    return global_auroc


def main():
    labeled = build_dataset()
    signals_path = extract_signals(labeled)

    sig = pd.read_parquet(signals_path)

    # allow_pickle not needed: arrays are numeric/strings (<U), file is self-generated (Task 12)
    npz = np.load(config.RESULTS_DIR / "hidden_states.npz")
    conditions = npz["conditions"]

    entity_hidden = {int(k.split("_")[-1]): npz[k] for k in npz.files if k.startswith("entity_layer_")}
    prompt_hidden = {int(k.split("_")[-1]): npz[k] for k in npz.files if k.startswith("prompt_layer_")}

    # PRIMARY contrast: KNOWN vs FABRICATED (condition-based), not the mixed
    # 3-condition behavioral label.
    mask, kf_labels = condition_contrast_labels(conditions, PRIMARY_POSITIVE, PRIMARY_NEGATIVE)

    rng = np.random.default_rng(0)
    shuffled_labels = kf_labels.copy()
    rng.shuffle(shuffled_labels)

    results_by_point = {}
    for point, hidden_by_layer in (("entity", entity_hidden), ("prompt", prompt_hidden)):
        global_auroc = _global_auroc_by_point(sig, point)

        hidden = {layer: arr[mask] for layer, arr in hidden_by_layer.items()}
        probe = probe_auroc_per_layer(hidden, kf_labels)
        probe_shuffled = probe_auroc_per_layer(hidden, shuffled_labels)

        results_by_point[point] = {
            "global_auroc": global_auroc,
            "probe": probe,
            "probe_shuffled": probe_shuffled,
        }

    # PROMPT point is primary: less confounded by the raw entity-name token.
    prompt_res = results_by_point["prompt"]
    entity_res = results_by_point["entity"]

    prompt_summary = go_no_go_summary(prompt_res["probe"], prompt_res["global_auroc"])
    entity_summary = go_no_go_summary(entity_res["probe"], entity_res["global_auroc"])

    shuffled_floor = max(
        (v for v in prompt_res["probe_shuffled"].values() if v == v),  # filter nan
        default=float("nan"),
    )

    plot_auroc_vs_depth(
        {**prompt_res["global_auroc"], "probe": prompt_res["probe"], "probe_shuffled": prompt_res["probe_shuffled"]},
        config.RESULTS_DIR / "auroc_vs_depth_known_vs_fab_prompt.png",
    )

    print("=== GO/NO-GO (PRIMARY: prompt token, KNOWN vs FABRICATED) ===")
    for k, v in prompt_summary.items():
        print(f"{k}: {v}")
    print(f"shuffled_label_floor (prompt probe, max over layers): {shuffled_floor}")

    print("\n=== GO/NO-GO (entity token, KNOWN vs FABRICATED, for comparison) ===")
    for k, v in entity_summary.items():
        print(f"{k}: {v}")

    print("\n=== CAVEATS ===")
    print(
        "1. The primary contrast is KNOWN-vs-FABRICATED (condition-based), not the\n"
        "   mixed 3-condition behavioral hallucination label. It is evaluated at both\n"
        "   the entity-name token and the post-question prompt token.\n"
        "2. The entity-token probe can be inflated by lexical/frequency differences\n"
        "   between real famous names (KNOWN) and mutated/invented strings\n"
        "   (FABRICATED), rather than genuine internal 'knowledge' separation.\n"
        "   Compare the entity-token probe AUROC against its shuffled-label floor\n"
        "   above, and prefer the prompt-token result as the less confounded signal.\n"
        "3. Layer-index caveat: ipr/entropy layers (0..N-1, MLP hooks) and probe\n"
        "   layers (0..N, residual-stream incl. embedding) are not the same axis;\n"
        "   see the module docstring / code comment above for details.\n"
        "4. FABRICATED names are token-matched to KNOWN names (Task 6) to mitigate\n"
        "   raw frequency confounds, but residual lexical differences may remain."
    )


if __name__ == "__main__":
    main()
