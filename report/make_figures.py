"""Generate the report figure from cross-validated scores of the final model.

Usage (from the repository root): python report/make_figures.py   (~3 min)
Writes report/figures/threshold.pdf. The out-of-fold scores are cached in
outputs/oof_full.npz; delete it to recompute them.

Uses the same protocol as model_selection.py: 5-fold stratified CV on the 80%
non-holdout rows, "full" preprocessing, logistic regression with lambda=1e-4.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_validation import cross_validate, logistic_model  # noqa: E402
from evaluation import classification_metrics, predict_labels  # noqa: E402
from evaluation import stratified_split  # noqa: E402
from preprocessing import load_data  # noqa: E402

SEED = 42
LAMBDA = 1e-4
SELECTED_THRESHOLD = 0.20
CACHE = "outputs/oof_full.npz"


def main():
    """Plot precision, recall, F1 and accuracy against the decision threshold."""
    x, _, y, _, _, names = load_data()
    tr, _ = stratified_split(y, val_ratio=0.2, seed=SEED)
    if os.path.exists(CACHE):
        scores = np.load(CACHE)["scores"]
    else:
        model = logistic_model(LAMBDA)
        oof, _ = cross_validate(x[tr], y[tr], names, "full", [model], k=5, seed=SEED)
        scores = oof[model["name"]]
        np.savez(CACHE, scores=scores)

    thresholds = np.linspace(0.02, 0.9, 89)
    metrics = [
        classification_metrics(y[tr], predict_labels(scores, t)) for t in thresholds
    ]
    plt.rcParams.update({"font.size": 8})
    plt.figure(figsize=(3.4, 2.1))
    for key, label in [
        ("f1", "F1"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("accuracy", "Accuracy"),
    ]:
        plt.plot(thresholds, [m[key] for m in metrics], label=label)
    plt.axvline(SELECTED_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
    plt.xlabel("Decision threshold on $\\sigma(x^\\top w)$")
    plt.ylabel("Out-of-fold score")
    plt.ylim(0, 1)
    plt.legend(
        fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False
    )
    plt.tight_layout(pad=0.2)
    plt.savefig("report/figures/threshold.pdf")
    for t in (SELECTED_THRESHOLD, 0.5):
        m = classification_metrics(y[tr], predict_labels(scores, t))
        print(
            f"t={t:.2f}: F1={m['f1']:.4f} P={m['precision']:.3f}"
            f" R={m['recall']:.3f} acc={m['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
