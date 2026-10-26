"""Ablation study of the preprocessing steps (see preprocessing.CONFIGS).

Usage: python ablation.py
Writes outputs/ablation.json and prints a results table.

Each config is fitted on the same stratified 80% training split and scored on
the same 20% validation split. Models: logistic regression (gradient descent,
checkpoints at several step counts to check convergence) and ridge regression.
The decision threshold maximizing F1 is chosen on the validation split for
every model, so all rows share the same (slightly optimistic) protocol.
"""

import json
import time

import numpy as np

from evaluation import (
    best_f1_threshold,
    classification_metrics,
    predict_labels,
    roc_auc,
    stratified_split,
)
from implementations import logistic_regression, ridge_regression, sigmoid
from preprocessing import CONFIGS, fit_preprocessor, load_data, transform

SEED = 42
GAMMA = 0.5
CHECKPOINTS = [500, 1000, 2000]
RIDGE_LAMBDA = 1e-5
THRESHOLDS = np.linspace(0.05, 0.95, 91)


def score(y_val, scores):
    """Validation metrics at the F1-maximizing threshold."""
    t, _ = best_f1_threshold(y_val, scores, THRESHOLDS)
    m = classification_metrics(y_val, predict_labels(scores, t))
    m.update({"threshold": float(t), "auc": float(roc_auc(y_val, scores))})
    return m


def main():
    """Run every preprocessing config with logistic and ridge regression."""
    x, _, y, _, _, names = load_data()
    tr, val = stratified_split(y, val_ratio=0.2, seed=SEED)
    results = []
    for config in CONFIGS:
        params = fit_preprocessor(x[tr], names, config)
        tx_tr, tx_val = transform(x[tr], params), transform(x[val], params)
        d = tx_tr.shape[1]

        w_r, _ = ridge_regression(y[tr], tx_tr, RIDGE_LAMBDA)
        m = score(y[val], tx_val @ w_r)
        m.update({"config": config, "model": "ridge", "features": d})
        results.append(m)
        print(
            f"{config:<12} D={d:4d} ridge          F1={m['f1']:.4f} AUC={m['auc']:.4f}"
        )

        w, done, start = np.zeros(d), 0, time.time()
        for steps in CHECKPOINTS:
            w, loss = logistic_regression(y[tr], tx_tr, w, steps - done, GAMMA)
            done = steps
            m = score(y[val], sigmoid(tx_val @ w))
            m.update(
                {
                    "config": config,
                    "model": f"logistic {steps} steps",
                    "features": d,
                    "train_loss": float(loss),
                    "seconds": round(time.time() - start, 1),
                }
            )
            results.append(m)
            print(
                f"{config:<12} D={d:4d} logistic {steps:4d}  F1={m['f1']:.4f}"
                f" P={m['precision']:.3f} R={m['recall']:.3f} acc={m['accuracy']:.4f}"
                f" AUC={m['auc']:.4f} loss={loss:.5f} t={m['threshold']:.2f}",
                flush=True,
            )

    with open("outputs/ablation.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
