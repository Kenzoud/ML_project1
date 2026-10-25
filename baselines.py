"""Compare baseline models on a stratified 80/20 train/validation split.

Usage: python baselines.py
Writes outputs/baselines.json and prints a results table.

Preprocessing is fitted on the training part only. Regularization strength
and decision threshold are both chosen on the validation part, so these
validation scores are slightly optimistic; cross-validation comes later.
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
from implementations import (
    least_squares,
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from preprocessing import fit_preprocessor, load_data, transform

SEED = 42
THRESHOLDS = np.linspace(0.05, 0.95, 91)
LAMBDAS_RIDGE = [1e-5, 1e-4, 1e-3, 1e-2]
LAMBDAS_LOGREG = [1e-5, 1e-4, 1e-3]
GAMMA = 0.5
MAX_ITERS = 500


def evaluate(name, y_val, scores, extra=None):
    """Tune the threshold on validation scores and collect metrics."""
    t, _ = best_f1_threshold(y_val, scores, THRESHOLDS)
    m = classification_metrics(y_val, predict_labels(scores, t))
    m.update(
        {"model": name, "threshold": float(t), "auc": float(roc_auc(y_val, scores))}
    )
    m.update(extra or {})
    print(
        f"{name:<28} F1={m['f1']:.4f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
        f"  acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}  t={t:.2f}"
    )
    return m


def main():
    """Run all baselines and save their validation metrics."""
    x, _, y, _, _, names = load_data()
    tr, val = stratified_split(y, val_ratio=0.2, seed=SEED)
    params = fit_preprocessor(x[tr], names)
    tx_tr, tx_val = transform(x[tr], params), transform(x[val], params)
    y_tr, y_val = y[tr], y[val]
    print(f"train {tx_tr.shape}, val {tx_val.shape}, positives {y_tr.mean():.4f}")
    w0 = np.zeros(tx_tr.shape[1])
    results = []

    m = classification_metrics(y_val, np.zeros_like(y_val))
    m.update({"model": "majority (all negative)"})
    print(f"{m['model']:<28} F1={m['f1']:.4f}  acc={m['accuracy']:.4f}")
    results.append(m)

    w, _ = least_squares(y_tr, tx_tr)
    results.append(evaluate("least squares", y_val, tx_val @ w))

    ridge = [
        evaluate(
            f"ridge lambda={l:g}", y_val, tx_val @ ridge_regression(y_tr, tx_tr, l)[0]
        )
        for l in LAMBDAS_RIDGE
    ]
    results += ridge

    start = time.time()
    w, loss = logistic_regression(y_tr, tx_tr, w0, MAX_ITERS, GAMMA)
    info = {"train_loss": float(loss), "seconds": round(time.time() - start, 1)}
    results.append(evaluate("logistic", y_val, sigmoid(tx_val @ w), info))

    for l in LAMBDAS_LOGREG:
        w, loss = reg_logistic_regression(y_tr, tx_tr, l, w0, MAX_ITERS, GAMMA)
        info = {"train_loss": float(loss), "lambda": l}
        results.append(
            evaluate(f"reg logistic lambda={l:g}", y_val, sigmoid(tx_val @ w), info)
        )

    with open("outputs/baselines.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
