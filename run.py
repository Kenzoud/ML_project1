"""Reproduce the selected AIcrowd submission.

Usage (from the repository root, with data in data/dataset/):
    python run.py

Model: "full" preprocessing (preprocessing.CONFIGS) + L2-regularized logistic
regression with gradient descent. The hyperparameters below were selected by
model_selection.py (5-fold stratified cross-validation on 80% of the labeled
rows; see README):
    LAMBDA: best mean fold F1 (all lambda <= 1e-3 are within 0.0005 F1).
    THRESHOLD: F1-maximizing threshold on the pooled out-of-fold scores.
run.py fits preprocessing and the model on all labeled rows, predicts the test
set at THRESHOLD and writes the CSV. Everything is deterministic (full-batch
gradient descent from zero weights).
"""

import numpy as np

from helpers import create_csv_submission
from implementations import reg_logistic_regression, sigmoid
from preprocessing import fit_preprocessor, load_data, transform

PREPROCESSING = "full"
LAMBDA = 1e-4
GAMMA = 0.5
MAX_ITERS = 500
THRESHOLD = 0.20
SUBMISSION_PATH = "outputs/submission.csv"


def main():
    """Fit on all labeled rows, predict the test set and write the submission."""
    x_train, x_test, y_train, _, test_ids, names = load_data()
    params = fit_preprocessor(x_train, names, PREPROCESSING)
    tx = transform(x_train, params)
    w, loss = reg_logistic_regression(
        y_train, tx, LAMBDA, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA
    )
    print(f"trained on {tx.shape[0]} rows, {tx.shape[1]} features, loss={loss:.5f}")

    test_scores = sigmoid(transform(x_test, params) @ w)
    y_pred = np.where(test_scores >= THRESHOLD, 1, -1)
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
    print(
        f"wrote {SUBMISSION_PATH}: {len(y_pred)} rows, {np.mean(y_pred == 1):.4f} positive"
    )


if __name__ == "__main__":
    main()
