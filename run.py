"""Reproduce the selected AIcrowd submission.

Usage (from the repository root, with data in data/dataset/):
    python run.py

Current model: "full" preprocessing (preprocessing.CONFIGS) + logistic
regression with gradient descent.
    1. Fit preprocessing and the model on 80% of the training rows and choose
       the decision threshold that maximizes F1 on the remaining 20%.
    2. Refit preprocessing and the model on all training rows.
    3. Predict the test set with the chosen threshold and write the CSV.
Everything is deterministic (fixed seed, full-batch gradient descent).
"""

import numpy as np

from evaluation import (
    best_f1_threshold,
    classification_metrics,
    predict_labels,
    stratified_split,
)
from helpers import create_csv_submission
from implementations import logistic_regression, sigmoid
from preprocessing import fit_preprocessor, load_data, transform

SEED = 42
PREPROCESSING = "full"
GAMMA = 0.5
MAX_ITERS = 500
THRESHOLDS = np.linspace(0.05, 0.95, 91)
SUBMISSION_PATH = "outputs/submission.csv"


def fit_and_score(x_fit, y_fit, feature_names, x_score):
    """Fit preprocessing and logistic regression, return probabilities on x_score."""
    params = fit_preprocessor(x_fit, feature_names, PREPROCESSING)
    tx = transform(x_fit, params)
    w, _ = logistic_regression(y_fit, tx, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA)
    return sigmoid(transform(x_score, params) @ w)


def main():
    """Select the threshold on a validation split, refit on all data, predict test."""
    np.random.seed(SEED)
    x_train, x_test, y_train, _, test_ids, names = load_data()

    tr, val = stratified_split(y_train, val_ratio=0.2, seed=SEED)
    val_scores = fit_and_score(x_train[tr], y_train[tr], names, x_train[val])
    threshold, _ = best_f1_threshold(y_train[val], val_scores, THRESHOLDS)
    m = classification_metrics(y_train[val], predict_labels(val_scores, threshold))
    print(
        f"validation: threshold={threshold:.2f} F1={m['f1']:.4f} acc={m['accuracy']:.4f}"
    )

    test_scores = fit_and_score(x_train, y_train, names, x_test)
    y_pred = np.where(predict_labels(test_scores, threshold) == 1, 1, -1)
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
    print(
        f"wrote {SUBMISSION_PATH}: {len(y_pred)} rows, {np.mean(y_pred == 1):.4f} positive"
    )


if __name__ == "__main__":
    main()
