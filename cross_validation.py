"""K-fold cross-validation with preprocessing refitted inside every fold.

A model is a dict:
    {"name": str, "fit": callable(y, tx) -> w, "link": callable(scores) -> scores}
where "link" maps tx @ w to the score that is thresholded (sigmoid for
logistic regression, identity for least squares / ridge).

cross_validate returns out-of-fold (OOF) scores: every training row is scored
by a model that never saw it, with preprocessing fitted without it either.
summarize then picks one threshold on the pooled OOF scores and reports the
metrics of each fold at that threshold.
"""

import numpy as np

from evaluation import (
    best_f1_threshold,
    classification_metrics,
    predict_labels,
    roc_auc,
    stratified_kfold,
)
from implementations import (
    logistic_regression,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from preprocessing import fit_preprocessor, transform

THRESHOLDS = np.linspace(0.05, 0.95, 91)


def logistic_model(lambda_, gamma=0.5, max_iters=500):
    """(Regularized) logistic regression by gradient descent, starting from 0."""

    def fit(y, tx):
        w0 = np.zeros(tx.shape[1])
        if lambda_ == 0:
            return logistic_regression(y, tx, w0, max_iters, gamma)[0]
        return reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)[0]

    return {"name": f"logistic lambda={lambda_:g}", "fit": fit, "link": sigmoid}


def ridge_model(lambda_):
    """Ridge regression on {0, 1} labels; scores are thresholded directly."""

    def fit(y, tx):
        return ridge_regression(y, tx, lambda_)[0]

    return {"name": f"ridge lambda={lambda_:g}", "fit": fit, "link": lambda s: s}


def cross_validate(x, y, feature_names, config, models, k=5, seed=42):
    """Compute out-of-fold scores of several models for one preprocessing config.

    Preprocessing is fitted once per fold (on the training folds only) and
    shared by all models.

    Args:
        x: numpy array (N, D), raw features.
        y: numpy array (N,), labels in {0, 1}.
        feature_names: list of D column names.
        config: preprocessing config name (see preprocessing.CONFIGS).
        models: list of model dicts (logistic_model, ridge_model).
        k: number of folds.
        seed: seed of the stratified fold assignment.

    Returns:
        oof: dict model name -> numpy array (N,) of out-of-fold scores.
        fold_of: numpy array (N,), index of the fold each row was scored in.
    """
    folds = stratified_kfold(y, k=k, seed=seed)
    fold_of = np.empty(len(y), dtype=int)
    oof = {m["name"]: np.empty(len(y)) for m in models}
    for i, val in enumerate(folds):
        fold_of[val] = i
        train = np.setdiff1d(np.arange(len(y)), val)
        params = fit_preprocessor(x[train], feature_names, config)
        tx_tr, tx_val = transform(x[train], params), transform(x[val], params)
        for m in models:
            w = m["fit"](y[train], tx_tr)
            oof[m["name"]][val] = m["link"](tx_val @ w)
    return oof, fold_of


def summarize(y, scores, fold_of, thresholds=THRESHOLDS):
    """Metrics of out-of-fold scores: pooled threshold, then per-fold mean/std.

    Returns:
        dict with the chosen "threshold", pooled "auc", and "<metric>_mean" /
        "<metric>_std" over folds for f1, precision, recall, accuracy and auc.
    """
    t, _ = best_f1_threshold(y, scores, thresholds)
    per_fold = []
    for i in np.unique(fold_of):
        rows = fold_of == i
        m = classification_metrics(y[rows], predict_labels(scores[rows], t))
        m["auc"] = roc_auc(y[rows], scores[rows])
        per_fold.append(m)
    out = {"threshold": float(t), "auc": float(roc_auc(y, scores))}
    for key in ("f1", "precision", "recall", "accuracy", "auc"):
        values = np.array([m[key] for m in per_fold])
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    out["f1_folds"] = [float(m["f1"]) for m in per_fold]
    return out
