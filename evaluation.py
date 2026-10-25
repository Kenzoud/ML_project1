"""Data splitting and classification metrics, in NumPy only.

Labels are {0, 1} throughout (1 = MICHD). Predictions from a score vector are
obtained with predict_labels(scores, threshold).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_split(y, val_ratio=0.2, seed=42):
    """Split indices into train/validation sets with the same class proportions.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        val_ratio: fraction of each class sent to the validation set.
        seed: random seed.

    Returns:
        train_idx, val_idx: sorted integer index arrays partitioning range(N).
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_val = int(round(val_ratio * len(idx)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def stratified_kfold(y, k=5, seed=42):
    """Build k stratified folds that together cover every sample exactly once.

    Unlike the lab's build_k_indices, leftover samples are not dropped:
    np.array_split spreads them over the first folds.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        k: number of folds.
        seed: random seed.

    Returns:
        List of k sorted integer index arrays (the validation fold of each split).
    """
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        for fold, part in zip(folds, np.array_split(idx, k)):
            fold.append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def predict_labels(scores, threshold):
    """Turn real-valued scores into {0, 1} labels: 1 where score >= threshold."""
    return (scores >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    """Count true/false positives/negatives for {0, 1} labels.

    Returns:
        dict with integer entries "tp", "fp", "fn", "tn".
    """
    y_true = np.asarray(y_true) == 1
    y_pred = np.asarray(y_pred) == 1
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
    }


def classification_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall and F1 of the positive class.

    Precision, recall and F1 are defined as 0 when their denominator is 0.

    Returns:
        dict with "accuracy", "precision", "recall", "f1" and the confusion counts.
    """
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **c,
    }


def roc_auc(y_true, scores):
    """Area under the ROC curve via the Mann-Whitney U statistic (ties averaged).

    It measures how well scores rank positives above negatives, independently
    of any threshold.
    """
    y_true = np.asarray(y_true) == 1
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within groups of tied scores.
    _, start, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for s, c in zip(start[counts > 1], counts[counts > 1]):
        ranks[order[s : s + c]] = s + (c + 1) / 2
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    return (ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_f1_threshold(y_true, scores, thresholds):
    """Pick the threshold that maximizes F1 on (y_true, scores).

    Use it on validation data only; the chosen threshold is then fixed for test.

    Returns:
        best_threshold, best_f1.
    """
    f1s = [
        classification_metrics(y_true, predict_labels(scores, t))["f1"]
        for t in thresholds
    ]
    best = int(np.argmax(f1s))
    return thresholds[best], f1s[best]
