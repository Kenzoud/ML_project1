"""Unit tests for evaluation.py."""

import numpy as np

import evaluation as ev


def test_stratified_split_partitions_and_keeps_class_ratio():
    y = np.r_[np.zeros(907), np.ones(93)]
    tr, val = ev.stratified_split(y, val_ratio=0.2, seed=0)
    np.testing.assert_array_equal(np.sort(np.r_[tr, val]), np.arange(1000))
    assert abs(y[val].mean() - y.mean()) < 0.01
    assert abs(len(val) - 200) <= 1


def test_stratified_split_is_seeded():
    y = np.r_[np.zeros(50), np.ones(10)]
    a = ev.stratified_split(y, seed=3)
    b = ev.stratified_split(y, seed=3)
    np.testing.assert_array_equal(a[1], b[1])


def test_kfold_covers_every_sample_once_with_uneven_sizes():
    y = np.r_[np.zeros(101), np.ones(10)]
    folds = ev.stratified_kfold(y, k=3, seed=0)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(111))
    assert [int(y[f].sum()) for f in folds] in ([4, 3, 3], [3, 4, 3], [3, 3, 4])


def test_classification_metrics_known_values():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = ev.classification_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    np.testing.assert_allclose(
        [m["accuracy"], m["precision"], m["recall"], m["f1"]],
        [6 / 8, 2 / 3, 2 / 3, 2 / 3],
    )


def test_metrics_without_positive_predictions():
    m = ev.classification_metrics(np.array([1, 0, 0]), np.zeros(3))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_roc_auc_known_values_and_ties():
    y = np.array([0, 0, 1, 1])
    np.testing.assert_allclose(ev.roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])), 0.75)
    np.testing.assert_allclose(ev.roc_auc(y, np.ones(4)), 0.5)
    np.testing.assert_allclose(ev.roc_auc(y, np.array([0.0, 0.1, 0.2, 0.3])), 1.0)


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.35, 0.9])
    t, f1 = ev.best_f1_threshold(y, scores, np.array([0.15, 0.32, 0.5]))
    assert t == 0.32 and f1 == 1.0
