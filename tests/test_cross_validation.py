"""Unit tests for cross_validation.py on synthetic data."""

import numpy as np
import pytest

from cross_validation import cross_validate, logistic_model, ridge_model, summarize

NAMES = ["SEQNO", "GENHLTH", "_BMI5"]


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    n = 1500
    genhlth = rng.choice([1, 2, 3, 4, 5, 9], size=n)
    bmi = rng.normal(27, 5, size=n)
    logit = -3 + 0.8 * (genhlth == 5) + 0.6 * (genhlth == 4) + 0.08 * (bmi - 27)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    x = np.c_[np.arange(n), genhlth, bmi].astype(float)
    return x, y


def test_every_row_scored_once_by_each_model(data):
    x, y = data
    models = [logistic_model(0, max_iters=50), ridge_model(1e-3)]
    oof, fold_of = cross_validate(x, y, NAMES, "full", models, k=3, seed=0)
    assert set(oof) == {m["name"] for m in models}
    assert sorted(np.unique(fold_of)) == [0, 1, 2]
    for scores in oof.values():
        assert scores.shape == y.shape and np.all(np.isfinite(scores))
    assert np.all((oof["logistic lambda=0"] > 0) & (oof["logistic lambda=0"] < 1))


def test_validation_rows_do_not_influence_their_fold(data):
    """Changing another validation row's features must not move row 7's score.

    Folds depend only on y, so they stay identical. If preprocessing or
    training used the validation fold, row 7's score would change.
    """
    x, y = data
    model = [logistic_model(0, max_iters=50)]
    oof, fold_of = cross_validate(x, y, NAMES, "full", model, k=3, seed=0)
    same_fold = np.flatnonzero(fold_of == fold_of[7])
    other = same_fold[same_fold != 7][0]
    x2 = x.copy()
    x2[other, 2] = 1000.0  # extreme BMI
    oof2, _ = cross_validate(x2, y, NAMES, "full", model, k=3, seed=0)
    assert oof["logistic lambda=0"][7] == oof2["logistic lambda=0"][7]


def test_summarize_reports_fold_statistics(data):
    x, y = data
    oof, fold_of = cross_validate(x, y, NAMES, "full", [ridge_model(1e-3)], k=3, seed=0)
    s = summarize(y, oof["ridge lambda=0.001"], fold_of)
    assert len(s["f1_folds"]) == 3
    np.testing.assert_allclose(s["f1_mean"], np.mean(s["f1_folds"]))
    assert 0.05 <= s["threshold"] <= 0.95
    assert 0.5 < s["auc"] <= 1.0
