"""Data loading and feature preprocessing.

Interface shared by every model:
    params = fit_preprocessor(x_train, feature_names)   # training rows only
    tx = transform(x, params)                           # train, validation, test

The current version is the *baseline*: it treats every column as numeric,
keeping BRFSS special codes (7, 9, 77, ...) as plain numbers. Improving that is
the first planned ablation.
"""

import csv
import os

import numpy as np

from helpers import load_csv_data

# Survey administration and identifier columns: they describe the interview,
# not the respondent, so they are dropped.
ADMIN_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
]


def load_data(data_dir="data/dataset", cache_path="data/cache.npz"):
    """Load the dataset, caching the parsed arrays (CSV parsing takes ~1 min).

    Args:
        data_dir: folder with x_train.csv, y_train.csv and x_test.csv.
        cache_path: .npz file written on the first call and read afterwards.

    Returns:
        x_train (N, 321), x_test (M, 321), y_train (N,) in {0, 1},
        train_ids (N,), test_ids (M,), feature_names (list of 321 str).
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        x_train, x_test, y_pm, train_ids, test_ids = (
            d["x_train"],
            d["x_test"],
            d["y_train"],
            d["train_ids"],
            d["test_ids"],
        )
    else:
        x_train, x_test, y_pm, train_ids, test_ids = load_csv_data(data_dir)
        np.savez(
            cache_path,
            x_train=x_train,
            x_test=x_test,
            y_train=y_pm,
            train_ids=train_ids,
            test_ids=test_ids,
        )
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        feature_names = next(csv.reader(f))[1:]  # drop the Id column
    y_train = (y_pm == 1).astype(float)
    return x_train, x_test, y_train, train_ids, test_ids, feature_names


def fit_preprocessor(x, feature_names, max_nan_frac=0.5):
    """Learn the baseline preprocessing parameters from training rows only.

    Steps: drop admin columns and columns with more than max_nan_frac missing
    values, impute NaN with the training median, drop constant columns, and
    standardize with the training mean and standard deviation.

    Args:
        x: numpy array of shape (N, D), raw training features.
        feature_names: list of D column names.
        max_nan_frac: maximum fraction of NaN for a column to be kept.

    Returns:
        dict of parameters consumed by transform().
    """
    nan_frac = np.isnan(x).mean(axis=0)
    keep = (nan_frac <= max_nan_frac) & ~np.isin(feature_names, ADMIN_COLUMNS)
    cols = np.flatnonzero(keep)
    medians = np.nanmedian(x[:, cols], axis=0)
    x_imp = _impute(x[:, cols], medians)
    std = x_imp.std(axis=0)
    nonconstant = std > 0
    cols, medians, std = cols[nonconstant], medians[nonconstant], std[nonconstant]
    mean = x_imp[:, nonconstant].mean(axis=0)
    return {
        "columns": cols,
        "names": [feature_names[i] for i in cols],
        "medians": medians,
        "mean": mean,
        "std": std,
    }


def transform(x, params):
    """Apply fitted preprocessing and prepend a bias column.

    Args:
        x: numpy array of shape (N, D), raw features (same columns as at fit time).
        params: output of fit_preprocessor().

    Returns:
        tx: numpy array of shape (N, 1 + number of kept columns).
    """
    x_sel = _impute(x[:, params["columns"]], params["medians"])
    x_std = (x_sel - params["mean"]) / params["std"]
    return np.c_[np.ones(len(x)), x_std]


def _impute(x, values):
    """Return a copy of x with NaN in column j replaced by values[j]."""
    x = x.copy()
    rows, cols = np.nonzero(np.isnan(x))
    x[rows, cols] = values[cols]
    return x
