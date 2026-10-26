"""Data loading and feature preprocessing.

Interface shared by every model:
    params = fit_preprocessor(x_train, feature_names, config)  # training rows only
    tx = transform(x, params)                                  # train, validation, test

All parameters (codes, levels, medians, means, ...) are learned from the rows
passed to fit_preprocessor, so validation and test rows never influence them.

CONFIGS lists the preprocessing variants compared in the ablation study, from
the original baseline ("baseline") to the full pipeline ("full").
"""

import csv
import hashlib
import os

import numpy as np

from feature_types import infer_feature_types
from helpers import load_csv_data

BASELINE_DROP = ["_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE",
                 "SEQNO", "_PSU"]  # fmt: skip

# Categorical levels rarer than this (fraction of training rows) are merged
# into the reference level, to avoid near-empty one-hot columns.
MIN_LEVEL_FRAC = 0.001

# Percentiles used to clip continuous features (limits heavy tails such as
# minutes of activity per week).
CLIP_PERCENTILES = (0.5, 99.5)

CONFIGS = {
    # Every column numeric, BRFSS codes kept as numbers, >50% missing dropped.
    "baseline": {
        "clean_codes": False,
        "max_nan_frac": 0.5,
        "one_hot": False,
        "missing_indicators": False,
        "clip": False,
        "dedupe": False,
    },
    # + special codes as missing, 88 -> 0, design weights and raw duplicates dropped.
    "clean_codes": {
        "clean_codes": True,
        "max_nan_frac": 0.5,
        "one_hot": False,
        "missing_indicators": False,
        "clip": False,
        "dedupe": False,
    },
    # + one-hot categorical columns, missing as its own level, all columns kept.
    "one_hot": {
        "clean_codes": True,
        "max_nan_frac": 1.0,
        "one_hot": True,
        "missing_indicators": False,
        "clip": False,
        "dedupe": True,
    },
    # + missing indicators and clipping for continuous columns.
    "full": {
        "clean_codes": True,
        "max_nan_frac": 1.0,
        "one_hot": True,
        "missing_indicators": True,
        "clip": True,
        "dedupe": True,
    },
}


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


def fit_preprocessor(x, feature_names, config="full"):
    """Learn preprocessing parameters from training rows only.

    Steps, depending on the config (see CONFIGS):
      1. Select columns: drop admin/weight/superseded columns and, optionally,
         columns with too many missing values.
      2. Clean codes: BRFSS "don't know"/"refused" codes become NaN and "none"
         codes (88, 888) become 0 (feature_types.py).
      3. Encode: one-hot for categorical columns, with missing as a level and
         the most frequent level as reference (dropped); numeric otherwise,
         optionally clipped; median-imputed, or, with a missing indicator,
         mean-imputed and standardized on observed rows.
      4. Drop constant and duplicate encoded columns, then standardize.

    Args:
        x: numpy array of shape (N, D), raw training features.
        feature_names: list of D column names.
        config: name of an entry in CONFIGS, or a dict with the same keys.

    Returns:
        dict of parameters consumed by transform().
    """
    cfg = CONFIGS[config] if isinstance(config, str) else config
    if cfg["clean_codes"]:
        specs = infer_feature_types(x, feature_names)
    else:
        specs = {
            n: {"kind": "drop" if n in BASELINE_DROP else "continuous"}
            for n in feature_names
        }
    columns = []
    for j, name in enumerate(feature_names):
        spec = specs[name]
        if spec["kind"] == "drop":
            continue
        col = _clean(x[:, j], spec)
        nan = np.isnan(col)
        if nan.mean() > cfg["max_nan_frac"] or nan.all():
            continue
        if spec["kind"] == "categorical" and cfg["one_hot"]:
            columns.append(_fit_categorical(j, name, spec, col))
        else:
            columns.append(_fit_numeric(j, name, spec, col, cfg))

    params = {"columns": columns}
    enc, names = _encode(x, params)
    keep = enc.std(axis=0) > 0
    if cfg["dedupe"]:
        keep &= _first_occurrences(enc)
    enc = enc[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    mean, std = enc.mean(axis=0), enc.std(axis=0)
    # Numeric columns with an indicator are already standardized on observed rows.
    prescaled = {
        c["name"] for c in columns if c["kind"] == "numeric" and c["indicator"]
    }
    for i, n in enumerate(names):
        if n in prescaled:
            mean[i], std[i] = 0.0, 1.0
    params.update({"keep": keep, "names": names, "mean": mean, "std": std})
    return params


def transform(x, params):
    """Apply fitted preprocessing and prepend a bias column.

    Args:
        x: numpy array of shape (N, D), raw features (same columns as at fit time).
        params: output of fit_preprocessor().

    Returns:
        tx: numpy array of shape (N, 1 + number of encoded features).
    """
    enc, _ = _encode(x, params)
    tx = np.empty((len(x), 1 + len(params["mean"])))
    tx[:, 0] = 1.0
    tx[:, 1:] = enc[:, params["keep"]]
    tx[:, 1:] -= params["mean"]
    tx[:, 1:] /= params["std"]
    return tx


def _clean(col, spec):
    """Return a float copy of col with missing codes as NaN and zero codes as 0."""
    col = col.astype(float)
    col[np.isin(col, spec.get("missing_codes", []))] = np.nan
    col[np.isin(col, spec.get("zero_codes", []))] = 0.0
    return col


def _fit_categorical(j, name, spec, col):
    """Choose the one-hot levels of a cleaned categorical column."""
    n = len(col)
    values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
    levels = list(values)
    freqs = list(counts / n)
    n_missing = np.isnan(col).sum()
    if n_missing:
        levels.append(np.nan)
        freqs.append(n_missing / n)
    reference = int(np.argmax(freqs))
    kept = [
        lv for i, (lv, f) in enumerate(zip(levels, freqs))
        if i != reference and f >= MIN_LEVEL_FRAC
    ]  # fmt: skip
    return {
        "index": j,
        "name": name,
        "spec": spec,
        "kind": "categorical",
        "levels": kept,
    }


def _fit_numeric(j, name, spec, col, cfg):
    """Compute clipping bounds, median and indicator flag of a numeric column."""
    observed = col[~np.isnan(col)]
    lo, hi = -np.inf, np.inf
    if cfg["clip"]:
        lo, hi = np.percentile(observed, CLIP_PERCENTILES)
    observed = np.clip(observed, lo, hi)
    indicator = cfg["missing_indicators"] and len(observed) < len(col)
    if indicator:
        # Standardize on observed rows and fill missing with their mean (0):
        # the indicator carries the missingness. Standardizing after imputation
        # would blow up columns observed for very few rows.
        fill, center, scale = np.mean(observed), np.mean(observed), np.std(observed)
    else:
        fill, center, scale = np.median(observed), 0.0, 1.0
    return {
        "index": j,
        "name": name,
        "spec": spec,
        "kind": "numeric",
        "clip": (lo, hi),
        "fill": fill,
        "center": center,
        "scale": scale if scale > 0 else 1.0,
        "indicator": indicator,
    }


def _encode(x, params):
    """Encode raw features column by column, before standardization.

    Returns:
        enc: numpy array of shape (N, number of encoded features).
        names: list of encoded feature names, e.g. "GENHLTH=2" or "BMI_missing".
    """
    blocks, names = [], []
    for c in params["columns"]:
        col = _clean(x[:, c["index"]], c["spec"])
        nan = np.isnan(col)
        if c["kind"] == "categorical":
            for lv in c["levels"]:
                blocks.append(nan if np.isnan(lv) else col == lv)
                names.append(
                    f"{c['name']}=" + ("missing" if np.isnan(lv) else f"{lv:g}")
                )
        else:
            lo, hi = c["clip"]
            value = np.where(nan, c["fill"], np.clip(col, lo, hi))
            blocks.append((value - c["center"]) / c["scale"])
            names.append(c["name"])
            if c["indicator"]:
                blocks.append(nan)
                names.append(f"{c['name']}_missing")
    enc = np.empty((len(x), len(blocks)))
    for i, b in enumerate(blocks):
        enc[:, i] = b
    return enc, names


def _first_occurrences(enc):
    """Boolean mask that is False for columns identical to an earlier column.

    Skip-logic makes many missing indicators identical (e.g. every follow-up
    question of the diabetes module is missing for the same respondents).
    """
    seen = set()
    mask = np.zeros(enc.shape[1], dtype=bool)
    for i in range(enc.shape[1]):
        h = hashlib.sha1(np.ascontiguousarray(enc[:, i]).tobytes()).hexdigest()
        mask[i] = h not in seen
        seen.add(h)
    return mask
