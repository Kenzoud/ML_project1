"""Infer how each BRFSS column should be cleaned and encoded.

BRFSS codes non-answers inside the value range, with a width that depends on
the question: 7/9 for one-digit answers, 77/99 for two digits, 777/999 for
three, 7777/9999 for four ("don't know" / "refused"). For day counts, 88 (or
888) means "none" and is mapped to 0. These rules are applied automatically;
OVERRIDES holds the columns where they are wrong, checked against the 2015
codebook (https://www.cdc.gov/brfss/annual_data/annual_2015.html).

Each column gets a spec:
    kind: "drop", "categorical" (one-hot) or "continuous" (numeric)
    missing_codes: values treated as missing
    zero_codes: values mapped to 0
"""

import csv

import numpy as np

ADMIN_COLUMNS = [
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
    "_PSU", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_DUALUSE",
    "_DUALCOR", "_LLCPWT",
]  # fmt: skip
SUPERSEDED_COLUMNS = [
    "WEIGHT2", "HEIGHT3",  # WTKG3, HTM4, _BMI5
    "ALCDAY5",  # DROCDY3_, _DRNKWEK
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",  # _FRUTSUM, _VEGESUM
    "EXEROFT1", "EXERHMM1", "EXEROFT2", "EXERHMM2", "STRENGTH",  # PA1MIN_, STRFREQ_
]  # fmt: skip

# Columns where the automatic rules are wrong.
OVERRIDES = {
    "_AGEG5YR": {"missing_codes": [14]},  # 14 = don't know / refused
    "DROCDY3_": {"kind": "continuous", "missing_codes": [900]},
    "_DRNKWEK": {"kind": "continuous", "missing_codes": [99900]},
    "MAXVO2_": {"kind": "continuous", "missing_codes": [999]},
    "FC60_": {"kind": "continuous", "missing_codes": [999]},
    "STRFREQ_": {"kind": "continuous", "missing_codes": [99]},
    "FTJUDA1_": {"kind": "continuous", "missing_codes": []},
    "_AGE80": {"kind": "continuous", "missing_codes": []},
    "HTIN4": {"missing_codes": [], "zero_codes": []},  # inches: 77, 88 are heights
    "DIABAGE2": {"missing_codes": [98, 99], "zero_codes": []},  # 77, 88 are ages
}

# A column with more distinct valid values than this is treated as continuous.
MAX_CATEGORIES = 10


def infer_spec(values):
    """Apply the automatic BRFSS coding rules to one column.

    Args:
        values: 1D numpy array, the observed (non-NaN) training values.

    Returns:
        dict with "kind", "missing_codes" and "zero_codes".
    """
    if len(values) == 0:
        return {"kind": "drop", "missing_codes": [], "zero_codes": []}
    vmax = values.max()
    integer = np.all(values == np.round(values))
    missing, zero = [], []
    if integer and vmax <= 9:
        missing = [9]
        if not np.any(values == 6):  # 7 is a real answer only after 1..6
            missing.append(7)
    elif integer and vmax <= 99:
        missing, zero = [77, 99], [88]
    elif integer and vmax <= 999:
        missing, zero = [777, 999], [888]
    elif integer and vmax <= 9999:
        missing, zero = [7777, 9999], [8888]
    valid = values[~np.isin(values, missing)]
    valid = np.where(np.isin(valid, zero), 0, valid)
    n_valid = len(np.unique(valid))
    if n_valid <= 1:
        kind = "drop"
    elif integer and n_valid <= MAX_CATEGORIES:
        kind = "categorical"
    else:
        kind = "continuous"
    present = lambda codes: [c for c in codes if np.any(values == c)]
    return {
        "kind": kind,
        "missing_codes": present(missing),
        "zero_codes": present(zero),
    }


def infer_feature_types(x, feature_names):
    """Build the spec of every column from training data.

    Args:
        x: numpy array of shape (N, D), raw training features.
        feature_names: list of D column names.

    Returns:
        dict mapping column name to its spec.
    """
    specs = {}
    for j, name in enumerate(feature_names):
        if name in ADMIN_COLUMNS or name in SUPERSEDED_COLUMNS:
            specs[name] = {"kind": "drop", "missing_codes": [], "zero_codes": []}
            continue
        col = x[:, j]
        specs[name] = infer_spec(col[~np.isnan(col)])
        specs[name].update(OVERRIDES.get(name, {}))
    return specs


def write_spec_table(x, y, feature_names, path):
    """Write the inferred specs with summary statistics to a CSV for review."""
    specs = infer_feature_types(x, feature_names)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["feature", "kind", "missing_codes", "zero_codes", "nan_frac",
             "n_values", "min", "max", "pos_rate_if_nan", "pos_rate_if_observed"]
        )  # fmt: skip
        for j, name in enumerate(feature_names):
            col, s = x[:, j], specs[name]
            nan = np.isnan(col)
            obs = col[~nan]
            w.writerow(
                [
                    name,
                    s["kind"],
                    " ".join(f"{c:g}" for c in s["missing_codes"]),
                    " ".join(f"{c:g}" for c in s["zero_codes"]),
                    f"{nan.mean():.3f}",
                    len(np.unique(obs)),
                    f"{obs.min():g}" if len(obs) else "",
                    f"{obs.max():g}" if len(obs) else "",
                    f"{y[nan].mean():.3f}" if nan.any() else "",
                    f"{y[~nan].mean():.3f}" if (~nan).any() else "",
                ]
            )
    return specs


if __name__ == "__main__":
    from preprocessing import load_data

    x_train, _, y_train, _, _, names = load_data()
    specs = write_spec_table(x_train, y_train, names, "docs/feature_types.csv")
    kinds = [s["kind"] for s in specs.values()]
    print({k: kinds.count(k) for k in sorted(set(kinds))})
