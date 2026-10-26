"""Unit tests for preprocessing.py on a small synthetic BRFSS-like table."""

import numpy as np
import pytest

from preprocessing import CONFIGS, fit_preprocessor, transform

NAMES = ["SEQNO", "GENHLTH", "PHYSHLTH", "_BMI5"]


@pytest.fixture()
def raw():
    rng = np.random.default_rng(0)
    n = 2000
    genhlth = rng.choice(
        [1, 2, 3, 4, 5, 7, 9], size=n, p=[0.2, 0.3, 0.3, 0.1, 0.06, 0.02, 0.02]
    )
    physhlth = rng.choice(np.r_[1:31, 77, 88, 99], size=n)
    bmi = rng.normal(27, 5, size=n)
    bmi[rng.random(n) < 0.1] = np.nan
    return np.c_[np.arange(n), genhlth, physhlth, bmi].astype(float)


@pytest.mark.parametrize("config", CONFIGS)
def test_transform_shape_and_finiteness(config, raw):
    params = fit_preprocessor(raw, NAMES, config)
    tx = transform(raw, params)
    assert tx.shape == (len(raw), 1 + len(params["names"]))
    assert np.all(np.isfinite(tx))
    np.testing.assert_array_equal(tx[:, 0], 1.0)
    assert not any(n.startswith("SEQNO") for n in params["names"])


def test_full_config_encodes_levels_codes_and_indicators(raw):
    params = fit_preprocessor(raw, NAMES, "full")
    names = params["names"]
    # GENHLTH one-hot: 7 and 9 merged into a "missing" level, mode (2 or 3) dropped.
    assert "GENHLTH=missing" in names
    assert "GENHLTH=7" not in names and "GENHLTH=9" not in names
    assert sum(n.startswith("GENHLTH=") for n in names) == 5
    assert "_BMI5_missing" in names and "PHYSHLTH_missing" in names


def test_zero_code_maps_to_zero_days(raw):
    params = fit_preprocessor(raw, NAMES, "clean_codes")
    col = params["names"].index("PHYSHLTH") + 1
    row88 = raw[:, 2] == 88
    row1 = raw[:, 2] == 1
    # 88 ("none") must encode below 1 day.
    tx = transform(raw, params)
    assert tx[row88, col].max() < tx[row1, col].min()


def test_parameters_come_from_fit_rows_only(raw):
    params = fit_preprocessor(raw[:1000], NAMES, "full")
    shifted = raw.copy()
    shifted[1000:, 3] += 100  # change BMI on rows not used for fitting
    np.testing.assert_array_equal(
        transform(raw[:1000], params), transform(shifted[:1000], params)
    )
    params2 = fit_preprocessor(shifted[:1000], NAMES, "full")
    np.testing.assert_array_equal(params["mean"], params2["mean"])


def test_unseen_level_maps_to_reference(raw):
    params = fit_preprocessor(raw, NAMES, "full")
    cols = [i + 1 for i, n in enumerate(params["names"]) if n.startswith("GENHLTH=")]
    reference = [v for v in (1, 2, 3, 4, 5) if f"GENHLTH={v}" not in params["names"]]
    assert len(reference) == 1
    new = raw[:2].copy()
    new[0, 1] = 6  # GENHLTH level never seen in training
    new[1, 1] = reference[0]
    tx = transform(new, params)
    np.testing.assert_array_equal(tx[0, cols], tx[1, cols])
