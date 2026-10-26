"""Unit tests for the BRFSS coding rules in feature_types.py."""

import numpy as np

from feature_types import infer_feature_types, infer_spec


def test_one_digit_answer_with_dont_know_and_refused():
    spec = infer_spec(np.array([1, 2, 3, 4, 5, 7, 9.0]))  # e.g. GENHLTH
    assert spec["kind"] == "categorical"
    assert sorted(spec["missing_codes"]) == [7, 9]


def test_seven_is_a_real_answer_when_six_exists():
    spec = infer_spec(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9.0]))  # e.g. EMPLOY1
    assert spec["missing_codes"] == [9]


def test_day_count_maps_88_to_zero():
    values = np.r_[np.arange(1, 31), 77, 88, 99].astype(float)  # e.g. PHYSHLTH
    spec = infer_spec(values)
    assert spec["kind"] == "continuous"
    assert spec["missing_codes"] == [77, 99]
    assert spec["zero_codes"] == [88]


def test_non_integer_column_has_no_codes():
    spec = infer_spec(np.array([12.5, 30.1, 45.0, 22.2]))
    assert spec["missing_codes"] == [] and spec["zero_codes"] == []


def test_constant_after_removing_codes_is_dropped():
    assert infer_spec(np.array([1, 1, 9.0]))["kind"] == "drop"


def test_admin_columns_and_overrides():
    x = np.array([[1.0, 77.0], [2.0, 60.0], [3.0, 88.0]])
    specs = infer_feature_types(x, ["SEQNO", "HTIN4"])
    assert specs["SEQNO"]["kind"] == "drop"
    assert specs["HTIN4"]["missing_codes"] == []
    assert specs["HTIN4"]["zero_codes"] == []
