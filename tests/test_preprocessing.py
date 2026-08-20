"""
Tests for webneuro_qc.preprocessing: normalize_missing_codes and
blank_strings_to_nan.

Run just this file:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.preprocessing import blank_strings_to_nan, normalize_missing_codes


def test_timeout_string_becomes_nan_and_is_flagged():
    df = pd.DataFrame({"emzcompk": [500, "TO", " to ", np.nan]})
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(df)

    assert clean_df.loc[0, "emzcompk"] == 500.0
    assert pd.isna(clean_df.loc[1, "emzcompk"])
    assert pd.isna(clean_df.loc[2, "emzcompk"])  # "  to  " matches case/whitespace-insensitively
    assert timeout_mask.loc[1, "emzcompk"]
    assert timeout_mask.loc[2, "emzcompk"]
    assert not unexpected_mask["emzcompk"].any()


def test_unexpected_non_numeric_string_is_flagged_not_dropped_silently():
    df = pd.DataFrame({"emzcompk": [500, "garbage", np.nan]})
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(df)

    assert pd.isna(clean_df.loc[1, "emzcompk"])
    assert not timeout_mask.loc[1, "emzcompk"]
    assert unexpected_mask.loc[1, "emzcompk"]


def test_numeric_columns_untouched():
    df = pd.DataFrame({"emzcompk": [500, 300, np.nan]})
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(df)

    assert clean_df["emzcompk"].tolist()[:2] == [500.0, 300.0]
    assert not timeout_mask["emzcompk"].any()
    assert not unexpected_mask["emzcompk"].any()


def test_empty_string_without_blank_strings_to_nan_is_flagged_unexpected():
    # documents the bug this fixes: an empty/whitespace string is NOT the
    # same as a truly blank CSV cell -- without cleaning first, it's
    # treated as garbage input, not ordinary missing data.
    df = pd.DataFrame({"vcrtne": [1200, "", " ", np.nan]})
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(df)

    assert pd.isna(clean_df.loc[1, "vcrtne"])
    assert pd.isna(clean_df.loc[2, "vcrtne"])
    assert not timeout_mask.loc[1, "vcrtne"]
    assert not timeout_mask.loc[2, "vcrtne"]
    assert unexpected_mask.loc[1, "vcrtne"]
    assert unexpected_mask.loc[2, "vcrtne"]


def test_blank_strings_to_nan_converts_empty_and_whitespace_only():
    df = pd.DataFrame({"vcrtne": [1200, "", " ", "\t\n", np.nan, "garbage"]})
    cleaned = blank_strings_to_nan(df)

    assert cleaned.loc[0, "vcrtne"] == 1200
    assert pd.isna(cleaned.loc[1, "vcrtne"])
    assert pd.isna(cleaned.loc[2, "vcrtne"])
    assert pd.isna(cleaned.loc[3, "vcrtne"])
    assert pd.isna(cleaned.loc[4, "vcrtne"])  # was already NaN
    assert cleaned.loc[5, "vcrtne"] == "garbage"  # not blank -- left alone


def test_blank_strings_to_nan_only_touches_given_columns():
    df = pd.DataFrame({"a": [""], "b": [""]})
    cleaned = blank_strings_to_nan(df, columns=["a"])

    assert pd.isna(cleaned.loc[0, "a"])
    assert cleaned.loc[0, "b"] == ""


def test_blank_strings_to_nan_then_normalize_avoids_unexpected_flag():
    # this is the fix: chaining blank_strings_to_nan() before
    # normalize_missing_codes() makes "" and " " behave exactly like a
    # cell that was empty in the raw CSV -- no timeout, no unexpected flag.
    df = pd.DataFrame({"vcrtne": [1200, "", " ", np.nan, "TO", "garbage"]})
    cleaned = blank_strings_to_nan(df)
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(cleaned)

    assert clean_df["vcrtne"].isna().tolist() == [False, True, True, True, True, True]
    assert not timeout_mask.loc[1, "vcrtne"]
    assert not timeout_mask.loc[2, "vcrtne"]
    assert timeout_mask.loc[4, "vcrtne"]  # "TO" still recognized as a timeout
    assert not unexpected_mask.loc[1, "vcrtne"]
    assert not unexpected_mask.loc[2, "vcrtne"]
    assert unexpected_mask.loc[5, "vcrtne"]  # real garbage is still flagged
