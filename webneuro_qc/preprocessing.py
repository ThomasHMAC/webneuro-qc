"""
PREPROCESSING -- normalize missing-data encodings to real NaN.

Problem: raw cells encode "missing" in more than one way, and none of
them are real NaN:
  - some tasks encode a timeout as the literal string "TO"
  - other variables (documented in the dictionary as "missing if none
    correct") are just left blank -- but "blank" in a CSV can mean an
    empty string "" or a whitespace-only string " ", not necessarily a
    truly empty cell
Left as-is, this breaks everything downstream:
  - .notna() reports these strings as present, so _present()/
    _both_present() wrongly treat a missing row as having real data
  - numeric comparisons (df["emzerrk"] <= df["emzoverk"]) either raise
    a TypeError or, if the column got silently object-dtyped, compare
    strings to numbers in a way pandas won't warn you about.

Fix: two-step normalization, applied before any checks run.
  1. blank_strings_to_nan() -- turn empty/whitespace-only string cells
     into real NaN, so they're treated as ordinary missing data instead
     of non-numeric junk.
  2. normalize_missing_codes() -- turn recognized timeout strings (e.g.
     "TO") into NaN and coerce the column to numeric. Any *other*
     non-numeric junk found along the way gets flagged separately
     (unexpected_value) instead of silently vanishing, so you don't
     mistake a real data error for a timeout.
"""

import numpy as np
import pandas as pd

DEFAULT_TIMEOUT_STRINGS = {"TO"}  # add variants here, e.g. "T.O.", "TIMEOUT"


def _is_string_cell(raw: pd.Series) -> pd.Series:
    """Bool mask, True where a cell literally holds a Python str."""
    return raw.map(lambda v: isinstance(v, str))


def blank_strings_to_nan(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Replace cells holding an empty or whitespace-only string (e.g. "", " ")
    with real NaN.

    Some variables (e.g. vcrtne, vcrtne2, g2avrtk -- documented in the
    dictionary as "missing if none correct") are left blank rather than
    given a sentinel like "TO". Run this before normalize_missing_codes()
    so those cells end up as ordinary missing data -- same as a cell that
    was empty in the raw CSV to begin with -- instead of tripping the
    unexpected_value check as if they were garbage input.

    Parameters
    ----------
    df      : raw dataframe (as read from file)
    columns : which columns to clean (default: all columns)

    Returns
    -------
    Copy of df with blank/whitespace-only string cells replaced by NaN.
    """
    if columns is None:
        columns = df.columns.tolist()

    clean_df = df.copy()
    for col in columns:
        raw = clean_df[col]
        is_str = _is_string_cell(raw)
        stripped = raw.where(is_str).astype("string").str.strip()
        is_blank = (is_str & (stripped == "")).fillna(False)
        clean_df[col] = raw.where(~is_blank, np.nan)

    return clean_df


def normalize_missing_codes(
    df: pd.DataFrame,
    columns: list = None,
    timeout_strings=DEFAULT_TIMEOUT_STRINGS,
):
    """
    Replace timeout sentinel strings with NaN and coerce the given
    columns to numeric dtype, so .notna()/.isna() and numeric
    comparisons behave correctly downstream.

    Parameters
    ----------
    df              : raw dataframe (as read from file)
    columns         : which columns to normalize (default: all columns)
    timeout_strings : set of strings that represent "timed out"
                       (case/whitespace-insensitive match)

    Returns
    -------
    clean_df        : copy of df with sentinel strings -> NaN and the
                       given columns coerced to numeric. Use THIS for
                       all downstream checks, not the raw df.
    timeout_mask    : bool DataFrame, True where a cell held a
                       recognized timeout string
    unexpected_mask : bool DataFrame, True where a cell held a
                       non-numeric, non-timeout string (i.e. something
                       that got coerced to NaN that ISN'T a known
                       timeout code -- worth a manual look, since it's
                       silently becoming "missing" otherwise)
    """
    if columns is None:
        columns = df.columns.tolist()

    norm_timeout_strings = {s.strip().upper() for s in timeout_strings}

    clean_df = df.copy()
    timeout_mask = pd.DataFrame(False, index=df.index, columns=columns)
    unexpected_mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        raw = df[col]
        is_str = _is_string_cell(raw)

        # normalize string cells for matching (strip + uppercase)
        str_upper = raw.where(is_str).astype("string").str.strip().str.upper()
        is_timeout = (is_str & str_upper.isin(norm_timeout_strings)).fillna(False)

        # blank out recognized timeout strings before numeric coercion
        prepped = raw.where(~is_timeout, np.nan)
        coerced = pd.to_numeric(prepped, errors="coerce")

        # a string that wasn't a recognized timeout code but still
        # failed numeric coercion -> flag it, don't just drop it
        unexpected = (is_str & ~is_timeout & coerced.isna()).fillna(False)

        clean_df[col] = coerced
        timeout_mask[col] = is_timeout
        unexpected_mask[col] = unexpected

    return clean_df, timeout_mask, unexpected_mask
