"""
Tests for webneuro_qc.checks.digit_span.check_digit_span.

Run just this file:
    pytest tests/test_digit_span.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.checks.digit_span import check_digit_span

DS_COLS = ["digitot", "digitsp"]


def _ds_df(**overrides) -> pd.DataFrame:
    n = max((len(v) for v in overrides.values()), default=1)
    data = {col: overrides.get(col, [np.nan] * n) for col in DS_COLS}
    return pd.DataFrame(data)


def test_invalid_value_1_or_2():
    df = _ds_df(digitsp=[1, 2, 4, np.nan])
    flags = check_digit_span(df)["ds_invalid_value_1_or_2"]
    assert flags[0] is True
    assert flags[1] is True
    assert flags[2] is False
    assert pd.isna(flags[3])


def test_missing_mismatch():
    df = _ds_df(
        digitot=[5, np.nan, np.nan],
        digitsp=[4, np.nan, 4],  # row2: digitsp present, digitot missing -> mismatch
    )
    flags = check_digit_span(df)["ds_missing_mismatch"]
    assert flags[0] == False  # both present -> fine
    assert flags[1] == False  # both missing -> fine
    assert flags[2] == True  # mismatched missingness -> flagged


def test_zero_logic_contradiction():
    df = _ds_df(
        digitot=[5, 0, 0, np.nan],
        digitsp=[0, 4, 0, np.nan],  # row0: sp=0,tot>0; row1: tot=0,sp>0; row2: both 0
    )
    flags = check_digit_span(df)["ds_zero_logic_contradiction"]
    assert flags[0] is True
    assert flags[1] is True
    assert flags[2] is False  # both zero -> consistent, not a contradiction
    assert pd.isna(flags[3])
