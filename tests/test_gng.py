"""
Tests for webneuro_qc.checks.gng.check_gng.

Run just this file:
    pytest tests/test_gng.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.checks.gng import check_gng

GNG_COLS = ["g2fpk", "g2fnk", "g2errk", "g2avrtk"]


def _gng_df(**overrides) -> pd.DataFrame:
    n = max((len(v) for v in overrides.values()), default=1)
    data = {col: overrides.get(col, [np.nan] * n) for col in GNG_COLS}
    return pd.DataFrame(data)


def test_fpk_out_of_range():
    df = _gng_df(g2fpk=[43, 42, np.nan])
    flags = check_gng(df)["gng_fpk_out_of_range"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_fnk_out_of_range():
    df = _gng_df(g2fnk=[127, 126, np.nan])
    flags = check_gng(df)["gng_fnk_out_of_range"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_errk_out_of_range():
    df = _gng_df(g2errk=[169, 168, np.nan])
    flags = check_gng(df)["gng_errk_out_of_range"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_errk_consistency():
    df = _gng_df(g2errk=[99, 10, np.nan], g2fnk=[5, 5, 5], g2fpk=[5, 5, 5])
    flags = check_gng(df)["gng_errk_consistency"]
    assert flags[0] is True  # 99 != 5 + 5 -> violation
    assert flags[1] is False  # 10 == 5 + 5 -> passes
    assert pd.isna(flags[2])  # g2errk missing -> not applicable


def test_avrtk_impossible():
    df = _gng_df(g2avrtk=[450, 450, np.nan], g2fnk=[126, 10, 126])
    flags = check_gng(df)["gng_avrtk_impossible"]
    assert flags[0] is True  # avrtk present while all green trials missed
    assert flags[1] is False
    assert pd.isna(flags[2])  # g2avrtk missing -> not applicable


def test_never_pressed_bug():
    df = _gng_df(g2fnk=[126, 10, np.nan], g2fpk=[0, 0, 0])
    flags = check_gng(df)["gng_never_pressed_bug"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])
