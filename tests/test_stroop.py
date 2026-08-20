"""
Tests for webneuro_qc.checks.stroop.check_stroop.

Run just this file:
    pytest tests/test_stroop.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.checks.stroop import STROOP_TIMEOUT_MS, check_stroop

STROOP_COLS = ["vcrtne", "vcrtne2", "vi_sco1", "vi_sco2"]


def _stroop_df(**overrides) -> pd.DataFrame:
    n = max((len(v) for v in overrides.values()), default=1)
    data = {col: overrides.get(col, [np.nan] * n) for col in STROOP_COLS}
    return pd.DataFrame(data)


def test_vcrtne_out_of_range():
    df = _stroop_df(vcrtne=[STROOP_TIMEOUT_MS + 1, STROOP_TIMEOUT_MS, np.nan])
    flags = check_stroop(df)["stroop_vcrtne_out_of_range"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_vcrtne2_out_of_range():
    df = _stroop_df(vcrtne2=[STROOP_TIMEOUT_MS + 1, STROOP_TIMEOUT_MS, np.nan])
    flags = check_stroop(df)["stroop_vcrtne2_out_of_range"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_rt1_present_score1_zero():
    df = _stroop_df(vcrtne=[1200, 1200, np.nan], vi_sco1=[0, 5, 0])
    flags = check_stroop(df)["stroop_rt1_present_score1_zero"]
    assert flags[0] is True  # RT present but score is 0 -> contradiction
    assert flags[1] is False
    assert pd.isna(flags[2])  # vcrtne missing -> not applicable


def test_rt2_present_score2_zero():
    df = _stroop_df(vcrtne2=[900, 900, np.nan], vi_sco2=[0, 5, 0])
    flags = check_stroop(df)["stroop_rt2_present_score2_zero"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_sco1_should_be_missing_bug():
    # vcrtne missing & vi_sco1 == 0 -> known bug
    df = _stroop_df(vcrtne=[np.nan, np.nan, 1200], vi_sco1=[0, 5, 0])
    flags = check_stroop(df)["stroop_sco1_should_be_missing_bug"]
    assert flags[0] is True  # missing RT + score 0 -> the bug
    assert flags[1] is False  # missing RT but nonzero score -> not the bug
    assert pd.isna(flags[2])  # vcrtne present -> check doesn't apply


def test_sco2_should_be_missing_bug():
    df = _stroop_df(vcrtne2=[np.nan, np.nan, 900], vi_sco2=[0, 5, 0])
    flags = check_stroop(df)["stroop_sco2_should_be_missing_bug"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])
