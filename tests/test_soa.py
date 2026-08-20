"""
Tests for webneuro_qc.checks.SoA.check_SoA_part_two.

Run just this file:
    pytest tests/test_soa.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.checks.SoA import SOA_TIMEOUT_MS, check_SoA_part_two

SOA_COLS = ["esoadur2", "esoaerr2", "scavr0t2"]


def _soa_df(**overrides) -> pd.DataFrame:
    """Build a SoA Part 2 dataframe; any column not overridden defaults to
    NaN for every row, sized to match the longest override."""
    n = max((len(v) for v in overrides.values()), default=1)
    data = {col: overrides.get(col, [np.nan] * n) for col in SOA_COLS}
    return pd.DataFrame(data)


def test_dur2_over_timeout():
    df = _soa_df(esoadur2=[SOA_TIMEOUT_MS + 1, SOA_TIMEOUT_MS, np.nan])
    flags = check_SoA_part_two(df)["SoA_dur2_over_timeout"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_dur2_eq_0():
    df = _soa_df(esoadur2=[0, 500, np.nan])
    flags = check_SoA_part_two(df)["SoA_dur2_eq_0"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_dur2_eq_0_and_below_5000_exclude_zero_zero_timeout_bug():
    # esoadur2==0 & esoaerr2==0 is the known bug (its own dedicated check
    # already covers it) -- these two generic checks should stay quiet on
    # that row instead of also firing, so one row doesn't produce three
    # overlapping issues for the same root cause
    df = _soa_df(esoadur2=[0], esoaerr2=[0])
    flags = check_SoA_part_two(df)
    assert flags.loc[0, "SoA_dur2_err2_zero_timeout_bug"] is True
    assert pd.isna(flags.loc[0, "SoA_dur2_eq_0"])
    assert pd.isna(flags.loc[0, "SoA_dur2_below_5000"])


def test_dur2_below_5000():
    df = _soa_df(esoadur2=[5000, 5001, np.nan])
    flags = check_SoA_part_two(df)["SoA_dur2_below_5000"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_err2_gt_25():
    df = _soa_df(esoaerr2=[26, 25, np.nan])
    flags = check_SoA_part_two(df)["SoA_err2_gt_25"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_err2_negative():
    df = _soa_df(esoaerr2=[-1, 0, np.nan])
    flags = check_SoA_part_two(df)["SoA_err2_negative"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_err2_non_integer():
    df = _soa_df(esoaerr2=[3.5, 3, np.nan])
    flags = check_SoA_part_two(df)["SoA_err2_non_integer"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_conn2_eq_0():
    df = _soa_df(scavr0t2=[0, 500, np.nan])
    flags = check_SoA_part_two(df)["SoA_conn2_eq_0"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_dur2_err2_zero_timeout_bug():
    df = _soa_df(esoadur2=[0, 500, 0, np.nan], esoaerr2=[0, 0, 3, 0])
    flags = check_SoA_part_two(df)["SoA_dur2_err2_zero_timeout_bug"]
    assert flags[0] is True  # both exactly 0 -> the known bug
    assert flags[1] is False  # dur2 nonzero -> not the bug
    assert flags[2] is False  # err2 nonzero -> not the bug
    assert pd.isna(flags[3])  # esoadur2 missing -> not applicable


def test_completed_no_err_response():
    # esoadur2 present (completed) but esoaerr2 missing -> impossible
    df = _soa_df(esoadur2=[50000, 50000, np.nan], esoaerr2=[np.nan, 3, np.nan])
    flags = check_SoA_part_two(df)["SoA_completed_no_err_response"]
    assert flags[0] == True
    assert flags[1] == False
    assert flags[2] == False  # esoadur2 missing -> not "completed", no contradiction


def test_completed_no_conn_time():
    df = _soa_df(esoadur2=[50000, 50000, np.nan], scavr0t2=[np.nan, 500, np.nan])
    flags = check_SoA_part_two(df)["SoA_completed_no_conn_time"]
    assert flags[0] == True
    assert flags[1] == False
    assert flags[2] == False


def test_completed_no_conn_time_excludes_zero_zero_timeout_bug():
    # esoadur2==0 & esoaerr2==0 is the known bug, not a real completion --
    # scavr0t2 missing here is expected, not a contradiction
    df = _soa_df(esoadur2=[0], esoaerr2=[0], scavr0t2=[np.nan])
    flags = check_SoA_part_two(df)["SoA_completed_no_conn_time"]
    assert flags[0] == False


def test_err_conn_missing_mismatch():
    df = _soa_df(
        esoaerr2=[3, np.nan, np.nan],
        scavr0t2=[500, np.nan, 500],  # row2: mismatched missingness
    )
    flags = check_SoA_part_two(df)["SoA_err_conn_missing_mismatch"]
    assert flags[0] == False  # both present -> fine
    assert flags[1] == False  # both missing -> fine
    assert flags[2] == True  # mismatched -> flagged


def test_err_conn_missing_mismatch_excludes_zero_zero_timeout_bug():
    # esoaerr2==0 (present) & scavr0t2 missing looks like a mismatch, but
    # it's the known zero/zero timeout bug, not a real contradiction
    df = _soa_df(esoadur2=[0], esoaerr2=[0], scavr0t2=[np.nan])
    flags = check_SoA_part_two(df)["SoA_err_conn_missing_mismatch"]
    assert flags[0] == False
