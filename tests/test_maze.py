"""
Tests for webneuro_qc.checks.maze.check_maze.

Run just this file (no other tasks' columns needed):
    pytest tests/test_maze.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc.checks.maze import MAZE_TIMEOUT_MS, check_maze

MAZE_COLS = ["emzcompk", "emzinitk", "emzerrk", "emzoverk", "emztrlsk"]


def _maze_df(**overrides) -> pd.DataFrame:
    """Build a Maze dataframe; any column not overridden defaults to NaN
    for every row, sized to match the longest override."""
    n = max((len(v) for v in overrides.values()), default=1)
    data = {col: overrides.get(col, [np.nan] * n) for col in MAZE_COLS}
    return pd.DataFrame(data)


def test_errk_not_gt_overk():
    df = _maze_df(emzerrk=[2, 5, np.nan], emzoverk=[5, 2, 5])
    flags = check_maze(df)["maze_errk_not_gt_overk"]
    assert flags[0] is True  # 2 <= 5 -> violation
    assert flags[1] is False  # 5 > 2 -> passes
    assert pd.isna(flags[2])  # emzerrk missing -> not applicable


def test_trlsk_below_floor():
    df = _maze_df(emztrlsk=[1, 2, np.nan])
    flags = check_maze(df)["maze_trlsk_below_floor"]
    assert flags[0] is True  # below the floor of 2
    assert flags[1] is False  # exactly at the floor
    assert pd.isna(flags[2])


def test_trlsk_implausibly_high():
    df = _maze_df(emztrlsk=[51, 50, np.nan])
    flags = check_maze(df)["maze_trlsk_implausibly_high"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_compk_not_gt_initk():
    df = _maze_df(emzcompk=[300, 500, np.nan], emzinitk=[400, 300, np.nan])
    flags = check_maze(df)["maze_compk_not_gt_initk"]
    assert flags[0] is True  # 300 <= 400 -> violation
    assert flags[1] is False  # 500 > 300 -> passes
    assert pd.isna(flags[2])


def test_compk_eq_0():
    df = _maze_df(emzcompk=[0, 500, np.nan])
    flags = check_maze(df)["maze_compk_eq_0"]
    assert flags[0] is True  # 0 <= 0 -> violation
    assert flags[1] is False  # 500 > 0 -> passes
    assert pd.isna(flags[2])


def test_compk_over_timeout():
    df = _maze_df(emzcompk=[MAZE_TIMEOUT_MS + 1, MAZE_TIMEOUT_MS, np.nan])
    flags = check_maze(df)["maze_compk_over_timeout"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_initk_over_timeout():
    df = _maze_df(emzinitk=[MAZE_TIMEOUT_MS + 1, MAZE_TIMEOUT_MS, np.nan])
    flags = check_maze(df)["maze_initk_over_timeout"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_partial_missing():
    df = _maze_df(
        emzcompk=[500, np.nan, np.nan],
        emzinitk=[400, np.nan, np.nan],
        emzerrk=[5, 3, np.nan],  # row1: only this one present -> partial
        emzoverk=[2, np.nan, np.nan],
        emztrlsk=[2, np.nan, np.nan],
    )
    flags = check_maze(df)["maze_partial_missing"]
    assert flags[0] == False  # all 5 present
    assert flags[1] == True  # only emzerrk present -> partial
    assert flags[2] == False  # all 5 missing


def test_errk_zero_impossible():
    # the maze path is hidden, so trial 1 can never be error-free ->
    # emzerrk==0 is impossible, not just unusual
    df = _maze_df(emzerrk=[0, 1, np.nan])
    flags = check_maze(df)["maze_errk_zero_impossible"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_overk_zero_review():
    # unlike errk, emzoverk==0 stays plausible -- the guaranteed error
    # doesn't have to be an overrun specifically -- so this is still a
    # soft review flag, not an impossibility
    df = _maze_df(emzoverk=[0, 1, np.nan])
    flags = check_maze(df)["maze_overk_zero_review"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_initk_zero_impossible():
    # emzinitk is time up to the end of the (guaranteed) final erroneous
    # trial, so it can never be exactly 0 either
    df = _maze_df(emzinitk=[0, 1, np.nan])
    flags = check_maze(df)["maze_initk_zero_impossible"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_trlsk_eq_2_impossible():
    # emztrlsk==2 requires trial 1 AND trial 2 to both be clean, but
    # trial 1 can never be clean -- so the true floor is 3, not 2
    df = _maze_df(emztrlsk=[2, 3, np.nan])
    flags = check_maze(df)["maze_trlsk_eq_2_impossible"]
    assert flags[0] is True
    assert flags[1] is False
    assert pd.isna(flags[2])


def test_negative_value():
    df = _maze_df(
        emzcompk=[-1, 500, np.nan],
        emzinitk=[np.nan, 400, np.nan],
        emzerrk=[np.nan, 5, -3],
        emzoverk=[np.nan, 2, np.nan],
        emztrlsk=[np.nan, 2, np.nan],
    )
    flags = check_maze(df)["maze_negative_value"]
    assert flags[0] == True  # emzcompk negative
    assert flags[1] == False  # all present and non-negative
    assert flags[2] == True  # emzerrk negative


def test_non_integer_value():
    df = _maze_df(
        emzerrk=[3.5, 3, np.nan],
        emzoverk=[1, 2.25, np.nan],
        emztrlsk=[2, 4, np.nan],
    )
    flags = check_maze(df)["maze_non_integer_value"]
    assert flags[0] == True  # emzerrk = 3.5
    assert flags[1] == True  # emzoverk = 2.25
    assert flags[2] == False  # all missing -> nothing non-integer to find
