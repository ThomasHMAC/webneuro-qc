"""MAZE checks."""

import pandas as pd

from ..helpers import _both_present, _flag, _present

MAZE_COLS = ["emzcompk", "emzinitk", "emzerrk", "emzoverk", "emztrlsk"]
MAZE_TIMEOUT_MS = 1_000_000  # ~16.7 min; the task times out at 16 min, and
# this is the dictionary's own stated max for
# the timing variables, so it's the ceiling
# to check against, not the round 16-min figure


def check_maze(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    # emzerrk must be strictly greater than emzoverk
    applicable = _both_present(df["emzerrk"], df["emzoverk"])
    flags["maze_errk_not_gt_overk"] = _flag(df["emzerrk"] <= df["emzoverk"], applicable)

    # emztrlsk floor is 2
    applicable = _present(df["emztrlsk"])
    flags["maze_trlsk_below_floor"] = _flag(df["emztrlsk"] < 2, applicable)

    # EXAMPLE new check: emztrlsk implausibly high (>50 trials to finish)
    applicable = _present(df["emztrlsk"])
    flags["maze_trlsk_implausibly_high"] = _flag(df["emztrlsk"] > 50, applicable)

    # emzcompk must be greater than 0
    applicable = _present(df["emzcompk"])
    flags["maze_compk_eq_0"] = _flag(df["emzcompk"] <= 0, applicable)

    # emzcompk must be strictly greater than emzinitk
    applicable = _both_present(df["emzcompk"], df["emzinitk"])
    flags["maze_compk_not_gt_initk"] = _flag(
        df["emzcompk"] <= df["emzinitk"], applicable
    )

    # timing variables shouldn't exceed the task's own timeout ceiling --
    # a value above this either means the timeout didn't fire correctly,
    # or the value is a logging/scoring error
    applicable = _present(df["emzcompk"])
    flags["maze_compk_over_timeout"] = _flag(
        df["emzcompk"] > MAZE_TIMEOUT_MS, applicable
    )

    applicable = _present(df["emzinitk"])
    flags["maze_initk_over_timeout"] = _flag(
        df["emzinitk"] > MAZE_TIMEOUT_MS, applicable
    )

    # partial missingness across the 5 shared-missingness variables
    present_counts = df[MAZE_COLS].notna().sum(axis=1)
    flags["maze_partial_missing"] = ~present_counts.isin([0, len(MAZE_COLS)])

    # the maze path is fully hidden, so the very first trial can never be
    # error-free -- an error somewhere is guaranteed, making emzerrk==0
    # a real impossibility, not just an unlikely edge case
    applicable = _present(df["emzerrk"])
    flags["maze_errk_zero_impossible"] = _flag(df["emzerrk"] == 0, applicable)

    # overruns are only ONE type of error, so a trial's guaranteed error
    # doesn't have to be an overrun specifically -- emzoverk==0 stays
    # plausible, just unusual enough to flag for manual review
    applicable = _present(df["emzoverk"])
    flags["maze_overk_zero_review"] = _flag(df["emzoverk"] == 0, applicable)

    # emzinitk is the time up to the end of the final erroneous trial --
    # since an erroneous trial is now guaranteed to exist, that trial takes
    # nonzero time, so emzinitk==0 is impossible for the same reason as
    # emzerrk==0 above (this also subsumes the narrower "errk>0 & initk==0"
    # contradiction: initk==0 is never valid regardless of errk's value)
    applicable = _present(df["emzinitk"])
    flags["maze_initk_zero_impossible"] = _flag(df["emzinitk"] == 0, applicable)

    # emztrlsk==2 (the documented range floor) requires trial 1 AND trial 2
    # to both be clean -- but trial 1 can never be clean, so the true
    # achievable floor is 3, not 2; this is the domain-knowledge companion
    # to maze_trlsk_below_floor above (which only catches the dictionary's
    # literal <2 floor)
    applicable = _present(df["emztrlsk"])
    flags["maze_trlsk_eq_2_impossible"] = _flag(df["emztrlsk"] == 2, applicable)

    # sanity check: none of the 5 Maze variables should ever be negative
    # (dictionary range floors are 0 for all of them, 2 for emztrlsk)
    negative_any = pd.Series(False, index=df.index)
    for c in MAZE_COLS:
        negative_any |= df[c] < 0
    flags["maze_negative_value"] = negative_any

    # emzerrk/emzoverk/emztrlsk are documented as integer-typed -- a
    # fractional value means the data got corrupted or mis-typed somewhere
    non_integer_any = pd.Series(False, index=df.index)
    for c in ["emzerrk", "emzoverk", "emztrlsk"]:
        non_integer_any |= df[c].notna() & (df[c] % 1 != 0)
    flags["maze_non_integer_value"] = non_integer_any

    return flags
