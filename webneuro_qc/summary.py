"""
Summary stats over a flags_df: violation/pass/timeout counts per check
and per task.
"""

import numpy as np
import pandas as pd

from .checks import TASK_COLUMN_GROUPS
from .metadata import CHECK_VARIABLES
from .preprocessing import DEFAULT_TIMEOUT_STRINGS, normalize_missing_codes


def get_timeout_mask(
    df: pd.DataFrame,
    timeout_strings=DEFAULT_TIMEOUT_STRINGS,
) -> pd.DataFrame:
    """
    Raw per-column timeout mask (True where a cell held a recognized
    timeout string) across every task column present in df.

    This is the same timeout_mask normalize_missing_codes computes
    internally, exposed on its own so callers (e.g. summarize()) can use
    it without re-running the full check pipeline.
    """
    column_task_map = {
        c: task_name
        for task_name, cols in TASK_COLUMN_GROUPS.items()
        for c in cols
        if c in df.columns
    }
    task_cols = list(column_task_map.keys())

    _, timeout_mask, _ = normalize_missing_codes(
        df, columns=task_cols, timeout_strings=timeout_strings
    )
    return timeout_mask


def summarize(
    flags_df: pd.DataFrame,
    only_relevant: bool = False,
    timeout_mask: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Quick counts of violations / passes / n-a per check.

    Columns:
      n_violations      -> rows that FAILED this check
      n_passed          -> rows that were evaluated and were fine
      n_timed_out       -> only present if timeout_mask is given: rows not
                            applicable specifically because every raw
                            variable the check needs was a recognized
                            timeout string (e.g. "TO")
      n_not_applicable  -> rows not applicable for any other reason (never
                            attempted, blank cell, etc.) -- or every not-
                            applicable row, if timeout_mask isn't given
    These always sum to len(flags_df) for every row.

    Parameters
    ----------
    only_relevant : if True, drop checks with zero violations so you
                    only see the checks that actually found something.
                    Useful once your dataset is bigger than a handful
                    of rows and the full table gets noisy.
    timeout_mask  : optional raw per-column timeout mask, e.g. from
                    get_timeout_mask(df). If given, splits
                    n_not_applicable into n_timed_out (genuine timeouts)
                    and n_not_applicable (everything else).
    """
    not_applicable = flags_df.isna()

    if timeout_mask is not None:
        timed_out = pd.DataFrame(False, index=flags_df.index, columns=flags_df.columns)
        for col in flags_df.columns:
            required_vars = [
                v for v in CHECK_VARIABLES.get(col, [col]) if v in timeout_mask.columns
            ]
            if required_vars:
                timed_out[col] = timeout_mask[required_vars].all(axis=1)

        summary = pd.DataFrame(
            {
                "n_violations": (flags_df == True).sum(),
                "n_passed": (flags_df == False).sum(),
                "n_timed_out": (not_applicable & timed_out).sum(),
                "n_not_applicable": (not_applicable & ~timed_out).sum(),
            }
        )
    else:
        summary = pd.DataFrame(
            {
                "n_violations": (flags_df == True).sum(),
                "n_passed": (flags_df == False).sum(),
                "n_not_applicable": not_applicable.sum(),
            }
        )

    summary = summary.sort_values("n_violations", ascending=False)
    if only_relevant:
        summary = summary[summary["n_violations"] > 0]
    return summary


def count_timeouts(
    df: pd.DataFrame,
    timeout_strings=DEFAULT_TIMEOUT_STRINGS,
) -> pd.Series:
    """
    Count, per task, how many rows have every one of that task's columns
    marked as a recognized timeout string (e.g. "TO").

    This is a genuine timeout count, distinct from summarize()'s
    n_not_applicable -- n_not_applicable also includes rows missing for
    other reasons (never attempted, blank cells, etc.), while this only
    counts rows where the timeout sentinel was actually present.

    Returns
    -------
    Series indexed by task name -> number of rows that timed out on that
    task. Only includes tasks that have at least one column present in df.
    """
    timeout_mask = get_timeout_mask(df, timeout_strings=timeout_strings)

    counts = {}
    for task_name, cols in TASK_COLUMN_GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        if cols:
            counts[task_name] = int(timeout_mask[cols].all(axis=1).sum())

    return pd.Series(counts, name="n_timed_out")


def apply_stroop_missing_score_correction(
    df: pd.DataFrame, flags_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply the confirmed correction for the Stroop "no response at all" bug:
    where vcrtne/vcrtne2 is missing and vi_sco1/vi_sco2 == 0, relabel the
    score to NaN (it should have been fully missing, not a literal 0).

    This is a real data correction, not just a flag -- use it once you've
    confirmed (as you have) that this pattern is always the bug in your
    data, not a genuine "tried everything, got zero right" result.

    Returns a corrected copy of df; the original is left untouched.
    """
    corrected = df.copy()

    sco1_bug = flags_df["stroop_sco1_should_be_missing_bug"] == True
    corrected.loc[sco1_bug.fillna(False), "vi_sco1"] = np.nan

    sco2_bug = flags_df["stroop_sco2_should_be_missing_bug"] == True
    corrected.loc[sco2_bug.fillna(False), "vi_sco2"] = np.nan

    return corrected
