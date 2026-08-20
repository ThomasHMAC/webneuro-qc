"""
Top-level entry points: run checks and build the tidy results table.
"""

import pandas as pd

from .checks import CHECK_TASK_MAP, MAZE_COLS, TASK_COLUMN_GROUPS
from .preprocessing import (
    DEFAULT_TIMEOUT_STRINGS,
    blank_strings_to_nan,
    normalize_missing_codes,
)
from .results import build_results_long


def run_maze_checks(
    df: pd.DataFrame,
    timeout_strings=DEFAULT_TIMEOUT_STRINGS,
    subject_id_col: str = None,
):
    """
    Standalone entry point for just the Maze variables -- doesn't require
    any of the other tasks' columns to be present in df.

    subject_id_col : name of the column in df holding subject/participant
                      IDs (if any). If given, the results table's
                      "participant_id" column will show that ID instead of
                      the row number.

    Returns (flags_df, results_long), same shape as run_all_checks.
    """
    clean_df, timeout_mask, unexpected_mask = normalize_missing_codes(
        blank_strings_to_nan(df, columns=MAZE_COLS),
        columns=MAZE_COLS,
        timeout_strings=timeout_strings,
    )
    flags_df = CHECK_TASK_MAP["Maze"](clean_df)
    task_labels = {col: "Maze" for col in flags_df.columns}
    subject_ids = df[subject_id_col] if subject_id_col else None
    column_task_map = {col: "Maze" for col in MAZE_COLS}
    results_long = build_results_long(
        flags_df,
        task_labels,
        unexpected_mask,
        subject_ids,
        column_task_map,
        timeout_mask,
    )
    return flags_df, results_long


def run_all_checks(
    df: pd.DataFrame,
    timeout_strings=DEFAULT_TIMEOUT_STRINGS,
    subject_id_col: str = None,
):
    """
    Run every check and return:
      flags_df     : DataFrame, same index as df, one column per check,
                     values True (violation) / False (passed) / <NA> (n/a)
      results_long : tidy DataFrame with one row per participant per
                     violation, plus one status="pass" row per participant
                     with no violations at all; columns = [participant_id,
                     task, status, variables, check, description]

    Automatically blanks out empty/whitespace-only string cells (see
    blank_strings_to_nan()) and normalizes timeout sentinel strings (e.g.
    "TO") to NaN before running any check -- see normalize_missing_codes().
    Any other non-numeric, non-timeout value found along the way is
    reported as its own "unexpected_value" issue rather than silently
    disappearing.

    subject_id_col : name of the column in df holding subject/participant
                      IDs (if any). If given, the results table's
                      "participant_id" column will show that ID instead of
                      the row number.
    """
    column_task_map = {
        c: task_name
        for task_name, cols in TASK_COLUMN_GROUPS.items()
        for c in cols
        if c in df.columns
    }
    task_cols = list(column_task_map.keys())

    subject_ids = df[subject_id_col] if subject_id_col else None

    df, timeout_mask, unexpected_mask = normalize_missing_codes(
        blank_strings_to_nan(df, columns=task_cols),
        columns=task_cols,
        timeout_strings=timeout_strings,
    )

    all_flags = []
    task_labels = {}

    for task_name, fn in CHECK_TASK_MAP.items():
        task_columns = TASK_COLUMN_GROUPS.get(task_name, [])
        if task_columns and not all(c in df.columns for c in task_columns):
            # this task's columns aren't in the dataset at all -- skip it
            # rather than KeyError on a column that was never there
            continue
        task_flags = fn(df)
        all_flags.append(task_flags)
        for col in task_flags.columns:
            task_labels[col] = task_name

    flags_df = pd.concat(all_flags, axis=1)
    results_long = build_results_long(
        flags_df,
        task_labels,
        unexpected_mask,
        subject_ids,
        column_task_map,
        timeout_mask,
    )

    return flags_df, results_long
