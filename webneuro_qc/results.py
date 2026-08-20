"""
Turn a flags_df from any check function into a tidy long-format table of
per-participant, per-task pass/fail results.
"""

import pandas as pd

from .metadata import CHECK_DESCRIPTIONS, CHECK_VARIABLES


def build_results_long(
    flags_df: pd.DataFrame,
    task_labels: dict = None,
    unexpected_mask: pd.DataFrame = None,
    subject_ids: pd.Series = None,
    column_task_map: dict = None,
    timeout_mask: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Turn a flags_df (from any check function or combination of them) into
    a tidy long-format table of pass/fail results.

    Parameters
    ----------
    flags_df        : tri-state DataFrame from one or more check_* functions
    task_labels     : optional {check_column_name: task_name} dict. If not
                       given, the check's own column name is used as the
                       task label.
    unexpected_mask : optional bool DataFrame from normalize_missing_codes,
                       surfaced as its own "unexpected_value" issues.
    subject_ids     : optional Series (same index as flags_df) of real
                       subject/participant IDs. If given, the output's
                       "subject" column uses these instead of the row's
                       positional index.
    column_task_map : optional {raw_column_name: task_name} dict, used to
                       label unexpected_mask/timeout_mask issues with the
                       task that raw column actually belongs to (falls back
                       to "PREPROCESSING" for unexpected_mask columns not
                       in the map; timeout_mask columns not in the map are
                       skipped, since a task can't be inferred for them).
    timeout_mask    : optional bool DataFrame from normalize_missing_codes,
                       surfaced as one "<task>_timeout" issue per subject
                       who has every column of a given task marked as a
                       recognized timeout string (not one row per column,
                       since a task's columns time out together).

    Returns
    -------
    DataFrame with columns [participant_id, task, status, variables, check,
    description], one row per participant per violation, PLUS one
    status="pass" row (blank variables/check/description) for every
    (participant, task) pair that had no violation -- so a participant who
    fails one task but is clean on another shows up as both a "fail" row
    for the failed task and a "pass" row for the clean one.
    """
    if task_labels is None:
        task_labels = {col: col for col in flags_df.columns}

    def _subject_for(row_id):
        if subject_ids is not None:
            return subject_ids.loc[row_id]
        return row_id

    records = []
    flagged_tasks_by_row = {}

    def _flag(row_id, task_name):
        flagged_tasks_by_row.setdefault(row_id, set()).add(task_name)

    for col in flags_df.columns:
        violated_mask = (
            flags_df[col] == True
        )  # noqa: E712 (tri-state, need literal compare)
        variables = ", ".join(CHECK_VARIABLES.get(col, [col]))
        task_name = task_labels.get(col, col)

        # Loop through the row labels where check was violated
        for row_id in flags_df.index[violated_mask.fillna(False)]:
            _flag(row_id, task_name)
            records.append(
                {
                    "participant_id": _subject_for(row_id),
                    "task": task_name,
                    "variables": variables,
                    "status": "fail",
                    "check": col,
                    "description": CHECK_DESCRIPTIONS.get(col, ""),
                }
            )

    if unexpected_mask is not None:
        column_task_map = column_task_map or {}
        for col in unexpected_mask.columns:
            task_name = column_task_map.get(col, "PREPROCESSING")
            # Loop through the row labels where unexpected value found
            for row_id in unexpected_mask.index[unexpected_mask[col]]:
                _flag(row_id, task_name)
                records.append(
                    {
                        "participant_id": _subject_for(row_id),
                        "variables": col,
                        "task": task_name,
                        "status": "fail",
                        "check": f"{col}_unexpected_value",
                        "description": (
                            f"{col} held a non-numeric value that isn't a "
                            f"recognized timeout code -- check raw data"
                        ),
                    }
                )

    if timeout_mask is not None:
        column_task_map = column_task_map or {}
        task_to_cols = {}
        for col in timeout_mask.columns:
            task_name = column_task_map.get(col)
            if task_name:
                task_to_cols.setdefault(task_name, []).append(col)

        # one row per subject per task, not per column -- a task's columns
        # are expected to time out together, so this avoids one duplicate
        # row per column for what is really a single timeout event
        for task_name, cols in task_to_cols.items():
            fully_timed_out = timeout_mask[cols].all(axis=1)
            for row_id in timeout_mask.index[fully_timed_out]:
                _flag(row_id, task_name)
                records.append(
                    {
                        "participant_id": _subject_for(row_id),
                        "task": task_name,
                        "status": "fail",
                        "variables": ", ".join(cols),
                        "check": f"{task_name}_timeout",
                        "description": (
                            f"Participant timed out on {task_name} (every "
                            f"variable recorded as a timeout sentinel)"
                        ),
                    }
                )

    # every task actually evaluated (i.e. has at least one check column in
    # flags_df) that a participant wasn't flagged on gets an explicit pass
    # row, so passing one task while failing another is visible rather than
    # the passed task being silently absent
    evaluated_tasks = set(task_labels.values())
    for row_id in flags_df.index:
        flagged_tasks = flagged_tasks_by_row.get(row_id, set())
        for task_name in evaluated_tasks - flagged_tasks:
            records.append(
                {
                    "participant_id": _subject_for(row_id),
                    "task": task_name,
                    "variables": "",
                    "status": "pass",
                    "check": "",
                    "description": "",
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "participant_id",
            "task",
            "status",
            "variables",
            "check",
            "description",
        ],
    )
