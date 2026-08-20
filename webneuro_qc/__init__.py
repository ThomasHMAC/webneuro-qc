"""
Cognitive task data-quality validation pipeline.

Implements the edge-case / consistency checks described in the task
data-dictionary notes for: Maze, Go/No-Go (GNG), Stroop, Digit Span
(Forward), and Switch of Attention (Part 2).

Usage
-----
    import pandas as pd
    from webneuro_qc import run_all_checks

    df = pd.read_csv("your_data.csv")
    flags_df, results_long = run_all_checks(df)

    # flags_df: same index as df, one boolean/NA column per check
    # results_long: tidy table of (row_id, task, status, check,
    #                description) with one row per participant per
    #                check outcome (pass or fail)

Notes on columns NOT covered here
----------------------------------
- g2sdrtk, vi_difrt: referenced in the data-dictionary notes but not
  present in the dataframe's column list you gave me. Stub checks are
  noted in webneuro_qc/checks/gng.py and stroop.py in case you add
  these columns later.
- esoadur1, esoaerr1, scavr0t1 (Switch of Attention, Part 1): not
  implemented -- only Part 2 (esoadur2, esoaerr2, scavr0t2) is covered.

Module map
----------
- helpers.py          tri-state flag helpers (_present, _flag, ...)
- preprocessing.py     timeout-sentinel normalization
- checks/maze.py       Maze checks
- checks/gng.py        Go/No-Go checks
- checks/stroop.py     Stroop checks
- checks/digit_span.py Digit Span (Forward) checks
- checks/SoA.py        Switch of Attention (Part 2) checks
- metadata.py          check descriptions + implicated columns
- runner.py            run_all_checks / run_maze_checks
- results.py           build_results_long
- summary.py           summarize / count_timeouts / get_timeout_mask / etc.
"""

from .checks import (
    DS_COLS,
    GNG_COLS,
    MAZE_COLS,
    MAZE_TIMEOUT_MS,
    SOA_COLS,
    SOA_TIMEOUT_MS,
    STROOP_COLS,
    STROOP_TIMEOUT_MS,
    TASK_COLUMN_GROUPS,
    check_digit_span,
    check_gng,
    check_maze,
    check_SoA_part_two,
    check_stroop,
)
from .metadata import CHECK_DESCRIPTIONS, CHECK_VARIABLES
from .preprocessing import DEFAULT_TIMEOUT_STRINGS, normalize_missing_codes
from .results import build_results_long
from .runner import run_all_checks, run_maze_checks
from .summary import (
    apply_stroop_missing_score_correction,
    count_timeouts,
    get_timeout_mask,
    summarize,
)

__all__ = [
    "run_all_checks",
    "run_maze_checks",
    "build_results_long",
    "summarize",
    "count_timeouts",
    "get_timeout_mask",
    "apply_stroop_missing_score_correction",
    "normalize_missing_codes",
    "DEFAULT_TIMEOUT_STRINGS",
    "check_maze",
    "check_gng",
    "check_stroop",
    "check_digit_span",
    "check_SoA_part_two",
    "MAZE_COLS",
    "MAZE_TIMEOUT_MS",
    "GNG_COLS",
    "STROOP_COLS",
    "STROOP_TIMEOUT_MS",
    "DS_COLS",
    "SOA_COLS",
    "SOA_TIMEOUT_MS",
    "TASK_COLUMN_GROUPS",
    "CHECK_DESCRIPTIONS",
    "CHECK_VARIABLES",
]
