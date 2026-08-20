"""
Integration tests for webneuro_qc.runner: run_maze_checks, run_all_checks,
summarize, apply_stroop_missing_score_correction.

Run just this file:
    pytest tests/test_runner.py -v
"""

import numpy as np
import pandas as pd

from webneuro_qc import (
    apply_stroop_missing_score_correction,
    count_timeouts,
    get_timeout_mask,
    run_all_checks,
    run_maze_checks,
    summarize,
)


def test_run_maze_checks_only_needs_maze_columns():
    # no GNG/Stroop/DigitSpan columns present -- should not raise
    df = pd.DataFrame(
        {
            "emzcompk": [300],
            "emzinitk": [400],  # violation: compk <= initk
            "emzerrk": [np.nan],
            "emzoverk": [np.nan],
            "emztrlsk": [np.nan],
        }
    )
    flags_df, results_long = run_maze_checks(df)

    assert flags_df.loc[0, "maze_compk_not_gt_initk"] is True
    assert (results_long["task"] == "Maze").all()
    assert "maze_compk_not_gt_initk" in results_long["check"].values


def test_run_maze_checks_surfaces_timeout_as_one_issue_row():
    df = pd.DataFrame(
        {
            "subject_id": ["P01", "P02"],
            "emzcompk": ["TO", 300],
            "emzinitk": ["TO", 250],
            "emzerrk": ["TO", 3],
            "emzoverk": ["TO", 1],
            "emztrlsk": ["TO", 4],
        }
    )
    _, results_long = run_maze_checks(df, subject_id_col="subject_id")

    timeout_rows = results_long[results_long["check"] == "Maze_timeout"]
    assert len(timeout_rows) == 1  # one row for P01, not one per column
    assert timeout_rows.iloc[0]["participant_id"] == "P01"
    assert timeout_rows.iloc[0]["task"] == "Maze"
    assert "P02" not in timeout_rows["participant_id"].values

    # P02 triggered no violations at all -- should still show up, as a pass
    pass_rows = results_long[results_long["status"] == "pass"]
    assert pass_rows.iloc[0]["participant_id"] == "P02"
    assert pass_rows.iloc[0]["task"] == "Maze"


def test_run_all_checks_reports_pass_per_task_not_just_per_participant():
    # one participant clean on GNG but with a Maze violation -- should show
    # a "fail" row for Maze AND a separate "pass" row for GNG, not just
    # silence on the task that was fine
    df = pd.DataFrame(
        {
            "emzcompk": [300],
            "emzinitk": [400],  # violation: compk <= initk
            "emzerrk": [np.nan],
            "emzoverk": [np.nan],
            "emztrlsk": [np.nan],
            "g2avrtk": [450],
            "g2errk": [10],
            "g2fnk": [5],
            "g2fpk": [5],
        }
    )
    _, results_long = run_all_checks(df)

    maze_rows = results_long[results_long["task"] == "Maze"]
    gng_rows = results_long[results_long["task"] == "GNG"]
    assert (maze_rows["status"] == "fail").any()
    assert len(gng_rows) == 1
    assert gng_rows.iloc[0]["status"] == "pass"


def test_run_maze_checks_with_subject_id_col():
    df = pd.DataFrame(
        {
            "subject_id": ["P01"],
            "emzcompk": [300],
            "emzinitk": [400],
            "emzerrk": [np.nan],
            "emzoverk": [np.nan],
            "emztrlsk": [np.nan],
        }
    )
    _, results_long = run_maze_checks(df, subject_id_col="subject_id")
    assert results_long.loc[0, "participant_id"] == "P01"


def test_run_all_checks_covers_every_task():
    df = pd.DataFrame(
        {
            "emzcompk": [300],
            "emzinitk": [400],
            "emzerrk": [np.nan],
            "emzoverk": [np.nan],
            "emztrlsk": [np.nan],
            "g2avrtk": [450],
            "g2errk": [np.nan],
            "g2fnk": [126],
            "g2fpk": [np.nan],
            "vcrtne": [np.nan],
            "vi_sco1": [0],
            "vcrtne2": [np.nan],
            "vi_sco2": [np.nan],
            "digitot": [0],
            "digitsp": [4],
        }
    )
    flags_df, results_long = run_all_checks(df)

    assert set(results_long["task"]) <= {"Maze", "GNG", "Stroop", "DigitSpan"}
    assert flags_df.loc[0, "maze_compk_not_gt_initk"] is True
    assert flags_df.loc[0, "ds_zero_logic_contradiction"] is True


def test_summarize_counts_add_up_to_row_count():
    df = pd.DataFrame(
        {
            "emzcompk": [300, np.nan],
            "emzinitk": [400, np.nan],
            "emzerrk": [np.nan, np.nan],
            "emzoverk": [np.nan, np.nan],
            "emztrlsk": [np.nan, np.nan],
        }
    )
    flags_df, _ = run_maze_checks(df)
    summary = summarize(flags_df)

    totals = summary["n_violations"] + summary["n_passed"] + summary["n_not_applicable"]
    assert (totals == len(flags_df)).all()


def test_summarize_splits_timed_out_from_not_applicable():
    df = pd.DataFrame(
        {
            # row0: full Maze timeout; row1: missing for some other reason
            # (never attempted -- no "TO" sentinel at all)
            "emzcompk": ["TO", np.nan],
            "emzinitk": ["TO", np.nan],
            "emzerrk": ["TO", np.nan],
            "emzoverk": ["TO", np.nan],
            "emztrlsk": ["TO", np.nan],
        }
    )
    flags_df, _ = run_maze_checks(df)
    timeout_mask = get_timeout_mask(df)
    summary = summarize(flags_df, timeout_mask=timeout_mask)

    assert "n_timed_out" in summary.columns
    assert summary.loc["maze_compk_not_gt_initk", "n_timed_out"] == 1
    assert summary.loc["maze_compk_not_gt_initk", "n_not_applicable"] == 1

    totals = (
        summary["n_violations"]
        + summary["n_passed"]
        + summary["n_timed_out"]
        + summary["n_not_applicable"]
    )
    assert (totals == len(flags_df)).all()


def test_summarize_without_timeout_mask_keeps_old_behavior():
    df = pd.DataFrame(
        {
            "emzcompk": ["TO", np.nan],
            "emzinitk": ["TO", np.nan],
            "emzerrk": ["TO", np.nan],
            "emzoverk": ["TO", np.nan],
            "emztrlsk": ["TO", np.nan],
        }
    )
    flags_df, _ = run_maze_checks(df)
    summary = summarize(flags_df)

    assert "n_timed_out" not in summary.columns
    assert summary.loc["maze_compk_not_gt_initk", "n_not_applicable"] == 2


def test_summarize_only_relevant_drops_zero_violation_checks():
    df = pd.DataFrame(
        {
            "emzcompk": [300],
            "emzinitk": [400],  # only this check fires
            "emzerrk": [np.nan],
            "emzoverk": [np.nan],
            "emztrlsk": [np.nan],
        }
    )
    flags_df, _ = run_maze_checks(df)
    summary = summarize(flags_df, only_relevant=True)

    assert (summary["n_violations"] > 0).all()
    assert "maze_compk_not_gt_initk" in summary.index


def test_count_timeouts_counts_full_timeout_rows_per_task():
    df = pd.DataFrame(
        {
            # row0: full Maze timeout: row1: no timeout, row2: partial (not a full timeout)
            "emzcompk": ["TO", 300, "TO"],
            "emzinitk": ["TO", 400, "TO"],
            "emzerrk": ["TO", 5, "TO"],
            "emzoverk": ["TO", 2, "TO"],
            "emztrlsk": ["TO", 2, np.nan],  # row2: not "TO" -> not a full timeout
            # GNG: no rows timed out
            "g2avrtk": [450, 450, 450],
            "g2errk": [10, 10, 10],
            "g2fnk": [5, 5, 5],
            "g2fpk": [5, 5, 5],
        }
    )
    counts = count_timeouts(df)

    assert counts["Maze"] == 1  # only row0: all 5 columns say "TO"
    assert counts["GNG"] == 0


def test_count_timeouts_only_includes_tasks_present_in_df():
    df = pd.DataFrame(
        {
            "emzcompk": ["TO"],
            "emzinitk": ["TO"],
            "emzerrk": ["TO"],
            "emzoverk": ["TO"],
            "emztrlsk": ["TO"],
        }
    )
    counts = count_timeouts(df)

    assert counts["Maze"] == 1
    assert "GNG" not in counts.index
    assert "Stroop" not in counts.index
    assert "DigitSpan" not in counts.index


def test_apply_stroop_missing_score_correction():
    df = pd.DataFrame({"vi_sco1": [0, 5], "vi_sco2": [0, 5]})
    flags_df = pd.DataFrame(
        {
            "stroop_sco1_should_be_missing_bug": [True, False],
            "stroop_sco2_should_be_missing_bug": [True, False],
        }
    )
    corrected = apply_stroop_missing_score_correction(df, flags_df)

    assert pd.isna(corrected.loc[0, "vi_sco1"])
    assert pd.isna(corrected.loc[0, "vi_sco2"])
    assert corrected.loc[1, "vi_sco1"] == 5
    assert corrected.loc[1, "vi_sco2"] == 5
    # original untouched
    assert df.loc[0, "vi_sco1"] == 0
