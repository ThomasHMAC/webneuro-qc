"""
CLI entry point for the QC pipeline.

The checks themselves live in the webneuro_qc/ package (see its
__init__.py for the module map) and are covered by tests/ (run
`pytest tests/test_maze.py -v` to test just one task, or `pytest` for
everything). This script just wires that package up to a real CSV file
on disk.

Usage
-----
    python3 main.py data/your_data.csv
    python3 main.py data/your_data.csv --subject-id-col record_id
    python3 main.py data/your_data.csv --output-csv qc.csv --out-flags flags.csv
    python3 main.py data/your_data.csv  # writes qc_results_<timestamp>.csv by default
"""

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

from webneuro_qc import (
    count_timeouts,
    get_timeout_mask,
    run_all_checks,
    run_maze_checks,
    summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="path to the raw data CSV")
    parser.add_argument(
        "--subject-id-col",
        default="record_id",
        help="column in the CSV holding subject/participant IDs (default: record_id)",
    )
    parser.add_argument(
        "--out-flags",
        help="if given, write the full per-check flags table to this CSV",
    )
    parser.add_argument(
        "--output-csv",
        default="qc_results.csv",
        help="path to write the tidy table to this CSV (default: qc_results.csv)",
    )
    parser.add_argument(
        "--only-relevant",
        action="store_true",
        help="in the printed summary, only show checks with >0 violations",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    subject_id_col = args.subject_id_col if args.subject_id_col in df.columns else None
    if args.subject_id_col and subject_id_col is None:
        print(
            f"warning: column '{args.subject_id_col}' not found in {args.csv_path}; "
            "falling back to row position for the 'participant_id' column",
            file=sys.stderr,
        )

    flags_df, results_long = run_all_checks(df, subject_id_col=subject_id_col)
    # flags_df, results_long = run_maze_checks(df, subject_id_col=subject_id_col)

    timeout_mask = get_timeout_mask(df)
    print(
        summarize(flags_df, only_relevant=args.only_relevant, timeout_mask=timeout_mask)
    )
    print()
    n_failed = (results_long["status"] == "fail").sum()
    print(f"{n_failed} issue(s) found across {len(df)} participants")
    print()
    print("timeouts per task:")
    print(count_timeouts(df))

    if args.out_flags:
        flags_df.to_csv(args.out_flags)
        print(f"wrote flags table to {args.out_flags}")

    path = Path(args.output_csv)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    results_long.sort_values(["participant_id", "task"]).to_csv(
        output_path, index=False
    )
    print(f"wrote tidy table to {output_path}")


if __name__ == "__main__":
    main()
