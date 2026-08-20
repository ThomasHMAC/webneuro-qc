"""SWITCH OF ATTENTION (Part 2) checks."""

import pandas as pd

from ..helpers import _both_present, _flag, _present

SOA_TIMEOUT_MS = 150_000  # dictionary's documented ceiling for esoadur2
SOA_COLS = ["esoadur2", "esoaerr2", "scavr0t2"]


def check_SoA_part_two(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    # KNOWN BUG: WebNeuro sometimes fails to label a genuine timeout with
    # "TO" and instead silently records esoadur2==0 & esoaerr2==0, as if
    # the task were attempted and instantly finished with zero errors.
    # Computed first so every other check below can exclude these rows --
    # otherwise the same known cause ends up reported multiple times under
    # different check names (e.g. also tripping "duration is 0", "duration
    # is suspiciously low", "no connection time despite completing").
    applicable = _both_present(df["esoadur2"], df["esoaerr2"])
    zero_zero_timeout_bug = (df["esoadur2"] == 0) & (df["esoaerr2"] == 0)
    flags["SoA_dur2_err2_zero_timeout_bug"] = _flag(zero_zero_timeout_bug, applicable)
    is_known_bug_row = zero_zero_timeout_bug.fillna(False)

    # timing variables shouldn't exceed the task's own timeout ceiling --
    # a value above this either means the timeout didn't fire correctly,
    # or the value is a logging/scoring error
    applicable = _present(df["esoadur2"])
    flags["SoA_dur2_over_timeout"] = _flag(df["esoadur2"] > SOA_TIMEOUT_MS, applicable)

    # esoadur2 duration must be greater than 0 -- excludes the known bug
    # rows above, since those are already explained by that check
    applicable = _present(df["esoadur2"]) & ~is_known_bug_row
    flags["SoA_dur2_eq_0"] = _flag(df["esoadur2"] <= 0, applicable)

    # esoadur2 duration should be greater than 5,000ms the median -- same
    # exclusion as above
    applicable = _present(df["esoadur2"]) & ~is_known_bug_row
    flags["SoA_dur2_below_5000"] = _flag(df["esoadur2"] <= 5000, applicable)

    # esoaerr2 must be <= 25 because it's capped at 25
    applicable = _present(df["esoaerr2"])
    flags["SoA_err2_gt_25"] = _flag(df["esoaerr2"] > 25, applicable)

    # error counts can't be negative
    applicable = _present(df["esoaerr2"])
    flags["SoA_err2_negative"] = _flag(df["esoaerr2"] < 0, applicable)

    # esoaerr2 is documented as integer-typed -- a fractional value means
    # the data got corrupted or mis-typed somewhere
    applicable = _present(df["esoaerr2"])
    flags["SoA_err2_non_integer"] = _flag(df["esoaerr2"] % 1 != 0, applicable)

    # scavr0t2 (average time between correct responses) can't be 0 or
    # negative -- same reasoning as esoadur2 above
    applicable = _present(df["scavr0t2"])
    flags["SoA_conn2_eq_0"] = _flag(df["scavr0t2"] <= 0, applicable)

    # esoadur2 present (and not the zero/zero bug above) means the
    # sequence was genuinely completed, which requires having pressed
    # keys -- so esoaerr2/scavr0t2 can't be missing in that case; this is
    # a real contradiction, not just unusual
    completed = df["esoadur2"].notna() & ~is_known_bug_row
    flags["SoA_completed_no_err_response"] = completed & df["esoaerr2"].isna()
    flags["SoA_completed_no_conn_time"] = completed & df["scavr0t2"].isna()

    # esoaerr2 ("missing if no response") and scavr0t2 ("missing if no
    # button pressed") share the same underlying trigger per the
    # dictionary, so they should go missing together -- except for the
    # zero/zero bug rows, where esoaerr2==0 (present) while scavr0t2 stays
    # missing is expected, not a mismatch
    both_present = df["esoaerr2"].notna() & df["scavr0t2"].notna()
    both_missing = df["esoaerr2"].isna() & df["scavr0t2"].isna()
    flags["SoA_err_conn_missing_mismatch"] = (
        ~(both_present | both_missing) & ~is_known_bug_row
    )

    return flags
