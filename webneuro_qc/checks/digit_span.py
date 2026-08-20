"""DIGIT SPAN (FORWARD) checks."""

import pandas as pd

from ..helpers import _both_present, _flag, _present

DS_COLS = ["digitot", "digitsp"]


def check_digit_span(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    applicable = _present(df["digitsp"])
    flags["ds_invalid_value_1_or_2"] = _flag(df["digitsp"].isin([1, 2]), applicable)

    # shared missingness: digitot / digitsp should be missing together
    both_present = df["digitot"].notna() & df["digitsp"].notna()
    both_missing = df["digitot"].isna() & df["digitsp"].isna()
    flags["ds_missing_mismatch"] = ~(both_present | both_missing)

    # logic contradiction: digitsp==0 <-> digitot==0
    applicable = _both_present(df["digitot"], df["digitsp"])
    flags["ds_zero_logic_contradiction"] = _flag(
        ((df["digitsp"] == 0) & (df["digitot"] > 0))
        | ((df["digitot"] == 0) & (df["digitsp"] > 0)),
        applicable,
    )

    return flags
