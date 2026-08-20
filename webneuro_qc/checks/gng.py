"""GO/NO-GO checks."""

import pandas as pd

from ..helpers import _both_present, _flag, _present

GNG_COLS = ["g2avrtk", "g2errk", "g2fnk", "g2fpk"]


def check_gng(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    applicable = _present(df["g2fpk"])
    flags["gng_fpk_out_of_range"] = _flag(df["g2fpk"] > 42, applicable)

    applicable = _present(df["g2fnk"])
    flags["gng_fnk_out_of_range"] = _flag(df["g2fnk"] > 126, applicable)

    applicable = _present(df["g2errk"])
    flags["gng_errk_out_of_range"] = _flag(df["g2errk"] > 168, applicable)

    # consistency: g2errk == g2fnk + g2fpk
    applicable = df[["g2errk", "g2fnk", "g2fpk"]].notna().all(axis=1)
    flags["gng_errk_consistency"] = _flag(
        df["g2errk"] != (df["g2fnk"] + df["g2fpk"]), applicable
    )

    # g2avrtk present while every green trial was missed -> impossible
    applicable = _both_present(df["g2avrtk"], df["g2fnk"])
    flags["gng_avrtk_impossible"] = _flag(
        df["g2avrtk"].notna() & (df["g2fnk"] == 126), applicable
    )

    # NOTE: g2sdrtk not in your dataframe -- if you add it, mirror the
    # g2avrtk check above and also add:
    #   flags["gng_sdrtk_avrtk_mismatch"] = _flag(
    #       df["g2sdrtk"].notna() != df["g2avrtk"].notna(),
    #       pd.Series(True, index=df.index),
    #   )

    # the "never pressed spacebar at all" bug: fnk=126 & fpk=0 should be
    # fully missing, not reported literally
    applicable = _both_present(df["g2fnk"], df["g2fpk"])
    flags["gng_never_pressed_bug"] = _flag(
        (df["g2fnk"] == 126) & (df["g2fpk"] == 0), applicable
    )

    return flags
