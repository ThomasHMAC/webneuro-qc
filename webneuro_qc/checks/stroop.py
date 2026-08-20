"""STROOP checks."""

import pandas as pd

from ..helpers import _both_present, _flag, _present

STROOP_TIMEOUT_MS = 30000  # corrected threshold; dictionary's 20000 is stale
STROOP_COLS = ["vcrtne", "vi_sco1", "vcrtne2", "vi_sco2"]


def check_stroop(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)

    # out-of-range RTs using the corrected (30s) timeout, not the stale dict value
    applicable = _present(df["vcrtne"])
    flags["stroop_vcrtne_out_of_range"] = _flag(
        df["vcrtne"] > STROOP_TIMEOUT_MS, applicable
    )

    applicable = _present(df["vcrtne2"])
    flags["stroop_vcrtne2_out_of_range"] = _flag(
        df["vcrtne2"] > STROOP_TIMEOUT_MS, applicable
    )

    # contradiction: an RT average implies >=1 correct trial, so score can't be 0
    applicable = _both_present(df["vcrtne"], df["vi_sco1"])
    flags["stroop_rt1_present_score1_zero"] = _flag(df["vi_sco1"] == 0, applicable)

    applicable = _both_present(df["vcrtne2"], df["vi_sco2"])
    flags["stroop_rt2_present_score2_zero"] = _flag(df["vi_sco2"] == 0, applicable)

    # NOTE: vi_difrt not in your dataframe -- if you add it, mirror:
    #   applicable = presence of vi_difrt OR (vcrtne/vcrtne2)
    #   flags["stroop_difrt_should_be_missing"] = _flag(
    #       df["vi_difrt"].notna() & (df["vcrtne"].isna() | df["vcrtne2"].isna()),
    #       df["vi_difrt"].notna(),
    #   )

    # KNOWN BUG (confirmed, not just "ambiguous"): vcrtne/vcrtne2 blank while
    # vi_sco1/vi_sco2 == 0 should have the score relabeled to missing too.
    # (Originally flagged as ambiguous per the dictionary's note that this
    # combo *could* be a genuine "tried everything, got zero right" result --
    # but per your confirmation, in this data it's the "no response at all"
    # bug and vi_sco1/vi_sco2 should be blanked to match.)
    applicable = _present(df["vi_sco1"]) & df["vcrtne"].isna()
    flags["stroop_sco1_should_be_missing_bug"] = _flag(df["vi_sco1"] == 0, applicable)

    applicable = _present(df["vi_sco2"]) & df["vcrtne2"].isna()
    flags["stroop_sco2_should_be_missing_bug"] = _flag(df["vi_sco2"] == 0, applicable)

    return flags
