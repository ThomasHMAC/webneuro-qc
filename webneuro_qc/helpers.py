"""
Small tri-state helpers shared by every check module.

A "check" result is tri-state, not boolean:
  True  -> check applicable AND violated
  False -> check applicable AND passed
  <NA>  -> check not applicable (required inputs missing)
"""

import pandas as pd


def _present(s: pd.Series) -> pd.Series:
    """True where value is present (not NaN)."""
    return s.notna()


def _both_present(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.notna() & b.notna()


def _flag(cond: pd.Series, applicable: pd.Series) -> pd.Series:
    """Build a tri-state result column (see module docstring)."""
    out = pd.Series(pd.NA, index=cond.index, dtype="object")
    out[applicable] = cond[applicable]
    return out
