"""Real-data mode - deliberate fixes on top of the faithful thesis logic.

Each fix addresses something seen in real port historian data that
the 2020 rules do not handle. The faithful mode (phase1/phase2 as verified
against ids.py) is left untouched; this module only adds steps around it.

    1. Level tags are not totalisers   - bin inventories average > 1000 so the
       thesis rules call them accumulators; they fall as much as they rise.
    2. Short blank gaps in totalisers  - interpolated (a totaliser only rises),
       so runs are not broken and tonnes are not lost.
    3. Totaliser jump cap              - a reading that glitches and recovers
       produced increments of up to 6e8 t; QUIRKS #1 meant nothing filtered them.
    4. Minimum run length              - 1-2 minute blips are not movements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .phase1 import ACCUMULATE

LEVEL = "LEVEL"


@dataclass
class CleanConfig:
    max_step: float = 250.0       # max totaliser rise per row (250 t / 10 s = 90,000 t/h)
    level_ratio: float = 0.3      # total fall / total rise above this -> a level, not a totaliser
    max_gap_rows: int = 30        # interpolate totaliser blanks up to 5 minutes long
    min_run_rows: int = 30        # ignore totaliser runs shorter than 5 minutes


def find_level_tags(clean: pd.DataFrame, tag_type: pd.Series, cfg: CleanConfig) -> list:
    """Accumulator-typed tags whose falls are a large share of their rises."""
    out = []
    for c in tag_type[tag_type == ACCUMULATE].index:
        x = clean[c].to_numpy(dtype=float)
        p, q = x[:-1], x[1:]
        with np.errstate(invalid="ignore"):
            ok = (p > 0) & (q > 0)
            d = np.where(ok, q - p, 0.0)
        small = np.abs(d) <= cfg.max_step            # ignore glitches and resets
        rise, fall = d[small & (d > 0)].sum(), -d[small & (d < 0)].sum()
        if rise == 0 or fall / rise > cfg.level_ratio:
            out.append(c)
    return out


def fill_totaliser_gaps(clean: pd.DataFrame, tags, cfg: CleanConfig) -> pd.DataFrame:
    """Linear interpolation across blank runs of at most max_gap_rows inside a totaliser."""
    out = clean.copy()
    for c in tags:
        s = out[c]
        na = s.isna()
        if not na.any():
            continue
        run_id = (~na).cumsum()
        run_len = na.groupby(run_id).transform("sum")
        filled = s.interpolate(limit_area="inside")
        out[c] = s.where(~(na & (run_len <= cfg.max_gap_rows)), filled)
    return out


def capped_increments(clean: pd.DataFrame, tag_type: pd.Series, cfg: CleanConfig) -> pd.DataFrame:
    """As phase1.increments, but a totaliser rise above max_step counts as 0."""
    x = clean.to_numpy(dtype=float)
    prev, cur = x[:-1], x[1:]
    with np.errstate(invalid="ignore"):
        ok = (cur > 0) & (prev > 0) & (cur > prev + 1) & (cur - prev <= cfg.max_step)
        acc_diff = np.where(ok, cur - prev, 0.0)
    is_acc = (tag_type.reindex(clean.columns) == ACCUMULATE).to_numpy()
    d = np.where(is_acc[None, :], acc_diff, cur)
    return pd.DataFrame(d, columns=clean.columns, index=pd.RangeIndex(1, len(x), name="Sequence"))
