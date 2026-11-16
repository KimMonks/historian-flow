"""Phase 1 - cleanse, classify, increments and activity.

Vectorised re-implementation of the thesis code in ids.py:

    get_type_noroute  -> cleanse() + classify()
    get_diff          -> increments()
    get_activity      -> activity()

Rows are numbered the way the original numbered them: the cleansed table uses
rows 0..n-1 and every later table (increments, activity, tote) uses rows
1..n-1, so a 'Sequence' value means the same thing in both versions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Tag types, same codes as the Django Tag model in the original project
ACCUMULATE = "ACCUMLATE"   # (sic) - kept identical to the original model value
CONTINUOUS = "CONTINUOUS"
GATE = "GATE"
COUNTER = "COUNTER"
PERCENT = "PERCENT"
POSITION = "POSITION"
OTHERS = "OTHERS"
ZERO = "ZERO"


@dataclass
class Phase1Result:
    clean: pd.DataFrame        # cleansed values, rows 0..n-1
    tag_type: pd.Series        # tag -> type code
    diff: pd.DataFrame         # per-interval increments (totalisers) or raw values, rows 1..n-1
    active: pd.DataFrame       # 0/1 activity, rows 1..n-1
    tote: pd.DataFrame         # running tonnes within an active run (totalisers) or raw value
    timestamps: pd.Series | None = None  # DateTime per original row, if supplied


def load_wide(path: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Read a wide historian extract (one row per interval, one column per tag)."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)  # -sig strips an Excel BOM
    ts = None
    if "DateTime" in df.columns:
        ts = df.pop("DateTime")
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df.astype(float), ts


def cleanse(raw: pd.DataFrame) -> pd.DataFrame:
    """Drop empty rows/tags, patch the first row and fill short zero dropouts.

    Matches get_type_noroute: a zero is replaced by the previous value when the
    previous value is positive and something non-zero appears within the next
    two readings (so 1-2 interval historian dropouts are bridged, real stops
    are not).
    """
    df = raw.loc[(raw != 0).any(axis=1)].reset_index(drop=True)
    df = df.loc[:, df.sum(axis=0) != 0]
    x = df.to_numpy(dtype=float, copy=True)
    n = len(x)
    if n > 1:
        first_zero = x[0] == 0
        x[0, first_zero] = x[1, first_zero]
    if n > 3:
        # The original walks row by row (rows 1..n-3). A zero is filled when the
        # previous value is positive and the 3-row window from it is non-zero.
        # That can only chain through runs of at most two zeros (a third zero
        # would make its own window sum to zero), so two array tests reproduce
        # the loop exactly.
        xn = np.nan_to_num(x)
        fwd = np.zeros_like(x)
        fwd[: n - 2] = xn[: n - 2] + xn[1: n - 1] + xn[2:]
        zero, pos = x == 0, x > 0
        cond = zero & (fwd > 0)
        cond[0] = False
        cond[n - 2:] = False
        f1 = np.zeros_like(cond)
        f1[1:] = cond[1:] & pos[:-1]                      # previous reading positive
        f2 = np.zeros_like(cond)
        f2[2:] = cond[2:] & cond[1:-1] & pos[:-2]         # second zero of a 2-zero gap
        src = x.copy()
        x[1:][f1[1:]] = src[:-1][f1[1:]]
        x[2:][f2[2:]] = src[:-2][f2[2:]]
    return pd.DataFrame(x, columns=df.columns)


def classify(clean: pd.DataFrame) -> pd.Series:
    """Infer each tag's signal type from its statistics (thesis rules, in order)."""
    nz = clean.replace(0, np.nan)
    mx, mn, mean = nz.max(), nz.min(), nz.mean()
    non_zero = clean.astype(bool).sum()
    types = {}
    for c in clean.columns:
        distinct = len(clean[c].round().unique()) - 1
        M, m, a = mx[c], mn[c], mean[c]
        if a > 1000:
            t = ACCUMULATE
        elif non_zero[c] == 0:
            t = ZERO
        elif a == 1 and M == 1 and distinct <= 2:
            t = GATE
        elif distinct == (M - m) / 2 + 1 or distinct == (M + 2 - m) / 2 + 1:
            t = COUNTER
        elif (M - a < 3) and (a < 6):
            t = CONTINUOUS
        elif (M < 101) and (M - a < 50):
            t = PERCENT
        elif (a > 100) and distinct < 100:
            t = POSITION
        else:
            t = OTHERS
        types[c] = t
    return pd.Series(types, name="tag_type")


def increments(clean: pd.DataFrame, tag_type: pd.Series) -> pd.DataFrame:
    """Totalisers -> per-interval increase; other tags pass through unchanged."""
    x = clean.to_numpy(dtype=float)
    prev, cur = x[:-1], x[1:]
    with np.errstate(invalid="ignore"):
        # NB: the last test is always true when the third one is - see QUIRKS.md #1
        ok = (cur > 0) & (prev > 0) & (cur > prev + 1) & (prev < cur + 100)
        acc_diff = np.where(ok, cur - prev, 0.0)
    is_acc = (tag_type.reindex(clean.columns) == ACCUMULATE).to_numpy()
    d = np.where(is_acc[None, :], acc_diff, cur)
    return pd.DataFrame(d, columns=clean.columns, index=pd.RangeIndex(1, len(x), name="Sequence"))


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    """Rows shifted down by k (k>0: previous rows, k<0: following rows), padded with 0."""
    out = np.zeros_like(a)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def activity(diff: pd.DataFrame, tag_type: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Activity flags and running tote, as get_activity.

    A row is active when its value is positive (or it sits between positive
    values) and the 4-back + 4-forward window sums to more than 3.
    """
    d = np.nan_to_num(diff.to_numpy(dtype=float))
    raw = diff.to_numpy(dtype=float)
    back = d + _shift(d, 1) + _shift(d, 2) + _shift(d, 3)
    fwd = d + _shift(d, -1) + _shift(d, -2) + _shift(d, -3)
    with np.errstate(invalid="ignore"):
        act = ((raw > 0) | ((back > 0) & (fwd > 0))) & (back + fwd > 3)

    # tote: totalisers accumulate within an active run; others carry the raw value
    is_acc = (tag_type.reindex(diff.columns) == ACCUMULATE).to_numpy()
    tote = np.where(act, raw, 0.0)
    acc_idx = np.flatnonzero(is_acc)
    for j in acc_idx:
        # tote[i] = tote[i-1] + d[i] while active; restarts at d[i] after an inactive
        # row or a previous jump >= 100; row 1 contributes 0. A grouped cumsum adds in
        # the same order as the original loop, so the floats are bit-identical.
        dj, aj = raw[:, j], act[:, j]
        with np.errstate(invalid="ignore"):
            keep = np.r_[False, dj[:-1] < 100]
        contrib = np.where(aj, dj, 0.0)
        contrib[0] = 0.0
        start = ~aj | ~keep | np.r_[True, ~aj[:-1]]
        start[0] = True
        seg = np.cumsum(start)
        tote[:, j] = np.where(aj, pd.Series(contrib).groupby(seg).cumsum().to_numpy(), 0.0)
    idx = diff.index
    return (pd.DataFrame(act.astype(np.int8), columns=diff.columns, index=idx),
            pd.DataFrame(tote, columns=diff.columns, index=idx))


def run_phase1(raw: pd.DataFrame, timestamps: pd.Series | None = None) -> Phase1Result:
    clean = cleanse(raw)
    if timestamps is not None:
        keep = (raw != 0).any(axis=1).to_numpy()
        timestamps = timestamps[keep].reset_index(drop=True)
    tag_type = classify(clean)
    diff = increments(clean, tag_type)
    active, tote = activity(diff, tag_type)
    return Phase1Result(clean, tag_type, diff, active, tote, timestamps)
