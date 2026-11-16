"""Operations-facing reports built on the pipeline outputs.

These are new (not in the thesis code): they turn movements and routes into the
tables an operations or reliability engineer would actually read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTERVAL_S = 10


def movement_log(movements: pd.DataFrame, members: pd.DataFrame, route_of: pd.Series,
                 timestamps: pd.Series | None) -> pd.DataFrame:
    """One row per reconstructed material movement."""
    runs = pd.concat([
        movements.assign(Slot=0)[["Movement", "Slot", "Tag", "Count", "Tote", "Sequence"]],
        members[["Movement", "Slot", "Tag", "Count", "Tote", "Sequence"]],
    ]).sort_values(["Movement", "Slot"])
    runs["first_row"] = (runs.Sequence - runs.Count).astype(int)
    runs["last_row"] = (runs.Sequence - 1).astype(int)
    g = runs.groupby("Movement")
    log = pd.DataFrame({
        "first_row": g.first_row.min(),
        "last_row": g.last_row.max(),
        "weightometers": g.Tag.agg(lambda s: " > ".join(s)),
        "n_scales": g.Tag.size(),
        "tonnes": g.Tote.median(),
        "tonnes_min": g.Tote.min(),
        "tonnes_max": g.Tote.max(),
        "duration_min": movements.set_index("Movement").Count * INTERVAL_S / 60,
    })
    log["scale_spread_pct"] = np.where(log.n_scales > 1,
                                       100 * (log.tonnes_max - log.tonnes_min) / log.tonnes, np.nan)
    log["rate_tph"] = log.tonnes / (log.duration_min / 60)
    log.insert(0, "route", route_of.to_numpy())
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        log.insert(1, "start", ts.iloc[log.first_row].to_numpy())
        log.insert(2, "end", ts.iloc[log.last_row].to_numpy())
    return log.round({"tonnes": 1, "tonnes_min": 1, "tonnes_max": 1, "duration_min": 1,
                      "scale_spread_pct": 2, "rate_tph": 0}).reset_index()


def route_summary(log: pd.DataFrame, route_keys: pd.DataFrame, tag_type: pd.Series) -> pd.DataFrame:
    """Per route: how often it ran, tonnes moved, typical rate, and its equipment tags."""
    tags = route_keys.set_index("Route Id")
    equip = tags.apply(lambda r: ", ".join(c for c in tags.columns if r[c]), axis=1)
    s = log.groupby("route").agg(movements=("Movement", "size"), tonnes=("tonnes", "sum"),
                                 median_rate_tph=("rate_tph", "median"),
                                 median_duration_min=("duration_min", "median"),
                                 weightometers=("weightometers", "first"))
    s["tags_bound"] = tags.sum(axis=1).reindex(s.index)
    s["equipment_tags"] = equip.reindex(s.index)
    return s.sort_values("tonnes", ascending=False).round(1).reset_index()


def tag_profile(p1) -> pd.DataFrame:
    """Per tag: inferred type and basic operating statistics."""
    nz = p1.clean.replace(0, np.nan)
    return pd.DataFrame({
        "type": p1.tag_type,
        "active_pct": (100 * p1.active.mean()).round(1),
        "min_nonzero": nz.min().round(2),
        "max": nz.max().round(2),
        "mean_nonzero": nz.mean().round(2),
    }).rename_axis("tag").reset_index()


def score_against_truth(log: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match each true movement (synthetic data only) to the reconstructed one
    covering the same weightometer and time window, and compare tonnes."""
    out = []
    for t in truth.itertuples():
        scales = [c.replace("tonnes_", "") for c in truth.columns
                  if c.startswith("tonnes_") and pd.notna(getattr(t, c))]
        true_t = np.nanmean([getattr(t, f"tonnes_{s}") for s in scales])
        cand = log[(log.first_row <= t.end_row + 5) & (log.last_row >= t.start_row - 5)
                   & log.weightometers.apply(lambda w: any(s + ".TOTALISER" in w for s in scales))]
        if len(cand):
            best = cand.iloc[(cand.tonnes - true_t).abs().argmin()]
            out.append({"true_route": t.route, "true_tonnes": round(true_t, 1),
                        "found_tonnes": best.tonnes, "found_scales": best.n_scales,
                        "error_pct": round(100 * (best.tonnes - true_t) / true_t, 2)})
        else:
            out.append({"true_route": t.route, "true_tonnes": round(true_t, 1),
                        "found_tonnes": np.nan, "found_scales": 0, "error_pct": np.nan})
    res = pd.DataFrame(out)
    stats = {
        "true_movements": len(res),
        "found": int(res.found_tonnes.notna().sum()),
        "found_all_scales": int((res.found_scales >= 2).sum()),
        "median_abs_tonnes_error_pct": float(res.error_pct.abs().median()),
        "reconstructed_movements": len(log),
    }
    return res, stats
