"""End-to-end pipeline and command line entry point.

    python -m historian_flow.pipeline data/synthetic_port.csv --out output/
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd

from . import clean, phase1, phase2, reports, routes_v2, routes_v3


@dataclass
class Result:
    p1: phase1.Phase1Result
    event_info: pd.DataFrame
    event_move: pd.DataFrame
    conf: pd.DataFrame
    movements: pd.DataFrame
    members: pd.DataFrame
    routes: pd.DataFrame
    route_keys: pd.DataFrame
    movement_log: pd.DataFrame
    route_summary: pd.DataFrame
    routes_v2: pd.DataFrame
    timings: dict
    routes_v3: pd.DataFrame | None = None
    mode: str = "faithful"


def _phase1_clean(raw, timestamps, cfg: clean.CleanConfig) -> phase1.Phase1Result:
    """Phase 1 with the real-data fixes (see clean.py)."""
    c = phase1.cleanse(raw)
    if timestamps is not None:
        keep = (raw != 0).any(axis=1).to_numpy()
        timestamps = timestamps[keep].reset_index(drop=True)
    tag_type = phase1.classify(c)
    tag_type[clean.find_level_tags(c, tag_type, cfg)] = clean.LEVEL
    acc = tag_type[tag_type == phase1.ACCUMULATE].index
    c = clean.fill_totaliser_gaps(c, acc, cfg)
    diff = clean.capped_increments(c, tag_type, cfg)
    active, tote = phase1.activity(diff, tag_type)
    return phase1.Phase1Result(c, tag_type, diff, active, tote, timestamps)


def run(raw: pd.DataFrame, timestamps: pd.Series | None = None, mode: str = "faithful",
        cfg: clean.CleanConfig | None = None) -> Result:
    """mode='faithful' reproduces the 2020 thesis code exactly; mode='clean'
    adds the real-data fixes in clean.py and chain-first route grouping."""
    cfg = cfg or clean.CleanConfig()
    t = {}
    t0 = time.perf_counter()
    p1 = _phase1_clean(raw, timestamps, cfg) if mode == "clean" else phase1.run_phase1(raw, timestamps)
    t["phase1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    info, move, conf = phase2.events(p1.active, p1.tote, p1.diff, p1.tag_type)
    if mode == "clean":
        move = move[move.Count >= cfg.min_run_rows].reset_index(drop=True)
    t["events"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    movements, members = phase2.match_movements(move)
    t["matching"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    routes, keys, route_of = phase2.build_routes(movements, members, p1.active, p1.tag_type)
    t["routes"] = time.perf_counter() - t0

    log = reports.movement_log(movements, members, route_of, p1.timestamps)
    summary = reports.route_summary(log, keys, p1.tag_type)
    t0 = time.perf_counter()
    aligned = routes_v2.aligned_tags(p1.active, p1.tag_type, log)
    label, v2 = routes_v2.group_routes(log, aligned)
    log.insert(1, "route_v2", label)
    t["routes_v2"] = time.perf_counter() - t0
    v3 = None
    if mode == "clean":
        t0 = time.perf_counter()
        cov = routes_v3.coverage(p1.active, log)
        label3, v3 = routes_v3.group_routes(log, cov, p1.tag_type)
        log.insert(1, "route_v3", label3)
        t["routes_v3"] = time.perf_counter() - t0
    t["total"] = sum(t.values())
    return Result(p1, info, move, conf, movements, members, routes, keys, log, summary, v2, t, v3, mode)


def save(res: Result, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    res.movement_log.to_csv(os.path.join(out, "movement_log.csv"), index=False)
    res.route_summary.to_csv(os.path.join(out, "route_summary_original_method.csv"), index=False)
    res.routes_v2.to_csv(os.path.join(out, "routes_v2.csv"), index=False)
    if res.routes_v3 is not None:
        res.routes_v3.to_csv(os.path.join(out, "routes_v3.csv"), index=False)
    reports.tag_profile(res.p1).to_csv(os.path.join(out, "tag_profile.csv"), index=False)
    res.routes.to_csv(os.path.join(out, "routes.csv"), index=False)
    res.event_move.to_csv(os.path.join(out, "event_move.csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct material movements from historian data")
    ap.add_argument("csv", help="wide historian extract: DateTime + one column per tag")
    ap.add_argument("--out", default="output")
    ap.add_argument("--mode", choices=["faithful", "clean"], default="faithful",
                    help="faithful = identical to the 2020 thesis code; clean = real-data fixes + route v3")
    ap.add_argument("--start", help="timestamp of the first row, if the file has no DateTime column "
                                    "(e.g. '2026-01-01 00:00:00')")
    ap.add_argument("--interval", default="10s", help="row interval when using --start (default 10s)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    raw, ts = phase1.load_wide(a.csv)
    if ts is None and a.start:
        ts = pd.Series(pd.date_range(a.start, periods=len(raw), freq=a.interval).strftime("%Y-%m-%d %H:%M:%S"))
    load_s = time.perf_counter() - t0
    res = run(raw, ts, mode=a.mode)
    save(res, a.out)
    print(f"{len(raw):,} rows x {raw.shape[1]} tags  (load {load_s:.2f}s, pipeline {res.timings['total']:.2f}s, mode {a.mode})")
    msg = f"{len(res.movements)} movements | routes: {len(res.routes)} (original method), {len(res.routes_v2)} (v2)"
    if res.routes_v3 is not None:
        msg += f", {len(res.routes_v3)} (v3; {int(res.routes_v3.confirmed.sum())} confirmed)"
    print(msg + f" -> {a.out}/")


if __name__ == "__main__":
    main()
