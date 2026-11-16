"""Synthetic bulk-handling port historian data.

Generates a wide, 10-second historian extract that looks like the data the
original thesis pipeline consumed (one row per 10 s, one column per tag),
together with a ground-truth table of the material movements that produced it.

The plant is fictional and generic:

    Inbound (unloading -> stockpile)
        CD1 -> C101 [BW101] -> C201 [BW201] -> C202 -> ST1
        CD1 -> C101 [BW101] -> C201 [BW201] -> C203 -> ST2
        CD2 -> C102 [BW102] -> C201 [BW201] -> C202 -> ST1
        CD2 -> C102 [BW102] -> C201 [BW201] -> C203 -> ST2

    Outbound (stockpile -> ship)
        RC1 -> C301 [BW301] -> C401 [BW401] -> C402 -> SL1
        RC1 -> C301 [BW301] -> C401 [BW401] -> C403 -> SL2
        RC2 -> C302 [BW302] -> C401 [BW401] -> C402 -> SL1
        RC2 -> C302 [BW302] -> C401 [BW401] -> C403 -> SL2

One inbound and one outbound route can run at the same time. Tags cover the
signal types the thesis classified: weightometer totalisers (accumulators),
belt speeds (continuous), run status (gate/binary), feeder and chute
percentages, machine travel positions, and noisy motor currents.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ROWS_PER_MIN = 6  # 10-second historian interval

INBOUND_ROUTES = {
    "IN-CD1-ST1": ["CD1", "C101", "C201", "C202", "ST1"],
    "IN-CD1-ST2": ["CD1", "C101", "C201", "C203", "ST2"],
    "IN-CD2-ST1": ["CD2", "C102", "C201", "C202", "ST1"],
    "IN-CD2-ST2": ["CD2", "C102", "C201", "C203", "ST2"],
}
OUTBOUND_ROUTES = {
    "OUT-RC1-SL1": ["RC1", "C301", "C401", "C402", "SL1"],
    "OUT-RC1-SL2": ["RC1", "C301", "C401", "C403", "SL2"],
    "OUT-RC2-SL1": ["RC2", "C302", "C401", "C402", "SL1"],
    "OUT-RC2-SL2": ["RC2", "C302", "C401", "C403", "SL2"],
}
CONVEYORS = ["C101", "C102", "C201", "C202", "C203", "C301", "C302", "C401", "C402", "C403"]
WEIGHTOMETERS = {"C101": "BW101", "C102": "BW102", "C201": "BW201",
                 "C301": "BW301", "C302": "BW302", "C401": "BW401"}
AMPS_ON = {"C101", "C201", "C301", "C401"}  # conveyors with a motor-current tag


@dataclass
class Movement:
    route: str
    start: int  # first row material is on the feeder
    end: int    # last row material is on the feeder (inclusive)
    rate_tph: float


def _schedule(rng, n_rows, routes, dur_min, rate, idle_min, trip_prob):
    """Back-to-back runs of randomly chosen routes with idle gaps and occasional trips."""
    moves, t = [], int(rng.uniform(5, 30) * ROWS_PER_MIN)
    names = list(routes)
    while True:
        route = names[rng.integers(len(names))]
        dur = int(rng.uniform(*dur_min) * ROWS_PER_MIN)
        r = rng.uniform(*rate)
        if rng.random() < trip_prob:  # a trip splits the run into two movements
            first = int(dur * rng.uniform(0.3, 0.7))
            gap = int(rng.uniform(4, 12) * ROWS_PER_MIN)
            parts = [(t, t + first - 1), (t + first + gap, t + dur + gap - 1)]
        else:
            parts = [(t, t + dur - 1)]
        if parts[-1][1] + 60 >= n_rows:
            break
        moves += [Movement(route, s, e, r) for s, e in parts]
        t = parts[-1][1] + int(rng.uniform(*idle_min) * ROWS_PER_MIN)
    return moves


def generate(days: float = 2.0, seed: int = 7, prefix: str = "",
             start: str = "2019-03-01 00:00:00") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (historian, truth).

    historian: DataFrame with a 'DateTime' column and one float column per tag.
    truth:     one row per movement with route, start/end rows and times, and
               the tonnes each weightometer on the route should have measured.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 60 * ROWS_PER_MIN)
    p = prefix

    moves = (_schedule(rng, n, INBOUND_ROUTES, (40, 120), (4000, 6000), (15, 60), 0.15)
             + _schedule(rng, n, OUTBOUND_ROUTES, (60, 180), (5000, 8000), (20, 90), 0.25))
    all_routes = {**INBOUND_ROUTES, **OUTBOUND_ROUTES}

    belt_on = {c: np.zeros(n, bool) for c in CONVEYORS}
    machine_on = {m: np.zeros(n, bool) for m in ["CD1", "CD2", "ST1", "ST2", "RC1", "RC2", "SL1", "SL2"]}
    flow = {bw: np.zeros(n) for bw in WEIGHTOMETERS.values()}  # tonnes per row
    truth = []

    for mv in moves:
        eq = all_routes[mv.route]
        belts = [e for e in eq if e in CONVEYORS]
        src, dst = eq[0], eq[-1]
        # start sequence: downstream belts start first, stop last
        k = len(belts)
        for pos, b in enumerate(belts):
            lead = 2 * (k - pos) + 2
            belt_on[b][max(0, mv.start - lead): min(n, mv.end + lead + 6)] = True
        machine_on[dst][max(0, mv.start - 2 * k - 4): min(n, mv.end + 2 * k + 8)] = True
        machine_on[src][mv.start: mv.end + 1] = True
        # material: transport lag of 1-2 rows per belt along the route
        per_row = mv.rate_tph / 360.0
        base = per_row * (1 + 0.04 * rng.standard_normal(mv.end - mv.start + 1))
        base = np.clip(base, per_row * 0.5, None)
        lag, tonnes = 0, {}
        for b in belts:
            if b in WEIGHTOMETERS:
                bw = WEIGHTOMETERS[b]
                s = mv.start + lag
                seg = base[: max(0, min(n, s + len(base)) - s)]
                flow[bw][s: s + len(seg)] += seg
                tonnes[p + bw] = float(seg.sum())
            lag += int(rng.integers(1, 3))
        truth.append({"route": p + mv.route, "start_row": mv.start, "end_row": mv.end,
                      "rate_tph": round(mv.rate_tph, 1), **{f"tonnes_{k}": v for k, v in tonnes.items()}})

    cols = {}
    # weightometer totalisers: large running counters with a per-scale calibration bias
    for bw, f in flow.items():
        bias = 1 + rng.normal(0, 0.012)
        tot = rng.uniform(1.2e6, 3.5e6) + np.cumsum(f * bias)
        drop = rng.random(n) < 1 / 3000  # historian bad-value dropouts read as 0
        cols[bw + ".TOTALISER"] = np.where(drop, 0.0, np.round(tot, 1))
    for c in CONVEYORS:
        on = belt_on[c]
        speed = np.where(on, np.round(4.4 + 0.08 * rng.standard_normal(n), 2), 0.0)
        speed[rng.random(n) < 1 / 5000] = 0.0
        cols[c + ".SPEED"] = speed
        cols[c + ".RUN"] = on.astype(float)
        if c in AMPS_ON:
            cols[c + ".MOTOR_AMPS"] = np.where(on, np.round(rng.normal(320, 35, n), 1), 0.0)
    for m in ["CD1", "CD2"]:
        cols[m + ".RUN"] = machine_on[m].astype(float)
        cols[m + ".FEEDER_RATE.PCT"] = np.where(machine_on[m], np.round(np.clip(rng.normal(72, 6, n), 40, 95), 1), 0.0)
    for m in ["ST1", "ST2", "RC1", "RC2"]:
        cols[m + ".RUN"] = machine_on[m].astype(float)
        # travel position: 10 m steps, only moves while the machine works; never 0
        stepping = machine_on[m] & (rng.random(n) < 1 / 30)
        steps = np.where(stepping, rng.choice([-10, 10], n), 0)
        cols[m + ".TRAVEL_POSN"] = (np.clip(400 + np.cumsum(steps), 120, 880)).astype(float)
    for m in ["SL1", "SL2"]:
        cols[m + ".RUN"] = machine_on[m].astype(float)
        cols[m + ".CHUTE_TRIM.PCT"] = np.where(machine_on[m], np.round(np.clip(rng.normal(55, 8, n), 20, 90), 1), 0.0)
    cols["SPARE.TAG"] = np.zeros(n)  # a dead tag the cleanser should drop

    hist = pd.DataFrame({p + k: v for k, v in cols.items()})
    ts = pd.date_range(start, periods=n, freq="10s")
    hist.insert(0, "DateTime", ts.strftime("%Y-%m-%d %H:%M:%S"))
    truth = pd.DataFrame(truth).sort_values("start_row").reset_index(drop=True)
    truth.insert(3, "start_time", ts[truth.start_row].strftime("%Y-%m-%d %H:%M:%S"))
    truth.insert(4, "end_time", ts[truth.end_row].strftime("%Y-%m-%d %H:%M:%S"))
    return hist, truth


def generate_site(areas: int = 1, days: float = 2.0, seed: int = 7):
    """Several independent plant areas side by side (for scaling tests)."""
    if areas == 1:
        return generate(days, seed)
    hs, ts = [], []
    for a in range(areas):
        h, t = generate(days, seed + a, prefix=f"A{a + 1}_")
        hs.append(h if a == 0 else h.drop(columns="DateTime"))
        ts.append(t)
    return pd.concat(hs, axis=1), pd.concat(ts, ignore_index=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write a synthetic port historian extract")
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--areas", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic_port.csv")
    a = ap.parse_args()
    h, t = generate_site(a.areas, a.days, a.seed)
    h.to_csv(a.out, index=False)
    t.to_csv(a.out.replace(".csv", "_truth.csv"), index=False)
    print(f"{a.out}: {len(h):,} rows x {h.shape[1] - 1} tags, {len(t)} movements")
