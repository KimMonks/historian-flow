"""Route discovery v2 - start/stop alignment (new, not in the thesis).

The thesis method binds every tag that is active for the whole of a movement.
Equipment from an unrelated route that happens to run at the same time (an
inbound train unloading while a ship loads) is active too, so it gets bound,
and each different overlap creates a new "route". That is the phantom-route
effect (about 5,000 routes found against about 500 configured).

v2 asks a sharper question: did this tag START and STOP with the movement?
A conveyor in the route starts a little before material arrives and stops a
little after it clears. Equipment from a concurrent route starts and stops on
its own schedule, and always-on signals (machine positions) never transition
at all, so neither is included.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .phase1 import ACCUMULATE


def _runs(col: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.concatenate([[0], col.astype(np.int8), [0]])
    d = np.diff(a)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1   # 0-based start/end rows


def aligned_tags(active: pd.DataFrame, tag_type: pd.Series, log: pd.DataFrame,
                 lead: int = 30, lag: int = 12, coverage: float = 0.9) -> pd.DataFrame:
    """Boolean matrix movements x tags: tag started and stopped with the movement.

    lead: rows a tag may start before / stop after the material (start-up and
          run-out sequences; 30 rows = 5 min at 10 s)
    lag:  rows a tag may start after / stop before the material
    """
    A = active.to_numpy().astype(bool)
    first = log.first_row.to_numpy() - 1      # activity labels start at 1
    last = log.last_row.to_numpy() - 1
    mid = (first + last) // 2
    span = last - first + 1
    out = np.zeros((len(log), A.shape[1]), dtype=bool)
    for j, c in enumerate(active.columns):
        if tag_type.get(c) == ACCUMULATE:
            continue
        S, E = _runs(A[:, j])
        if len(S) == 0:
            continue
        k = np.searchsorted(S, mid, side="right") - 1
        ok = k >= 0
        k = np.clip(k, 0, None)
        s, e = S[k], E[k]
        covers = ok & (e >= mid)
        overlap = np.minimum(e, last) - np.maximum(s, first) + 1
        out[:, j] = (covers & (overlap >= coverage * span)
                     & (s >= first - lead) & (s <= first + lag)
                     & (e <= last + lead) & (e >= last - lag))
    return pd.DataFrame(out, columns=active.columns, index=log.index)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def group_routes(log: pd.DataFrame, aligned: pd.DataFrame,
                 similarity: float = 0.8) -> tuple[pd.Series, pd.DataFrame]:
    """Group movements into routes. Returns (route label per movement, route table).

    Within each set of weightometers, equipment signatures are clustered:
      * signatures at least `similarity` alike (Jaccard) are one route - a couple
        of coincidentally aligned extra tags do not create a new route, while a
        different branch (another stacker, another shiploader) does;
      * a signature that is a subset of exactly one route is a fragment of it
        (e.g. part of a run split by a trip, where some belts kept running).
    A route's equipment is the tags present in at least half of its movements.
    """
    eq = [frozenset(aligned.columns[aligned.loc[i].to_numpy()]) for i in log.index]
    ws = [frozenset(w.split(" > ")) for w in log.weightometers]
    labels = [None] * len(log)
    routes = []                                   # (weightometers, core tags, member positions)
    for w in dict.fromkeys(ws):                   # each weightometer set, in first-seen order
        pos = [k for k in range(len(log)) if ws[k] == w]
        freq: dict = {}
        for k in pos:
            freq[eq[k]] = freq.get(eq[k], 0) + 1
        cents: list = []
        for s, _ in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0]))):
            if not any(_jaccard(s, c) >= similarity for c in cents):
                cents.append(s)
        # fold fragments: a centroid that is a strict subset of exactly one other
        cents = [c for c in cents if sum(c < d for d in cents) != 1]
        groups: dict = {}
        for k in pos:
            sup = [c for c in cents if eq[k] <= c]
            best = sup[0] if len(sup) == 1 else max(cents, key=lambda c: _jaccard(eq[k], c))
            groups.setdefault(best, []).append(k)
        for members in groups.values():
            core = frozenset(t for t in aligned.columns
                             if sum(t in eq[k] for k in members) * 2 >= len(members))
            routes.append((w, core, members))
    routes.sort(key=lambda r: min(r[2]))
    table = []
    for n, (w, core, members) in enumerate(routes, start=1):
        name = f"R{n:03d}"
        for k in members:
            labels[k] = name
        sub = log.iloc[members]
        table.append({"route_v2": name, "movements": len(members),
                      "tonnes": round(float(sub.tonnes.sum()), 1),
                      "median_rate_tph": round(float(sub.rate_tph.median()), 0),
                      "weightometers": sub.weightometers.iloc[0],
                      "n_tags": len(w) + len(core), "equipment": ", ".join(sorted(core))})
    lab = pd.Series(labels, index=log.index, name="route_v2")
    return lab, pd.DataFrame(table).sort_values("tonnes", ascending=False).reset_index(drop=True)
