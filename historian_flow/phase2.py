"""Phase 2 - events, movement matching and route discovery.

Re-implements, with the same rules, the thesis functions in ids.py:

    get_events          -> events()
    move_tags/new_tag   -> match_movements()
    event_routes + bind_all_acc_routes + compare_all_tags -> build_routes()

The big speed-ups come from:
  * events: one vectorised pass instead of a rows x tags x tags Python loop
  * movement matching: a set lookup instead of rescanning every earlier
    movement for every candidate (new_tag was O(n^2))
  * route binding: boolean-matrix subset tests instead of per-tag bitarrays
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .phase1 import ACCUMULATE

DELAY = 40  # rows (~6.7 min at 10 s) to look ahead for the same tonnes on another scale
START, END = "Start", "End"


# --------------------------------------------------------------------- events
def run_lengths(active: np.ndarray) -> np.ndarray:
    """Length of the current active run at each row (0 when inactive)."""
    a = active.astype(np.int64)
    c = np.cumsum(a, axis=0)
    reset = np.maximum.accumulate(np.where(a == 0, c, 0), axis=0)
    return c - reset


def events(active: pd.DataFrame, tote: pd.DataFrame, diff: pd.DataFrame,
           tag_type: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Start/End events for every tag, the totaliser 'movement' events and the
    tag co-activity matrix (get_events)."""
    A = active.to_numpy()
    L, m = A.shape
    cols = np.asarray(active.columns)
    E = run_lengths(A).astype(float)
    E[0, A[0] == 0] = np.nan          # original leaves row 1 undefined when inactive

    # an event fires where activity changes, and always on the last row
    change = np.zeros_like(A, dtype=bool)
    change[1:] = A[1:] != A[:-1]
    change[-1] = True
    change[0] = False
    r, c = np.nonzero(change)          # row-major: by sequence, then tag order
    prev = r - 1
    types = tag_type.reindex(cols).to_numpy()
    info = pd.DataFrame({
        "Sequence": r + 1,             # rows are labelled 1..L
        "Tag": cols[c],
        "Count": E[prev, c],
        "Increment": diff.to_numpy()[prev, c],
        "Event": np.where(A[prev, c] == 0, START, END),
        "Tote": tote.to_numpy()[prev, c],
        "Type": types[c],
    })
    move = info[(info.Type == ACCUMULATE) & (info.Tote > 1)].reset_index(drop=True)

    A2 = A[1:].astype(np.float64)                   # float matmul uses BLAS; exact for counts < 2**53
    conf = (A2.T @ A2).astype(np.int64)
    conf[0, :] = 0                     # original loop skips the first tag as a partner
    conf = pd.DataFrame(conf, index=cols, columns=cols)
    return info, move, conf


# ------------------------------------------------------------------ movements
def match_movements(move: pd.DataFrame, delay: int = DELAY) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group totaliser runs that carried the same material (move_tags).

    Two runs on different weightometers match when the second one ends within
    `delay` rows, its duration is within 10% and its tonnes are within 7.5%
    (under 1000 t) or 15% (1000 t and over). Each run joins at most one movement.

    Returns (movements, members): one row per movement (its primary run) and
    one row per matched secondary run.
    """
    seq = move.Sequence.to_numpy()
    tag = move.Tag.to_numpy()
    cnt = move.Count.to_numpy(dtype=float)
    tot = move.Tote.to_numpy(dtype=float)
    n = len(move)
    used: set[tuple] = set()
    mv_rows, mem_rows = [], []
    for i in range(n):
        if (tag[i], seq[i]) in used:
            continue
        used.add((tag[i], seq[i]))
        mid = len(mv_rows)
        matched = 0
        for j in range(1, delay):
            k = i + j
            if k >= n or seq[k] - seq[i] > delay:
                break
            if (tag[k], seq[k]) in used or tag[i] == tag[k]:
                continue
            if ((cnt[i] - cnt[k]) ** 2) ** 0.5 < cnt[i] * 0.1:
                comp = ((tot[i] - tot[k]) ** 2) ** 0.5
                if (tot[i] < 1000 and comp < tot[i] * 0.075) or (tot[i] >= 1000 and comp < tot[i] * 0.15):
                    used.add((tag[k], seq[k]))
                    matched += 1
                    mem_rows.append((mid, matched, tag[k], cnt[k], tot[k], seq[k]))
        mv_rows.append((mid, seq[i], tag[i], cnt[i], tot[i], matched))
    movements = pd.DataFrame(mv_rows, columns=["Movement", "Sequence", "Tag", "Count", "Tote", "Matched"])
    members = pd.DataFrame(mem_rows, columns=["Movement", "Slot", "Tag", "Count", "Tote", "Sequence"])
    return movements, members


# --------------------------------------------------------------------- routes
@dataclass
class Route:
    route_id: str
    matched: int
    slots: list = field(default_factory=list)   # [tag, max count, tonnes] per totaliser
    occurs: int = 0
    bound: np.ndarray | None = None              # accumulated binding counts per tag


class RouteBuilder:
    """Discovers routes: the set of tags that are active for the whole of a movement."""

    def __init__(self, active: pd.DataFrame, tag_type: pd.Series):
        self.A = active.to_numpy().astype(bool)
        self.cols = list(active.columns)
        self.col_idx = {c: i for i, c in enumerate(self.cols)}
        self.non_acc = (tag_type.reindex(self.cols) != ACCUMULATE).to_numpy()
        self.routes: list[Route] = []

    def bind(self, tags: list, count, seq) -> np.ndarray:
        """bind_all_acc_routes: tags active throughout the movement window."""
        count, seq = int(count), int(seq)
        lo = seq - count                       # window = rows lo..seq-1 (labels)
        W = self.A[max(lo, 1) - 1: seq - 1]
        if W.shape[0] != count:
            raise ValueError("movement window runs off the start of the data")
        bound = np.zeros(len(self.cols), dtype=np.int64)
        ba = np.zeros(count, dtype=bool)
        for t in tags:                         # cumulative OR, as the original
            ba = W[:, self.col_idx[t]] | ba
            bound[self.col_idx[t]] += int(ba.sum())
        covered = ~(ba[:, None] & ~W).any(axis=0)   # active whenever ba is
        in_set = np.zeros(len(self.cols), dtype=bool)
        in_set[[self.col_idx[t] for t in tags]] = True
        bound[self.non_acc & covered & ~in_set] += int(ba.sum())
        return bound

    def find(self, bound: np.ndarray, tags: list) -> int | None:
        """compare_all_tags: first route whose totalisers are all in this
        movement and whose binding pattern equals this movement's."""
        tagset = set(tags)
        pattern = bound != 0
        for idx, r in enumerate(self.routes):
            if all(s[0] in tagset for s in r.slots[: r.matched + 1]):
                if np.array_equal(r.bound != 0, pattern):
                    return idx
        return None

    def add(self, mv, members) -> str:
        tags = [mv.Tag] + list(members.Tag)
        count, seq, delay = mv.Count, mv.Sequence, 0
        for s in members.Sequence:
            if seq < s:
                seq, delay = s, s - mv.Sequence
        bound = self.bind(tags, count + delay, seq)
        idx = self.find(bound, tags)
        if idx is None:
            r = Route(f"ROUTE{len(self.routes) + 1}", int(mv.Matched),
                      [[mv.Tag, mv.Count, mv.Tote]], 0, bound)
            for mem in members.itertuples():
                r.slots.append([mem.Tag, mem.Count, mem.Tote])
                if mem.Count > r.slots[0][1]:      # first slot keeps the largest run
                    r.slots[0][1] = mem.Count
                if mem.Tote > r.slots[0][2]:
                    r.slots[0][2] = mem.Tote
            self.routes.append(r)
            return r.route_id
        r = self.routes[idx]
        # NB: re-binds on the primary's own window, not the lag-extended one (QUIRKS.md #3)
        r.bound = r.bound + self.bind(tags, mv.Count, mv.Sequence)
        if r.slots[0][1] < mv.Count:
            r.slots[0][1] = mv.Count
        if r.slots[0][2] < mv.Tote:
            r.slots[0][2] = mv.Tote
        for j, mem in enumerate(members.itertuples(), start=1):
            if j < len(r.slots):
                if r.slots[j][1] < mem.Count:
                    r.slots[j][1] = mem.Count
                if r.slots[j][2] < mem.Tote:
                    r.slots[j][2] += mem.Tote          # (sic) adds - QUIRKS.md #2
        r.occurs += 1
        return r.route_id

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(routes, route_keys) in the original layout: Route Id, Matched, Occurs,
        then (tag, count, tonnes) per totaliser slot, then binding counts per tag."""
        width = max([len(r.slots) for r in self.routes] + [1])
        recs, keys = [], []
        for r in self.routes:
            rec = {"Route Id": r.route_id, "Matched": r.matched, "Occurs": r.occurs}
            for s in range(width):
                tag, cnt, wgt = r.slots[s] if s < len(r.slots) else (np.nan, np.nan, np.nan)
                rec[f"Tot{s + 1}"], rec[f"Tot{s + 1}Count"], rec[f"Tot{s + 1}Tonnes"] = tag, cnt, wgt
            recs.append(rec)
            keys.append(r.bound)
        routes = pd.DataFrame(recs)
        bound = pd.DataFrame(np.array(keys).reshape(len(keys), len(self.cols)), columns=self.cols)
        routes = pd.concat([routes, bound], axis=1)
        route_keys = pd.concat([routes[["Route Id"]], bound != 0], axis=1)
        return routes, route_keys


def build_routes(movements: pd.DataFrame, members: pd.DataFrame, active: pd.DataFrame,
                 tag_type: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rb = RouteBuilder(active, tag_type)
    grouped = dict(tuple(members.groupby("Movement"))) if len(members) else {}
    empty = members.iloc[0:0]
    route_of = [rb.add(mv, grouped.get(mv.Movement, empty)) for mv in movements.itertuples()]
    routes, keys = rb.tables()
    return routes, keys, pd.Series(route_of, index=movements.index, name="Route Id")
