"""Route discovery v3 - chain-first (real-data mode, new).

Findings on real port historian data that shaped this:
  * Belts keep running between pours and stockpile changes, so tags seldom
    start and stop with an individual movement - v2's alignment test rarely
    fires on real data.
  * A big port runs many belts at once (a typical belt is on during ~45% of all
    movements), so "running throughout" (the thesis test) binds dozens of
    unrelated tags.
  * The weightometers that saw the same tonnes are the strongest evidence.

So a route is identified by its chain of weightometers. A partial chain (a
scale missed the match) is folded into the full chain it belongs to, and the
route's equipment is described by specificity: tags running through this
route's movements far more often than through movements in general.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .phase1 import ACCUMULATE


def coverage(active: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each movement's window during which each tag was active."""
    A = active.to_numpy().astype(np.int32)
    C = np.vstack([np.zeros((1, A.shape[1]), np.int32), np.cumsum(A, axis=0)])
    f, l = log.first_row.to_numpy() - 1, log.last_row.to_numpy() - 1   # activity rows start at 1
    return pd.DataFrame((C[l + 1] - C[f]) / (l - f + 1)[:, None], columns=active.columns, index=log.index)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def _split(on: pd.DataFrame, base: pd.Series, confirm: int, similarity: float = 0.5,
           off: pd.DataFrame | None = None, min_frac: float = 0.0) -> np.ndarray:
    """Split one chain's movements into branches by their non-generic equipment.

    Candidate tags run through some of the chain's movements but not others
    (20-80%) and are not running during most movements plant-wide. A branch is
    kept only if at least `confirm` movements share it; others join the
    nearest kept branch."""
    share = on.mean()
    cand = share[(share >= 0.2) & (share <= 0.8) & (base < 0.5)].index
    if off is not None and len(cand):
        # the tag must be clearly OFF (not just partly on) when not part of the branch
        clear = ((on[cand]) | (off[cand])).mean()
        cand = clear[clear >= 0.9].index
    if len(cand) == 0:
        return np.zeros(len(on), dtype=int)
    sig = [frozenset(cand[r]) for r in on[cand].to_numpy()]
    freq = pd.Series(sig).value_counts()
    cents = []
    for s in freq.index:
        if not any(_jaccard(s, c) >= similarity for c in cents):
            cents.append(s)
    assign = np.array([max(range(len(cents)), key=lambda k: _jaccard(s, cents[k])) for s in sig])
    counts = np.bincount(assign, minlength=len(cents))
    keep = [k for k in range(len(cents)) if counts[k] >= max(confirm, min_frac * len(on))]
    if len(keep) <= 1:
        return np.zeros(len(on), dtype=int)
    final = np.array([max(keep, key=lambda k: _jaccard(s, cents[k])) for s in sig])
    remap = {k: i for i, k in enumerate(sorted(set(final), key=lambda k: -counts[k]))}
    return np.array([remap[k] for k in final])


def group_routes(log: pd.DataFrame, cov: pd.DataFrame, tag_type: pd.Series,
                 min_cover: float = 0.9, min_share: float = 0.8, min_lift: float = 1.5,
                 confirm: int = 3, split_branches: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Returns (route label per movement, route table)."""
    chains = log.weightometers.apply(lambda w: frozenset(w.split(" > ")))
    counts = chains.value_counts()

    # fold a partial multi-scale chain into the clearly dominant chain containing it
    parent = {}
    for s, n in counts.items():
        if len(s) < 2:
            continue                      # single-scale runs are often genuine (merging flows)
        sup = sorted(((counts[t], t) for t in counts.index if s < t), key=lambda x: -x[0])
        if sup and sup[0][0] >= n and (len(sup) == 1 or sup[0][0] >= 1.5 * sup[1][0]):
            parent[s] = sup[0][1]

    def root(s):
        while s in parent:
            s = parent[s]
        return s

    route_chain = chains.apply(root)
    nonacc = [c for c in cov.columns if tag_type.get(c) != ACCUMULATE]
    on = cov[nonacc] >= min_cover
    off = cov[nonacc] <= 0.1
    base = on.mean().clip(lower=0.02)

    # branches: same scales, different destination equipment (e.g. stacker 1 vs 2)
    branch = pd.Series(0, index=log.index)
    if split_branches:
        for c in route_chain.unique():
            idx = log.index[(route_chain == c).to_numpy()]
            if len(idx) < 2 * confirm:
                continue
            branch.loc[idx] = _split(on.loc[idx], base, confirm, off=off.loc[idx], min_frac=0.2)

    keys = list(zip(route_chain, branch))
    size = pd.Series(keys).value_counts()
    names = {k: f"R{i + 1:03d}" for i, k in enumerate(size.index)}
    label = pd.Series([names[k] for k in keys], index=log.index, name="route")

    rows = []
    for (c, b), name in names.items():
        m = (label == name).to_numpy()
        sub = log[m]
        share = on[m].mean()
        lift = share / base
        spec = share[(share >= min_share) & (lift >= min_lift)].sort_values(ascending=False)
        rows.append({
            "route": name, "movements": int(m.sum()), "confirmed": bool(m.sum() >= confirm),
            "weightometers": " + ".join(sorted(c)), "n_scales": len(c), "branch": int(b),
            "partial_chains_folded": int((m & (chains != route_chain).to_numpy()).sum()),
            "tonnes": round(float(sub.tonnes.sum()), 0),
            "median_tonnes": round(float(sub.tonnes.median()), 0),
            "median_rate_tph": round(float(sub.rate_tph.median()), 0),
            "median_scale_spread_pct": round(float(sub.scale_spread_pct.median()), 2) if len(c) > 1 else np.nan,
            "specific_equipment": ", ".join(spec.index[:15]),
        })
    table = pd.DataFrame(rows)
    return label, table
