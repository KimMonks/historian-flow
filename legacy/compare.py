"""Compare the rebuild against outputs saved by run_original.py.

    python3 legacy/compare.py <input_nodt.csv> <original_out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from historian_flow import phase1, phase2  # noqa: E402

inp, od = sys.argv[1], sys.argv[2]
rd = lambda n: pd.read_csv(os.path.join(od, n + ".csv"), index_col=0)
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def num_equal(a, b, rtol=1e-8):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=1e-6, equal_nan=True)


raw = pd.read_csv(inp, dtype=float)
p1 = phase1.run_phase1(raw)
info, move, conf = phase2.events(p1.active, p1.tote, p1.diff, p1.tag_type)
movements, members = phase2.match_movements(move)
routes, keys, route_of = phase2.build_routes(movements, members, p1.active, p1.tag_type)

# ---- phase 1
o = rd("clean")
check("cleansed values", list(o.columns) == list(p1.clean.columns) and num_equal(o, p1.clean),
      f"{o.shape}")
o = rd("tag_type").iloc[0]
check("tag types", (o == p1.tag_type.reindex(o.index)).all(),
      ", ".join(f"{k}={v}" for k, v in p1.tag_type.value_counts().items()))
o = rd("diff")
check("increments", num_equal(o, p1.diff) and list(o.index) == list(p1.diff.index), f"{o.shape}")
o = rd("activity")
check("activity flags", num_equal(o, p1.active), f"{int(o.to_numpy().sum()):,} active cells")
o = rd("tote")
check("running tote", num_equal(o, p1.tote))

# ---- events
o = rd("event_info").reset_index(drop=True)
same = (len(o) == len(info) and (o.Sequence.to_numpy() == info.Sequence.to_numpy()).all()
        and (o.Tag.to_numpy() == info.Tag.to_numpy()).all() and (o.Event.to_numpy() == info.Event.to_numpy()).all()
        and (o.Type.to_numpy() == info.Type.to_numpy()).all()
        and all(num_equal(o[c], info[c]) for c in ["Count", "Increment", "Tote"]))
check("all events", same, f"{len(o):,} events")
o = rd("event_move").reset_index(drop=True)
same = (len(o) == len(move) and (o.Tag.to_numpy() == move.Tag.to_numpy()).all()
        and all(num_equal(o[c], move[c]) for c in ["Sequence", "Count", "Tote"]))
check("totaliser movement events", same, f"{len(o)} events")
o = rd("conf")
check("co-activity matrix", num_equal(o, conf))

# ---- movement matching (original wide layout -> primary + member slots)
o = rd("move_tags").reset_index(drop=True)
ok = len(o) == len(movements) and (o.Tag.to_numpy() == movements.Tag.to_numpy()).all() \
    and num_equal(o.Matched, movements.Matched) and num_equal(o.Tote, movements.Tote) \
    and num_equal(o.Sequence, movements.Sequence)
mem_o = []
for i, r in o.iterrows():
    for s in range(int(r.Matched)):
        base = 6 + 4 * s
        mem_o.append((i, s + 1, r[str(base)], float(r[str(base + 1)]), float(r[str(base + 2)]), float(r[str(base + 3)])))
mem_o = pd.DataFrame(mem_o, columns=["Movement", "Slot", "Tag", "Count", "Tote", "Sequence"])
ok = ok and len(mem_o) == len(members) and (mem_o.Tag.to_numpy() == members.Tag.to_numpy()).all() \
    and all(num_equal(mem_o[c], members[c]) for c in ["Movement", "Slot", "Count", "Tote", "Sequence"])
check("movement matching", ok, f"{len(o)} movements, {len(mem_o)} matched secondary runs")

# ---- routes (original columns are positional: 3 + 3*slot = tag, +1 count, +2 tonnes)
o = rd("event_routes").reset_index(drop=True)
ok = len(o) == len(routes) and (o["Route Id"].to_numpy() == routes["Route Id"].to_numpy()).all() \
    and num_equal(o.Matched, routes.Matched) and num_equal(o.Occurs, routes.Occurs)
width = max(int(routes.Matched.max()) + 1, 1)
for s in range(width):
    ot, oc, ow = o.iloc[:, 3 + 3 * s], o.iloc[:, 4 + 3 * s], o.iloc[:, 5 + 3 * s]
    nt, nc, nw = routes[f"Tot{s + 1}"], routes[f"Tot{s + 1}Count"], routes[f"Tot{s + 1}Tonnes"]
    has = nt.notna().to_numpy()
    ok = ok and (ot[has].to_numpy() == nt[has].to_numpy()).all() \
        and num_equal(oc[has], nc[has]) and num_equal(ow[has], nw[has])
tagcols = list(p1.active.columns)
ok = ok and num_equal(o[tagcols], routes[tagcols])
check("route discovery", ok, f"{len(o)} routes")
o = rd("event_route_keys").reset_index(drop=True)
check("route keys", (o[tagcols].astype(bool).to_numpy() == keys[tagcols].to_numpy()).all())

n_fail = sum(not r[1] for r in results)
print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
sys.exit(1 if n_fail else 0)
