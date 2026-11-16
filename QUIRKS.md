# Quirks found in the original `ids.py`

The rebuild reproduces all of these on purpose, so its output matches the 2020 code
exactly. Each one is a candidate for a deliberate fix in a later version.
Line numbers refer to `ids.py` (June 2020).

| # | Where | What happens | Effect |
|---|-------|--------------|--------|
| 1 | `get_diff` (l.1204) | `prev < cur + 100` is always true when `cur > prev + 1`, so it never filters anything. The intent was probably `cur < prev + 100`. | Totaliser jumps (rollovers, spikes) are not removed at the increment stage. `get_activity` partly compensates by restarting the tote after a jump of 100 or more. |
| 2 | `event_routes`, existing route (l.902) | Secondary-scale tonnes use `+=` where the first slot uses `=`, so the recorded tonnes keep growing each time a route repeats. | Route weight limits drift upward, which loosens the "invalid value" alert test in `test_normal`. |
| 3 | `event_routes`, existing route (l.889) | A new route is bound over the movement window extended by the scale lag. An existing route is re-bound over the primary scale's window only. | The binding counts for repeated routes cover a different window from the first occurrence. |
| 4 | `compare_all_tags` (l.551) | A movement matches a route only if its tag pattern equals the route's accumulated pattern (the OR of all earlier movements). | Once a route picks up one extra tag, identical later movements no longer match and spawn a new route. **A main driver of phantom routes**, together with #5. |
| 5 | `bind_all_acc_routes` (l.150) | Every non-totaliser tag active for the whole window is bound, including equipment from a different route that happens to run at the same time, and always-on signals such as machine positions. | Route identity depends on what else the plant was doing. On the synthetic site this gives 32 routes against 8 real ones (339 against 64 at 8 areas). See `routes_v2.py`. |
| 6 | `bind_all_acc_routes` | Totaliser windows are OR-ed cumulatively, so each totaliser's bound count depends on its position in the list. | Counts are order dependent (cosmetic for route keys, which only use non-zero). |
| 7 | `event_routes` column set-up (l.824-831) | `PCA9*` and `PCA10*` are inserted at positions already used by `PCA8*`, so the labels of slots 8-10 are shuffled. The positional writes are correct. | Cosmetic: the saved files carry misleading column names for slots 8-10. |
| 8 | `handle()` preamble | All-zero rows are dropped without resetting the index, but every later step uses `.at[i]` and assumes rows 0..n-1. | A single all-zero row in the input would break the run. The rebuild resets the index. |
| 9 | `move_tags` (l.490) | Duration and tonnes tolerances are relative to the first scale only, and nothing checks that the two scales are physically connected. | With several independent circuits, unrelated runs with similar timing and tonnes are sometimes paired (57 of 1,272 movements in the 8-area, 7-day test). |

## Confirmed on real port data

- **#1 matters in practice.** Totaliser readings that glitch to a small value and then recover produce
  enormous increments and "movements" of hundreds of millions of tonnes. Fixed in `--mode clean`.
- **Level tags.** Bin inventories have a mean over 1,000 and are typed as accumulators, which creates
  false movements. Fixed in `--mode clean`.
- **#4/#5 at plant scale.** Almost every movement becomes its own route, because a typical belt is
  running during a large share of all movements. See `routes_v3.py`.
