# historian-flow

**Reconstruct material movements and conveyor routes from raw industrial historian data.**

Plants such as ports, mines and processing facilities store years of historian data
(OSIsoft/AVEVA PI, Honeywell PHD and similar): thousands of tags sampled every few seconds.
Very little of it is ever analysed. historian-flow turns a raw tag extract into an
account of what the plant actually did:

- **Movement log**: every material movement, with start and end times, the weightometers that
  saw it, tonnes on each scale, scale agreement (a built-in reconciliation check), duration and rate
- **Route map**: which scale chains and equipment run together, how often and how much they move
- **Tag profile**: each tag's inferred signal type (totaliser, speed, run status, percentage,
  position, level) and its operating range

It needs no plant model, P&IDs or tag documentation. Routes and equipment are inferred from the data.

```
17,280 rows x 47 tags  (load 0.09s, pipeline 0.14s, mode clean)
49 movements | routes: 32 (original method), 8 (v2), 8 (v3; 8 confirmed) -> output/
```

## Background

This is a rebuild of the calibration stage from my Master's research thesis:

> Monks, Kimberley (2020). *Cyber Intrusion Detection in Port Bulk Handling Operations*.
> Research Thesis, Master of Computer and Information Science, University of South Australia.

The thesis learned a port's normal material movements and routes from historian data, as a
baseline for detecting cyber intrusions in operational technology. The 2020 implementation
was a Django application that took about 12 hours to process 8 hours of data. This rebuild
reproduces its results exactly, runs thousands of times faster, and adds fixes developed
on real port data.

## Quick start

```bash
pip install pandas numpy pytest
python -m pytest -q                                          # 6 passed
python -m historian_flow.pipeline data/synthetic_port.csv --mode clean --out output
```

Open `output/movement_log.csv` and `output/routes_v3.csv` in Excel or pandas.

Make more demo data:

```bash
python -m historian_flow.synthetic --days 7 --areas 4 --out data/site.csv
```

## Your own data

One CSV in **wide** format: one row per 10-second interval, one column per tag, numeric values,
blanks for bad or missing readings. A `DateTime` first column is recommended. Without it, pass
`--start "2026-01-01 00:00:00" --interval 10s`.

Include the weightometer totalisers (essential), plus conveyor speeds and run status, feeder
rates, and machine status for the same area. In PI DataLink this is a *Sampled Data* query
at a 10 s interval.

```bash
python -m historian_flow.pipeline my_extract.csv --mode clean --out results
```

A week of data across several hundred tags runs in seconds on a laptop.

## Two modes

| Mode | Use it for |
|------|-----------|
| `--mode faithful` (default) | Reproducing the 2020 thesis: identical results to the original code |
| `--mode clean` | **Real plant data**: adds the fixes below and chain-first route discovery |

What clean mode adds, and why. Each fix came from running the original logic on real port data:

1. **Level tags are not totalisers.** Bin inventories average over 1,000, so the thesis rules
   class them as accumulators. A tag whose falls exceed 30% of its rises is reclassified as `LEVEL`.
2. **Short blank gaps in totalisers are interpolated** (up to 5 min), so runs aren't broken
   and tonnes aren't lost.
3. **Totaliser jump cap** (250 t per 10 s). A reading that glitches low and then recovers otherwise
   produces "movements" of hundreds of millions of tonnes.
4. **Runs under 5 minutes are ignored.**
5. **Chain-first routes (`routes_v3.py`).** On real plants belts keep running between movements,
   and a typical belt is on during a large share of all movements. "Running throughout a
   movement" therefore binds many unrelated tags, which is the thesis's *phantom route* effect.
   v3 identifies a route by the chain of weightometers that saw the same tonnes, folds partial
   chains into full ones, splits a chain into branches only on clearly on/off destination
   equipment, and describes each route by its *specific* equipment (tags running through its
   movements far more than through movements generally).

On real port data clean mode reduced the reported routes by more than 95%. Most of the
remaining routes recur and cover nearly all movements, and their inferred equipment matches the
plant's physical layout (for example, each car dumper's scale linked to its own apron feeder).

**Known limit:** where two flows merge onto one belt, the downstream scale carries the combined
tonnes, so one-to-one tonnage matching can't pair them. Those runs appear as single-scale routes.

## How it works

| Stage | File | What happens |
|---|---|---|
| Cleanse | `phase1.py` | drop empty rows and tags, bridge 1-2 reading dropouts |
| Classify | `phase1.py` | infer each tag's type from its statistics |
| Increments and activity | `phase1.py` | totaliser deltas; when each tag is active; running tonnes |
| Events | `phase2.py` | start/stop events for every tag; totaliser runs |
| Movement matching | `phase2.py` | pair runs on different scales with matching duration and tonnes |
| Routes (thesis method) | `phase2.py` | tags active throughout each movement |
| Routes v2 | `routes_v2.py` | tags that start and stop with the movement (best on clean, simulated data) |
| Real-data fixes | `clean.py` | level tags, gap fill, jump cap, minimum run |
| Routes v3 | `routes_v3.py` | chain-first routes with specific equipment (best on real data) |
| Reports | `reports.py` | movement log, route summaries, tag profile |
| Demo plant | `synthetic.py` | fictional port with 8 routes and ground truth |

## Verified against the original

The original 2020 code was run **unchanged** in a rebuilt 2020 environment (Python 3.8,
pandas 1.0.5, numpy 1.18.5, bitarray 1.2.2), with small stand-ins for Django (`legacy/`).
Every intermediate table (cleansed values, tag types, increments, activity, tote, events,
movement events, co-activity matrix, movement matching, routes and route keys) was compared
with the rebuild's.

| Synthetic input | Original | Rebuild | Speed-up | Tables identical |
|---|---|---|---|---|
| 12 h, 46 tags | 192 s | 0.06 s | ~3,100x | 11/11 |
| 12 h, 92 tags, extra dropouts | 575 s | 0.11 s | ~5,300x | 11/11 |
| 2 days, 46 tags | 748 s | 0.16 s | ~4,700x | 11/11 |

`QUIRKS.md` lists behaviour in the original code that faithful mode reproduces on purpose,
several of which explain the phantom routes. To rerun the comparison, place the original
`ids.py` beside `legacy/run_original.py` (see its docstring).

## Results on the synthetic plant

| Site | Movements found | Median tonnes error | Routes: thesis method | Routes: v3 confirmed | True routes |
|---|---|---|---|---|---|
| 1 area, 2 days | 49 / 49 | 1.0% | 32 | 8 | 8 |
| 8 areas, 7 days | 1,326 / 1,326 | 0.6% | 1,142 | 64 | 64 |

*The demo plant is synthetic, so treat these figures as proof that the logic works, not as
field accuracy.*

## Roadmap

- Many-to-one tonnage matching for merging and splitting flows
- Restrict matching to physically connected scales
- Port the thesis's monitoring stage (live comparison against learned routes, alerting) onto routes v3

## Licence

MIT. See `LICENSE`.
