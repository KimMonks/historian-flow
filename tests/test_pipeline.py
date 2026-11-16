"""Quick checks on synthetic data (run: python -m pytest -q)."""
import numpy as np
import pandas as pd

from historian_flow import phase1, pipeline, reports, routes_v2, synthetic


def _run(days=1.0, seed=3):
    h, truth = synthetic.generate(days=days, seed=seed)
    raw = h.drop(columns="DateTime").astype(float)
    return pipeline.run(raw, h.DateTime), truth


def test_tag_types():
    res, _ = _run()
    t = res.p1.tag_type
    assert (t[[c for c in t.index if c.endswith(".TOTALISER")]] == phase1.ACCUMULATE).all()
    assert (t[[c for c in t.index if c.endswith(".SPEED")]] == phase1.CONTINUOUS).all()
    assert "SPARE.TAG" not in t.index          # dead tag dropped


def test_short_dropouts_are_bridged():
    x = pd.DataFrame({"a": [5.0, 5, 0, 5, 5, 0, 0, 5, 5, 0, 0, 0, 5, 5, 5],
                      "posn": [400.0] * 15})   # keeps rows from being all-zero
    c = phase1.cleanse(x)["a"].tolist()
    assert c[2] == 5 and c[5] == 5 and c[6] == 5   # 1- and 2-reading gaps filled
    assert c[9] == 0 and c[10] == 0 and c[11] == 0  # a 3-reading stop is real


def test_movements_recovered():
    res, truth = _run(days=2, seed=7)
    _, stats = reports.score_against_truth(res.movement_log, truth)
    assert stats["found"] == stats["true_movements"]
    assert stats["median_abs_tonnes_error_pct"] < 2.0


def test_routes_v2_are_pure():
    res, truth = _run(days=2, seed=7)
    al = routes_v2.aligned_tags(res.p1.active, res.p1.tag_type, res.movement_log)
    lab, tab = routes_v2.group_routes(res.movement_log, al)
    assert len(tab) == truth.route.nunique()


def test_clean_mode_fixes_glitches_and_levels():
    h, truth = synthetic.generate(days=2, seed=7)
    raw = h.drop(columns="DateTime").astype(float)
    tot = "BW101.TOTALISER"
    raw.loc[5000, tot] = 34.0                      # a glitch low reading, then recovery
    lvl = np.abs(np.sin(np.arange(len(raw)) / 300)) * 5000 + 1500
    raw["BIN1.INV"] = lvl                          # a bin level: rises and falls, mean > 1000
    faithful = pipeline.run(raw, h.DateTime)
    cleaned = pipeline.run(raw, h.DateTime, mode="clean")
    assert faithful.movement_log.tonnes.max() > 1e6           # the glitch leaks through in 2020 logic
    assert cleaned.movement_log.tonnes.max() < 50_000
    assert cleaned.p1.tag_type["BIN1.INV"] == "LEVEL"
    assert cleaned.p1.tag_type[tot] == phase1.ACCUMULATE


def test_clean_mode_routes_v3():
    res, truth = _run(days=2, seed=7)
    h, _ = synthetic.generate(days=2, seed=7)
    r = pipeline.run(h.drop(columns="DateTime").astype(float), h.DateTime, mode="clean")
    assert r.routes_v3.confirmed.sum() == truth.route.nunique()
