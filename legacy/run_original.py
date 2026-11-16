"""Run the original 2020 thesis ids.py pipeline (the 'all' run type, phases 1-3 calibration)
unchanged, on a wide CSV, saving every intermediate table and step timings.

Usage (Python 3.8 + pandas 1.0.5 + bitarray 1.2.2, as in 2020):
    python3.8 -I run_original.py <path/to/ids.py> <input.csv> <out_dir>
"""
import importlib.util, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))
import pandas as pd

ids_path, csv_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("original_ids", ids_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cmd = mod.Command()

T = {}
def step(name, fn, *a):
    t0 = time.perf_counter(); r = fn(*a); T[name] = time.perf_counter() - t0
    print(f"{name}: {T[name]:.1f}s", flush=True)
    return r

# same preamble as handle() for run_type 'all'
df_orig = pd.read_csv(csv_path, dtype=float)
df_orig = df_orig.loc[(df_orig != 0).any(axis=1)]

df, df_diff, df_tag_type, df_tag_sig, accum_col = step("get_type_noroute", cmd.get_type_noroute, df_orig)
df_activity, df_tote, df_ba_activity, df_activity_count = step("get_activity", cmd.get_activity, df_diff, df_tag_type)
df_event, df_conf, df_event_move, df_event_info = step("get_events", cmd.get_events, df_activity, df_tag_type, df_tote, df_diff)
df_event_move.to_csv(os.path.join(out, "event_move.csv"))
df_move_tags = step("move_tags", cmd.move_tags, df_event_move)
df_event_routes, df_event_route_keys = step("event_routes", cmd.event_routes, df_move_tags, df_activity, df_tag_type)

for name, d in [("clean", df), ("diff", df_diff), ("tag_type", df_tag_type), ("activity", df_activity),
                ("tote", df_tote), ("event_info", df_event_info), ("conf", df_conf),
                ("move_tags", df_move_tags), ("event_routes", df_event_routes),
                ("event_route_keys", df_event_route_keys)]:
    d.to_csv(os.path.join(out, name + ".csv"), float_format="%.10g")
json.dump(T, open(os.path.join(out, "timings.json"), "w"), indent=1)
print("done", json.dumps(T))
