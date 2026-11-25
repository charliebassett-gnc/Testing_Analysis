"""
Analyse Hunter SE field-test logs copied from the Jetson (robot_control/logs).

Folder layout (everything lives next to this script):
  analyse_runs.py
  incoming\\            copy the Jetson logs in here (loose files or a whole folder)
  raw_data\\<date>\\     the script moves the logs here, sorted by the date in their file name
  results\\<date>\\      attempts.csv, overview.png and runN.png for that day
  waypoints\\           waypoint CSVs (falls back to the JetsonControl repo's waypoints folder)

Usage (laptop PowerShell):
  python analyse_runs.py                    sort incoming\\, analyse any new or unanalysed dates
  python analyse_runs.py --date 2026-11-25  re-analyse one date
  python analyse_runs.py --all              re-analyse every date
  python analyse_runs.py --date 2026-11-25 --runs 17 21

Each Jetson run (nav_wrap_test_runN + control_test_runN + metadata) can hold
several automatic attempts: drive forward under control, take over manually,
reposition, let it converge again. The script splits every run into attempts
and reports how each one converged onto the path.

How an attempt is detected
  The logs don't record who was driving, so it is inferred:
  - Forward speed along the robot's heading is computed from the positions.
  - An attempt is a stretch of forward driving near the commanded speed
    (>= 50 %) lasting at least --min-attempt seconds. Reversing or stopping
    splits attempts.
  - "auto_corr" is the correlation between the Jetson's steering command and
    the measured yaw rate. Close to +1 means the Jetson was steering;
    near 0 or negative suggests that stretch was manual driving.

Lateral offset = perpendicular distance from the NAV-960 position to the path
(+ = left of the path, looking along the direction of travel).

All runN.png graphs from one date share the same time axis so controllers
can be compared directly.

Requires: numpy, matplotlib  (pip install numpy matplotlib)
"""

import argparse
import csv
import filecmp
import glob
import json
import math
import os
import re
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
INCOMING = os.path.join(HERE, "incoming")
RAW = os.path.join(HERE, "raw_data")
RESULTS = os.path.join(HERE, "results")
WAYPOINT_DIRS = [os.path.join(HERE, "waypoints"),
                 r"C:\Users\cbasset\Documents\JetsonControl\waypoints"]

LOG_HZ = 50.0
CONVERGED_M = 0.10       # |offset| below this counts as on the line
CONVERGED_HOLD_S = 2.0   # ...and must stay there this long
TIME_STEP_S = 5          # shared time axis is rounded up to a multiple of this
MIN_OFFSET_AXIS_M = 0.5  # offset axis always shows at least +/- this

LOG_NAME = re.compile(r"_run\d+_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}")


# ---------------------------------------------------------------- sorting incoming

def sort_incoming():
    """Move dated Jetson log files from incoming\\ into raw_data\\<date>\\.
    Returns the set of dates that received new files."""
    os.makedirs(INCOMING, exist_ok=True)
    new_dates, skipped, clashes, dups = set(), [], [], 0
    for root, dirs, files in os.walk(INCOMING):
        for d in dirs:
            if d.lower() == "analysis":  # old outputs from the previous script version
                skipped.append(os.path.relpath(os.path.join(root, d), INCOMING) + "\\ (old analysis output, delete it)")
        dirs[:] = [d for d in dirs if d.lower() != "analysis"]
        for name in files:
            src = os.path.join(root, name)
            m = LOG_NAME.search(name)
            if not m:
                skipped.append(os.path.relpath(src, INCOMING))
                continue
            date = m.group(1)
            dst_dir = os.path.join(RAW, date)
            dst = os.path.join(dst_dir, name)
            os.makedirs(dst_dir, exist_ok=True)
            if os.path.exists(dst):
                if filecmp.cmp(src, dst, shallow=False):
                    os.remove(src)  # already have it
                    dups += 1
                else:
                    clashes.append(os.path.relpath(src, INCOMING))
                continue
            shutil.move(src, dst)
            new_dates.add(date)

    # tidy up folders left empty in incoming\
    for root, dirs, files in os.walk(INCOMING, topdown=False):
        if root != INCOMING and not os.listdir(root):
            os.rmdir(root)

    for d in sorted(new_dates):
        print(f"Moved new logs into raw_data\\{d}")
    if dups:
        print(f"Removed {dups} file(s) from incoming\\ that were already in raw_data")
    if clashes:
        print("\nWARNING: these files differ from ones already in raw_data and were left in incoming\\:")
        for c in clashes:
            print("   ", c)
    if skipped:
        print("\nLeft in incoming\\ (not a dated Jetson log):")
        for s in skipped:
            print("   ", s)
    return new_dates


# ---------------------------------------------------------------- loading

def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan
                         for r in rows]) for k in rows[0]}


def find_runs(log_dir):
    runs = {}
    for p in glob.glob(os.path.join(log_dir, "nav_wrap_test_run*_*.csv")):
        m = re.search(r"nav_wrap_test_run(\d+)_(.+)\.csv$", os.path.basename(p))
        if m:
            runs[int(m.group(1))] = (p, m.group(2))
    return dict(sorted(runs.items()))


def load_path(name, wp_dirs):
    for d in wp_dirs:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            pts = []
            with open(p) as f:
                for line in f:
                    parts = line.strip().split(",")
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except (ValueError, IndexError):
                        pass
            if len(pts) >= 2:
                return np.array(pts)
    m = re.match(r"line_(\d+(?:\.\d+)?)m", name)
    if m:  # line_15m.csv only exists on the Jetson; it is a straight line along x
        return np.array([(0.0, 0.0), (float(m.group(1)), 0.0)])
    return None


# ---------------------------------------------------------------- geometry

def path_errors(x, y, yaw_deg, path):
    """Signed lateral offset (m, + = left of path) and heading error (deg)."""
    a, b = path[:-1], path[1:]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    seg_yaw = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    off = np.empty(len(x))
    herr = np.empty(len(x))
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        t = np.clip(((p - a) * d).sum(1) / L2, 0, 1)
        q = a + t[:, None] * d
        dist = np.hypot(*(p - q).T)
        k = int(np.argmin(dist))
        cross = d[k, 0] * (p[1] - a[k, 1]) - d[k, 1] * (p[0] - a[k, 0])
        off[i] = math.copysign(dist[k], cross)
        herr[i] = (yaw_deg[i] - seg_yaw[k] + 180) % 360 - 180
    return off, herr


def smooth_diff(v, n):
    """Central difference over n samples, per second."""
    out = np.full(len(v), np.nan)
    h = n // 2
    out[h:-h] = (v[2 * h:] - v[:-2 * h]) / (2 * h / LOG_HZ)
    out[:h] = out[h]
    out[-h:] = out[-h - 1]
    return out


def segments(forward, reversing, min_len, max_gap):
    """Forward-driving stretches. Short dips (<= max_gap samples) are bridged
    unless the robot reversed during the dip."""
    raw, start = [], None
    for i, m in enumerate(np.append(forward, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            raw.append([start, i])
            start = None
    merged = []
    for s in raw:
        if merged and s[0] - merged[-1][1] <= max_gap and not reversing[merged[-1][1]:s[0]].any():
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged if s[1] - s[0] >= min_len]


# ---------------------------------------------------------------- analysis

def analyse_run(n, nav_path, tag, log_dir, args):
    """Returns a dict with the run's data and attempt rows, or None."""
    meta_p = nav_path.replace(".csv", "_metadata.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    ctrl_p = glob.glob(os.path.join(log_dir, f"control_test_run{n}_{tag}.csv"))

    nav = read_csv(nav_path)
    if not nav or len(nav["time"]) < 50:
        print(f"run {n}: too short, skipped")
        return None
    ctrl = read_csv(ctrl_p[0]) if ctrl_p else {}

    t = nav["time"] - nav["time"][0]
    x, y, yaw = nav["raw_x"], nav["raw_y"], np.unwrap(np.radians(nav["yaw"]))
    yaw_deg = np.degrees(yaw)
    N = len(t)
    steer = ctrl.get("steering_command", np.full(N, np.nan))[:N]
    steer = np.pad(steer, (0, N - len(steer)), constant_values=np.nan)

    v_cmd = float(meta.get("velocity", 0.5))
    mode = meta.get("controller_mode", "?")
    path_name = meta.get("waypoints_file", "?")
    path = load_path(path_name, WAYPOINT_DIRS)
    if path is None:
        print(f"run {n}: waypoint file {path_name} not found in {WAYPOINT_DIRS[0]}; offsets skipped")

    w = max(4, int(args.smooth * LOG_HZ) // 2 * 2)
    vx, vy = smooth_diff(x, w), smooth_diff(y, w)
    v_fwd = vx * np.cos(yaw) + vy * np.sin(yaw)
    yaw_rate = smooth_diff(yaw_deg, w)

    segs = segments(v_fwd > 0.5 * v_cmd, v_fwd < -0.1,
                    int(args.min_attempt * LOG_HZ), int(1.5 * LOG_HZ))

    off = herr = None
    if path is not None:
        off, herr = path_errors(x, y, yaw_deg, path)

    rows = []
    for k, (i0, i1) in enumerate(segs, 1):
        sl = slice(i0, i1)
        dist = float(np.nansum(np.hypot(np.diff(x[sl]), np.diff(y[sl]))))
        s, r = steer[sl], yaw_rate[sl]
        ok = ~np.isnan(s) & ~np.isnan(r)
        # Only meaningful when the Jetson actually asked for some steering
        corr = round(float(np.corrcoef(s[ok], r[ok])[0, 1]), 2) \
            if ok.sum() > 20 and s[ok].std() > 0.02 and r[ok].std() > 1e-3 else ""
        row = dict(run=n, attempt=k, path=path_name, mode=mode,
                   v_cmd=v_cmd, t_start=round(t[i0], 1), duration_s=round(t[i1 - 1] - t[i0], 1),
                   distance_m=round(dist, 1), auto_corr=corr, note="")
        if off is not None:
            o, h = off[sl], herr[sl]
            hold = int(CONVERGED_HOLD_S * LOG_HZ)
            inside = np.abs(o) < CONVERGED_M
            conv = next((j for j in range(len(o) - hold + 1) if inside[j:j + hold].all()), None)
            tail = o[-int(3 * LOG_HZ):]
            row.update(start_offset_m=round(o[0], 2), start_heading_deg=round(h[0], 1),
                       converge_s=round(conv / LOG_HZ, 1) if conv is not None else "",
                       converge_m=round(float(np.nansum(np.hypot(np.diff(x[i0:i0 + conv + 1]), np.diff(y[i0:i0 + conv + 1])))), 1) if conv is not None else "",
                       overshoot_m=round(float(-np.min(o * np.sign(o[0]))) if abs(o[0]) > CONVERGED_M else 0.0, 2),
                       rms_offset_m=round(float(np.sqrt(np.nanmean(o ** 2))), 3),
                       max_offset_m=round(float(np.nanmax(np.abs(o))), 2),
                       end_offset_m=round(float(np.nanmedian(tail)), 3))
            notes = []
            if abs(h[0]) > 90:
                notes.append("starts facing away from path: start may still be manual")
            if conv is None and abs(o[0]) > CONVERGED_M:
                notes.append("never converged")
            row["note"] = "; ".join(notes)
        if corr != "" and corr < 0.3:
            row["note"] = "; ".join(filter(None, [row["note"], "steering and yaw rate don't match: may be manual"]))
        rows.append(row)

    return dict(n=n, t=t, x=x, y=y, off=off, segs=segs, path=path, path_name=path_name,
                mode=mode, v_cmd=v_cmd, rows=rows)


# ---------------------------------------------------------------- plotting

def plot_run(run, t_max, out_dir):
    n, t, x, y, off, segs = run["n"], run["t"], run["x"], run["y"], run["off"], run["segs"]
    path, path_name = run["path"], run["path_name"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 2]})
    ax[0].plot(x, y, color="0.75", lw=1, label="whole log (incl. manual)")
    if path is not None:
        ax[0].plot(path[:, 0], path[:, 1], "k--", lw=2, label=f"path {path_name}")
    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, (i0, i1) in enumerate(segs):
        c = colours[k % 10]
        ax[0].plot(x[i0:i1], y[i0:i1], color=c, lw=2, label=f"attempt {k + 1}")
        ax[0].plot(x[i0], y[i0], "o", color=c)
        if off is not None:
            ax[1].plot(t[i0:i1] - t[i0], off[i0:i1], color=c, lw=1.5, label=f"attempt {k + 1}")
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].set_title(f"Run {n}: {path_name}, {run['mode']}, {run['v_cmd']} m/s  "
                    f"({len(segs)} attempt{'s' if len(segs) != 1 else ''}, dots = start)")
    ax[0].grid(True); ax[0].legend(fontsize=7, ncol=2)
    ax[1].axhspan(-CONVERGED_M, CONVERGED_M, color="g", alpha=0.1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlim(0, t_max)
    # At least +/-MIN_OFFSET_AXIS_M so small tracking errors aren't blown up
    lo, hi = ax[1].get_ylim()
    ax[1].set_ylim(min(lo, -MIN_OFFSET_AXIS_M), max(hi, MIN_OFFSET_AXIS_M))
    ax[1].set_xlabel("Time since attempt start (s)"); ax[1].set_ylabel("Path error (m)")
    ax[1].grid(True)
    if segs and off is not None:
        ax[1].legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"run{n}.png"), dpi=100)
    plt.close(fig)


def plot_overview(runs, out_dir):
    overview = {}
    for run in runs:
        ov = overview.setdefault(run["path_name"], {"path": run["path"], "lines": []})
        for k, (i0, i1) in enumerate(run["segs"], 1):
            ov["lines"].append((f"r{run['n']}.{k} {run['mode']} {run['v_cmd']}",
                                run["x"][i0:i1], run["y"][i0:i1]))
    paths = list(overview.items())
    fig, axs = plt.subplots(len(paths), 1, figsize=(11, 5 * len(paths)), squeeze=False)
    for a, (name, ov) in zip(axs[:, 0], paths):
        if ov["path"] is not None:
            a.plot(ov["path"][:, 0], ov["path"][:, 1], "k--", lw=2.5, label="path")
        for label, xs, ys in ov["lines"]:
            a.plot(xs, ys, lw=1.2, label=label)
            a.plot(xs[0], ys[0], "o", ms=3, color=a.lines[-1].get_color())
        a.set_title(f"{name}: every automatic attempt (dots = start)")
        a.set_aspect("equal", "datalim"); a.grid(True); a.legend(fontsize=6, ncol=4)
        a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "overview.png"), dpi=100); plt.close(fig)


# ---------------------------------------------------------------- one date

COLS = ["run", "attempt", "path", "mode", "v_cmd", "t_start", "duration_s", "distance_m", "auto_corr",
        "start_offset_m", "start_heading_deg", "converge_s", "converge_m", "overshoot_m",
        "rms_offset_m", "max_offset_m", "end_offset_m", "note"]


def analyse_date(date, args):
    log_dir = os.path.join(RAW, date)
    out_dir = os.path.join(RESULTS, date)
    print(f"\n=== {date} ===")

    # Every run of the day is analysed so the shared time axis is the same
    # whether or not --runs picks a subset to re-plot.
    runs = [r for r in (analyse_run(n, p, tag, log_dir, args)
                        for n, (p, tag) in find_runs(log_dir).items()) if r]
    all_rows = [r for run in runs for r in run["rows"]]
    if not all_rows:
        print("No runs with automatic attempts found.")
        return
    longest = max(r["duration_s"] for r in all_rows)
    t_max = TIME_STEP_S * math.ceil(longest / TIME_STEP_S)

    os.makedirs(out_dir, exist_ok=True)
    if args.runs:
        runs = [run for run in runs if run["n"] in args.runs]
    for run in runs:
        plot_run(run, t_max, out_dir)
    rows = [r for run in runs for r in run["rows"]]
    if not args.runs:  # a subset shouldn't overwrite the full day's table/overview
        plot_overview(runs, out_dir)
        with open(os.path.join(out_dir, "attempts.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            wr.writeheader(); wr.writerows(rows)

    hdr = f"{'run':>4} {'att':>3} {'path':12} {'mode':4} {'v':>3} {'t0':>6} {'dur':>5} {'dist':>5} {'auto':>5} " \
          f"{'off0':>6} {'hdg0':>6} {'conv_s':>6} {'conv_m':>6} {'over':>5} {'rms':>6} {'end':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:>4} {r['attempt']:>3} {r['path'][:12]:12} {r['mode'][:4]:4} {r['v_cmd']:>3} "
              f"{r['t_start']:>6} {r['duration_s']:>5} {r['distance_m']:>5} {r['auto_corr']:>5} "
              f"{r.get('start_offset_m', ''):>6} {r.get('start_heading_deg', ''):>6} {str(r.get('converge_s', '')):>6} "
              f"{str(r.get('converge_m', '')):>6} {r.get('overshoot_m', ''):>5} {r.get('rms_offset_m', ''):>6} {r.get('end_offset_m', ''):>6}")
    saved = "run graphs" if args.runs else "attempts.csv, overview.png and run graphs"
    print(f"\nSaved {saved} (time axis 0-{t_max} s) to results\\{date}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="re-analyse this date (YYYY-MM-DD)")
    ap.add_argument("--all", action="store_true", help="re-analyse every date in raw_data")
    ap.add_argument("--runs", type=int, nargs="*", help="only these run numbers")
    ap.add_argument("--min-attempt", type=float, default=3.0, help="shortest attempt in seconds (default 3)")
    ap.add_argument("--smooth", type=float, default=0.5, help="speed/yaw-rate smoothing window in s (default 0.5)")
    args = ap.parse_args()

    for d in (INCOMING, RAW, RESULTS, WAYPOINT_DIRS[0]):
        os.makedirs(d, exist_ok=True)

    new_dates = sort_incoming()
    all_dates = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))

    if args.date:
        if args.date not in all_dates:
            print(f"No raw_data\\{args.date} folder. Dates available: {', '.join(all_dates) or 'none'}")
            return
        dates = [args.date]
    elif args.all:
        dates = all_dates
    else:
        dates = [d for d in all_dates
                 if d in new_dates or not os.path.isfile(os.path.join(RESULTS, d, "attempts.csv"))]

    if not dates:
        print("Nothing new to analyse. Copy Jetson logs into incoming\\, or use --date / --all to re-run.")
        return
    for d in dates:
        analyse_date(d, args)

    print("\nauto = steering-vs-yaw-rate correlation (near +1 = Jetson steering; low = probably manual)")
    print(f"conv = time/distance until |offset| < {CONVERGED_M} m for {CONVERGED_HOLD_S} s (blank = never)")


if __name__ == "__main__":
    main()
