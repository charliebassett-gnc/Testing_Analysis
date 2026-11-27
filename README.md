# Testing_Analysis

Analysis of AgileX Hunter SE field tests. The robot runs the path following stack from
[JetsonControl](https://github.com/charliebassett-gnc/JetsonControl) on a Jetson Orin Nano,
with position and heading from a Trimble NAV-960. This repo turns the Jetson's run logs into
per-attempt convergence metrics and graphs so steering controllers (PID vs ONNX/RL) can be compared.

Browse [`results/`](results) to see the graphs from each test day.

IF CLONING REPOSITORY MANUALLY ADD FOLDERS THAT ARE MISSING AS SHOWN BELOW  

## Folder layout

```
Testing_Analysis/
  analyse_runs.py      the analysis script
  incoming/            drop Jetson logs here (not committed)
  raw_data/<date>/     logs sorted by test date (not committed; archive on the shared Drive)
  results/<date>/      attempts.csv, overview.png, runN.png (and noise_distribution.png) for each test day
  waypoints/           waypoint CSVs for the test paths
```

## Setup

Python 3 with numpy and matplotlib:

```powershell
pip install numpy matplotlib
```

## Workflow

1. Connect laptop and Jetson to same wifi network and copy the logs off the Jetson (run this in **laptop PowerShell**, not in an SSH session):

   ```powershell
   scp -r trimble@trimble.local:~/trimble-agis-fyp/robot_control/logs "$HOME\Documents\Testing_Analysis\incoming\"
   ```

   Loose files or whole folders are both fine.

2. Run the script (from any folder):

   ```powershell
   python analyse_runs.py
   ```

   It moves every `*_runN_YYYY-MM-DD_HH-MM*` file into `raw_data/<date>/`, then analyses any
   date that's new or has no results yet, writing to `results/<date>/`.

3. Commit the new `results/<date>/` folder so the team can see it, and upload
   `raw_data/<date>/` to the shared Google Drive.

The script never overwrites files. A file that's identical to one already in `raw_data` is
removed from `incoming`. A file with the same name but different contents, or anything that isn't
a dated Jetson log, stays in `incoming` and is listed in the output.

### Options

| Option | Effect |
| --- | --- |
| `--date 2026-11-25` | Re-analyse one date |
| `--all` | Re-analyse every date in `raw_data` |
| `--runs 17 21` | Only redraw these runs' graphs (with `--date`; `attempts.csv` and `overview.png` are left alone) |
| `--min-attempt 3` | Shortest automatic attempt in seconds |
| `--smooth 0.5` | Speed and yaw-rate smoothing window in seconds |

## What the outputs mean

The Jetson logs one run from Start to Stop. During a run the operator may take manual control,
drive the robot to a new start position and hand back, so one run can contain several
**attempts**. The logs don't record who was driving, so attempts are inferred:

- Forward speed along the robot's heading is computed from the logged positions.
- An attempt is forward driving at 50 % or more of the commanded speed for at least
  `--min-attempt` seconds. Reversing or stopping starts a new attempt.
- `auto_corr` is the correlation between the Jetson's steering command and the measured yaw rate.
  Near +1 means the Jetson was steering; low or negative suggests manual driving.

**Lateral offset** is the perpendicular distance from the NAV-960's reported position to the
path, positive to the left of the path looking along the direction of travel. That point is
wherever the receiver's antenna offset puts it (ideally the rear-axle centre). It's not the
Jetson's `cross_track_error`, which is measured from a point projected 5 m ahead.

### Injected sensor noise

Runs from 2026-11-27 on can have synthetic noise injected into the NAV values (see JetsonControl's
*Sensor Noise Injection*). The run metadata then has a `sensor_noise` block (`enabled`, `setting`
such as `HIGH:pos,yaw`, `channels`, `level`, std/bias per channel, `seed`) and the Jetson also
writes a `sensor_noise_runN_*.csv` with the clean and noisy NAV `x, y, yaw, yaw_rate, v` each tick
and an `active` flag. Its run number is counted separately from `nav_wrap_test`, so the script
matches it by the timestamp in the file name and identical tick times.

With noise on, `nav_wrap_test`'s position and heading are the noisy values the controller saw.
The script removes the noise exactly (noisy ? clean, rotated into the local path frame), so
**every metric and the solid lines in the graphs use the true position and heading**. The noisy
position and offset are drawn faintly in `runN.png`. Without a matching `sensor_noise` log the
noisy position is used and a warning is printed.

### Graphs

- `runN.png`: the trajectory with each attempt coloured (dots mark the start), and lateral
  offset against time since each attempt started. The green band is ±0.10 m.
  All graphs from one day share the same time axis (0 to the day's longest attempt, rounded up
  to 5 s) so controllers can be compared directly. The offset axis shows at least ±0.5 m.
  Runs with injected noise also show the noisy position the Jetson saw (thin grey) and the
  noisy offset (faint, same colour); the noise setting is in the title.
- `overview.png`: every automatic attempt from the day overlaid, one panel per path.
- `noise_distribution.png` (only on days with noise): histograms of the injected noise
  (noisy ? clean from `sensor_noise_run*.csv`, ticks with noise active, NAV units), one row per
  channel (`x`, `y`, `yaw`, `yaw_rate`, `v`) and one column per level (`LOW`/`MED`/`HIGH`), pooled
  over the runs with that channel on. Each panel overlays the normal fitted to the data (solid)
  and the configured N(bias, std) from the metadata (dashed), with mean, std and % within 1 std.
  Every preset uses the same seed, so runs at one level replay the same sequence and pooled
  samples repeat (the panel shows the number of distinct draws).

### attempts.csv

| Column | Meaning |
| --- | --- |
| `run`, `attempt` | Jetson run number and attempt within it |
| `path`, `mode`, `v_cmd` | Waypoint file, controller (`pid`/`onnx`) and commanded speed (m/s), from the run's metadata |
| `noise` | Injected sensor noise setting from the metadata (`OFF`, or e.g. `HIGH:pos,yaw`); blank for logs from before noise was recorded |
| `t_start`, `duration_s`, `distance_m` | When the attempt started in the log, how long it lasted, how far it travelled |
| `auto_corr` | Steering-command vs yaw-rate correlation (blank if the Jetson barely steered). Injected noise makes the steering noisy, so it is low on noisy runs |
| `start_offset_m`, `start_heading_deg` | Offset and heading error to the path at the start |
| `converge_s`, `converge_m` | Time and distance until \|offset\| < 0.10 m held for 2 s (blank = never) |
| `overshoot_m` | Furthest it went past the line on the other side |
| `rms_offset_m`, `max_offset_m` | RMS and maximum \|offset\| over the attempt |
| `end_offset_m` | Median offset over the last 3 s |
| `note` | Warnings, such as a start that may still be manual or a run that never converged |

## Waypoints

Each run's metadata names its waypoint file, which is looked up in `waypoints/` and then in the
JetsonControl repo's `waypoints` folder. Add any new test path's CSV to `waypoints/`.
`line_<N>m.csv` is treated as a straight line along x if the file is missing.
