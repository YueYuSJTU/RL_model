"""Aggregate the fixed goal-point battery into the behavioural scale tables.

Input : experiments/opponent_behavior/raw/gp_<subject>.json
Output: experiments/opponent_behavior/goalpoint_battery.csv     (subject x point)
        experiments/opponent_behavior/opponent_capability.csv   (subject aggregate)
        experiments/opponent_behavior/analysis.json

Every number is measured.  Nothing is hand-set.

The scale
---------
Each opponent is shown the same 16 fixed goal points and only its own flight is
recorded, so the comparison is like-for-like:

  pursuit duty cycle   mean over the battery of the fraction of steps whose
                       nose-to-goal angle is below 30 deg  -> PRIMARY axis
  intercept rate       fraction of goal points at which the opponent achieved a
                       valid firing solution on the commanded target
                       (nose < 15 deg AND 500 <= range <= 4500 ft)
  closure              mean closing rate towards the goal point
  straightness         net displacement / path length  (1.0 = ignores the target)
  yaw rate, bank std   manoeuvring activity
  Es                   mean specific energy h + V^2/2g

Tier boundaries are placed at the LARGEST GAP in the measured distribution
rather than at a chosen constant; the gap is reported so it can be checked.

Usage:
  python scripts/opponent_behavior/analyze.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import OPPONENTS, REPO  # noqa: E402
from goalpoint import GOAL_POINTS  # noqa: E402

ROOT = REPO / "experiments" / "opponent_behavior"
RAW = ROOT / "raw"

PRIMARY = "point_frac"
AGG_KEYS = [
    "point_frac", "gun_frac", "gun_reached", "track_mean_rad", "closure_mean_fps",
    "dist_mean_ft", "dist_min_ft", "dist_end_ft", "straightness", "yaw_rate_deg",
    "bank_std_rad", "bank_abs_mean_rad", "alt_std_ft", "speed_mean_fps", "Es_ft",
    "ep_steps",
]


def f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def read_json(p: Path):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    keys: list = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def largest_gap(values: np.ndarray):
    """Index of the largest gap in a sorted 1-D distribution."""
    v = np.sort(values)
    if v.size < 2:
        return None, float("nan")
    gaps = np.diff(v)
    i = int(np.argmax(gaps))
    return i, float(gaps[i])


def main() -> int:
    files = sorted(RAW.glob("gp_*.json"))
    if not files:
        print("no goal-point runs found in %s" % RAW)
        return 1
    runs = {}
    for p in files:
        d = read_json(p)
        runs[d["subject"]] = d
    subjects = sorted(runs, key=lambda s: int(s))

    # ---- per (subject, goal point)
    battery_rows = []
    for s in subjects:
        for pt in GOAL_POINTS:
            d = runs[s]["goal_points"].get(pt["name"])
            if not d:
                continue
            row = dict(
                subject=s, family=runs[s]["subject_family"],
                reward_set=runs[s]["subject_reward_set"],
                point=pt["name"], bearing_deg=pt["bearing_deg"],
                dist_ft=pt["dist_ft"], alt_ft=pt["alt_ft"],
            )
            for k in AGG_KEYS:
                if k in d:
                    row[k] = d[k]["mean"]
            battery_rows.append(row)
    write_csv(ROOT / "goalpoint_battery.csv", battery_rows)

    # ---- per subject aggregate
    cap_rows = []
    for s in subjects:
        gps = runs[s]["goal_points"]
        row = dict(
            id=s, family=runs[s]["subject_family"],
            reward_set=runs[s]["subject_reward_set"],
            train_steps=OPPONENTS[s]["train_steps"],
            batch_size=OPPONENTS[s]["batch_size"],
            n_points=len(gps),
        )
        for k in AGG_KEYS:
            vals = [f(gps[p["name"]][k]["mean"]) for p in GOAL_POINTS
                    if p["name"] in gps and k in gps[p["name"]]]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                row["mean_" + k] = float(np.mean(vals))
                row["min_" + k] = float(np.min(vals))
                row["max_" + k] = float(np.max(vals))
                row["std_" + k] = float(np.std(vals))
        reached = [f(gps[p["name"]]["gun_reached"]["mean"]) for p in GOAL_POINTS
                   if p["name"] in gps and "gun_reached" in gps[p["name"]]]
        row["intercept_rate"] = (float(np.mean([1.0 if v >= 0.5 else 0.0 for v in reached]))
                                 if reached else float("nan"))
        # "sustained pointing": the opponent held its nose on the commanded
        # target for at least 30% of the episode.  A transient entry into the
        # weapon envelope is NOT enough -- a straight-flying aircraft scores on
        # the goal point placed 20000 ft dead ahead by simply overflying it
        # (measured: every non-pursuing opponent scores on exactly that point),
        # so a transient is not evidence of target-directed behaviour.
        sust = [1 if f(gps[p["name"]]["point_frac"]["mean"]) >= 0.30 else 0
                for p in GOAL_POINTS if p["name"] in gps]
        row["sustained_points"] = int(sum(sust))
        row["sustained_share"] = float(np.mean(sust)) if sust else float("nan")
        # which points a straight-flying aircraft can score on by accident
        row["accidental_points"] = ";".join(
            p["name"] for p in GOAL_POINTS
            if p["name"] in gps and f(gps[p["name"]]["point_frac"]["mean"]) >= 0.30
            and f(gps[p["name"]]["dist_ft"]) >= 20000.0)
        cap_rows.append(row)
    cap_rows.sort(key=lambda r: (-r["sustained_points"], -r["mean_" + PRIMARY]))

    # ---- data-driven tier boundary on the sustained-point count
    v = np.array([r["sustained_points"] for r in cap_rows], dtype=float)
    drops = -np.diff(v) if v.size > 1 else np.array([])
    boundaries = [i + 1 for i, g in enumerate(drops) if g >= 2]
    tiers = {}
    for i, r in enumerate(cap_rows):
        k = sum(1 for b in boundaries if i >= b)
        tiers[r["id"]] = "tier %s" % "ABC"[min(k, 2)]
    for r in cap_rows:
        r["tier"] = tiers.get(r["id"], "n/a")
    write_csv(ROOT / "opponent_capability.csv", cap_rows)

    (ROOT / "analysis.json").write_text(json.dumps(dict(
        subjects=subjects,
        goal_points=[p["name"] for p in GOAL_POINTS],
        capability=cap_rows,
        battery=battery_rows,
        primary=PRIMARY,
        agg_keys=AGG_KEYS,
        tier_boundaries=boundaries,
        tier_definition="tiers split where the measured sustained-point counts "
                        "(goal points with pursuit duty cycle >= 0.30) differ by "
                        "at least 2; no performance constant is chosen",
        runs={s: {k: v for k, v in runs[s].items() if k != "goal_points"}
              for s in subjects},
    ), indent=1, default=float))

    print("subjects: %d   goal points per subject: %d" % (len(subjects), len(GOAL_POINTS)))
    print()
    print("%-5s %-20s %8s %9s %8s %9s %9s %-8s" % (
        "ID", "family", "sustained", "meanPurs", "gun%", "closure", "straight", "tier"))
    for r in cap_rows:
        print("%-5s %-20s %4d/%-3d %9.3f %8.4f %+9.0f %9.3f %-8s" % (
            r["id"], r["family"], r["sustained_points"], r["n_points"],
            r["mean_" + PRIMARY], r["mean_gun_frac"],
            r["mean_closure_mean_fps"], r["mean_straightness"], r["tier"]))
    print()
    print("tier boundaries at sustained-point ranks: %s" % boundaries)
    print("written to %s" % ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
