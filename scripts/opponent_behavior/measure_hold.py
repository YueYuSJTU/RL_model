"""Measure each opponent's own altitude/speed hold, to pick the low-pressure probe.

The study needs a non-threatening, stable reference opponent so that the
"low pressure" behaviour of each subject can be measured.  A scripted
altitude-hold controller was attempted first and rejected on measured grounds
(see calibrate_level_probe.py: a constant command cannot trim the model, and a
feedback loop drifts out of the linear regime within 60 s).  This script
measures the alternative: how well each opponent holds altitude and speed on
its own, against a scripted random probe that cannot threaten it.

Output: experiments/opponent_behavior/altitude_hold.csv  (+ stdout table)

Usage:
  python scripts/opponent_behavior/measure_hold.py [--episodes 3]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from common import (OPPONENTS, REPO, RandomProbe, _predict, load_policy,  # noqa: E402
                    make_eval_env)

OUT_CSV = REPO / "experiments" / "opponent_behavior" / "altitude_hold.csv"


def hold(rel: str, n: int, seed: int = 11) -> dict:
    m1 = load_policy(rel)
    probe = RandomProbe(seed=1)
    vec = make_eval_env(seed=seed)
    alt, spd, roll = [], [], []
    for _ in range(n):
        obs = vec.reset()
        half = obs.shape[1] // 2
        s1 = s2 = None
        e1 = np.ones(1, dtype=bool)
        e2 = np.ones(1, dtype=bool)
        done = False
        while not done:
            a1, s1 = _predict(m1, obs[:, :half], s1, e1)
            a2, s2 = _predict(probe, obs[:, half:], s2, e2)
            obs, _r, d, info = vec.step(np.concatenate([a1, a2], axis=-1))
            e1 = d
            e2 = d
            ms = info[0]["metrics_step"]
            tr = info[0]["trajectory"]
            alt.append(ms["altitude_sl_ft"])
            spd.append(ms["u_fps"])
            roll.append(tr["self"]["roll_rad"])
            if bool(d[0]):
                done = True
    vec.close()
    a = np.array(alt)
    u = np.array(spd)
    r = np.degrees(np.abs(np.array(roll)))
    return dict(
        alt_mean=float(a.mean()), alt_std=float(a.std()), alt_min=float(a.min()),
        alt_max=float(a.max()), spd_mean=float(u.mean()), spd_std=float(u.std()),
        roll_abs_mean_deg=float(r.mean()), n_steps=int(a.size),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()

    rows = []
    print("%-5s %-42s %8s %8s %8s %8s %7s %7s" % (
        "ID", "reward set", "alt_mean", "alt_std", "alt_min", "V_mean", "V_std", "|roll|"))
    for tag in sorted(OPPONENTS, key=lambda t: (OPPONENTS[t]["family"], t)):
        r = hold(OPPONENTS[tag]["path"], args.episodes)
        r["id"] = tag
        r["path"] = OPPONENTS[tag]["path"]
        r["family"] = OPPONENTS[tag]["family"]
        r["reward_set"] = OPPONENTS[tag]["reward_set"]
        rows.append(r)
        print("%-5s %-42s %8.0f %8.0f %8.0f %8.0f %7.1f %7.1f" % (
            tag, r["reward_set"][:42], r["alt_mean"], r["alt_std"], r["alt_min"],
            r["spd_mean"], r["spd_std"], r["roll_abs_mean_deg"]), flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwritten: %s" % OUT_CSV, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
