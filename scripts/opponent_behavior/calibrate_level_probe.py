"""Calibrate the scripted low-pressure (altitude/speed-hold) probe by measurement.

Two stages, both purely empirical:

  stage 1 - open loop.  Sweep 42 constant (elevator, throttle) pairs.  Result
            (kept as evidence): NO constant command trims this F-16 model.
            elevator -0.06 climbs to ~30000 ft, elevator >= -0.03 dives to the
            ground, and throttle does not change that qualitatively.  The
            phugoid mode diverges, so an open-loop probe cannot represent
            "neutral level flight".

  stage 2 - closed loop.  Sweep the gains of ClosedLoopLevelProbe, which holds
            altitude and speed using the opponent-side state the environment
            exposes in info['metrics_step'].  Selection: no ground crash, then
            smallest |altitude drift|, then smallest altitude standard
            deviation, then speed closest to the initial cruise speed.

The selected gains are written to
experiments/opponent_behavior/level_probe_calibration.json and read back by
common.load_level_probe(), so the probe used for the study is reproducible and
its constants are measured.

Usage:
  python scripts/opponent_behavior/calibrate_level_probe.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

from common import (REPO, ClosedLoopLevelProbe, OpenLoopLevelProbe,  # noqa: E402
                    _predict, make_eval_env, write_json)

OUT = REPO / "experiments" / "opponent_behavior" / "level_probe_calibration.json"

ELEVATORS = [-0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12]
THROTTLES = [0.2, 0.3, 0.4, 0.5, 0.65, 0.8]

KP_ALT = [3.0e-5, 1.0e-4, 2.0e-4, 4.0e-4]
KD_ALT = [3.0e-4, 1.0e-3, 3.0e-3]
KP_SPD = [2.0e-3, 5.0e-3, 1.0e-2]
THR_TRIM = [0.4, 0.6]


def fly_pair(probe1, probe2, n_ep: int = 2):
    """Drive both aircraft with scripted probes and return both traces.

    The environment exposes only four aircraft-state numbers per step
    (self alt/speed and opponent alt/speed), which is enough to feed both
    probes: probe1 is fed the self numbers, probe2 the opponent numbers.
    """
    vec = make_eval_env(seed=7)
    rec = {1: [], 2: []}
    crashed = 0
    for _ in range(n_ep):
        obs = vec.reset()
        half = obs.shape[1] // 2
        s1 = s2 = None
        e1 = np.ones(1, dtype=bool)
        e2 = np.ones(1, dtype=bool)
        done = False
        while not done:
            a1, s1 = _predict(probe1, obs[:, :half], s1, e1)
            a2, s2 = _predict(probe2, obs[:, half:], s2, e2)
            obs, _r, dones, info = vec.step(np.concatenate([a1, a2], axis=-1))
            e1 = dones
            e2 = dones
            ms = info[0]["metrics_step"]
            rec[1].append((ms["altitude_sl_ft"], ms["u_fps"]))
            rec[2].append((ms["oppo_altitude_sl_ft"], ms["oppo_u_fps"]))
            if hasattr(probe1, "observe"):
                probe1.observe({"alt": ms["altitude_sl_ft"], "spd": ms["u_fps"]})
            if hasattr(probe2, "observe"):
                probe2.observe({"alt": ms["oppo_altitude_sl_ft"], "spd": ms["oppo_u_fps"]})
            if bool(dones[0]):
                done = True
                ei = info[0].get("env_info", {}) or {}
                if ei.get("win") == 0.5:
                    crashed += 1
    vec.close()
    return rec, crashed


def stats_of(rec):
    a = np.array([p[0] for p in rec])
    u = np.array([p[1] for p in rec])
    return dict(
        alt_mean=float(a.mean()), alt_std=float(a.std()), alt_min=float(a.min()),
        alt_end=float(a[-1]), alt_drift=float(abs(a[-1] - a[0])),
        speed_mean=float(u.mean()), speed_end=float(u[-1]),
        speed_std=float(u.std()),
    )


def stage1_open_loop():
    rows = []
    t0 = time.time()
    print("--- stage 1: open-loop constant commands ---", flush=True)
    for elev, thr in itertools.product(ELEVATORS, THROTTLES):
        rec, crashed = fly_pair(OpenLoopLevelProbe(elev, thr), OpenLoopLevelProbe(elev, thr), 2)
        s = stats_of(rec[1])
        s.update(elevator=elev, throttle=thr, ground_crash_episodes=crashed)
        rows.append(s)
        print("  elev=%+0.2f thr=%.2f  alt %7.0f+-%5.0f  drift %6.0f  V %6.0f  minAlt %6.0f"
              % (elev, thr, s["alt_mean"], s["alt_std"], s["alt_drift"],
                 s["speed_mean"], s["alt_min"]), flush=True)
    viable = [r for r in rows if r["ground_crash_episodes"] == 0 and r["alt_min"] > 2000.0]
    best_drift = min((r["alt_drift"] for r in viable), default=float("inf"))
    print("  open-loop best achievable |drift| = %.0f ft over 60 s\n" % best_drift, flush=True)
    return rows, best_drift, time.time() - t0


def stage2_closed_loop():
    rows = []
    t0 = time.time()
    print("--- stage 2: closed-loop gain search ---", flush=True)
    for kp_alt, kd_alt, kp_spd, thr_trim in itertools.product(KP_ALT, KD_ALT, KP_SPD, THR_TRIM):
        g = dict(kp_alt=kp_alt, kd_alt=kd_alt, kp_spd=kp_spd, thr_trim=thr_trim,
                 alt_target_ft=10000.0)
        rec, crashed = fly_pair(ClosedLoopLevelProbe(g), ClosedLoopLevelProbe(g), 2)
        s = stats_of(rec[1])
        s.update(gains=g, ground_crash_episodes=crashed)
        rows.append(s)
        print("  kp_alt=%.0e kd_alt=%.0e kp_spd=%.0e thr=%.1f  alt %6.0f+-%4.0f  "
              "drift %5.0f  V %5.0f+-%3.0f  minAlt %6.0f  crash=%d"
              % (kp_alt, kd_alt, kp_spd, thr_trim, s["alt_mean"], s["alt_std"],
                 s["alt_drift"], s["speed_mean"], s["speed_std"], s["alt_min"],
                 crashed), flush=True)
    return rows, time.time() - t0


def main() -> int:
    open_rows, open_best, t1 = stage1_open_loop()
    closed_rows, t2 = stage2_closed_loop()

    viable = [r for r in closed_rows if r["ground_crash_episodes"] == 0 and r["alt_min"] > 3000.0]
    if not viable:
        print("no viable closed-loop gains found", flush=True)
        return 1
    best = min(viable, key=lambda r: (r["alt_drift"], r["alt_std"], r["speed_std"]))
    summary = dict(
        selected_gains=best["gains"],
        selected_stats={k: v for k, v in best.items() if k != "gains"},
        selection_rule="no ground crash and min altitude > 3000 ft; then minimise "
                       "|alt_end - alt_start|, then altitude std, then speed std",
        open_loop_best_drift_ft=open_best,
        open_loop_evidence=open_rows,
        closed_loop_sweep=closed_rows,
        wall_s_stage1=t1,
        wall_s_stage2=t2,
    )
    write_json(OUT, summary)
    print("\nSELECTED closed-loop gains: %s" % best["gains"], flush=True)
    print("  -> alt %.0f +- %.0f ft (drift %.0f ft), V %.0f +- %.0f ft/s"
          % (best["alt_mean"], best["alt_std"], best["alt_drift"],
             best["speed_mean"], best["speed_std"]), flush=True)
    print("  open-loop best possible drift was %.0f ft (no constant command trims)"
          % open_best, flush=True)
    print("written: %s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
