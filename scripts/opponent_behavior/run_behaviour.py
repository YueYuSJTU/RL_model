"""Behaviour-distribution measurement: how an opponent habitually flies.

This measures the opponent's OWN motion and control habits, not its success at
any task.  The opponent sits in the opponent seat and the environment's native
"jsbsim" opponent mode is used (goal_point_prob = 0), so the opponent's
observation describes the other aircraft -- i.e. the opponent can react to it if
it wants to.  The other aircraft is a scripted constant-command drone, identical
for every opponent, so the stimulus is standardised.

Recorded every step: the opponent's own control commands (all four channels),
attitude, altitude, speed and position, plus the counterpart's position.
Everything downstream is computed from those.

Metric families
---------------
VERTICAL   climb_rate_mean, climb_time_frac, alt_trend, alt_p10/p90, alt_min
LATERAL    turn_rate_mean, turn_bias, left_time_frac, bank_mean_abs, bank_p90,
           hard_bank_frac, roll_reversal_per_min
CONTROL    per channel: mean, std, mean_abs, sat_frac (|u|>0.9);
           control_activity (mean |du| per step), throttle_mean, throttle_std,
           rudder_usage
ENERGY     speed_mean, speed_std, Es_mean, Es_trend, energy_exchange_corr
REACTION   los_point_frac (nose within 30 deg of the counterpart),
           los_closure_mean, react_corr (correlation between the line-of-sight
           angle and the turn rate: positive = turns towards the counterpart),
           react_gain (slope, 1/s), react_r2

Usage:
  python scripts/opponent_behavior/run_behaviour.py --subject 11 --out ...
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from common import G_FT_S2, OPPONENTS, _predict, load_policy, make_eval_env, write_json  # noqa: E402
from goalpoint import DRONE_ACTION, DRONE_ALT_FT, SUBJ_ALT_FT, SUBJ_NED_NE_FT, TrimDrone  # noqa: E402

DT = 0.1
LOS_POINT_RAD = math.radians(30.0)
ROLL_DEADBAND_RAD = 0.02          # ~1.1 deg of roll change per step
REACT_MIN_TURN_STD_DEG_S = 1.0    # variability floor for the reaction metrics
CHANNELS = ["aileron", "elevator", "rudder", "throttle"]


def _get_task(vec):
    e = vec.envs[0]
    for _ in range(10):
        if hasattr(e, "task"):
            return e.task
        if hasattr(e, "unwrapped") and hasattr(e.unwrapped, "task"):
            return e.unwrapped.task
        if hasattr(e, "env"):
            e = e.env
            continue
        break
    raise RuntimeError("could not locate the task object")


def pin_geometry(vec):
    """Identical starting state for every opponent; no goal point involved."""
    task = _get_task(vec)

    def fixed_geometry() -> None:
        task._init_pair_cached = True
        task._init_self_ned_xy_ft = (0.0, 0.0)
        task._init_oppo_ned_ne_ft = tuple(float(v) for v in SUBJ_NED_NE_FT)
        task._init_self_alt_ft = float(DRONE_ALT_FT)
        task._init_oppo_alt_ft = float(SUBJ_ALT_FT)
        task._init_self_heading_deg = 0.0
        task._init_oppo_heading_deg = 0.0

    task._maybe_init_pair_geometry = fixed_geometry
    # native "jsbsim" opponent mode: the opponent's observation is the REAL
    # other aircraft, so reacting to it is possible
    vec.env_method("update_task_parameters", goal_point_prob=0.0)


def run_free(vec, subject, repeats: int = 1) -> List[Dict[str, Any]]:
    drone = TrimDrone()
    out: List[Dict[str, Any]] = []
    for _ in range(repeats):
        pin_geometry(vec)
        obs = vec.reset()
        half = obs.shape[1] // 2
        s2 = None
        e2 = np.ones(1, dtype=bool)
        rows: List[Dict[str, float]] = []
        done = False
        while not done:
            a1, _ = _predict(drone, obs[:, :half], None, np.ones(1, dtype=bool))
            a2, s2 = _predict(subject, obs[:, half:], s2, e2)
            obs, _r, dones, info = vec.step(np.concatenate([a1, a2], axis=-1))
            e2 = dones
            ms = info[0].get("metrics_step")
            tr = info[0].get("trajectory")
            if ms is not None and tr is not None:
                rows.append(dict(
                    u_ail=float(a2[0, 0]), u_ele=float(a2[0, 1]),
                    u_rud=float(a2[0, 2]), u_thr=float(a2[0, 3]),
                    alt=float(tr["oppo"]["z_ft"]),
                    roll=float(tr["oppo"]["roll_rad"]),
                    pitch=float(tr["oppo"]["pitch_rad"]),
                    yaw=float(tr["oppo"]["yaw_deg"]),
                    x=float(tr["oppo"]["x_ft"]), y=float(tr["oppo"]["y_ft"]),
                    cx=float(tr["self"]["x_ft"]), cy=float(tr["self"]["y_ft"]),
                    cz=float(tr["self"]["z_ft"]),
                    speed=float(ms["oppo_u_fps"]),
                    hp=float(ms["hp_oppo"]),
                ))
            if bool(dones[0]):
                done = True
                out.append(dict(steps=info[0].get("env_info", {}).get("steps_used"),
                                series=rows))
    return out


def behaviour_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """All behaviour-distribution statistics for one episode."""
    if len(rows) < 5:
        return {}
    g = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    t = np.arange(len(rows)) * DT

    # ---------------- vertical ----------------
    roc = np.gradient(g["alt"], DT)
    alt_slope = float(np.polyfit(t, g["alt"], 1)[0]) if len(t) > 2 else 0.0

    # ---------------- lateral ----------------
    dyaw = np.diff(g["yaw"])
    dyaw = (dyaw + 180.0) % 360.0 - 180.0            # wrap to [-180, 180)
    turn_rate = np.degrees(np.radians(dyaw) / DT)     # deg/s, positive = right
    bank = np.degrees(g["roll"])
    # roll reversals need a deadband: without one, numerical jitter around zero
    # is counted as a reversal (measured: 47 reversals/min for a policy whose
    # mean bank is 3.2 deg and turn rate 0.3 deg/s, which is impossible)
    droll = np.diff(g["roll"])
    active = np.abs(droll) > ROLL_DEADBAND_RAD
    sign = np.sign(droll[active])
    reversals = int(np.sum((sign[1:] * sign[:-1]) < 0)) if sign.size > 1 else 0
    dur_min = max(1e-9, len(rows) * DT / 60.0)

    # ---------------- control ----------------
    ctrl: Dict[str, float] = {}
    for name, key in zip(CHANNELS, ["u_ail", "u_ele", "u_rud", "u_thr"]):
        v = g[key]
        ctrl["ctrl_%s_mean" % name] = float(v.mean())
        ctrl["ctrl_%s_std" % name] = float(v.std())
        ctrl["ctrl_%s_mean_abs" % name] = float(np.abs(v).mean())
        ctrl["ctrl_%s_sat_frac" % name] = float((np.abs(v) > 0.9).mean())
        ctrl["ctrl_%s_activity" % name] = float(np.abs(np.diff(v)).mean()) if v.size > 1 else 0.0
    ctrl["control_activity"] = float(np.mean(
        [ctrl["ctrl_%s_activity" % n] for n in CHANNELS]))

    # ---------------- energy ----------------
    Es = g["alt"] + g["speed"] ** 2 / (2.0 * G_FT_S2)
    Es_slope = float(np.polyfit(t, Es, 1)[0]) if len(t) > 2 else 0.0
    dh, dv = np.diff(g["alt"]), np.diff(g["speed"])
    ee_corr = float(np.corrcoef(dh, -dv)[0, 1]) if dh.size > 2 and dh.std() > 1e-9 and dv.std() > 1e-9 else 0.0

    # ---------------- reaction to the counterpart ----------------
    los_n = g["cy"] - g["y"]
    los_e = g["cx"] - g["x"]
    bearing_world = np.degrees(np.arctan2(los_e, los_n))          # compass deg
    los_angle = (bearing_world - g["yaw"] + 180.0) % 360.0 - 180.0  # + = target right
    rng = np.hypot(los_n, los_e)
    closure = -np.gradient(rng, DT)
    tr_mid = turn_rate
    la_mid = (los_angle[:-1] + los_angle[1:]) / 2.0
    # A correlation between the line-of-sight angle and the turn rate is only
    # meaningful if the subject actually varies its turn rate.  Measured: a
    # policy with turn-rate std of ~0.3 deg/s produced R2 = 0.87 out of
    # numerical drift, which is not a response.  A variability floor is
    # therefore applied and the result is flagged when it is not met.
    tr_std = float(tr_mid.std())
    measurable = bool(tr_std >= REACT_MIN_TURN_STD_DEG_S and la_mid.std() > 1e-6)
    cc = float(np.corrcoef(la_mid, tr_mid)[0, 1]) if (measurable and la_mid.size > 2) else 0.0
    gain = float(np.polyfit(la_mid, tr_mid, 1)[0]) if (measurable and la_mid.size > 2) else 0.0
    r2 = cc ** 2 if measurable else 0.0

    return dict(
        # vertical
        climb_rate_mean=float(roc.mean()),
        climb_time_frac=float((roc > 0).mean()),
        alt_trend=alt_slope,
        alt_p10=float(np.percentile(g["alt"], 10)),
        alt_p90=float(np.percentile(g["alt"], 90)),
        alt_min=float(g["alt"].min()),
        alt_mean=float(g["alt"].mean()),
        # lateral
        turn_rate_mean=float(np.abs(turn_rate).mean()),
        turn_bias=float(turn_rate.mean()),
        left_time_frac=float((turn_rate < 0).mean()),
        bank_mean_abs=float(np.abs(bank).mean()),
        bank_std=float(bank.std()),
        bank_p90=float(np.percentile(np.abs(bank), 90)),
        hard_bank_frac=float((np.abs(bank) > 60.0).mean()),
        roll_reversal_per_min=float(reversals / dur_min),
        pitch_mean=float(np.degrees(g["pitch"]).mean()),
        # control
        **ctrl,
        # energy
        speed_mean=float(g["speed"].mean()),
        speed_std=float(g["speed"].std()),
        Es_mean=float(Es.mean()),
        Es_trend=Es_slope,
        energy_exchange_corr=ee_corr,
        # reaction
        los_point_frac=float((np.abs(los_angle) < np.degrees(LOS_POINT_RAD)).mean()),
        los_closure_mean=float(closure.mean()),
        turn_rate_std=tr_std,
        react_corr=cc,
        react_gain=gain,
        react_r2=r2,
        react_measurable=1.0 if measurable else 0.0,
        ep_steps=float(len(rows)),
        ep_seconds=float(len(rows)) * DT,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, choices=sorted(OPPONENTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    vec = make_eval_env(seed=args.seed)
    subject = load_policy(OPPONENTS[args.subject]["path"])
    eps = run_free(vec, subject, repeats=args.repeats)
    vec.close()

    per_ep = [behaviour_metrics(e["series"]) for e in eps]
    per_ep = [p for p in per_ep if p]
    agg = {k: float(np.mean([p[k] for p in per_ep])) for k in per_ep[0]}
    write_json(Path(args.out), dict(
        subject=args.subject, subject_path=OPPONENTS[args.subject]["path"],
        subject_family=OPPONENTS[args.subject]["family"],
        subject_reward_set=OPPONENTS[args.subject]["reward_set"],
        repeats=args.repeats, seed=args.seed, wall_s=time.time() - t0,
        metrics=agg, per_episode=per_ep,
    ))
    print("[behaviour] %-4s climb=%+7.1f ft/s climbTime=%.2f turnRate=%5.1f deg/s "
          "bias=%+6.2f |bank|=%5.1f hardBank=%.2f rev/min=%4.1f act=%.4f "
          "thr=%4.2f point=%.2f clos=%+6.0f reactR2=%.2f  %.0fs" % (
              args.subject, agg["climb_rate_mean"], agg["climb_time_frac"],
              agg["turn_rate_mean"], agg["turn_bias"], agg["bank_mean_abs"],
              agg["hard_bank_frac"], agg["roll_reversal_per_min"],
              agg["control_activity"], agg["ctrl_throttle_mean"],
              agg["los_point_frac"], agg["los_closure_mean"], agg["react_r2"],
              time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
