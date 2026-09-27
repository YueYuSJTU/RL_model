"""Goal-point battery probe: measure an opponent's behaviour on FIXED targets.

Design (requested): every benchmark opponent is shown the SAME designed set of
fixed goal points, so the only thing that varies between opponents is the
opponent itself.  No external model is used as a measuring stick: every
reported number is a quantity of the subject's own flight.

How it is wired, using only the repository's existing machinery:

  * the subject sits in the OPPONENT seat (model2);
  * the environment is put in its native "goal_point" mode
    (`update_task_parameters(goal_point_prob=1.0)`), and
    `TrackingTask._cal_oppo_state` then builds the OPPONENT's relative-geometry
    observation from `self.goal_point` instead of from the other aircraft
    (task_tracking.py:849-850).  The subject therefore "sees" the goal point as
    its target;
  * `self.goal_point` is fixed per episode by patching the instance method
    `get_goal_point` (called once per episode at task_tracking.py:1043).  No
    file under jsb_env/** is modified;
  * the initial geometry is likewise pinned by patching the instance method
    `_maybe_init_pair_geometry`, so every opponent starts from an identical
    state and only the goal point differs.

The ego seat holds a scripted, zero-learning "trim drone" whose only job is to
keep the simulation valid.  It is placed ~20000 ft away and climbing, so it
never enters gun range of the subject, and it is identical for every subject.

Measured quantities (all of the SUBJECT):
  straightness   net displacement / path length      (1.0 = dead straight)
  yaw_rate_deg   mean |heading change| per step
  bank_std_rad   std of roll angle
  alt_std_ft     std of altitude
  Es_ft          mean specific energy h + V^2/2g
  point%         fraction of steps whose nose-to-goal angle < 30 deg
  gun%           fraction of steps with a valid firing solution on the goal
                 point (nose < 15 deg, 500 <= range <= 4500 ft)
  closure fps    mean closing rate towards the goal point
  dist mean/min  range to the goal point
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common import (EVAL_ENV_CFG, G_FT_S2, _predict, make_eval_env)  # noqa: E402

# --------------------------------------------------------------------------
# Fixed initial geometry (identical for every subject and every goal point)
# --------------------------------------------------------------------------
SUBJ_NED_NE_FT = (0.0, -20000.0)   # subject starts 20000 ft west of the drone
SUBJ_ALT_FT = 10000.0
SUBJ_HEADING_DEG = 0.0
DRONE_ALT_FT = 13000.0
DRONE_HEADING_DEG = 0.0
# constant command that makes the drone climb away from the subject's airspace
DRONE_ACTION = (0.0, -0.05, 0.0, 0.55)

GUN_RANGE_FT = 4500.0
GUN_ANGLE_RAD = math.radians(15.0)
POINT_ANGLE_RAD = math.radians(30.0)

MAX_STEPS = 600  # 60 s at 10 Hz


def build_battery() -> List[Dict[str, Any]]:
    """Designed goal-point battery: bearing x range x altitude coverage.

    Bearings are measured from the subject's starting position, in the local
    NED frame (0 deg = north = the subject's initial heading).
    """
    pts: List[Dict[str, Any]] = []
    seen = set()

    def add(name: str, bearing_deg: float, dist_ft: float, alt_ft: float) -> None:
        key = (round(bearing_deg, 3), round(dist_ft, 3), round(alt_ft, 3))
        if key in seen:
            return
        seen.add(key)
        pts.append(dict(name=name, bearing_deg=bearing_deg, dist_ft=dist_ft,
                        alt_ft=alt_ft))

    # 1) bearing sweep at a fixed range and altitude (8 points)
    for b in (0, 45, 90, 135, 180, -135, -90, -45):
        add("bearing_%+04d" % b, b, 12000.0, 14000.0)
    # 2) range sweep, dead ahead at the same altitude (adds 2 new)
    for d in (6000.0, 12000.0, 20000.0):
        add("range_%05d" % int(d), 0.0, d, 14000.0)
    # 3) altitude sweep, dead ahead at the same range (adds 2 new)
    for h in (8000.0, 14000.0, 20000.0):
        add("alt_%05d" % int(h), 0.0, 12000.0, h)
    # 4) cross terms: beam bearings at the range/altitude extremes (4 new)
    for b in (90, -90):
        for h in (8000.0, 20000.0):
            add("cross_%+04d_%05d" % (b, int(h)), b, 20000.0, h)
    return pts


GOAL_POINTS = build_battery()


def goal_point_ned(pt: Dict[str, Any]):
    """Goal point in the local NED frame (north, east, altitude) in feet."""
    b = math.radians(pt["bearing_deg"])
    return (SUBJ_NED_NE_FT[0] + pt["dist_ft"] * math.cos(b),
            SUBJ_NED_NE_FT[1] + pt["dist_ft"] * math.sin(b),
            pt["alt_ft"])


# --------------------------------------------------------------------------
# Scripted ego-seat drone
# --------------------------------------------------------------------------
class TrimDrone:
    """Zero-learning constant-command drone used to keep the sim valid."""

    def __init__(self, action: Sequence[float] = DRONE_ACTION):
        self.action = np.array(action, dtype=np.float32)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = int(np.asarray(observation).shape[0]) if np.asarray(observation).ndim > 1 else 1
        return np.tile(self.action, (n, 1)), None


# --------------------------------------------------------------------------
# Environment wiring (instance-level patching; no repository file is edited)
# --------------------------------------------------------------------------
def _get_task(vec):
    e = vec.envs[0]
    seen = 0
    while seen < 10:
        if hasattr(e, "task"):
            return e.task
        if hasattr(e, "unwrapped") and hasattr(e.unwrapped, "task"):
            return e.unwrapped.task
        if hasattr(e, "env"):
            e = e.env
            seen += 1
            continue
        break
    raise RuntimeError("could not locate the task object on the environment")


def configure_goalpoint_env(vec, pt: Optional[Dict[str, Any]]):
    """Pin the initial geometry and the goal point for the next episodes."""
    from jsbgym_m import properties as prp  # jsb_env adds itself to sys.path

    task = _get_task(vec)

    def fixed_geometry() -> None:
        task._init_pair_cached = True
        task._init_self_ned_xy_ft = (0.0, 0.0)
        task._init_oppo_ned_ne_ft = tuple(float(v) for v in SUBJ_NED_NE_FT)
        task._init_self_alt_ft = float(DRONE_ALT_FT)
        task._init_oppo_alt_ft = float(SUBJ_ALT_FT)
        task._init_self_heading_deg = float(DRONE_HEADING_DEG)
        task._init_oppo_heading_deg = float(SUBJ_HEADING_DEG)

    task._maybe_init_pair_geometry = fixed_geometry

    target = pt if pt is not None else GOAL_POINTS[0]
    gx, gy, gh = goal_point_ned(target)

    def fixed_goal_point(_gp=prp.Vector3(gx, gy, gh)):
        return _gp

    task.get_goal_point = fixed_goal_point
    # native goal-point mode: the opponent-side observation describes the goal
    vec.env_method("update_task_parameters", goal_point_prob=1.0)
    return (gx, gy, gh)


# --------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------
def rollout_goalpoint(vec, subject, pt: Dict[str, Any], repeats: int = 1) -> List[Dict[str, Any]]:
    """Run one goal point for `repeats` episodes and return the subject's traces."""
    gp = np.array(goal_point_ned(pt), dtype=float)
    drone = TrimDrone()
    episodes: List[Dict[str, Any]] = []
    for _ in range(repeats):
        configure_goalpoint_env(vec, pt)
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
                ox = float(tr["oppo"]["x_ft"])
                oy = float(tr["oppo"]["y_ft"])
                oz = float(tr["oppo"]["z_ft"])
                d = float(np.linalg.norm(np.array([ox, oy, oz]) - gp))
                rows.append(dict(
                    track=float(ms["oppo_track_angle_rad"]),
                    dist=d,
                    alt=oz,
                    u=float(ms["oppo_u_fps"]),
                    roll=float(tr["oppo"]["roll_rad"]),
                    yaw=float(tr["oppo"]["yaw_deg"]),
                    x=ox, y=oy, z=oz,
                    hp_subject=float(ms["hp_oppo"]),
                    hp_drone=float(ms["hp_self"]),
                ))
            if bool(dones[0]):
                done = True
                ei = info[0].get("env_info", {}) or {}
                episodes.append(dict(win=ei.get("win"), steps=ei.get("steps_used"),
                                     hp_self=ei.get("HP_self"),
                                     hp_oppo=ei.get("HP_oppo"), series=rows))
    return episodes


def goalpoint_descriptors(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Descriptors of the subject for one goal point, all measured."""
    if not rows:
        return {}
    dist = np.array([r["dist"] for r in rows], dtype=float)
    track = np.array([r["track"] for r in rows], dtype=float)
    alt = np.array([r["alt"] for r in rows], dtype=float)
    u = np.array([r["u"] for r in rows], dtype=float)
    roll = np.array([r["roll"] for r in rows], dtype=float)
    yaw = np.array([r["yaw"] for r in rows], dtype=float)
    x = np.array([r["x"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=float)

    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    net = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    dyaw = np.abs(np.diff(yaw))
    dyaw = np.minimum(dyaw, 360.0 - dyaw)

    closure = -np.diff(dist) / 0.1 if dist.size > 1 else np.zeros(1)
    in_gun = (track <= GUN_ANGLE_RAD) & (dist >= 500.0) & (dist <= GUN_RANGE_FT)

    return dict(
        straightness=net / path if path > 1e-6 else 1.0,
        yaw_rate_deg=float(dyaw.mean()) if dyaw.size else 0.0,
        bank_std_rad=float(roll.std()),
        bank_abs_mean_rad=float(np.abs(roll).mean()),
        alt_std_ft=float(alt.std()),
        speed_mean_fps=float(u.mean()),
        Es_ft=float((alt + u * u / (2.0 * G_FT_S2)).mean()),
        point_frac=float((track < POINT_ANGLE_RAD).mean()),
        gun_frac=float(in_gun.mean()),
        gun_reached=1.0 if bool(in_gun.any()) else 0.0,
        track_mean_rad=float(track.mean()),
        closure_mean_fps=float(closure.mean()),
        dist_mean_ft=float(dist.mean()),
        dist_min_ft=float(dist.min()),
        dist_end_ft=float(dist[-1]),
        ep_steps=float(len(rows)),
    )


def summarize_goalpoint(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_ep = [goalpoint_descriptors(e["series"]) for e in episodes if e["series"]]
    per_ep = [p for p in per_ep if p]
    if not per_ep:
        return {}
    keys = sorted(per_ep[0].keys())
    out: Dict[str, Any] = {}
    for k in keys:
        v = np.array([p[k] for p in per_ep], dtype=float)
        out[k] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    out["n_episodes"] = len(per_ep)
    out["per_episode"] = per_ep
    out["drone_hp_end"] = float(np.mean([e["hp_self"] for e in episodes]))
    out["subject_hp_end"] = float(np.mean([e["hp_oppo"] for e in episodes]))
    out["avg_steps"] = float(np.mean([e["steps"] for e in episodes]))
    return out
