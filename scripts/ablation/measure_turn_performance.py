"""Probe the turn/manoeuvre performance of one subject at the JSBSim level.

Purpose: decide how the corner-speed metric of the paper's evaluation table can
be defined on measured grounds.  The heading-rate proxy used by
run_behaviour.py is unreliable when the aircraft rolls while turning, so this
probe reads the quantities JSBSim itself computes every step:

  velocities/vtrue-fps       true airspeed
  velocities/vc-kts          calibrated airspeed
  forces/load-factor         aerodynamic normal load factor n
  inertia/weight-lbs         current weight
  atmosphere/rho-slugs_ft3   air density
  aero/alpha-deg             angle of attack

and derives the physical turn rate of a coordinated turn,

  omega = g * sqrt(n^2 - 1) / V      [rad/s]

which is compared with the heading rate read from the trajectory.

Read-only: jsb_env/**, src/** and scripts/opponent_behavior/** are imported,
never modified.  The environment is instrumented by wrapping the instance
method task.task_step.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/measure_turn_performance.py \
      --subject "Opp1-goalpoint=opponent_pool/test_pool/1" --repeats 2

Output: experiments/model_behavior/turn/<name>.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "opponent_behavior"))

from common import load_policy, make_eval_env, write_json          # noqa: E402
from goalpoint import TrimDrone                                    # noqa: E402
from run_behaviour import _get_task, pin_geometry                  # noqa: E402

import jsb_env.jsbgym_m.properties as prp                          # noqa: E402

OUT = Path("experiments") / "model_behavior" / "turn"
G_FT_S2 = 32.174

PROPS = {
    "n": prp.Property("forces/load-factor", "normal load factor"),
    "vcas": prp.Property("velocities/vc-kts", "calibrated airspeed [kts]"),
    "weight": prp.Property("inertia/weight-lbs", "weight [lb]"),
    "rho": prp.Property("atmosphere/rho-slugs_ft3", "air density [slug/ft^3]"),
    "alpha": prp.Property("aero/alpha-deg", "angle of attack [deg]"),
    "qbar": prp.Property("aero/qbar-psf", "dynamic pressure [psf]"),
}


def get_opponent_sim(vec):
    """Locate the environment object that owns the two Simulation instances."""
    e = vec.envs[0]
    for _ in range(10):
        if hasattr(e, "opponent_sim"):
            return e
        if hasattr(e, "env"):
            e = e.env
            continue
        if hasattr(e, "unwrapped") and hasattr(e.unwrapped, "opponent_sim"):
            return e.unwrapped
        break
    raise RuntimeError("could not locate the environment owning the simulations")


def fly(vec, subject, repeats):
    """Free-flight episodes (same protocol as the behaviour study) with
    JSBSim-level instrumentation of the subject, which sits in seat 2."""
    env = get_opponent_sim(vec)
    drone = TrimDrone()
    eps = []
    for _ in range(repeats):
        pin_geometry(vec)
        obs = vec.reset()
        half = obs.shape[1] // 2
        s2, e2 = None, np.ones(1, dtype=bool)
        rows = []
        done = False
        while not done:
            a1, _ = drone.predict(obs[:, :half], None, np.ones(1, dtype=bool))
            a2, s2 = subject.predict(obs[:, half:], state=s2, episode_start=e2)
            obs, _r, dones, info = vec.step(np.concatenate([a1, a2], axis=-1))
            e2 = dones
            sim = env.opponent_sim
            tr = info[0]["trajectory"]["oppo"]
            V = float(sim[prp.vtrue_fps])
            n = float(sim[PROPS["n"]])
            rows.append(dict(
                V=V, vcas=float(sim[PROPS["vcas"]]), n=n,
                weight=float(sim[PROPS["weight"]]), rho=float(sim[PROPS["rho"]]),
                alpha=float(sim[PROPS["alpha"]]), qbar=float(sim[PROPS["qbar"]]),
                alt=float(sim[prp.altitude_sl_ft]), roll=float(tr["roll_rad"]),
                pitch=float(tr["pitch_rad"]), yaw=float(tr["yaw_deg"]),
                throttle=float(a2[0, 3]),
                omega_phys=float(G_FT_S2 * math.sqrt(max(n * n - 1.0, 0.0)) / V
                                 * 180.0 / math.pi) if V > 1 else 0.0))
            if bool(dones[0]):
                done = True
        eps.append(rows)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", action="append", required=True,
                    help="NAME=RELATIVE/PATH, repeatable")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(args.subject):
        name, path = spec.split("=", 1)
        out = OUT / ("%s.json" % name)
        if out.exists():
            print("skip (exists):", out, flush=True)
            continue
        t0 = time.time()
        vec = make_eval_env(seed=args.seed + i)
        subject = load_policy(path)
        eps = fly(vec, subject, args.repeats)
        vec.close()
        write_json(out, dict(subject=name, subject_path=path, seed=args.seed + i,
                             wall_s=time.time() - t0, episodes=eps))
        allrows = [r for ep in eps for r in ep]
        V = np.array([r["V"] for r in allrows])
        n = np.array([r["n"] for r in allrows])
        w = np.array([r["omega_phys"] for r in allrows])
        yaw = np.array([[r["yaw"] for r in ep] for ep in eps], dtype=float)
        dyaw = np.diff(yaw, axis=1)
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        head_rate = np.abs(dyaw / 0.1)
        print("%-18s wall=%4.0fs steps=%d  V=%5.0f[%4.0f..%4.0f] kcas=%5.0f  n=%4.2f(max %4.2f, p95 %4.2f) "
              "omega_phys: mean %4.1f p95 %5.1f max %5.1f | heading rate p95 %5.1f"
              % (name, time.time() - t0, len(allrows), V.mean(), V.min(), V.max(),
                 np.array([r["vcas"] for r in allrows]).mean(), n.mean(), n.max(),
                 np.percentile(n, 95), w.mean(), np.percentile(w, 95), w.max(),
                 np.percentile(head_rate, 95)), flush=True)


if __name__ == "__main__":
    main()
