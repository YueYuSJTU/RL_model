"""Reward-independent capability profiling of the evaluated models.

Motivation
----------
The reviewers objected that several metrics of the benchmark table (gun
opportunity ratio, mean track angle, total damage, mean specific energy) are
quantities that the training reward is defined on, so comparing models on them
is partly circular.  This script measures, in the SAME engagements as the
benchmark evaluation and with the same protocol (randomised initial states,
goal_point_prob = 0), four quantities that appear in NO reward term of the
evaluated policy (shape = stage2, task_tracking.py:382-438 plus the terminal
override at task_tracking.py:1000-1015):

  dEs                 mean specific-energy margin  mean(Es_ego) - mean(Es_oppo),
                      Es = h + V^2/(2g).  The reward compares angles and
                      ranges, never the energies of the two aircraft.
  turn_rate_mean      mean turn rate of a coordinated turn,
                      omega = g*sqrt(n^2-1)/V, with n read from JSBSim
                      (forces/load-factor).  No reward term contains a turn
                      rate or a load factor.
  react_r2            fraction of the policy's turning explained by the
                      line-of-sight angle to the opponent (squared correlation
                      between the LOS angle and the turn rate).  Measures
                      whether the policy steers BY the opponent; the reward
                      contains no turn-rate term, so it cannot be optimised
                      directly.
  corner_adherence    mean exp(-lambda*|V - Vc|/Vc) over manoeuvring steps
                      (n >= 2), with the corner speed evaluated at the
                      instantaneous altitude and weight,
                      Vc = sqrt(2*n_max*W/(rho*S*CL_max)), n_max = 9 (the F-16
                      flight-control limit in the JSBSim model) and
                      CL_max = 1.6 (the lift limit at the 25 deg angle-of-
                      attack limit of the model).  No reward term contains the
                      airspeed (the closure-rate term depends on the range
                      RATE, not on V).

Read-only: jsb_env/**, src/** and scripts/opponent_behavior/** are imported,
never modified.  Run with PYTHONDONTWRITEBYTECODE=1.

Usage (sharded):
  for i in 0 1 2 3 4 5 6 7; do
    PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/measure_capability_profile.py \
        --group paper --episodes 200 --shards 8 --shard $i &
  done
  PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/measure_capability_profile.py --merge --group paper

Output: experiments/capability_profile/<group>/shard<i>.jsonl  and merged.json / summary.md
"""
from __future__ import annotations

import argparse
import glob
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

from stable_baselines3.common.vec_env import DummyVecEnv          # noqa: E402

from common import _predict, load_policy                          # noqa: E402
from src.environments.make_env import create_env                  # noqa: E402
from src.utils.serialization import load_config                   # noqa: E402

import jsb_env.jsbgym_m.properties as prp                         # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "scripts", "ablation"))
from run_model_behaviour import (ABLATION, OPPONENTS, PAPER,      # noqa: E402
                                 LAMBDA, MANOEUVRE_BANK_DEG)

OUT_ROOT = Path("experiments") / "capability_profile"
G_FT_S2 = 32.174
S_WING_FT2 = 300.0        # metrics/wingarea of the JSBSim F-16 model
CL_MAX = 1.6              # lift limit at the 25 deg alpha limit of the model
N_MAX = 9.0               # fcs/elevator-cmd-limiter, 9G in the model
MANOEUVRE_N = 2.0         # "manoeuvring" for the corner-speed metric
LOS_POINT_RAD = math.radians(30.0)

P_N = prp.Property("forces/load-factor", "normal load factor")
P_RHO = prp.Property("atmosphere/rho-slugs_ft3", "air density")
P_W = prp.Property("inertia/weight-lbs", "weight")
P_KCAS = prp.Property("velocities/vc-kts", "calibrated airspeed")
P_NZ = prp.Property("accelerations/Nz", "load factor Nz")
P_NPILOT = prp.Property("accelerations/n-pilot-z-norm", "pilot normal load factor")

GROUPS = {"paper": PAPER, "ablation": ABLATION, "opponents": OPPONENTS}
GROUPS["all"] = dict(OPPONENTS, **ABLATION, **PAPER)

# optional benchmark-opponent file (see load_opponents)
BENCH_FILE = None


def get_env_with_sims(vec):
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


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def correlate(a, b, min_std):
    """Squared correlation, but only when the explanatory series actually varies
    (same variability floor as the opponent behaviour study)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 5 or b.size < 5:
        return 0.0, False, 0.0
    if a.std() < min_std or b.std() < 1e-6:
        return 0.0, False, 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0, False, 0.0
    gain = float(np.polyfit(a, b, 1)[0])
    return c * c, True, gain


def run_pair(model1_path, model2_path, n_episodes, seed=0):
    """One matchup, mirroring Evaluator.run_match but reading the simulators."""
    env_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))
    env_cfg["render_mode"] = None
    env_cfg["use_vec_normalize"] = False
    vec = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv)
    vec.training = False
    vec.norm_reward = False
    vec.env_method("update_task_parameters", goal_point_prob=0.0)
    env = get_env_with_sims(vec)
    m1 = load_policy(model1_path)
    m2 = load_policy(model2_path)

    eps = []
    rng = np.random.default_rng(seed)
    try:
        for _ in range(n_episodes):
            obs = vec.reset()
            half = obs.shape[1] // 2
            s1 = s2 = None
            e1 = np.ones(1, dtype=bool)
            e2 = np.ones(1, dtype=bool)
            steps = []
            done = False
            win = None
            while not done:
                a1, s1 = _predict(m1, obs[:, :half], s1, e1)
                a2, s2 = _predict(m2, obs[:, half:], s2, e2)
                obs, _r, dones, info = vec.step(np.concatenate([a1, a2], axis=-1))
                e1 = dones
                e2 = dones
                sim, osim = env.sim, env.opponent_sim
                tr = info[0]["trajectory"]
                trs, tro = tr["self"], tr["oppo"]
                V = float(sim[prp.vtrue_fps])
                Vo = float(osim[prp.vtrue_fps])
                n = float(sim[P_N])
                h = float(sim[prp.altitude_sl_ft])
                ho = float(osim[prp.altitude_sl_ft])
                rho = float(sim[P_RHO])
                W = float(sim[P_W])
                roll = float(trs["roll_rad"])
                # line of sight from the ego to the opponent, in the ego's frame
                los_n = tro["x_ft"] - trs["x_ft"]
                los_e = tro["y_ft"] - trs["y_ft"]
                bearing = math.degrees(math.atan2(los_e, los_n))
                los_angle = wrap_pi(math.radians((bearing - trs["yaw_deg"] + 180.0) % 360.0 - 180.0))
                omega = (G_FT_S2 * math.sqrt(max(n * n - 1.0, 0.0)) / V * 180.0 / math.pi) if V > 1 else 0.0
                omega_signed = math.copysign(omega, roll) if abs(roll) > math.radians(10.0) else 0.0
                vc = math.sqrt(2.0 * N_MAX * W / (rho * S_WING_FT2 * CL_MAX)) if rho > 0 else 0.0
                steps.append(dict(
                    V=V, Vo=Vo, n=n, nz=float(sim[P_NZ]), npilot=float(sim[P_NPILOT]),
                    alt=h, alt_o=ho, kcas=float(sim[P_KCAS]),
                    roll=roll, los=los_angle, yaw=float(trs["yaw_deg"]),
                    omega=omega, omega_signed=omega_signed,
                    vc=vc, Es=h + V * V / (2 * G_FT_S2), Es_o=ho + Vo * Vo / (2 * G_FT_S2),
                    dist=float(info[0]["metrics_step"]["distance_oppo_ft"])))
                if bool(dones[0]):
                    done = True
                    win = float(info[0].get("env_info", {}).get("win", 0) or 0)
            eps.append(dict(steps=steps, win=win))
    finally:
        vec.close()

    return aggregate(eps)


def aggregate(eps):
    n_ep = len(eps)
    out = dict(
        n_episodes=n_ep,
        win_rate=float(np.mean([e["win"] == 1 for e in eps])),
        draw_rate=float(np.mean([e["win"] == 0 for e in eps])),
        loss_rate=float(np.mean([e["win"] == -1 for e in eps])),
        dEs=0.0, turn_rate_mean=0.0, turn_rate_man=0.0, n_mean=0.0, n_p95=0.0,
        react_r2=0.0, react_r2_phys=0.0, react_gain=0.0, los_point_frac=0.0,
        corner_adherence=None, corner_steps=0, speed_mean=0.0, kcas_mean=0.0,
        alt_mean=0.0, vc_mean=0.0, dist_mean=0.0, measurable_frac=0.0)
    if not eps:
        return out
    dEs, turn, turn_man, nm, np95, r2, r2p, gn_list, losf, corner, csteps = ([] for _ in range(11))
    spd, kcas, alt, vcm, dst, meas = ([] for _ in range(6))
    for e in eps:
        s = e["steps"]
        if len(s) < 5:
            continue
        V = np.array([x["V"] for x in s]); n = np.array([x["n"] for x in s])
        om = np.array([x["omega"] for x in s]); oms = np.array([x["omega_signed"] for x in s])
        los = np.array([x["los"] for x in s])
        yaw = np.array([x["yaw"] for x in s])
        dyaw = np.diff(yaw)
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        yaw_rate = dyaw / 0.1                       # deg/s, same proxy as the opponent study
        vc = np.array([x["vc"] for x in s])
        dEs.append(float(np.mean([x["Es"] for x in s]) - np.mean([x["Es_o"] for x in s])))
        turn.append(float(om.mean()))
        man = n >= MANOEUVRE_N
        turn_man.append(float(om[man].mean()) if man.any() else 0.0)
        nm.append(float(n.mean())); np95.append(float(np.percentile(n, 95)))
        los_mid = np.radians((los[:-1] + los[1:]) / 2.0)
        r2.append(correlate(yaw_rate, los_mid, 1.0)[0])
        rr, m1, gn = correlate(oms[1:], los_mid, 1.0)
        r2p.append(rr); gn_list.append(gn); meas.append(1.0 if m1 else 0.0)
        losf.append(float(np.mean(np.abs(los) < LOS_POINT_RAD)))
        if man.any():
            corner.append(float(np.mean(np.exp(-LAMBDA * np.abs(V[man] - vc[man]) / vc[man]))))
            csteps.append(int(man.sum()))
        spd.append(float(V.mean())); kcas.append(float(np.mean([x["kcas"] for x in s])))
        alt.append(float(np.mean([x["alt"] for x in s]))); vcm.append(float(vc.mean()))
        dst.append(float(np.mean([x["dist"] for x in s])))
    out.update(dEs=float(np.mean(dEs)), turn_rate_mean=float(np.mean(turn)),
               turn_rate_man=float(np.mean(turn_man)), n_mean=float(np.mean(nm)),
               n_p95=float(np.mean(np95)), react_r2=float(np.mean(r2)),
               react_r2_phys=float(np.mean(r2p)), react_gain=float(np.mean(gn_list)),
               los_point_frac=float(np.mean(losf)),
               corner_adherence=float(np.mean(corner)) if corner else None,
               corner_steps=int(np.sum(csteps)), speed_mean=float(np.mean(spd)),
               kcas_mean=float(np.mean(kcas)), alt_mean=float(np.mean(alt)),
               vc_mean=float(np.mean(vcm)), dist_mean=float(np.mean(dst)),
               measurable_frac=float(np.mean(meas)),
               nz_mean=float(np.mean([x["nz"] for e in eps for x in e["steps"]])),
               nz_p95=float(np.percentile([x["nz"] for e in eps for x in e["steps"]], 95)),
               npilot_mean=float(np.mean([x["npilot"] for e in eps for x in e["steps"]])),
               npilot_p95=float(np.percentile([x["npilot"] for e in eps for x in e["steps"]], 95)))
    return out


def load_opponents():
    """The benchmark opponents to measure against.

    Defaults to the opponent set of the paper's benchmark evaluation.  A
    different set can be given with --bench-file (the same JSON format used by
    the benchmark evaluation scripts), so that the profile is measured against
    exactly the opponents reported in the outcome table.
    """
    if BENCH_FILE:
        d = json.load(open(BENCH_FILE))
        return {k: (v if os.path.isabs(v) else os.path.join(ROOT, v)) for k, v in d.items()}
    return OPPONENTS


def pairs_for(group):
    """The evaluated arms against the benchmark opponents."""
    models = GROUPS[group]
    return [(mname, mpath, oname, opath)
            for mname, mpath in models.items()
            for oname, opath in load_opponents().items()]


def run_shard(group, episodes, shards, shard, out_name=None):
    out_dir = OUT_ROOT / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / ((out_name or "shard%d" % shard) + ".jsonl")
    done = set()
    if out_file.exists():
        for line in open(out_file):
            try:
                r = json.loads(line)
                done.add((r["model"], r["opponent"]))
            except Exception:
                pass
    pairs = pairs_for(group)
    mine = [p for i, p in enumerate(pairs) if i % shards == shard]
    print("[shard %d/%d] %s: %d pairs (%d already done)"
          % (shard, shards, group, len(mine), len(done)), flush=True)
    with open(out_file, "a") as fh:
        for i, (mname, mpath, oname, opath) in enumerate(mine):
            if (mname, oname) in done:
                print("  skip %s vs %s" % (mname, oname), flush=True)
                continue
            t0 = time.time()
            rec = dict(model=mname, model_path=mpath, opponent=oname, opponent_path=opath,
                       n_episodes=episodes)
            try:
                rec["metrics"] = run_pair(mpath, opath, episodes, seed=1000 * shard + i)
                rec["ok"] = True
            except Exception as exc:                                    # noqa: BLE001
                rec["ok"] = False
                rec["error"] = repr(exc)[:500]
                import traceback
                rec["traceback"] = traceback.format_exc()[-2000:]
            rec["seconds"] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            m = rec.get("metrics", {})
            print("  %-10s vs %-16s win=%5.1f%% dEs=%+6.0f turn=%5.2f n=%4.2f R2=%.2f corner=%s (%.0fs)"
                  % (mname, oname, 100 * m.get("win_rate", 0), m.get("dEs", 0),
                     m.get("turn_rate_mean", 0), m.get("n_mean", 0), m.get("react_r2", 0),
                     ("%.3f" % m["corner_adherence"]) if m.get("corner_adherence") is not None else "n/a",
                     rec["seconds"]), flush=True)


def merge(group):
    out_dir = OUT_ROOT / group
    recs = []
    for fp in sorted(glob.glob(str(out_dir / "shard*.jsonl"))):
        for line in open(fp):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    ok = [r for r in recs if r.get("ok")]
    json.dump(recs, open(out_dir / "merged.json", "w"), indent=1)
    keys = ["win_rate", "draw_rate", "loss_rate", "dEs", "turn_rate_mean", "turn_rate_man",
            "n_mean", "n_p95", "react_r2", "react_r2_phys", "los_point_frac", "corner_adherence",
            "speed_mean", "kcas_mean", "alt_mean", "vc_mean", "dist_mean", "measurable_frac"]
    lines = ["# capability profile (group=%s, %d episodes per matchup)\n" % (group, ok[0]["n_episodes"] if ok else 0),
             "| model | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for model in sorted({r["model"] for r in ok}):
        rs = [r for r in ok if r["model"] == model]
        n = len(rs)
        cells = []
        for k in keys:
            vals = [r["metrics"][k] for r in rs if r["metrics"].get(k) is not None]
            cells.append("%.4g" % (sum(vals) / len(vals)) if vals else "n/a")
        lines.append("| %s | %s |" % (model, " | ".join(cells)))
    open(out_dir / "summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote %s" % (out_dir / "summary.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(GROUPS))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--bench-file", default=None,
                    help="JSON {name: path} of benchmark opponents to measure against")
    args = ap.parse_args()
    global BENCH_FILE
    BENCH_FILE = args.bench_file
    if args.merge:
        merge(args.group)
    else:
        run_shard(args.group, args.episodes, args.shards, args.shard, args.out_name)


if __name__ == "__main__":
    main()
