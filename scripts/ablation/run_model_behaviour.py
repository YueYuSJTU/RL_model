"""Measure the models evaluated in the paper with the SAME reward-free behaviour
protocol that was used for the benchmark opponents (scripts/opponent_behavior).

Why: the reviewer objected that metrics such as energy, track angle or gun
opportunity are quantities the training reward is defined on, so reporting them
as a capability comparison is circular.  The metrics produced here are measured
in free flight with NO reward, NO engagement outcome and a scripted
constant-command drone that is identical for every subject, i.e. exactly the
protocol behind the paper's Table of opponent behaviour.  For the final policy
(shape=stage2, task_tracking.py:382-438 plus the terminal override at
task_tracking.py:1000-1015) none of these quantities appears in the reward:

  mean turn rate           - turn rate appears in no reward term
  reaction R^2 / gain      - the correlation between the line-of-sight angle and
                             the turn rate measures whether the policy steers by
                             the opponent; the reward contains no turn-rate term
  corner-speed adherence   - the reward contains no speed term at all (the
                             closure-rate term depends on the range RATE V_c,
                             not on the airspeed)
  Delta Es                 - engagement-level energy margin, taken from the
                             existing benchmark records; the reward contains no
                             term that compares the energies of the two aircraft

Read-only guarantee: jsb_env/**, src/** and scripts/opponent_behavior/** are
imported, never modified.  Run with PYTHONDONTWRITEBYTECODE=1.

Usage:
  PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/run_model_behaviour.py --group opponents
  PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/run_model_behaviour.py --group paper --repeats 5
  PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/run_model_behaviour.py --analyze

Output:
  experiments/model_behavior/raw/mb_<group>_<name>.json   per-subject metrics + samples
  experiments/model_behavior/corner_speed.json           empirical manoeuvre envelope
  experiments/model_behavior/summary.json/.csv           one row per subject
"""
from __future__ import annotations

import argparse
import csv
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

from common import load_policy, make_eval_env, write_json          # noqa: E402
from run_behaviour import run_free, behaviour_metrics              # noqa: E402

OUT = Path("experiments") / "model_behavior"
RAW = OUT / "raw"

# --------------------------------------------------------------------------
# Subjects
# --------------------------------------------------------------------------
PAPER = {
    "BDC-LSTM": "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35",
    "BDC-MLP": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
    "BD": "experiments/20260506_105525_best_train_mlp2_no_curri_have_pool/stage1/20260506_163252_cycle_9",
    "BC": "experiments/20260506_211933_best_train_mlp2_have_curri_no_pool/stage2/20260507_155441_cycle_30",
    "Baseline": "experiments/20250616_221656/stage2/20250616_221824_TrackingTask_ppo_1layer1",
}

# Opponent-pool-initialization ablation arms of Section 5 (sec:init_ablation)
ABLATION = {
    "Init-with": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
    "Init-without": "experiments/20260924_225859_selfplay_init/stage2/20260926_124349_cycle_30",
}

# The eight benchmark opponents of the paper (opponent_pool/test_pool), used both
# as reference subjects and to estimate the manoeuvre envelope, so that the
# corner speed is a property of the aircraft/environment rather than of the
# models being evaluated.
OPPONENTS = {
    "Opp1-goalpoint": "opponent_pool/test_pool/1",
    "Opp2-combat": "opponent_pool/test_pool/616",
    "Opp3-combat": "opponent_pool/test_pool/2",
    "Opp4-combat": "opponent_pool/test_pool/3",
    "Opp5-level": "opponent_pool/test_pool/11",
    "Opp6-level": "opponent_pool/test_pool/16",
    "Opp7-level": "opponent_pool/test_pool/17",
    "Opp8-level": "opponent_pool/test_pool/23",
}

GROUPS = {"paper": PAPER, "ablation": ABLATION, "opponents": OPPONENTS}
GROUPS["all"] = dict(OPPONENTS, **ABLATION, **PAPER)

# corner-speed adherence: exp(-lambda * |V - Vc| / Vc), lambda from metrics.md
LAMBDA = 5.0
MANOEUVRE_BANK_DEG = 30.0      # only turning flight counts for adherence
ENVELOPE_BIN_FPS = 25.0
ENVELOPE_MIN_SAMPLES = 200


def step_samples(series):
    """Per-step (speed, |bank|, |turn rate|) of the subject, from its own trace."""
    rows = []
    n = len(series)
    for i in range(1, n):
        dyaw = series[i]["yaw"] - series[i - 1]["yaw"]
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        rows.append((float(series[i]["speed"]),
                     abs(math.degrees(series[i]["roll"])),
                     abs(dyaw) / 0.1))
    return rows


def measure(name, path, repeats, seed):
    t0 = time.time()
    vec = make_eval_env(seed=seed)
    subject = load_policy(path)
    eps = run_free(vec, subject, repeats=repeats)
    vec.close()
    per_ep = [p for p in (behaviour_metrics(e["series"]) for e in eps) if p]
    agg = {k: float(np.mean([p[k] for p in per_ep])) for k in per_ep[0]}
    samples = []
    for e in eps:
        samples.extend(step_samples(e["series"]))
    return agg, per_ep, samples, time.time() - t0


def run_group(group, repeats, seed0):
    subjects = GROUPS[group]
    RAW.mkdir(parents=True, exist_ok=True)
    for i, (name, path) in enumerate(subjects.items()):
        out = RAW / ("mb_%s_%s.json" % (group, name.replace("/", "_")))
        if out.exists():
            print("skip (exists):", out, flush=True)
            continue
        agg, per_ep, samples, wall = measure(name, path, repeats, seed0 + i)
        write_json(out, dict(subject=name, group=group, subject_path=path,
                             repeats=repeats, seed=seed0 + i, wall_s=wall,
                             metrics=agg, per_episode=per_ep,
                             samples=[[round(v, 2) for v in s] for s in samples]))
        print("[%s] %-16s turn=%5.2f deg/s trStd=%5.2f |bank|=%5.1f hard=%4.2f rev=%5.1f "
              "reactR2=%.2f meas=%d speed=%5.0f Es=%6.0f (%4.0fs)"
              % (group, name, agg["turn_rate_mean"], agg["turn_rate_std"], agg["bank_mean_abs"],
                 agg["hard_bank_frac"], agg["roll_reversal_per_min"], agg["react_r2"],
                 int(agg["react_measurable"]), agg["speed_mean"], agg["Es_mean"], wall),
              flush=True)


def envelope(samples):
    """Empirical manoeuvre envelope omega_env(V): 95th percentile of the turn
    rate observed in each speed bin, pooled over the reference subjects."""
    V = np.array([s[0] for s in samples], dtype=float)
    W = np.array([s[2] for s in samples], dtype=float)
    lo, hi = 150.0, 1000.0
    edges = np.arange(lo, hi + ENVELOPE_BIN_FPS, ENVELOPE_BIN_FPS)
    curve = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (V >= a) & (V < b)
        if int(m.sum()) >= ENVELOPE_MIN_SAMPLES:
            curve.append(dict(v_mid=float(0.5 * (a + b)), n=int(m.sum()),
                              v_mean=float(V[m].mean()),
                              omega_p95=float(np.percentile(W[m], 95)),
                              omega_mean=float(W[m].mean())))
    if not curve:
        return None
    k = int(np.argmax([c["omega_p95"] for c in curve]))
    vc = curve[k]["v_mid"]
    # refine with a local parabola through the peak and its two neighbours
    if 0 < k < len(curve) - 1:
        xs = np.array([curve[k - 1]["v_mid"], curve[k]["v_mid"], curve[k + 1]["v_mid"]])
        ys = np.array([curve[k - 1]["omega_p95"], curve[k]["omega_p95"], curve[k + 1]["omega_p95"]])
        c = np.polyfit(xs, ys, 2)
        if c[0] < 0:
            vc = float(-c[1] / (2 * c[0]))
    return dict(v_corner_fps=vc, lambda_=LAMBDA, bin_fps=ENVELOPE_BIN_FPS,
                min_samples=ENVELOPE_MIN_SAMPLES, curve=curve)


def adherence(samples, vc):
    """Corner-speed adherence over the turning part of the flight."""
    if not samples or not vc:
        return None, 0
    V = np.array([s[0] for s in samples], dtype=float)
    bank = np.array([s[1] for s in samples], dtype=float)
    m = bank >= MANOEUVRE_BANK_DEG
    if int(m.sum()) == 0:
        return None, 0
    val = float(np.mean(np.exp(-LAMBDA * np.abs(V[m] - vc) / vc)))
    return val, int(m.sum())


def analyse():
    files = sorted(glob.glob(str(RAW / "mb_*.json")))
    if not files:
        raise SystemExit("no raw measurements; run --group first")
    recs = [json.load(open(f)) for f in files]
    ref = [r for r in recs if r["group"] == "opponents"]
    env = envelope([s for r in (ref or recs) for s in r["samples"]])
    write_json(OUT / "corner_speed.json", dict(
        estimated_from=sorted(r["subject"] for r in (ref or recs)), **env))
    vc = env["v_corner_fps"]
    print("\nempirical corner speed Vc = %.0f ft/s (peak of the 95th-percentile "
          "turn-rate envelope, %d subjects)" % (vc, len(ref or recs)))
    for c in env["curve"]:
        print("   V=%5.0f fps  n=%6d  omega_p95=%5.2f deg/s  omega_mean=%5.2f"
              % (c["v_mid"], c["n"], c["omega_p95"], c["omega_mean"]))

    rows = []
    for r in recs:
        a, n_man = adherence(r["samples"], vc)
        m = r["metrics"]
        rows.append(dict(subject=r["subject"], group=r["group"],
                         turn_rate_mean=m["turn_rate_mean"], turn_rate_std=m["turn_rate_std"],
                         bank_mean_abs=m["bank_mean_abs"], hard_bank_frac=m["hard_bank_frac"],
                         roll_reversal_per_min=m["roll_reversal_per_min"],
                         control_activity=m["control_activity"],
                         react_r2=m["react_r2"] if m["react_measurable"] else None,
                         react_gain=m["react_gain"] if m["react_measurable"] else None,
                         react_measurable=bool(m["react_measurable"]),
                         los_point_frac=m["los_point_frac"],
                         speed_mean=m["speed_mean"], Es_mean=m["Es_mean"],
                         corner_adherence=a, corner_steps=n_man))
    write_json(OUT / "summary.json", dict(v_corner_fps=vc, subjects=rows))
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n%-18s %6s %6s %6s %6s %7s %6s" % ("subject", "group", "turn", "R2", "corner", "speed", "Es"))
    for r in rows:
        print("%-18s %6s %6.2f %6s %6s %7.0f %6.0f" % (
            r["subject"], r["group"], r["turn_rate_mean"],
            ("%.2f" % r["react_r2"]) if r["react_r2"] is not None else "  n/a",
            ("%.3f" % r["corner_adherence"]) if r["corner_adherence"] is not None else "  n/a",
            r["speed_mean"], r["Es_mean"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=sorted(GROUPS), default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.group:
        run_group(args.group, args.repeats, args.seed0)
    if args.analyze or not args.group:
        analyse()


if __name__ == "__main__":
    main()
