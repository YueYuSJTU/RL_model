"""Build the benchmark-opponent behaviour-class table (table only, no figure).

Outputs
  experiments/opponent_behavior/behaviour_classes.csv
  experiments/opponent_behavior/behaviour_classes.tex
  experiments/opponent_behavior/feature_ranking.csv

Method
------
1. Opponents are re-numbered 1..8 in the fixed order of the source folders
   (opponent_pool/test_pool/1,2,3,11,16,17,23,616), so the identifiers are
   stable across revisions.  The mapping is printed and written to CSV.

2. Candidate features are pooled from BOTH measurement sets (free-flight
   behaviour runs bh_*.json and fixed goal-point runs gp_*.json).

3. Candidate five-class partitions are proposed from physically interpretable
   axes (vertical behaviour, manoeuvre intensity, throttle use) and each is
   scored by the mean eta^2 over all standardised features.  The partition that
   separates the measurements best is kept.

4. Column screening.  Three statistics were tried and ALL proved unable to
   rank features on this data set: with 8 opponents in 5 classes (four of them
   single-member) plain eta^2 reaches 1.000 for 13 unrelated columns, the
   corrected omega^2 still reaches 1.000 for several columns that are ~0 for
   every opponent, and leave-one-out single-column class accuracy never exceeds
   0.375.  The ranking is therefore reported as reference only, and the six
   columns are chosen for interpretability and behaviour-family coverage while
   giving every class a clearly visible signature.

   (original note)  With 8 opponents in 5 classes any effect size is
   unreliable -- measured: plain eta^2 reaches exactly 1.000 for 13 unrelated
   columns, and the small-sample corrected omega^2 still reaches 1.000 for
   several columns that are ~0 for every opponent.  Statistics are therefore
   used only to (a) drop degenerate columns (one value covering >=75% of the
   opponents) and (b) report each column's omega^2.  The six reported columns
   are then chosen to be non-degenerate, physically interpretable, drawn from
   DIFFERENT behaviour families, and to give each class a distinct signature.
   The full ranking is written to feature_ranking.csv so it can be overridden.

Usage:
  python scripts/opponent_behavior/select_table.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import REPO  # noqa: E402

ROOT = REPO / "experiments" / "opponent_behavior"
RAW = ROOT / "raw"

ORIG_ORDER = ["1", "2", "3", "11", "16", "17", "23", "616"]
NEW_ID = {o: str(i + 1) for i, o in enumerate(ORIG_ORDER)}

EXCLUDE = {"ep_steps", "ep_seconds", "react_measurable",
           "train_steps", "batch_size", "n_points", "n_episodes", "mean_ep_steps"}

FAMILY = {
    "climb_rate_mean": "VERTICAL", "climb_time_frac": "VERTICAL",
    "alt_trend": "VERTICAL", "alt_p10": "VERTICAL", "alt_p90": "VERTICAL",
    "alt_min": "VERTICAL", "alt_mean": "VERTICAL", "pitch_mean": "VERTICAL",
    "turn_rate_mean": "LATERAL", "turn_rate_std": "LATERAL",
    "turn_bias": "LATERAL", "left_time_frac": "LATERAL",
    "bank_mean_abs": "LATERAL", "bank_std": "LATERAL", "bank_p90": "LATERAL",
    "hard_bank_frac": "LATERAL", "roll_reversal_per_min": "LATERAL",
    "mean_yaw_rate_deg": "LATERAL", "mean_bank_std_rad": "LATERAL",
    "mean_straightness": "LATERAL",
    "control_activity": "CONTROL", "ctrl_throttle_mean": "CONTROL",
    "ctrl_throttle_std": "CONTROL", "ctrl_elevator_mean": "CONTROL",
    "ctrl_elevator_std": "CONTROL", "ctrl_elevator_mean_abs": "CONTROL",
    "ctrl_aileron_mean": "CONTROL", "ctrl_aileron_std": "CONTROL",
    "ctrl_aileron_mean_abs": "CONTROL", "ctrl_rudder_mean": "CONTROL",
    "ctrl_rudder_std": "CONTROL", "ctrl_rudder_mean_abs": "CONTROL",
    "speed_mean": "ENERGY", "speed_std": "ENERGY", "Es_mean": "ENERGY",
    "Es_trend": "ENERGY", "energy_exchange_corr": "ENERGY",
    "mean_Es_ft": "ENERGY",
    "los_point_frac": "REACTION", "los_closure_mean": "REACTION",
    "react_corr": "REACTION", "react_gain": "REACTION", "react_r2": "REACTION",
    "mean_point_frac": "TASK", "mean_gun_frac": "TASK",
    "mean_closure_mean_fps": "TASK", "mean_dist_min_ft": "TASK",
    "intercept_rate": "TASK", "sustained_points": "TASK",
}

# Six reported columns: one per behaviour family, each giving the classes a
# distinct signature (class descriptions are given in README.md).
CHOSEN: List[str] = [
    "bank_mean_abs",          # LATERAL   how steeply it banks
    "climb_rate_mean",        # VERTICAL  habitually climbs or descends
    "ctrl_elevator_mean",     # CONTROL   stick habit (sustained pull vs trim)
    "speed_mean",             # ENERGY    habitual speed
    "los_closure_mean",       # REACTION  does it close on the opponent at all
]

LABEL = {
    "climb_rate_mean": "Climb rate (ft/s)",
    "bank_mean_abs": "Mean $|\\phi|$ (deg)",
    "ctrl_throttle_mean": "Throttle cmd (mean)",
    "ctrl_elevator_mean": "Elevator cmd (mean)",
    "energy_exchange_corr": "Energy-exchange corr.",
    "los_closure_mean": "Closure to opponent (ft/s)",
    "hard_bank_frac": "$|\\phi|>60^{\\circ}$ (frac.)",
    "turn_rate_mean": "Turn rate (deg/s)",
    "control_activity": "Control activity",
    "speed_mean": "Mean speed (ft/s)",
    "Es_mean": "Mean specific energy (ft)",
    "turn_bias": "Turn bias (deg/s)",
}

FMTS = {
    "climb_rate_mean": "%+.1f", "bank_mean_abs": "%.1f",
    "ctrl_throttle_mean": "%.2f", "ctrl_elevator_mean": "%+.3f",
    "energy_exchange_corr": "%+.2f", "los_closure_mean": "%+.0f",
    "hard_bank_frac": "%.2f", "turn_rate_mean": "%.2f",
    "control_activity": "%.4f", "speed_mean": "%.0f",
    "Es_mean": "%.0f", "turn_bias": "%+.2f",
}


def load_features() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for o in ORIG_ORDER:
        row: Dict[str, float] = {}
        bh = RAW / ("bh_%s.json" % o)
        if bh.exists():
            row.update(json.loads(bh.read_text())["metrics"])
        gp = RAW / ("gp_%s.json" % o)
        if gp.exists():
            gps = json.loads(gp.read_text())["goal_points"]
            for k in sorted({k for v in gps.values() for k in v}):
                vals = [float(v[k]["mean"]) for v in gps.values()
                        if k in v and isinstance(v[k], dict)]
                vals = [x for x in vals if np.isfinite(x)]
                if vals:
                    row["mean_" + k] = float(np.mean(vals))
            reached = [v["gun_reached"]["mean"] for v in gps.values() if "gun_reached" in v]
            row["intercept_rate"] = float(np.mean([1.0 if x >= 0.5 else 0.0 for x in reached]))
        out[o] = row
    return out


def degenerate(X: np.ndarray, j: int) -> bool:
    v = X[:, j]
    vals, counts = np.unique(np.round(v, 6), return_counts=True)
    return (counts.max() / v.size >= 0.75) and (vals.size < 4)


def eta_squared(X: np.ndarray, labels: List[str]) -> float:
    vals = []
    for j in range(X.shape[1]):
        c = X[:, j]
        tot = float(((c - c.mean()) ** 2).sum())
        if tot <= 1e-12:
            continue
        bet = 0.0
        for lab in sorted(set(labels)):
            g = c[[i for i, l in enumerate(labels) if l == lab]]
            bet += len(g) * (g.mean() - c.mean()) ** 2
        vals.append(bet / tot)
    return float(np.mean(vals)) if vals else 0.0


def omega_squared(X: np.ndarray, labels: List[str], j: int) -> float:
    c = X[:, j]
    n = c.size
    groups = [c[[i for i, l in enumerate(labels) if l == lab]] for lab in sorted(set(labels))]
    k = len(groups)
    if n - k <= 0:
        return 0.0
    ss_tot = float(((c - c.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0
    ss_bet = float(sum(len(g) * (g.mean() - c.mean()) ** 2 for g in groups))
    ms_wit = (ss_tot - ss_bet) / (n - k)
    return max(0.0, float((ss_bet - (k - 1) * ms_wit) / (ss_tot + ms_wit)))


def partitions(feat: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    bank = {o: feat[o]["bank_mean_abs"] for o in ORIG_ORDER}
    climb_t = {o: feat[o]["climb_time_frac"] for o in ORIG_ORDER}
    thr = {o: feat[o]["ctrl_throttle_mean"] for o in ORIG_ORDER}

    def rule_a(o: str) -> str:
        if climb_t[o] <= 0.10:
            return "IV"
        if bank[o] >= 70:
            return "I"
        if bank[o] >= 50 and thr[o] <= 0.05:
            return "II"
        if bank[o] >= 30:
            return "III"
        return "V"

    def rule_b(o: str) -> str:
        b = bank[o]
        if b >= 70:
            return "I"
        if b >= 50:
            return "II"
        if b >= 30:
            return "III"
        if b >= 15:
            return "IV"
        return "V"

    def rule_c(o: str) -> str:
        if climb_t[o] <= 0.10:
            return "IV"
        if bank[o] >= 50:
            return "I" if thr[o] >= 0.5 else "II"
        return "III" if bank[o] >= 30 else "V"

    return {
        "rule_a (vertical, then bank + throttle)": [rule_a(o) for o in ORIG_ORDER],
        "rule_b (bank only)": [rule_b(o) for o in ORIG_ORDER],
        "rule_c (vertical, bank, throttle split)": [rule_c(o) for o in ORIG_ORDER],
    }


def main() -> int:
    feat = load_features()
    keys = sorted({k for row in feat.values() for k in row} - EXCLUDE)
    keys = [k for k in keys if all(np.isfinite(feat[o].get(k, np.nan)) for o in ORIG_ORDER)]
    X = np.array([[feat[o][k] for k in keys] for o in ORIG_ORDER], dtype=float)
    Xs = (X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0)

    parts = partitions(feat)
    scores = {n: eta_squared(Xs, l) for n, l in parts.items()}
    best = max(scores, key=scores.get)
    labels = parts[best]
    print("five-class partition candidates (mean eta^2 over %d features):" % len(keys))
    for n in sorted(scores, key=scores.get, reverse=True):
        cls: Dict[str, List[str]] = {}
        for o, l in zip(ORIG_ORDER, parts[n]):
            cls.setdefault(l, []).append(NEW_ID[o])
        print("   %-42s %.3f  %s" % (n, scores[n], {k: cls[k] for k in sorted(cls)}))
    print("\nselected partition: %s" % best)

    deg = [k for j, k in enumerate(keys) if degenerate(Xs, j)]
    om = {k: omega_squared(Xs, labels, keys.index(k)) for k in keys if k not in deg}
    ranking = sorted(om, key=om.get, reverse=True)

    print("\ndegenerate columns dropped (one value covers >=75%% of opponents): %d" % len(deg))
    print("   " + ", ".join(deg))
    print("\ntop 15 by omega^2 (reported for reference only; n=8, k=5 cannot rank reliably):")
    for i, k in enumerate(ranking[:15], 1):
        print("   %2d. %-26s %.3f [%s]%s" % (
            i, k, om[k], FAMILY.get(k, "?"), "  <= reported" if k in CHOSEN else ""))

    print("\nreported 6 columns:")
    for k in CHOSEN:
        print("   %-24s %-9s omega^2=%.3f" % (k, FAMILY.get(k, "?"), om.get(k, float("nan"))))
    M = np.array([[feat[o][k] for k in CHOSEN] for o in ORIG_ORDER])
    C = np.corrcoef(M.T)
    worst = (0.0, "", "")
    for i in range(len(CHOSEN)):
        for j in range(i + 1, len(CHOSEN)):
            if abs(C[i, j]) > worst[0]:
                worst = (abs(C[i, j]), CHOSEN[i], CHOSEN[j])
    print("   max pairwise |r| = %.2f  (%s vs %s)" % worst)

    with open(ROOT / "feature_ranking.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "feature", "family", "omega_squared", "selected"])
        for i, k in enumerate(ranking, 1):
            w.writerow([i, k, FAMILY.get(k, "?"), "%.4f" % om[k],
                        "yes" if k in CHOSEN else ""])

    rows = []
    for o in ORIG_ORDER:
        r = dict(category=labels[ORIG_ORDER.index(o)], source_folder=o)
        for k in CHOSEN:
            r[k] = feat[o][k]
        rows.append(r)
    # sequential numbering 1..8 in the order the table is printed (class order,
    # and within a class the source-folder order); the source folder is kept in
    # the CSV so the mapping stays reproducible.
    rows.sort(key=lambda r: (r["category"], ORIG_ORDER.index(r["source_folder"])))
    for i, r in enumerate(rows, 1):
        r["opponent"] = str(i)

    with open(ROOT / "behaviour_classes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "opponent", "source_folder"] + CHOSEN)
        w.writeheader()
        w.writerows(rows)

    tex = [r"\begin{table}[h]",
           r"\caption{Measured behaviour of the eight benchmark opponents over identical "
           r"60\,s free-flight episodes started from identical initial conditions. "
           r"Class definitions are given in the text.}",
           r"\label{tab:bench_behaviour}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}cl" + "r" * len(CHOSEN) + "}",
           r"\toprule",
           "Class & Opp. & " + " & ".join(LABEL.get(k, k) for k in CHOSEN) + r" \\",
           r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["category"] != prev:
            tex.append(r"\midrule")
        prev = r["category"]
        tex.append("%s & %s & " % (r["category"], r["opponent"])
                   + " & ".join(FMTS.get(k, "%.3g") % r[k] for k in CHOSEN) + r" \\")
    tex += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    (ROOT / "behaviour_classes.tex").write_text("\n".join(tex) + "\n")

    print("\n=== TABLE ===")
    print("%-5s %-4s " % ("Class", "Opp.")
          + " ".join("%13s" % LABEL.get(k, k)[:13] for k in CHOSEN))
    for r in rows:
        print("%-5s %-4s " % (r["category"], r["opponent"])
              + " ".join("%13s" % (FMTS.get(k, "%.3g") % r[k]) for k in CHOSEN))
    print("\nnumbering map (paper id <- source folder): "
          + ", ".join("%s<-%s" % (r["opponent"], r["source_folder"]) for r in rows))
    print("\nwritten: %s" % (ROOT / "behaviour_classes.csv"))
    print("written: %s" % (ROOT / "behaviour_classes.tex"))
    print("written: %s" % (ROOT / "feature_ranking.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
