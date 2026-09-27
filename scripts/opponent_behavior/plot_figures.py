"""Figures for the benchmark-opponent behavioural scale (fixed goal-point battery).

Reads the CSVs written by analyze.py and produces:

  figures/fig1_behaviour_scale.png
      (a) capability ladder      - pursuit duty cycle per opponent, with the
                                   spread over the 16 goal points
      (b) descriptor profile     - the measured behaviour vector per opponent
      (c) per-goal-point map     - which situations separate the opponents

  figures/fig2_behaviour_map.png
      straightness vs pursuit duty cycle, showing that the tiers are separated
      by the opponents' own measured behaviour (no reference model involved)

Usage:
  python scripts/opponent_behavior/plot_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import REPO  # noqa: E402

ROOT = REPO / "experiments" / "opponent_behavior"
FIG = ROOT / "figures"

BLUE = "#307bb5"
RED = "#c43f3c"
GREY = "#EAEAEA"
ORANGE = "#e08a1e"
GREEN = "#2e7d5b"


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "figure.dpi": 300,
    })


# descriptors shown in panel (b): label, csv key, format, group
PROFILE = [
    ("pursuit duty cycle", "mean_point_frac", "{:.2f}", 0),
    ("intercept rate", "intercept_rate", "{:.2f}", 0),
    ("gun frac.", "mean_gun_frac", "{:.3f}", 0),
    ("mean closure (ft/s)", "mean_closure_mean_fps", "{:+.0f}", 0),
    ("straightness", "mean_straightness", "{:.2f}", 1),
    ("yaw rate (deg/step)", "mean_yaw_rate_deg", "{:.2f}", 1),
    ("bank $\\sigma$ (rad)", "mean_bank_std_rad", "{:.2f}", 1),
    ("alt $\\sigma$ (ft)", "mean_alt_std_ft", "{:.0f}", 1),
    ("$E_s$ (ft)", "mean_Es_ft", "{:.0f}", 2),
    ("min range (ft)", "mean_dist_min_ft", "{:.0f}", 2),
]
GROUP_NAMES = ["Target pursuit", "Manoeuvring", "Energy / geometry"]


def fig1(cap, battery, ana):
    order = [r["id"] for r in cap]
    fig = plt.figure(figsize=(13.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 4.1, 3.4], wspace=0.42)

    # ---------------- (a) ladder
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))
    mean = np.array([f(r["mean_point_frac"]) for r in cap])
    lo = np.array([f(r["min_point_frac"]) for r in cap])
    hi = np.array([f(r["max_point_frac"]) for r in cap])
    tier_col = {"tier A": BLUE, "tier B": ORANGE, "tier C": GREEN}
    cols = [tier_col.get(r["tier"], "grey") for r in cap]
    ax.hlines(y, lo, hi, color="#9aa0a6", lw=1.6, zorder=1)
    ax.scatter(mean, y, s=95, c=cols, edgecolor="black", linewidth=1.0, zorder=3)
    for i, r in enumerate(cap):
        ax.annotate("%d/%d" % (round(f(r["intercept_rate"]) * f(r["n_points"])),
                               int(f(r["n_points"]))),
                    (mean[i], y[i]), textcoords="offset points", xytext=(11, -2),
                    fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(["opp. %s" % s for s in order])
    ax.set_xlabel("pursuit duty cycle\n(fraction of steps pointing at the commanded target)")
    ax.set_title("(a) Behavioural scale\n(bar = range over the 16 goal points;\n"
                 "label = goal points with a firing solution)", fontsize=9)
    gi = ana.get("tier_gap_index")
    if gi is not None:
        ax.axhline(gi + 0.5, color="k", ls="--", lw=1.4)
        ax.text(ax.get_xlim()[1], gi + 0.62, "tier boundary\n(largest gap in the data)",
                fontsize=7, ha="right", va="bottom")
    ax.grid(axis="x", alpha=0.25, ls="--")
    ax.set_axisbelow(True)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------- (b) descriptor profile
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [p[0] for p in PROFILE]
    groups = [p[3] for p in PROFILE]
    fmts = [p[2] for p in PROFILE]
    M = np.array([[f(r[k]) for _, k, _, _ in PROFILE] for r in cap], dtype=float)
    Z = np.zeros_like(M)
    for j in range(M.shape[1]):
        c = M[:, j]
        sd = c.std(ddof=0)
        Z[:, j] = (c - c.mean()) / sd if sd > 1e-12 else 0.0
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    im = ax2.imshow(np.clip(Z, -2, 2), aspect="auto", cmap="RdBu_r", norm=norm)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=40, ha="right")
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels(["opp. %s" % s for s in order])
    for i in range(len(order)):
        for j in range(len(labels)):
            ax2.text(j, i, fmts[j].format(M[i, j]), ha="center", va="center",
                     fontsize=6.6)
    for j in range(1, len(labels)):
        if groups[j] != groups[j - 1]:
            ax2.axvline(j - 0.5, color="k", lw=1.1)
    for gi_, gname in enumerate(GROUP_NAMES):
        cs = [j for j in range(len(labels)) if groups[j] == gi_]
        ax2.text(np.mean(cs), -0.92, gname, ha="center", va="bottom",
                 fontsize=8.5, weight="bold")
    ax2.set_ylim(len(order) - 0.5, -0.5)
    ax2.set_title("(b) Measured behaviour vector\n(colour = z-score within the pool; "
                  "cells show values)", fontsize=9)
    cb = fig.colorbar(im, ax=ax2, fraction=0.032, pad=0.015)
    cb.set_label("z-score", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)

    # ---------------- (c) per-goal-point map
    ax3 = fig.add_subplot(gs[0, 2])
    pts = ana["goal_points"]
    grid = np.full((len(order), len(pts)), np.nan)
    for i, s in enumerate(order):
        for j, p in enumerate(pts):
            m = [b for b in battery if b["subject"] == s and b["point"] == p]
            if m:
                grid[i, j] = f(m[0]["point_frac"])
    im3 = ax3.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0,
                     vmax=max(0.4, float(np.nanmax(grid))))
    ax3.set_yticks(range(len(order)))
    ax3.set_yticklabels(["opp. %s" % s for s in order])
    ax3.set_xticks(range(len(pts)))
    ax3.set_xticklabels([p.replace("bearing_", "b").replace("range_", "r")
                         .replace("alt_", "h").replace("cross_", "x")
                         for p in pts], rotation=60, ha="right", fontsize=6.2)
    for i in range(len(order)):
        for j in range(len(pts)):
            if np.isfinite(grid[i, j]):
                ax3.text(j, i, "%.2f" % grid[i, j], ha="center", va="center",
                         fontsize=5.6,
                         color="white" if grid[i, j] < 0.5 * np.nanmax(grid) else "black")
    cb3 = fig.colorbar(im3, ax=ax3, fraction=0.032, pad=0.015)
    cb3.set_label("pursuit duty cycle", fontsize=7.5)
    cb3.ax.tick_params(labelsize=6.5)
    ax3.set_title("(c) Where the opponents differ\n(pursuit duty cycle per goal point)",
                  fontsize=9)
    ax3.set_ylim(len(order) - 0.5, -0.5)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig1_behaviour_scale.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2(cap):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    tier_col = {"tier A": BLUE, "tier B": ORANGE, "tier C": GREEN}
    for r in cap:
        x = f(r["mean_straightness"])
        y = f(r["mean_point_frac"])
        ax.scatter(x, y, s=140, color=tier_col.get(r["tier"], "grey"),
                   edgecolor="black", linewidth=1.1, zorder=3)
        ax.annotate(r["id"], (x, y), textcoords="offset points", xytext=(10, 5),
                    fontsize=9, weight="bold")
    ax.set_xlabel("straightness  (net displacement / path length;\n"
                  "1.0 = flies a straight line and ignores the target)")
    ax.set_ylabel("pursuit duty cycle")
    ax.set_title("Opponents separate by their own measured behaviour", fontsize=10)
    ax.grid(alpha=0.25, ls="--")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor=c, edgecolor="black", label=k)
               for k, c in tier_col.items()]
    ax.legend(handles=handles, loc="best", frameon=False, title="tier",
              title_fontsize=8)
    out = FIG / "fig2_behaviour_map.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    style()
    cap = read_csv(ROOT / "opponent_capability.csv")
    battery = read_csv(ROOT / "goalpoint_battery.csv")
    with open(ROOT / "analysis.json", encoding="utf-8") as fh:
        ana = json.load(fh)
    p1 = fig1(cap, battery, ana)
    p2 = fig2(cap)
    print("written: %s" % p1)
    print("written: %s" % p2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
