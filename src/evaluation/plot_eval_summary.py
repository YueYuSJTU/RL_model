import os
from typing import Dict, Any, List

import numpy as np


def _annotate_bars(
    ax,
    bars,
    fmt: str = "{:.2f}",
    rotation: int = 0,
    pad_frac: float = 0.03,
    top_margin_frac: float = 0.12,
) -> None:
    """Annotate bars and keep labels inside the axes.

    - pad_frac: vertical padding as fraction of y-range
    - top_margin_frac: extra headroom to avoid text clipping
    """
    heights = [b.get_height() for b in bars]
    if not heights:
        return

    ymin, ymax = ax.get_ylim()
    # If default limits, expand based on bar heights
    hmax = max(heights)
    hmin = min(heights)
    if ymax <= ymin:
        ymin, ymax = 0.0, 1.0

    # Ensure some headroom
    data_min = min(ymin, hmin, 0.0)
    data_max = max(ymax, hmax)
    yrng = max(1e-9, data_max - data_min)
    new_top = data_max + top_margin_frac * yrng
    ax.set_ylim(data_min, new_top)

    ymin, ymax = ax.get_ylim()
    yrng = max(1e-9, ymax - ymin)
    pad = pad_frac * yrng

    for b in bars:
        h = b.get_height()
        y = h + pad
        # Clamp inside plotting area
        y = min(y, ymax - pad)
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=rotation,
            clip_on=True,
        )


def plot_opponent_metrics(
    opponents: Dict[str, Dict[str, Any]],
    out_path: str,
    overall_average=None,
    self_name: str = "SELF",
    oppo_name: str = "OPPO",
) -> str:
    """Plot per-opponent rows for quick comparison.

    Layout requirements implemented:
    - Each row corresponds to one opponent (<=5 expected).
    - Win rate is printed as text in the row title (not plotted).
    - Last row plots overall_average (if provided).
    - Blue bars for self, red bars for opponent.
    - Each metric subplot shows grouped bars (self vs opponent) side-by-side.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(opponents.keys())
    if not names and overall_average is None:
        raise ValueError("No opponents to plot")

    # Metrics to plot (grouped self vs opponent)
    # key_self, key_oppo, title, fmt
    metrics = [
        ("avg_hp", "avg_hp_oppo", "HP", "{:.2f}"),
        (
            "gun_opportunity_time_ratio",
            "gun_opportunity_time_ratio_oppo",
            "GunOpp time ratio",
            "{:.2%}",
        ),
        ("track_angle_mean", "track_angle_mean_oppo", "Track angle mean (rad)", "{:.2f}"),
        ("specific_energy_mean", "specific_energy_mean_oppo", "SpecificEnergy mean", "{:.1f}"),
    ]

    # rows: opponents + optional overall
    rows: List[tuple[str, Dict[str, Any]]] = [(n, opponents[n]) for n in names]
    if overall_average is not None:
        rows.append(("overall_average", overall_average))

    nrows = len(rows)
    ncols = len(metrics)

    fig_w = max(12, 2.8 * ncols)
    fig_h = max(3.0, 1.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    blue = "#1f77b4"
    red = "#d62728"

    for r, (row_name, row_data) in enumerate(rows):
        # Row header includes win rate (self) as number only
        if row_name == "overall_average":
            row_title = "overall_average"
        else:
            win_rate = float(row_data.get("win_rate", 0.0) or 0.0)
            row_title = f"{row_name} | win_rate(self)={win_rate:.2%}"

        for c, (k_self, k_oppo, title, fmt) in enumerate(metrics):
            ax = axes[r, c]

            # Add light red background for the last row
            if r == nrows - 1:
                ax.set_facecolor("#ff9393")

            v_self = float(row_data.get(k_self, 0.0) or 0.0)
            v_oppo = float(row_data.get(k_oppo, 0.0) or 0.0)

            x = np.array([0.0])
            # narrower bars with more whitespace
            width = 0.22
            bars1 = ax.bar(x - width / 2, [v_self], width=width, color=blue, label=self_name)
            bars2 = ax.bar(x + width / 2, [v_oppo], width=width, color=red, label=oppo_name)
            # add a bit of x padding so bars don't feel cramped
            ax.set_xlim(-0.6, 0.6)

            # Titles only on first row
            if r == 0:
                ax.set_title(title, fontsize=10)

            # Row label on first col
            if c == 0:
                ax.set_ylabel(row_title, fontsize=9)

            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.2)

            _annotate_bars(ax, bars1, fmt=fmt)
            _annotate_bars(ax, bars2, fmt=fmt)

            # Add legend once
            if r == 0 and c == ncols - 1:
                ax.legend(loc="upper right", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.suptitle("Evaluation summary (SELF=blue, OPPO=red)", fontsize=12)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
