import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SceneArrays:
    t: np.ndarray  # (T,)
    target_point_ft: np.ndarray  # (3,)
    red_pos_ft: np.ndarray  # (T, R, 3)
    red_valid_mask: np.ndarray  # (T, R)
    blue_pos_ft: np.ndarray  # (T, B, 3)
    blue_valid_mask: np.ndarray  # (T, B)


def _to_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(a)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def frames_to_arrays(scene_frames: List[Dict[str, Any]]) -> SceneArrays:
    if not scene_frames:
        raise ValueError("scene_frames is empty")

    t = _to_numpy([float(f.get("t", 0.0)) for f in scene_frames], dtype=np.float32)

    target_point_ft = _to_numpy(scene_frames[0]["target_point_ft"], dtype=np.float32)
    if target_point_ft.shape != (3,):
        target_point_ft = target_point_ft.reshape(3)

    red_counts = [int(_to_numpy(f["red_pos_ft"]).shape[0]) for f in scene_frames]
    blue_counts = [int(_to_numpy(f["blue_pos_ft"]).shape[0]) for f in scene_frames]
    max_red = max(red_counts) if red_counts else 0
    max_blue = max(blue_counts) if blue_counts else 0

    red_pos_ft = np.zeros((len(scene_frames), max_red, 3), dtype=np.float32)
    red_valid_mask = np.zeros((len(scene_frames), max_red), dtype=bool)
    blue_pos_ft = np.zeros((len(scene_frames), max_blue, 3), dtype=np.float32)
    blue_valid_mask = np.zeros((len(scene_frames), max_blue), dtype=bool)

    for i, f in enumerate(scene_frames):
        r = _to_numpy(f["red_pos_ft"], dtype=np.float32).reshape(-1, 3)
        b = _to_numpy(f["blue_pos_ft"], dtype=np.float32).reshape(-1, 3)

        if r.shape[0] > 0:
            red_pos_ft[i, : r.shape[0]] = r
            red_valid_mask[i, : r.shape[0]] = True

        if b.shape[0] > 0:
            blue_pos_ft[i, : b.shape[0]] = b
            blue_valid_mask[i, : b.shape[0]] = True

    return SceneArrays(
        t=t,
        target_point_ft=target_point_ft,
        red_pos_ft=red_pos_ft,
        red_valid_mask=red_valid_mask,
        blue_pos_ft=blue_pos_ft,
        blue_valid_mask=blue_valid_mask,
    )


def dump_scene_json(scene_frames: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scene_frames, f, ensure_ascii=False, indent=2, default=_default)


def dump_scene_npz(scene_frames: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = frames_to_arrays(scene_frames)
    np.savez_compressed(
        out_path,
        t=arrays.t,
        target_point_ft=arrays.target_point_ft,
        red_pos_ft=arrays.red_pos_ft,
        red_valid_mask=arrays.red_valid_mask,
        blue_pos_ft=arrays.blue_pos_ft,
        blue_valid_mask=arrays.blue_valid_mask,
    )


def plot_scene_3d_trajectory(
    scene_frames: List[Dict[str, Any]],
    title: Optional[str] = None,
) -> None:
    arrays = frames_to_arrays(scene_frames)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    def _plot_team(pos: np.ndarray, valid: np.ndarray, color: str, label_prefix: str):
        team_n = pos.shape[1]
        for j in range(team_n):
            m = valid[:, j]
            if not np.any(m):
                continue
            xs = pos[m, j, 0]
            ys = pos[m, j, 1]
            zs = pos[m, j, 2]
            ax.plot(xs, ys, zs, color=color, alpha=0.7, linewidth=1.5, label=f"{label_prefix}{j}")

    _plot_team(arrays.red_pos_ft, arrays.red_valid_mask, color="tab:red", label_prefix="red_")
    _plot_team(arrays.blue_pos_ft, arrays.blue_valid_mask, color="tab:blue", label_prefix="blue_")

    tp = arrays.target_point_ft
    ax.scatter(tp[0], tp[1], tp[2], color="black", marker="x", s=80, label="target")

    ax.set_xlabel("North (ft)")
    ax.set_ylabel("East (ft)")
    ax.set_zlabel("Altitude (ft)")
    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # 注意：按你的需求，这里不自动退出/不自动 close。
    # 如果你仍希望保存图片，再手动调用 fig.savefig(...)
    if title is not None:
        fig.canvas.manager.set_window_title(title)

    fig.tight_layout()
    plt.show()
