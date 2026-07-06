import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline scene visualizer")
    p.add_argument(
        "--env_id",
        type=str,
        default="F16-AttackDefendPointTask-Demo-NoFG-v0",
        help="Gymnasium env id",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: experiments/scene_viz/<timestamp>",
    )
    p.add_argument("--seed", type=int, default=None, help="Optional env seed")
    p.add_argument("--max_steps", type=int, default=600, help="Safety cap on steps")
    return p.parse_args()


def _default_out_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / "scene_viz" / ts


def _collect_scene_frames(env, max_steps: int) -> List[Dict[str, Any]]:
    scene_frames: List[Dict[str, Any]] = []

    obs, info = env.reset()
    if isinstance(info, dict) and info.get("scene") is not None:
        scene_frames.append(info["scene"])

    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if isinstance(info, dict) and info.get("scene") is not None:
            scene_frames.append(info["scene"])
        steps += 1

    return scene_frames


def main() -> None:
    args = _parse_args()

    import gymnasium as gym
    import jsb_env.jsbgym_m  # noqa: F401

    out_dir = Path(args.out_dir) if args.out_dir is not None else _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode=None)
    try:
        if args.seed is not None:
            env.reset(seed=args.seed)
        scene_frames = _collect_scene_frames(env, max_steps=args.max_steps)
    finally:
        env.close()

    if not scene_frames:
        raise RuntimeError("No scene frames collected. Does this env populate info['scene']?")

    from src.scene_viz.plot_scene_3d_trajectory import dump_scene_json, plot_scene_3d_trajectory

    # dump_scene_json(scene_frames, out_dir / "scene.json")
    plot_scene_3d_trajectory(scene_frames, title='F16-AttackDefendPointTask-Demo-NoFG-v0')

    print(f"Saved JSON to: {out_dir / 'scene.json'}")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
