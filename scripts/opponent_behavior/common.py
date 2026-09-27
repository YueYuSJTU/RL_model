"""Shared constants and helpers for the benchmark-opponent behaviour study.

Scope
-----
This package measures the EIGHT benchmark opponents themselves.  No external
trained model is used as a measuring stick; every reported number is a quantity
of the opponent's own flight.  See README.md.

Read-only guarantee
-------------------
Nothing here writes to jsb_env/**, src/environments/**, src/agents/**,
src/evaluation/** or src/utils/custom_callback.py.  Those trees are imported,
never modified.  Run every script with PYTHONDONTWRITEBYTECODE=1 (the provided
shell drivers do this) so that Python does not drop __pycache__ entries into
them either.  All artifacts go to experiments/opponent_behavior/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from src.agents.make_agent import load_agent  # noqa: E402
from src.agents.model_wrapper import ObsAdaptingModel  # noqa: E402
from src.environments.make_env import create_env  # noqa: E402

G_FT_S2 = 32.174

# --------------------------------------------------------------------------
# One fixed evaluation environment for the whole study.  TrackingTask/stage2 is
# only a container here: the reward is irrelevant because the subject is driven
# in the environment's native goal-point mode and the drone flies a constant
# command.  What matters is that a single, identical configuration is used for
# every opponent, so no behavioural difference can come from the scenario.
# --------------------------------------------------------------------------
EVAL_ENV_CFG: Dict[str, Any] = {
    "plane": "F16",
    "task": "TrackingTask",
    "shape": "stage2",
    "render_mode": None,
    "use_vec_normalize": False,
    "norm_obs": False,
    "norm_reward": False,
    "clip_obs": 10.0,
}

# --------------------------------------------------------------------------
# The eight benchmark opponents.  Provenance verified by md5 of best_model.zip
# and by each folder's train_config.yaml / evaluations.npz (see README.md).
# --------------------------------------------------------------------------
OPPONENTS: Dict[str, Dict[str, Any]] = {
    "1": dict(
        path="opponent_pool/test_pool/1",
        family="protected-init",
        reward_set="GoalPointTask / stage1 (navigation only)",
        train_steps=15_000_000,
        batch_size=128,
        note="goalpoint navigation policy; the pool's protected opponent id 1",
    ),
    "2": dict(
        path="opponent_pool/test_pool/2",
        family="protected-init",
        reward_set="TrackingTask / stage2 (combat)",
        train_steps=40_000_000,
        batch_size=128,
        note="protected opponent id 2",
    ),
    "3": dict(
        path="opponent_pool/test_pool/3",
        family="protected-init",
        reward_set="TrackingTask / stage2 (combat)",
        train_steps=500_000_000,
        batch_size=128,
        note="protected opponent id 3",
    ),
    "11": dict(
        path="opponent_pool/test_pool/11",
        family="level-flight-batch",
        reward_set="TrackingInitTask / stage3 (flight quality only)",
        train_steps=600_000,
        batch_size=256,
        note="same file as opponent_pool/pool2/11",
    ),
    "16": dict(
        path="opponent_pool/test_pool/16",
        family="level-flight-batch",
        reward_set="TrackingInitTask / stage4 (flight quality only)",
        train_steps=600_000,
        batch_size=256,
        note="same file as opponent_pool/pool2/16",
    ),
    "17": dict(
        path="opponent_pool/test_pool/17",
        family="level-flight-batch",
        reward_set="TrackingInitTask / stage5 (flight quality only)",
        train_steps=600_000,
        batch_size=256,
        note="same file as opponent_pool/pool2/17",
    ),
    "23": dict(
        path="opponent_pool/test_pool/23",
        family="level-flight-batch",
        reward_set="TrackingInitTask / stage3 (flight quality only)",
        train_steps=600_000,
        batch_size=256,
        note="same file as opponent_pool/pool2/3 -- NOT the same model as "
             "test_pool/3 (md5 469281b7 vs 3543907a)",
    ),
    "616": dict(
        path="opponent_pool/test_pool/616",
        family="legacy-run",
        reward_set="TrackingTask / stage2 (combat)",
        train_steps=605_000_000,
        batch_size=256,
        note="experiments/20250616_221656/stage2/... ; the model evaluated as the "
             "paper's Baseline",
    ),
}


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def _read_yaml(rel_path: str, name: str) -> Dict[str, Any]:
    import yaml

    with open(REPO / rel_path / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_eval_env(seed: Optional[int] = None):
    """One environment per process with the study's fixed configuration."""
    vec = create_env(dict(EVAL_ENV_CFG), training=False, vec_env_cls=DummyVecEnv)
    vec.training = False
    vec.norm_reward = False
    # start from the non-curriculum setting; goalpoint.py then selects the
    # environment's native goal-point mode explicitly
    vec.env_method("update_task_parameters", goal_point_prob=0.0)
    if seed is not None:
        vec.seed(seed)
    return vec


def load_policy(rel_path: str):
    """Load a trained policy exactly the way Evaluator.run_match does."""
    agent_cfg = _read_yaml(rel_path, "agent_config.yaml")
    env_cfg = _read_yaml(rel_path, "env_config.yaml")
    fake_env = create_env(env_cfg, training=False, vec_env_kwargs=None)
    model = load_agent(
        env=fake_env,
        agent_class=agent_cfg.get("algorithm", "PPO"),
        path=str(REPO / rel_path / "best_model"),
        device=agent_cfg["device"],
        agent_cfg=agent_cfg,
    )
    return ObsAdaptingModel(model, env_cfg)


def _predict(model, obs, state, episode_start):
    try:
        return model.predict(obs, state=state, episode_start=episode_start,
                             deterministic=True)
    except TypeError:
        return model.predict(obs, deterministic=True)[0], None


# --------------------------------------------------------------------------
# Scripted probes retained as measurement records (see README.md)
# --------------------------------------------------------------------------
class OpenLoopLevelProbe:
    """Constant-command probe.

    Kept as EVIDENCE that a fixed command does not trim this F-16 model: with
    zero control surfaces it dives to the ground, and with elevator -0.06 it
    climbs past 30000 ft.  Used by calibrate_level_probe.py; not used as a
    reference opponent.
    """

    def __init__(self, elevator: float = 0.0, throttle: float = 0.4):
        self.action = np.array([0.0, float(elevator), 0.0, float(throttle)],
                               dtype=np.float32)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = int(np.asarray(observation).shape[0]) if np.asarray(observation).ndim > 1 else 1
        return np.tile(self.action, (n, 1)), None


class ClosedLoopLevelProbe:
    """Scripted altitude/speed-hold probe.

    Retained as EVIDENCE of why a scripted 'straight-and-level drone' is not
    used as the reference opponent: measured, it holds ~9900 ft for the first
    ~35 s and then departs (the phugoid is unstable and elevator authority
    large enough to recover drives the model into a deep stall).  The gains are
    calibrated by calibrate_level_probe.py.
    """

    DEFAULT_GAINS = dict(kp_alt=2.0e-5, kd_alt=5.0e-3, kp_spd=2.0e-3,
                         alt_target_ft=10000.0, thr_trim=0.38,
                         elev_trim=-0.038, elev_limit=0.08)

    def __init__(self, gains: Optional[Dict[str, float]] = None,
                 spd_target_fps: float = 0.0, dt: float = 0.1):
        g = dict(self.DEFAULT_GAINS)
        if gains:
            g.update(gains)
        self.kp_alt = float(g["kp_alt"])
        self.kd_alt = float(g["kd_alt"])
        self.kp_spd = float(g["kp_spd"])
        self.alt_target = float(g["alt_target_ft"])
        self.thr_trim = float(g.get("thr_trim", 0.38))
        self.elev_trim = float(g.get("elev_trim", -0.038))
        self.elev_limit = float(g.get("elev_limit", 0.08))
        self.dt = float(dt)
        self.spd_target = float(spd_target_fps)
        self._last_alt: Optional[float] = None
        self.elev_cmd = 0.0
        self.thr_cmd = self.thr_trim

    def observe(self, state: Dict[str, float]) -> None:
        alt = float(state["alt"])
        spd = float(state["spd"])
        if self.spd_target <= 0.0:
            self.spd_target = spd
        roc_fps = 0.0 if self._last_alt is None else (alt - self._last_alt) / self.dt
        self._last_alt = alt
        cmd = self.elev_trim + self.kp_alt * (alt - self.alt_target) + self.kd_alt * roc_fps
        self.elev_cmd = float(np.clip(cmd, -self.elev_limit, self.elev_limit))
        self.thr_cmd = float(np.clip(
            self.thr_trim + self.kp_spd * (self.spd_target - spd), 0.0, 1.0))

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = int(np.asarray(observation).shape[0]) if np.asarray(observation).ndim > 1 else 1
        a = np.array([0.0, self.elev_cmd, 0.0, self.thr_cmd], dtype=np.float32)
        return np.tile(a, (n, 1)), None


class RandomProbe:
    """Zero-learning uniform-random probe (used by measure_hold.py)."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        n = int(np.asarray(observation).shape[0]) if np.asarray(observation).ndim > 1 else 1
        a = self.rng.uniform(-1.0, 1.0, size=(n, 3))
        t = self.rng.uniform(0.0, 1.0, size=(n, 1))
        return np.concatenate([a, t], axis=1).astype(np.float32), None


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
