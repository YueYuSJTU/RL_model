import math
from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Type

import gymnasium as gym
import numpy as np

import jsbgym_m.properties as prp
from jsbgym_m.simulation import Simulation
from jsbgym_m.properties import Property
from jsbgym_m.tasks import Shaping, FlightTask


@dataclass(frozen=True)
class TeamSpec:
    n_red: int
    n_blue: int
    max_red: int = 5
    max_blue: int = 5


class AttackDefendPointTask(FlightTask):
    """Multi-aircraft task (team vs team) oriented around attacking/defending a target point.

    This task is designed for demo / deployment-style rollouts, not training throughput.
    It provides per-aircraft low-level actions (aileron/elevator/rudder/throttle) and
    padded observations with masks.

    Notes:
    - The environment is responsible for maintaining multiple Simulation instances and
      calling the task with all sims each step.
    - This task intentionally does NOT inherit from TrackingTask to avoid 1v1 assumptions.
    """

    # per-aircraft low-level actions (same as TrackingTask)
    action_variables = (
        prp.aileron_cmd,
        prp.elevator_cmd,
        prp.rudder_cmd,
        prp.throttle_cmd,
    )

    DEFAULT_EPISODE_TIME_S = 120.0

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft,
        *,
        team_spec: TeamSpec,
        target_point_ft: Tuple[float, float, float] = (8000.0, 0.0, 10000.0),
        target_radius_ft: float = 1500.0,
        hold_time_s: float = 0.0,
        positive_rewards: bool = True,
        obs_config: Optional[dict] = None,
    ):
        # Keep parameters
        self.step_frequency_hz = float(step_frequency_hz)
        self.aircraft = aircraft
        self.team_spec = team_spec
        self.positive_rewards = positive_rewards

        self.target_point_ft = np.array(target_point_ft, dtype=np.float64)
        self.target_radius_ft = float(target_radius_ft)
        self.hold_time_s = float(hold_time_s)

        # episode clock
        self.max_time_s = float(self.DEFAULT_EPISODE_TIME_S)
        self._t = 0.0
        self._hold_elapsed = 0.0

        # Define per-aircraft state variables (minimal, stable). We will expose only base + actions.
        self.state_variables = (
            *FlightTask.base_state_variables,
            *self.action_variables,
        )
        self._make_state_class()

        # We do not reuse assessor/reward shaping here; for demo we keep reward simple.
        # Still satisfy FlightTask interface by providing a stub assessor-like storage.
        from jsbgym_m import assessors
        from jsbgym_m import rewards

        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=prp.altitude_sl_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=0.1,
            ),
            # add an airspeed error relative to cruise speed component?
        )

        assessor = assessors.AssessorImpl(base_components=base_components, positive_rewards=True)
        super().__init__(assessor=assessor)

        # masks are handled at env level; task provides per-sim state extraction helper

    # --- Gym space helpers (padded, Dict) ---
    def get_action_space(self) -> gym.Space:
        act_lows = np.array([p.min for p in self.action_variables], dtype=np.float64)
        act_highs = np.array([p.max for p in self.action_variables], dtype=np.float64)

        red_low = np.tile(act_lows[None, :], (self.team_spec.max_red, 1))
        red_high = np.tile(act_highs[None, :], (self.team_spec.max_red, 1))
        blue_low = np.tile(act_lows[None, :], (self.team_spec.max_blue, 1))
        blue_high = np.tile(act_highs[None, :], (self.team_spec.max_blue, 1))

        red = gym.spaces.Box(low=red_low, high=red_high, dtype=np.float64)
        blue = gym.spaces.Box(low=blue_low, high=blue_high, dtype=np.float64)
        return gym.spaces.Dict({"red": red, "blue": blue})

    def get_state_space(self) -> gym.Space:
        # normalize to [-1,1] by bounding mins/maxs from BoundedProperty
        lows = np.array([p.min for p in self.state_variables], dtype=np.float64)
        highs = np.array([p.max for p in self.state_variables], dtype=np.float64)

        red = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.team_spec.max_red, len(self.state_variables)), dtype=np.float64)
        blue = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.team_spec.max_blue, len(self.state_variables)), dtype=np.float64)
        mask_red = gym.spaces.Box(low=0.0, high=1.0, shape=(self.team_spec.max_red,), dtype=np.float64)
        mask_blue = gym.spaces.Box(low=0.0, high=1.0, shape=(self.team_spec.max_blue,), dtype=np.float64)

        # Include target point (for controllers) and simple progress signals
        target = gym.spaces.Box(low=-1e9, high=1e9, shape=(3,), dtype=np.float64)
        return gym.spaces.Dict(
            {
                "red": red,
                "blue": blue,
                "red_mask": mask_red,
                "blue_mask": mask_blue,
                "target_point_ft": target,
            }
        )

    # --- Initial conditions ---
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        # Env will provide per-aircraft ICs directly; task doesn't own ICs in multi-sim mode.
        return None

    # --- Episode lifecycle ---
    def reset_episode_clock(self):
        self._t = 0.0
        self._hold_elapsed = 0.0

    def extract_state_vector(self, sim: Simulation) -> np.ndarray:
        raw = np.array([sim[p] for p in self.state_variables], dtype=np.float64)
        return self._normalize_observation(raw)

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        mins = np.array([p.min for p in self.state_variables], dtype=np.float64)
        maxs = np.array([p.max for p in self.state_variables], dtype=np.float64)
        finite_max = 1e4
        mins = np.where(np.isneginf(mins), -finite_max, mins)
        maxs = np.where(np.isposinf(maxs), finite_max, maxs)
        ranges = np.where((maxs - mins) > 1e-10, (maxs - mins), 1e-10)
        norm = 2.0 * (obs - mins) / ranges - 1.0
        return np.clip(norm, -1.0, 1.0)

    # --- Terminal / scoring ---
    def step_clock(self, dt_s: float):
        self._t += float(dt_s)

    def compute_progress(self, red_positions_ft: np.ndarray) -> Dict[str, float]:
        # red_positions_ft: (n_red, 3)
        if red_positions_ft.size == 0:
            min_dist = float("inf")
        else:
            dists = np.linalg.norm(red_positions_ft - self.target_point_ft[None, :], axis=1)
            min_dist = float(np.min(dists))

        in_radius = 1.0 if min_dist <= self.target_radius_ft else 0.0
        if in_radius > 0.5:
            self._hold_elapsed += 1.0 / self.step_frequency_hz
        else:
            self._hold_elapsed = 0.0

        return {
            "min_red_dist_to_target_ft": float(min_dist),
            "hold_elapsed_s": float(self._hold_elapsed),
            "time_s": float(self._t),
        }

    def is_success(self) -> bool:
        if self.hold_time_s <= 0.0:
            return self._hold_elapsed > 0.0
        return self._hold_elapsed >= self.hold_time_s

    def is_time_up(self) -> bool:
        return self._t >= self.max_time_s

    # --- Required by FlightTask but unused in multi-sim mode ---
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int):
        raise NotImplementedError("Use Multi-team env; this task is not single-sim step-driven.")

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        raise NotImplementedError("Use Multi-team env; this task is not single-sim reset-driven.")

    def _is_terminal(self, sim: Simulation) -> bool:
        raise NotImplementedError

    def _reward_terminal_override(self, reward, sim: Simulation):
        return reward
