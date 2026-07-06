from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import jsbgym_m.properties as prp
from jsbgym_m.aircraft import Aircraft, f16
from jsbgym_m.coordinate import GPS_NED
from jsbgym_m.simulation import Simulation
from jsbgym_m.task_attack_defend_point import AttackDefendPointTask, TeamSpec
from jsbgym_m.tasks import Shaping


class MultiTeamJsbSimEnv(gym.Env):
    """Multi-aircraft (N vs M) demo environment.

    - Teams: red (attack) and blue (defend), each 1..5 aircraft.
    - Actions: Dict with padded per-aircraft low-level commands.
    - Observations: Dict with padded per-aircraft normalized state vectors and masks.

    This env is intended for rollout / visualization / deployment demos. It is not
    wired into the SB3 training pipeline by default.
    """

    JSBSIM_DT_HZ: int = 60
    metadata = {
        "render_modes": ["human", "anim3d"],
        "render_fps": 60,
    }

    def __init__(
        self,
        aircraft: Aircraft = f16,
        opponent_aircraft: Aircraft = f16,
        *,
        n_red: int = 5,
        n_blue: int = 5,
        max_red: int = 5,
        max_blue: int = 5,
        agent_interaction_freq: int = 10,
        render_mode: Optional[str] = None,
        target_point_ft: Tuple[float, float, float] = (8000.0, 0.0, 10000.0),
        target_radius_ft: float = 1500.0,
        hold_time_s: float = 0.0,
        episode_time_s: float = 120.0,
    ):
        if not (1 <= n_red <= max_red):
            raise ValueError("n_red must be within [1,max_red]")
        if not (1 <= n_blue <= max_blue):
            raise ValueError("n_blue must be within [1,max_blue]")
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError("agent_interaction_freq must be <= JSBSIM_DT_HZ")

        self.render_mode = render_mode
        self.aircraft = aircraft
        self.opponent_aircraft = opponent_aircraft

        self.sim_steps_per_agent_step = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.step_frequency_hz = float(agent_interaction_freq)

        self.team_spec = TeamSpec(n_red=n_red, n_blue=n_blue, max_red=max_red, max_blue=max_blue)

        self.task = AttackDefendPointTask(
            shaping_type=Shaping.STANDARD,
            step_frequency_hz=self.step_frequency_hz,
            aircraft=aircraft,
            team_spec=self.team_spec,
            target_point_ft=target_point_ft,
            target_radius_ft=target_radius_ft,
            hold_time_s=hold_time_s,
            positive_rewards=True,
        )
        self.task.max_time_s = float(episode_time_s)

        self.observation_space = self.task.get_state_space()
        self.action_space = self.task.get_action_space()

        self.red_sims: List[Simulation] = []
        self.blue_sims: List[Simulation] = []

        self._ned = GPS_NED(unit="ft")
        self._viewer = None  # lazy

    # ---- sim helpers ----
    def _init_new_sim(self, dt_hz: int, aircraft: Aircraft, init_conditions: Dict):
        return Simulation(sim_frequency_hz=dt_hz, aircraft=aircraft, init_conditions=init_conditions)

    def _default_ic(self, heading_deg: float, offset_north_ft: float, offset_east_ft: float, altitude_ft: float = 10000.0):
        # Use base ICs in FlightTask and adjust lat/lon approximately; good enough for local demo.
        lat0 = 51.3781
        lon0 = -2.3273
        north_m = offset_north_ft * 0.3048
        east_m = offset_east_ft * 0.3048
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * max(1e-6, math.cos(math.radians(lat0)))
        dlat = north_m / meters_per_deg_lat
        dlon = east_m / meters_per_deg_lon

        return {
            prp.initial_altitude_ft: float(altitude_ft),
            prp.initial_terrain_altitude_ft: 0.00000001,
            prp.initial_longitude_geoc_deg: float(lon0 + dlon),
            prp.initial_latitude_geod_deg: float(lat0 + dlat),
            prp.initial_u_fps: float(self.aircraft.get_cruise_speed_fps()),
            prp.initial_v_fps: 0.0,
            prp.initial_w_fps: 0.0,
            prp.initial_p_radps: 0.0,
            prp.initial_q_radps: 0.0,
            prp.initial_r_radps: 0.0,
            prp.initial_roc_fpm: 0.0,
            prp.initial_heading_deg: float(heading_deg),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Recreate sims each episode (demo-friendly, simpler than reinitialise many sockets)
        for s in self.red_sims + self.blue_sims:
            s.close()
        self.red_sims = []
        self.blue_sims = []

        # Simple spawn: red west of origin heading east; blue east heading west
        for i in range(self.team_spec.n_red):
            ic = self._default_ic(heading_deg=90.0, offset_north_ft=i * 800.0, offset_east_ft=-8000.0, altitude_ft=10000.0)
            self.red_sims.append(self._init_new_sim(self.JSBSIM_DT_HZ, self.aircraft, ic))
        for i in range(self.team_spec.n_blue):
            ic = self._default_ic(heading_deg=270.0, offset_north_ft=i * 800.0, offset_east_ft=8000.0, altitude_ft=10000.0)
            self.blue_sims.append(self._init_new_sim(self.JSBSIM_DT_HZ, self.opponent_aircraft, ic))

        # Set NED origin from the first red aircraft (same pattern as TrackingTask)
        ecef0 = [
            float(self.red_sims[0][prp.ecef_x_ft]),
            float(self.red_sims[0][prp.ecef_y_ft]),
            float(self.red_sims[0][prp.ecef_z_ft]),
        ]
        lla0 = self._ned.ecef2geo(*ecef0)
        self._ned.setNEDorigin(*lla0)

        self.task.reset_episode_clock()

        obs = self._build_obs()
        info = {"task": self._task_info()}
        return obs, info

    def _build_obs(self):
        red = np.zeros((self.team_spec.max_red, len(self.task.state_variables)), dtype=np.float64)
        blue = np.zeros((self.team_spec.max_blue, len(self.task.state_variables)), dtype=np.float64)
        red_mask = np.zeros((self.team_spec.max_red,), dtype=np.float64)
        blue_mask = np.zeros((self.team_spec.max_blue,), dtype=np.float64)

        for i, sim in enumerate(self.red_sims):
            red[i] = self.task.extract_state_vector(sim)
            red_mask[i] = 1.0
        for i, sim in enumerate(self.blue_sims):
            blue[i] = self.task.extract_state_vector(sim)
            blue_mask[i] = 1.0

        return {
            "red": red,
            "blue": blue,
            "red_mask": red_mask,
            "blue_mask": blue_mask,
            "target_point_ft": self.task.target_point_ft.copy(),
        }

    def _task_info(self):
        red_pos = self._get_positions_ft(self.red_sims)
        progress = self.task.compute_progress(red_pos)
        return {
            **progress,
            "success": bool(self.task.is_success()),
            "time_up": bool(self.task.is_time_up()),
            "n_red": int(self.team_spec.n_red),
            "n_blue": int(self.team_spec.n_blue),
        }

    def _get_positions_ft(self, sims: List[Simulation]) -> np.ndarray:
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for sim in sims:
            ned = self._ned.ecef2ned(
                float(sim[prp.ecef_x_ft]),
                float(sim[prp.ecef_y_ft]),
                float(sim[prp.ecef_z_ft]),
            )
            xs.append(float(ned[0]))
            ys.append(float(ned[1]))
            zs.append(float(sim[prp.altitude_sl_ft]))
        return np.stack([xs, ys, zs], axis=1).astype(np.float64)

    def step(self, action: Dict[str, np.ndarray]):
        # Apply actions to alive sims
        red_act = np.asarray(action["red"], dtype=np.float64)
        blue_act = np.asarray(action["blue"], dtype=np.float64)

        for i, sim in enumerate(self.red_sims):
            a = red_act[i]
            for prop, cmd in zip(self.task.action_variables, a):
                sim[prop] = float(cmd)
        for i, sim in enumerate(self.blue_sims):
            a = blue_act[i]
            for prop, cmd in zip(self.task.action_variables, a):
                sim[prop] = float(cmd)

        for _ in range(self.sim_steps_per_agent_step):
            for sim in self.red_sims:
                sim.run()
            for sim in self.blue_sims:
                sim.run()

        self.task.step_clock(1.0 / self.step_frequency_hz)

        obs = self._build_obs()
        task_info = self._task_info()
        info = {"task": task_info}

        # scene-level trajectory payload for offline visualization
        # positions are in local NED frame (ft): [north, east, altitude_sl_ft]
        info["scene"] = {
            "t": float(task_info.get("time_s", 0.0)),
            "red_pos_ft": self._get_positions_ft(self.red_sims),
            "blue_pos_ft": self._get_positions_ft(self.blue_sims),
            "target_point_ft": self.task.target_point_ft.copy(),
            "red_mask": obs["red_mask"].copy(),
            "blue_mask": obs["blue_mask"].copy(),
        }

        success = self.task.is_success()
        terminated = bool(success)
        truncated = bool(self.task.is_time_up() and not success)

        # Simple demo reward: encourage red towards target, discourage time
        min_dist = info["task"]["min_red_dist_to_target_ft"]
        reward = -0.0001 * float(min_dist)
        if success:
            reward += 100.0

        if self.render_mode in ("human", "anim3d"):
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        # TODO: extend Enhanced3DVisualiser to multi-sim; for now, render first red vs first blue if present
        if self.render_mode is None:
            return
        if not self.red_sims:
            return

        from jsbgym_m.enhanced_visualiser import Enhanced3DVisualiser

        if self._viewer is None:
            self._viewer = Enhanced3DVisualiser(self.red_sims[0], tuple())
        opp = self.blue_sims[0] if self.blue_sims else None
        self._viewer.plot(self.red_sims[0], opp)

    def close(self):
        for s in self.red_sims + self.blue_sims:
            s.close()
        self.red_sims = []
        self.blue_sims = []
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
