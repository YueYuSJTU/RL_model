import gymnasium as gym
import numpy as np
import random
import types
import math
import enum
import warnings
from pyquaternion import Quaternion
from collections import namedtuple
import jsbgym_m.properties as prp
from jsbgym_m import assessors, rewards, utils
from jsbgym_m.simulation import Simulation
from jsbgym_m.properties import BoundedProperty, Property
from jsbgym_m.aircraft import Aircraft
from jsbgym_m.rewards import RewardStub, Reward
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type
from jsbgym_m.tasks import HeadingControlTask, Shaping, FlightTask
from jsbgym_m.task_advanced import TrajectoryTask
from jsbgym_m.coordinate import GPS_utils, GPS_NED

class FlyTask(FlightTask):
    """
    
    """
    DEFAULT_EPISODE_TIME_S = 30.0
    THROTTLE_CMD = 0.4
    MIXTURE_CMD = 0.5
    INITIAL_HEADING_DEG = 0.0
    ALTITUDE_SCALING_FT = 5
    PQR_SCALING_RADPS = 0.5

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        # self.extra_state_variables = (
        #     self.altitude_error_ft,
        #     self.track_error_deg,
        #     prp.sideslip_deg,
        # )
        self.state_variables = (
            FlightTask.base_state_variables 
            # + self.extra_state_variables
        )
        self.action_variables = (
            prp.aileron_cmd,
            prp.elevator_cmd,
            prp.rudder_cmd,
            prp.throttle_cmd,
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="no_p",
                prop=prp.p_radps,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.PQR_SCALING_RADPS,
            ),
            rewards.AsymptoticErrorComponent(
                name="no_q",
                prop=prp.q_radps,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.PQR_SCALING_RADPS,
            ),
            rewards.AsymptoticErrorComponent(
                name="no_r",
                prop=prp.r_radps,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.PQR_SCALING_RADPS,
            ),
            # add an airspeed error relative to cruise speed component?
        )
        return base_components

    def _select_assessor(
        self,
        base_components: Tuple[rewards.RewardComponent, ...],
        shaping_components: Tuple[rewards.RewardComponent, ...],
        shaping: Shaping,
    ) -> assessors.AssessorImpl:
        shaping_components = (
            rewards.SmoothingComponent(
                name="smooth altitude",
                props=[prp.altitude_sl_ft],
                state_variables=self.state_variables,
                scaling_factor=self.ALTITUDE_SCALING_FT,
                is_potential_based=True,
            ),
            # rewards.SmoothingComponent(
            #     name="action_penalty",
            #     props=[prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd],
            #     state_variables=self.action_variables,
            #     is_potential_based=True,
            #     cmp_scale=1.0,
            # )
        )
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            raise NotImplementedError(
                f"Shaping {shaping} not implemented for FlyTask"
            )
        
    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
        }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._decrement_steps_left(sim)
        self._debug_print(sim)
    
    def _debug_print(self, sim: Simulation) -> None:
        print()
        print(f"action: {sim[prp.aileron_cmd]:.4f}, {sim[prp.elevator_cmd]:.4f}, {sim[prp.rudder_cmd]:.4f}, {sim[prp.throttle_cmd]:.4f}")
        print(f"trueac: {sim[prp.aileron_left]:.4f}, {sim[prp.elevator]:.4f}, {sim[prp.rudder]:.4f}, {sim[prp.throttle_Aug]:.4f}")
        print(f"pqr: {sim[prp.p_radps]:.4f}, {sim[prp.q_radps]:.4f}, {sim[prp.r_radps]:.4f}")

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        # state_quality = sim[self.last_assessment_reward]
        # # TODO: issues if sequential?
        # state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY
        return terminal_step
    
    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation
    ) -> rewards.Reward:
        # if self._altitude_out_of_bounds(sim) and not self.positive_rewards:
        #     # if using negative rewards, need to give a big negative reward on terminal
        #     return self._get_out_of_bounds_reward(sim)
        # else:
            return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            prp.roll_rad,
            prp.sideslip_deg,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )