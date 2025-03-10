import gymnasium as gym
import numpy as np
import random
import types
import math
import enum
import warnings
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
from jsbgym_m.coordinate import GPS_utils

class SmoothHeadingTask(HeadingControlTask):
    """
    SmoothHeadingTask is a task designed to control the heading of an aircraft smoothly.
    It extends the HeadingControlTask and includes additional state variables and reward components.
    Attributes:
        ACTION_PENALTY_SCALING (int): Scaling factor for action penalty.
    Args:
        shaping_type (Shaping): The type of shaping used for the task.
        step_frequency_hz (float): The number of agent interaction steps per second.
        aircraft (Aircraft): The aircraft used in the simulation.
        episode_time_s (float, optional): The duration of an episode in seconds. Defaults to HeadingControlTask.DEFAULT_EPISODE_TIME_S.
        positive_rewards (bool, optional): Whether to use positive rewards. Defaults to True.
    Methods:
        _make_base_reward_components() -> Tuple[rewards.RewardComponent, ...]:
            Creates the base reward components for the task.
    Attention:
        Do NOT Support EXTRA_SEQUENTIAL mode
    """

    ACTION_PENALTY_SCALING = 50000

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = HeadingControlTask.DEFAULT_EPISODE_TIME_S, 
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (
            self.altitude_error_ft,
            self.track_error_deg,
            prp.sideslip_deg,
        )
        self.state_variables = (
            FlightTask.base_state_variables 
            + self.extra_state_variables 
            + HeadingControlTask.action_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super(HeadingControlTask, self).__init__(assessor)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
            ),
            rewards.AsymptoticErrorComponent(
                name="travel_direction",
                prop=self.track_error_deg,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.TRACK_ERROR_SCALING_DEG,
            ),
            rewards.AsymptoticErrorComponent(
                name="action_penalty",
                prop=prp.elevator_cmd,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ACTION_PENALTY_SCALING,
            ),
        )
        return base_components
    
class TurnHeadingControlTask(SmoothHeadingTask):
    """
    TurnHeadingControlTask inherited from SmoothHeadingTask
    """

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _get_target_track(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_track_deg.min, self.target_track_deg.max)
    

class TrajectoryTask(FlightTask):
    """
    control the trajectory of an aircraft.
    """

    TARGET_xPOSITION_FT = 3000
    TARGET_yPOSITION_FT = 5000
    TARGET_zPOSITION_FT = 200
    THROTTLE_CMD = 0.7
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 90.0
    ALTITUDE_SCALING_FT = 100
    X_POSITION_SCALING_MT = 2500
    Y_POSITION_SCALING_MT = 4000
    ACTION_PENALTY_SCALING = 0.1
    ROLL_ERROR_SCALING_RAD = 0.15  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.0
    VERTICAL_SPEED_SCALING_FPS = 1
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 3000  # terminate if altitude error exceeds this
    NAVIGATION_TOLERANCE = 500       # terminate if relative error is less than this
    enu_Xposition_ft = BoundedProperty(
        "position/positionX-ft",
        "current track [ft]",
        -10000,
        10000,
    )
    enu_Yposition_ft = BoundedProperty(
        "position/positionY-ft",
        "current track [ft]",
        -10000,
        10000,
    )
    enu_Zposition_ft = BoundedProperty(
        "position/positionZ-ft",
        "current altitude [ft]",
        -10000,
        10000,
    )
    target_Xposition = BoundedProperty(
        "target/positionX-ft",
        "desired track [ft]",
        -10000,
        10000,
    )
    target_Yposition = BoundedProperty(
        "target/positionY-ft",
        "desired track [ft]",
        -10000,
        10000,
    )
    target_Altitude = BoundedProperty(
        "target/altitude-ft",
        "desired altitude [ft]",
        prp.altitude_sl_ft.min,
        prp.altitude_sl_ft.max,
    )
    # position_error_ft = BoundedProperty(
    #     "error/position-error-ft",
    #     "error to desired track [ft]",
    #     0,
    #     20000,
    # )

    x_error_ft = BoundedProperty(
        "error/x-error-ft",
        "error to desired x-position [ft]",
        -20000,
        20000,
    )
    y_error_ft = BoundedProperty(
        "error/y-error-ft",
        "error to desired y-position [ft]",
        -20000,
        20000,
    )
    altitude_error_ft = BoundedProperty(
        "error/altitude-error-ft",
        "error to desired altitude [ft]",
        prp.altitude_sl_ft.min,
        prp.altitude_sl_ft.max,
    )
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.extra_state_variables = (
            self.altitude_error_ft,
            self.x_error_ft,
            self.y_error_ft,
            # prp.sideslip_deg,
            # prp.v_down_fps,
        )
        self.state_variables = (
            FlightTask.base_state_variables + self.action_variables
            + self.extra_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        self.coordinate_transform = GPS_utils(unit='ft')
        self.target_theta = 0
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            # rewards.AsymptoticErrorComponent(
            #     name="altitude_error",
            #     prop=self.altitude_error_ft,
            #     state_variables=self.state_variables,
            #     target=0.0,
            #     is_potential_based=False,
            #     scaling_factor=self.ALTITUDE_SCALING_FT,
            # ),
            # rewards.AsymptoticErrorComponent(
            #     name="position_error",
            #     prop=self.position_error_ft,
            #     state_variables=self.state_variables,
            #     target=0.0,
            #     is_potential_based=False,
            #     scaling_factor=self.POSITION_SCALING_MT,
            # ),
            # # rewards.AsymptoticErrorComponent(
            # #     name="action_penalty",
            # #     prop=prp.elevator_cmd,
            # #     state_variables=self.state_variables,
            # #     target=0.0,
            # #     is_potential_based=False,
            # #     scaling_factor=self.ACTION_PENALTY_SCALING,
            # # ),
            # # rewards.AsymptoticErrorComponent(
            # #     name="vertival_speed",
            # #     prop=prp.v_down_fps,
            # #     state_variables=self.state_variables,
            # #     target=0.0,
            # #     is_potential_based=False,
            # #     scaling_factor=self.VERTICAL_SPEED_SCALING_FPS,
            # # ),
            rewards.ScaledAsymptoticErrorComponent(
                name="altitude_error",
                prop=self.altitude_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.ALTITUDE_SCALING_FT,
                cmp_scale=0.3,
            ),
            rewards.ScaledAsymptoticErrorComponent(
                name="x_position_error",
                prop=self.x_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.X_POSITION_SCALING_MT,
                cmp_scale=0.3,
            ),
            rewards.ScaledAsymptoticErrorComponent(
                name="y_position_error",
                prop=self.y_error_ft,
                state_variables=self.state_variables,
                target=0.0,
                is_potential_based=False,
                scaling_factor=self.Y_POSITION_SCALING_MT,
                cmp_scale=0.3,
            ),
            # rewards.ScaledAsymptoticErrorComponent(
            #     name="p_reward",
            #     prop=prp.p_radps,
            #     state_variables=self.state_variables,
            #     target=0.0,
            #     is_potential_based=False,
            #     scaling_factor=self.ACTION_PENALTY_SCALING,
            #     cmp_scale=0.2,
            # ),
        )
        return base_components
    
    def _select_assessor(
        self,
        base_components: Tuple[rewards.RewardComponent, ...],
        shaping_components: Tuple[rewards.RewardComponent, ...],
        shaping: Shaping,
    ) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        # else:
        #     raise ValueError(f"Unsupported shaping type: {shaping}")
        else:
            # wings_level = rewards.AsymptoticErrorComponent(
            #     name="wings_level",
            #     prop=prp.roll_rad,
            #     state_variables=self.state_variables,
            #     target=0.0,
            #     is_potential_based=True,
            #     scaling_factor=self.ROLL_ERROR_SCALING_RAD,
            # )
            # no_sideslip = rewards.AsymptoticErrorComponent(
            #     name="no_sideslip",
            #     prop=prp.sideslip_deg,
            #     state_variables=self.state_variables,
            #     target=0.0,
            #     is_potential_based=True,
            #     scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG,
            # )
            action_penalty = rewards.SmoothingComponent(
                name="action_penalty",
                props=[prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd],
                state_variables=self.state_variables,
                is_potential_based=True,
                cmp_scale=0.3,
            )
            # potential_based_components = (wings_level, no_sideslip)
            potential_based_components = (action_penalty,)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(
                base_components,
                potential_based_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            raise ValueError(f"Unsupported shaping type: {shaping}")
        # elif shaping is Shaping.EXTRA_SEQUENTIAL:
        #     altitude_error, travel_direction = base_components
        #     # make the wings_level shaping reward dependent on facing the correct direction
        #     dependency_map = {wings_level: (travel_direction,)}
        #     return assessors.ContinuousSequentialAssessor(
        #         base_components,
        #         potential_based_components,
        #         potential_dependency_map=dependency_map,
        #         positive_rewards=self.positive_rewards,
        #     )
    
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
        self._update_enu_position(sim)
        self._update_position_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)
        # self._coordinate_debug(sim)
    
    def _coordinate_debug(self, sim: Simulation) -> None:
        x, y, z = sim[prp.ecef_x_ft], sim[prp.ecef_y_ft], sim[prp.ecef_z_ft]
        alt_geod = sim[prp.altitude_geod_ft]
        lat, lon, alt = sim[prp.lat_geod_deg], sim[prp.lng_geoc_deg], sim[prp.altitude_sl_ft]
        x_enu, y_enu = sim[prp.dist_travel_lon_m]/0.3048, sim[prp.dist_travel_lat_m]/0.3048

        geo2ecef = self.coordinate_transform.geo2ecef(lat, lon, alt_geod)
        ecef2enu = self.coordinate_transform.ecef2enu(x, y, z)
        ecef2geo = self.coordinate_transform.ecef2geo(x, y, z)
        geo2enu = self.coordinate_transform.geo2enu(lat, lon, alt_geod)

        print("====================DEBUG:COORDINATE====================")
        print("x, y, z:", x, y, z)
        print("alt_geod:", alt_geod)
        print("lat, lon, alt:", lat, lon, alt)
        print("x_enu, y_enu:", x_enu, y_enu)

        print("geo2ecef:", geo2ecef)
        print("ecef2enu:", ecef2enu)
        print("ecef2geo:", ecef2geo)
        print("geo2enu:", geo2enu)

    def _update_enu_position(self, sim: Simulation):
        x, y, z = sim[prp.ecef_x_ft], sim[prp.ecef_y_ft], sim[prp.ecef_z_ft]
        x_enu, y_enu, z_enu = self.coordinate_transform.ecef2enu(x, y, z)
        sim[self.enu_Xposition_ft] = x_enu
        sim[self.enu_Yposition_ft] = y_enu
        sim[self.enu_Zposition_ft] = z_enu
    
    def get_enu_position(self, sim: Simulation) -> list[float, float, float]:
        return sim[self.enu_Xposition_ft], sim[self.enu_Yposition_ft], sim[self.enu_Zposition_ft]

    def _update_position_error(self, sim: Simulation):
        position = prp.Vector2(sim[self.enu_Xposition_ft], sim[self.enu_Yposition_ft])
        target_position = prp.Vector2(sim[self.target_Xposition], sim[self.target_Yposition])
        sim[self.x_error_ft] = position.get_x() - target_position.get_x()
        sim[self.y_error_ft] = position.get_y() - target_position.get_y()

    def _update_altitude_error(self, sim: Simulation):
        z_position = sim[self.enu_Zposition_ft]
        target_z_ft = self._get_target_position("z")
        error_ft = z_position - target_z_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        # TODO: issues if sequential?
        return terminal_step or self._state_out_of_bounds(sim) or self._arrive_at_navigation_point(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT
    
    def _state_out_of_bounds(self, sim: Simulation) -> bool:
        state_out_of_bounds = sim[self.last_agent_reward] < self.MIN_STATE_QUALITY
        if not self.positive_rewards:
            state_out_of_bounds = False
        return state_out_of_bounds or self._altitude_out_of_bounds(sim)
    
    def _arrive_at_navigation_point(self, sim: Simulation) -> bool:
        x_arrive = abs(sim[self.x_error_ft]) < self.NAVIGATION_TOLERANCE
        y_arrive = abs(sim[self.y_error_ft]) < self.NAVIGATION_TOLERANCE
        return x_arrive and y_arrive

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[self.steps_left]) * -1.0
        return RewardStub(reward_scalar, reward_scalar)
    
    def _arrive_at_navigation_point_reward(self, sim: Simulation) -> rewards.Reward:
        bonus = 1 + sim[self.steps_left]
        return RewardStub(bonus, bonus)

    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation
    ) -> rewards.Reward:
        if self._state_out_of_bounds(sim) and not self.positive_rewards:
            # if using negative rewards, need to give a big negative reward on terminal
            return self._get_out_of_bounds_reward(sim)
        else:
            if self._arrive_at_navigation_point(sim):
                return self._arrive_at_navigation_point_reward(sim)
            else:
                return reward
    
    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        self.init_ecef_position = [sim[prp.ecef_x_ft], 
                                   sim[prp.ecef_y_ft], 
                                   sim[prp.ecef_z_ft]]
        lla_position = self.coordinate_transform.ecef2geo(*self.init_ecef_position)
        self.coordinate_transform.setENUorigin(*lla_position)
        # self._random_target_position()
        # self._circle_target_position()
        self._line_target_position()
        sim[self.target_Xposition] = self._get_target_position("x")
        sim[self.target_Yposition] = self._get_target_position("y")

    def _random_target_position(self) -> None:
        self.TARGET_xPOSITION_FT = np.random.uniform(-10000, -9000) if random.random() < 0.5 else np.random.uniform(9000, 10000)
        self.TARGET_yPOSITION_FT = np.random.uniform(-10000, -9000) if random.random() < 0.5 else np.random.uniform(9000, 10000)
        self.TARGET_zPOSITION_FT = np.random.uniform(-1000, 1000)

    def _circle_target_position(self) -> None:
        self.target_theta += 0.01
        self.TARGET_xPOSITION_FT = 5000 * math.cos(self.target_theta)
        self.TARGET_yPOSITION_FT = 5000 * math.sin(self.target_theta)
        self.TARGET_zPOSITION_FT = 200

    def _line_target_position(self) -> None:
        self.TARGET_xPOSITION_FT = np.random.uniform(-5000, 5000)
        self.TARGET_yPOSITION_FT = 9000
        self.TARGET_zPOSITION_FT = 200

    def _get_target_position(self, flag: str) -> float:
        # use the same, initial heading every episode
        if flag == "x":
            return self.TARGET_xPOSITION_FT
        elif flag == "y":
            return self.TARGET_yPOSITION_FT
        elif flag == "z":
            return self.TARGET_zPOSITION_FT
        else:
            raise ValueError(f"Unsupported flag in get target position: {flag}")

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            self.target_Xposition,
            self.target_Yposition,
            self.x_error_ft,
            self.y_error_ft,
            self.altitude_error_ft,
            # prp.roll_rad,
            # prp.sideslip_deg,
            self.last_agent_reward,
            self.last_assessment_reward,
            self.steps_left,
        )