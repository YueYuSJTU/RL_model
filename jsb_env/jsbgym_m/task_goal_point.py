import gymnasium as gym
import numpy as np
import random
import types
import math
import enum
import os
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
from jsbgym_m.task_tracking import TrackingTask
from jsbgym_m.coordinate import GPS_utils, GPS_NED
from stable_baselines3 import PPO


class GoalPointTask(TrackingTask):
    """
    The plane can fly to a goal point. The point can move or stay still.
    
    It remove the opponent aircraft state variables and related 
    reward components from TrackingTask.
    """

    DEFAULT_EPISODE_TIME_S = 60.0
    INITIAL_HEADING_DEG = 0
    THROTTLE_CMD = 0.4
    MIXTURE_CMD = 0.8
    ARRIVE_RADIUS_FT = 500.0  # within 500 ft is considered as arrived

    def __init__(
        self,
        shaping_type: Shaping,
        step_frequency_hz: float,
        aircraft: Aircraft,
        episode_time_s: float = DEFAULT_EPISODE_TIME_S,
        positive_rewards: bool = True,
        goal_point_mode: str = 'dynamic'  # 'static' or 'dynamic'
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        :param goal_point_mode: The movement mode of the goal point.
        """
        
        self.extra_state_variables = (
            self.distance_oppo_ft,
            self.track_angle_rad,
            self.bearing_accountingRollPitch_rad,
            self.elevation_accountingRollPitch_rad,
            self.bearing_pointMass_rad,
            self.elevation_pointMass_rad,
            # goal point模式下不存在adverse_angle
            # adverse_angle_rad,
            self.closure_rate,
        )
        self.state_variables = (
            FlightTask.base_state_variables
            + self.tracking_state_variables
            + self.extra_state_variables
            # + self.oppo_state_variables
            + self.action_variables
        )
        super().__init__(
            shaping_type=shaping_type,
            step_frequency_hz=step_frequency_hz,
            aircraft=aircraft,
            episode_time_s=episode_time_s,
            positive_rewards=positive_rewards,
        )
        self.opponent = self._create_opponent(model="goal_point")
        self.goal_point_mode = goal_point_mode
        # self.goal_point_position = np.array([5000.0, 5000.0, 5000.0])
        # self.goal_point_velocity = np.array([400.0, 0.0, 0.0])
    
    def _create_opponent(self, model: str = "goal_point"):
        """
        Create the opponent aircraft.
        """
        if model == "goal_point":
            return model
        elif model == "jsbsim":
            return "jsbsim"
        else:
            raise ValueError("Unsupported opponent model: {}".format(model))
    
    def make_assessor(self, shaping_type: Shaping) -> assessors.AssessorImpl:
        """
        Create the assessor for the task.

        :param shaping_type: the type of shaping to use
        :return: the assessor
        """
        if Shaping.is_stage_type(shaping_type):
            stage_number = int(shaping_type[5:])  # 提取数字部分
            base_components = ()
            shaping_components = ()

            if stage_number == 1:
                base_components = (
                    rewards.ScaledAsymptoticErrorComponent(
                        name="distance_reward",
                        prop=self.distance_oppo_ft,
                        state_variables=self.state_variables,
                        is_potential_based=False,
                        target=0.0,
                        scaling_factor=2000,  # Negative for reward
                        cmp_scale=2.0,
                    ),
                    rewards.ScaledAsymptoticErrorComponent(
                        name="heading_error",
                        prop=self.track_angle_rad,
                        state_variables=self.state_variables,
                        is_potential_based=False,
                        target=0.0,
                        scaling_factor=0.2,
                        cmp_scale=1.0,
                    ),
                    # rewards.ScaledAsymptoticErrorComponent(
                    #     name="altitude_error",
                    #     prop=prp.altitude_sl_ft,
                    #     state_variables=self.state_variables,
                    #     is_potential_based=False,
                    #     target=5000.0,  # target altitude
                    #     scaling_factor=1.0 / 2000.0,
                    #     cmp_scale=1.0,
                    # ),
                    # rewards.ScaledAsymptoticErrorComponent(
                    #     name="velocity_error",
                    #     prop=prp.vtrue_fps,
                    #     state_variables=self.state_variables,
                    #     is_potential_based=False,
                    #     target=self.aircraft.get_cruise_speed_fps(),
                    #     scaling_factor=1.0 / 100.0,
                    #     cmp_scale=1.0,
                    # ),
                )
                shaping_components = (
                    rewards.SmoothingComponent(
                        name="action_penalty",
                        props=[prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd],
                        state_variables=self.state_variables,
                        is_potential_based=True,
                        list_length=10,
                        cmp_scale=0.5,
                    ),
                )
            if not base_components and not shaping_components:
                raise ValueError(f"Reward function of {shaping_type} is not defined")
        else:
            raise ValueError(f"Unsupported shaping type: {shaping_type} , you should use 'stage*' as shaping type")
        
        return assessors.AssessorImpl(
            base_components,
            shaping_components,
            positive_rewards=self.positive_rewards,
        )
    
    # def get_state_space(self) -> gym.Space:
    #     state_lows = np.array([-1 for _ in self.state_variables])
    #     state_highs = np.array([1 for _ in self.state_variables])
        
    #     # 将观测空间扩展到原来的两倍
    #     doubled_state_lows = np.concatenate([state_lows, state_lows])
    #     doubled_state_highs = np.concatenate([state_highs, state_highs])
        
    #     return gym.spaces.Box(low=doubled_state_lows, high=doubled_state_highs, dtype=np.float64)

    # def get_action_space(self) -> gym.Space:
    #     action_lows = np.array([act_var.min for act_var in self.action_variables])
    #     action_highs = np.array([act_var.max for act_var in self.action_variables])
        
    #     # 将动作空间扩展到原来的两倍
    #     doubled_action_lows = np.concatenate([action_lows, action_lows])
    #     doubled_action_highs = np.concatenate([action_highs, action_highs])
        
    #     return gym.spaces.Box(low=doubled_action_lows, high=doubled_action_highs, dtype=np.float64)

    def task_step(
        self, sim: Simulation, action: Sequence[float], sim_steps: int, opponent_sim: Simulation=None
    ) -> Tuple[NamedTuple, float, bool, Dict]:
        if len(action) == len(self.action_variables):
            self_action = action
            # opponent_action = np.random.uniform(-1, 1, size=4)
        elif len(action) == 2*len(self.action_variables):
            self_action = action[:len(self.action_variables)]
            # opponent_action = action[len(self.action_variables):]
        else:
            raise ValueError(
                f"Action length {len(action)} does not match the expected length {len(self.action_variables)} or {2*len(self.action_variables)}"
            )
        
        # if self.opponent == "jsbsim":
        #     if opponent_sim is None:
        #         raise ValueError("Opponent_sim is None. ")
        #     # opponent_action = self._get_opponent_action(opponent_sim)
        #     for prop, command in zip(self.action_variables, opponent_action):
        #         opponent_sim[prop] = command

        # input actions
        for prop, command in zip(self.action_variables, self_action):
            sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            self.update_goal_point()
            sim.run()

        self._update_custom_properties(sim, opponent_sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        opponent_state = np.zeros_like(state)
        terminated = self._is_terminal(sim, opponent_sim)
        truncated = False
        reward = self.assessor.assess(state, self.last_state, terminated)
        reward_components = self.assessor.assess_components(state, self.last_state, terminated)
        env_info = None
        if terminated:
            reward = self._reward_terminal_override(reward, sim, opponent_sim)
            if sim[self.distance_oppo_ft] <= self.ARRIVE_RADIUS_FT:
                # 成功到达目标点
                win = 1
            else:
                win = 0
            env_info = {"win": win, "steps_used": self.steps_left.max - sim[self.steps_left]}
        if self.debug:
            self._validate_state(state, terminated, truncated, self_action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {"reward": reward_components, "env_info": env_info}
        observation = np.concatenate([np.array(state), np.array(opponent_state)])
        observation = self.observation_normalization(observation)

        return observation, reward.agent_reward(), terminated, False, info
    
    def update_goal_point(self):
        if self.goal_point_mode == 'dynamic':
            # Update position based on velocity
            dt = 1.0 / self.step_frequency_hz
            self.goal_point_position += self.goal_point_velocity * dt
        # For 'static' mode, do nothing, position remains constant
        return self.goal_point_position

    # def observation_normalization(self, observation: np.ndarray) -> np.ndarray:
    #     """
    #     Normalize observation values to the range [-1, 1] based on BoundedProperty min and max values.
        
    #     :param observation: Raw observation vector
    #     :return: Normalized observation vector
    #     """
    #     # Get min and max values for all state variables
    #     mins = np.array([prop.min for prop in self.state_variables])
    #     maxs = np.array([prop.max for prop in self.state_variables])
        
    #     # Handle the doubled observation space (self and opponent aircraft)
    #     if len(observation) == 2 * len(self.state_variables):
    #         mins = np.concatenate([mins, mins])
    #         maxs = np.concatenate([maxs, maxs])
        
    #     # Replace infinite values with large but finite values
    #     finite_max = 1e4  # A large but finite value
    #     mins = np.where(np.isneginf(mins), -finite_max, mins)
    #     maxs = np.where(np.isinf(maxs), finite_max, maxs)
        
    #     # Calculate ranges, avoiding division by zero
    #     ranges = maxs - mins
    #     ranges = np.where(ranges > 1e-10, ranges, 1e-10)
        
    #     # Normalize to [-1, 1]
    #     normalized_obs = 2 * (observation - mins) / ranges - 1
        
    #     # Clip values to ensure they stay in the range [-1, 1]
    #     normalized_obs = np.clip(normalized_obs, -1.0, 1.0)
        
    #     return normalized_obs

    def get_opponent_initial_conditions(self) -> Dict[Property, float]:
        """
        Get the initial conditions for the opponent aircraft.
        """
        base_oppo_initial_conditions = (
            types.MappingProxyType(  # MappingProxyType makes dict immutable
                {
                    prp.initial_altitude_ft: 5000,
                    prp.initial_terrain_altitude_ft: 0.00000001,
                    prp.initial_longitude_geoc_deg: -2.3273,
                    prp.initial_latitude_geod_deg: 51.4381,  # corresponds to UoBath
                }
            )
        )
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(), # 这里后续应该改成目标飞机的巡航速度
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: 180,
        }
        return {**base_oppo_initial_conditions, **extra_conditions}


    # def get_initial_conditions(self) -> Dict[Property, float]:
    #     extra_conditions = {
    #         prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
    #         prp.initial_v_fps: 0,
    #         prp.initial_w_fps: 0,
    #         prp.initial_p_radps: 0,
    #         prp.initial_q_radps: 0,
    #         prp.initial_r_radps: 0,
    #         prp.initial_roc_fpm: 0,
    #         prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
    #     }
    #     return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
        self._cal_self_position(sim, opponent_sim)
        self._update_extra_properties(sim, opponent_sim)
        # self._cal_oppo_state(sim, opponent_sim)
        self._update_HP(sim, opponent_sim)
        self._update_steps_left(sim, opponent_sim)

    def _cal_self_position(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
        """
        Calculate the position of the self aircraft.
        """
        self_position = self.coordinate_transform.ecef2ned(
            sim[prp.ecef_x_ft],
            sim[prp.ecef_y_ft],
            sim[prp.ecef_z_ft]
        )
        sim[self.ned_Xposition_ft] = self_position[0]
        sim[self.ned_Yposition_ft] = self_position[1]
        # Z position 使用海拔高度
        # sim[self.ned_Zposition_ft] = self_position[2]

        if opponent_sim is not None:
            self.goal_point_position = np.array([
                opponent_sim[self.ned_Xposition_ft],
                opponent_sim[self.ned_Yposition_ft],
                opponent_sim[prp.altitude_sl_ft]
            ])
            opponent_position = self.update_goal_point()
            sim[self.oppo_x_ft] = opponent_sim[self.ned_Xposition_ft] = opponent_position[0]
            sim[self.oppo_y_ft] = opponent_sim[self.ned_Yposition_ft] = opponent_position[1]
            sim[self.oppo_altitude_sl_ft] = opponent_sim[prp.altitude_sl_ft] = opponent_position[2]
    
    # def get_position(self, sim: Simulation) -> Tuple[float, float, float]:
    #     """
    #     Get the position of the self aircraft.
    #     """
    #     return (
    #         sim[self.ned_Xposition_ft],
    #         sim[self.ned_Yposition_ft],
    #         sim[prp.altitude_sl_ft],
    #         sim[self.oppo_x_ft],
    #         sim[self.oppo_y_ft],
    #         sim[self.oppo_altitude_sl_ft]
    #     )

    # def _cal_oppo_state(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
    #     """
    #     Calculate the state of the opponent aircraft.
    #     """
    #     # get raw data
    #     if self.opponent == "goal_point":
    #         if opponent_sim is None:
    #             raise ValueError("Opponent_sim is None. You should give it when calculating opponent state.")
    #         # opponent_position = self.coordinate_transform.ecef2ned(
    #         #     opponent_sim[prp.ecef_x_ft],
    #         #     opponent_sim[prp.ecef_y_ft],
    #         #     opponent_sim[prp.ecef_z_ft]
    #         # )
    #         self._update_extra_properties(sim=opponent_sim, opponent_sim=sim)
    #         sim[self.oppo_x_ft] = opponent_sim[self.ned_Xposition_ft]
    #         sim[self.oppo_y_ft] = opponent_sim[self.ned_Yposition_ft]
    #         sim[self.oppo_altitude_sl_ft] = opponent_sim[prp.altitude_sl_ft]
    #         sim[self.oppo_roll_rad] = opponent_sim[prp.roll_rad]
    #         sim[self.oppo_pitch_rad] = opponent_sim[prp.pitch_rad]
    #         sim[self.oppo_heading_deg] = opponent_sim[prp.heading_deg]
    #         sim[self.oppo_u_fps] = opponent_sim[prp.u_fps]
    #         sim[self.oppo_v_fps] = opponent_sim[prp.v_fps]
    #         sim[self.oppo_w_fps] = opponent_sim[prp.w_fps]
    #         sim[self.oppo_p_radps] = opponent_sim[prp.p_radps]
    #         sim[self.oppo_q_radps] = opponent_sim[prp.q_radps]
    #         sim[self.oppo_r_radps] = opponent_sim[prp.r_radps]
    #         sim[self.oppo_alpha_deg] = opponent_sim[prp.alpha_deg]
    #         sim[self.oppo_beta_deg] = opponent_sim[prp.beta_deg]
    #         sim[self.oppo_vtrue_fps] = opponent_sim[prp.vtrue_fps]
    #         sim[self.oppo_track_angle_rad] = opponent_sim[self.track_angle_rad]
    #         sim[self.oppo_bearing_accountingRollPitch_rad] = opponent_sim[self.bearing_accountingRollPitch_rad]
    #         sim[self.oppo_elevation_accountingRollPitch_rad] = opponent_sim[self.elevation_accountingRollPitch_rad]
    #         sim[self.oppo_bearing_pointMass_rad] = opponent_sim[self.bearing_pointMass_rad]
    #         sim[self.oppo_elevation_pointMass_rad] = opponent_sim[self.elevation_pointMass_rad]

    #         opponent_sim[self.oppo_x_ft] = sim[self.ned_Xposition_ft]
    #         opponent_sim[self.oppo_y_ft] = sim[self.ned_Yposition_ft]
    #         opponent_sim[self.oppo_altitude_sl_ft] = sim[prp.altitude_sl_ft]
    #         opponent_sim[self.oppo_roll_rad] = sim[prp.roll_rad]
    #         opponent_sim[self.oppo_pitch_rad] = sim[prp.pitch_rad]
    #         opponent_sim[self.oppo_heading_deg] = sim[prp.heading_deg]
    #         opponent_sim[self.oppo_u_fps] = sim[prp.u_fps]
    #         opponent_sim[self.oppo_v_fps] = sim[prp.v_fps]
    #         opponent_sim[self.oppo_w_fps] = sim[prp.w_fps]
    #         opponent_sim[self.oppo_p_radps] = sim[prp.p_radps]
    #         opponent_sim[self.oppo_q_radps] = sim[prp.q_radps]
    #         opponent_sim[self.oppo_r_radps] = sim[prp.r_radps]
    #         opponent_sim[self.oppo_alpha_deg] = sim[prp.alpha_deg]
    #         opponent_sim[self.oppo_beta_deg] = sim[prp.beta_deg]
    #         opponent_sim[self.oppo_vtrue_fps] = sim[prp.vtrue_fps]
    #         opponent_sim[self.oppo_track_angle_rad] = sim[self.track_angle_rad]
    #         opponent_sim[self.oppo_bearing_accountingRollPitch_rad] = sim[self.bearing_accountingRollPitch_rad]
    #         opponent_sim[self.oppo_elevation_accountingRollPitch_rad] = sim[self.elevation_accountingRollPitch_rad]
    #         opponent_sim[self.oppo_bearing_pointMass_rad] = sim[self.bearing_pointMass_rad]
    #         opponent_sim[self.oppo_elevation_pointMass_rad] = sim[self.elevation_pointMass_rad]

    #     else:
    #         raise ValueError("Unsupported opponent model: {}".format(self.opponent))

        
    def _update_extra_properties(self, sim: Simulation, opponent_sim: Simulation) -> None:
        """
        Update the extra properties.
        """
        own_position = prp.Vector3(
            sim[self.ned_Xposition_ft],
            sim[self.ned_Yposition_ft],
            sim[prp.altitude_sl_ft]
        )
        oppo_position = prp.Vector3(
            opponent_sim[self.ned_Xposition_ft],
            opponent_sim[self.ned_Yposition_ft],
            opponent_sim[prp.altitude_sl_ft]
        )

        if sim[self.steps_left] == self.steps_left.max:
            pre_distance = (own_position-oppo_position).Norm()
        else:
            pre_distance = sim[self.distance_oppo_ft]
        sim[self.distance_oppo_ft] = (own_position-oppo_position).Norm()
        time = 1 / self.step_frequency_hz
        sim[self.closure_rate] = (pre_distance - sim[self.distance_oppo_ft]) / time

        sim[self.bearing_pointMass_rad] = prp.Vector3.cal_angle(
            (oppo_position-own_position).project_to_plane("xy"), 
            prp.Vector3(x=1, y=0, z=0)
        )
        dlt_x, dlt_y, dlt_z = (oppo_position-own_position).get_xyz()
        sim[self.elevation_pointMass_rad] = math.atan2(dlt_z, math.sqrt(dlt_x**2+dlt_y**2))
        
        # 这里取-dlt_z是因为ned坐标系的z轴朝下
        R = Quaternion(0, dlt_x, dlt_y, -dlt_z)
        Q = prp.Eular2Quaternion(
            psi=sim[prp.psi_rad],
            theta=sim[prp.pitch_rad],
            phi=sim[prp.roll_rad]
        )
        Rb = Q.inverse * R * Q
        rbx, rby, rbz = Rb.vector
        sim[self.track_angle_rad] = prp.Vector3.cal_angle(
            prp.Vector3(rbx, rby, rbz),
            prp.Vector3(1, 0, 0)
        )

        opponent_sim[self.adverse_angle_rad] = prp.Vector3.cal_angle(
            prp.Vector3(rbx, rby, rbz),
            prp.Vector3(-1, 0, 0)
        )
        # print(f"opponent adverse angle: {opponent_sim[self.adverse_angle_rad]}")

        sim[self.bearing_accountingRollPitch_rad] = math.atan2(rby, rbx)
        sim[self.elevation_accountingRollPitch_rad] = math.atan2(rbz, math.sqrt(rbx**2+rby**2))


    def _update_HP(self, sim: Simulation, opponent_sim: Simulation) -> None:
        """
        Update the HP of the aircraft and opponent aircraft.
        """
        # # update opponent HP
        # if sim[self.track_angle_rad] <= math.radians(2) and 500 <= sim[self.distance_oppo_ft] <= 3000:
        #     damage = (3000 - sim[self.distance_oppo_ft]) / 2500 / self.step_frequency_hz
        #     if opponent_sim[self.aircraft_HP] > 0:
        #         opponent_sim[self.aircraft_HP] -= damage
        #     else:
        #         opponent_sim[self.aircraft_HP] = 0
        # if opponent_sim[prp.altitude_sl_ft] <= 1:
        #     opponent_sim[self.aircraft_HP] = 0

        # update self HP
        # if opponent_sim[self.track_angle_rad] <= math.radians(2) and 500 <= sim[self.distance_oppo_ft] <= 3000:
        #     damage = (3000 - sim[self.distance_oppo_ft]) / 2500 / self.step_frequency_hz
        #     if sim[self.aircraft_HP] > 0:
        #         sim[self.aircraft_HP] -= damage
        #     else:
        #         sim[self.aircraft_HP] = 0
        if sim[prp.altitude_sl_ft] <= 1:
            sim[self.aircraft_HP] = 0
        

    # def _update_steps_left(self, sim: Simulation, opponent_sim: Simulation) -> None:
    #     sim[self.steps_left] -= 1
    #     opponent_sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation, opponent_sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        HP_is_zero = self._is_hp_zero(sim, opponent_sim)
        reach_goal = sim[self.distance_oppo_ft] <= self.ARRIVE_RADIUS_FT
        return terminal_step or HP_is_zero or reach_goal
    
    def _is_hp_zero(self, sim: Simulation, opponent_sim: Simulation) -> bool:
        # print(f"self HP: {sim[self.aircraft_HP]}, opponent HP: {sim[self.opponent_HP]}")
        return sim[self.aircraft_HP] <= 0
    
    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation, opponent_sim: Simulation
    ) -> rewards.Reward:
        reach_goal = sim[self.distance_oppo_ft] <= self.ARRIVE_RADIUS_FT
        add_reward = 0
        if reach_goal:
            add_reward += 100.0
        if self._is_hp_zero(sim, opponent_sim):
            add_reward -= 100.0
        reward.set_additional_reward(add_reward)
        # print(f"debug: add_rwd:{add_reward}, self_HP:{sim[self.aircraft_HP]}, opponent_HP:{sim[self.opponent_HP]}")
        return reward

    def observe_first_state(self, sim: Simulation, opponent_sim: Simulation=None) -> np.ndarray:
        self._new_episode_init(sim, opponent_sim)
        self._update_custom_properties(sim, opponent_sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        # opponent_state = self.State(*(opponent_sim[prop] for prop in self.state_variables))
        opponent_state = np.zeros_like(state)
        observation = np.concatenate([np.array(state), np.array(opponent_state)])
        observation = self.observation_normalization(observation)
        return observation

    def _new_episode_init(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
        super()._new_episode_init(sim, opponent_sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        
        self.init_ecef_position = [sim[prp.ecef_x_ft], 
                                   sim[prp.ecef_y_ft], 
                                   sim[prp.ecef_z_ft]]
        lla_position = self.coordinate_transform.ecef2geo(*self.init_ecef_position)
        self.coordinate_transform.setNEDorigin(*lla_position)
        sim[self.aircraft_HP] = self.HP

        # Initialize goal point
        if opponent_sim is not None:
            # Set opponent sim's initial state so it can be used for storage
            self.goal_point_position = self.coordinate_transform.ecef2ned(
                opponent_sim[prp.ecef_x_ft],
                opponent_sim[prp.ecef_y_ft],
                opponent_sim[prp.ecef_z_ft]
            )
            opponent_sim[self.ned_Xposition_ft] = self.goal_point_position[0]
            opponent_sim[self.ned_Yposition_ft] = self.goal_point_position[1]
            
            if self.goal_point_mode == 'dynamic':
                # Random velocity similar to an aircraft's cruise speed
                cruise_speed_fps = self.aircraft.get_cruise_speed_fps()
                speed = random.uniform(cruise_speed_fps * 0.5, cruise_speed_fps * 1.2)
                # Random direction in 3D
                phi = random.uniform(0, 2 * math.pi) # Azimuth
                theta = random.uniform(-math.pi/6, math.pi/6) # Elevation
                self.goal_point_velocity = np.array([
                    speed * math.cos(theta) * math.cos(phi),
                    speed * math.cos(theta) * math.sin(phi),
                    speed * math.sin(theta)
                ])
            else: # static
                self.goal_point_velocity = np.array([0.0, 0.0, 0.0])
            
            opponent_sim[self.steps_left] = self.steps_left.max
        
        if self.opponent == "jsbsim":
            if opponent_sim is None:
                raise ValueError("Opponent_sim is None. You should give it when restart a new episode.")
            # super()._new_episode_init(opponent_sim) # We manually init opponent_sim above

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            self.distance_oppo_ft,
            self.track_angle_rad,
            # self.adverse_angle_rad,
            self.closure_rate,
            # self.opponent_HP,
            self.steps_left,
        )


def logistic(x, alpha, x0):
    arg = alpha * (x - x0)
    if arg > 700:  # exp(700)接近浮点上限
        return 1.0
    elif arg < -700:
        return 0.0
    else:
        return 1.0 / (1.0 + math.exp(-arg))

def GammaB(distance):
    if distance < 1950:
        return betaB(distance) * logistic(distance, 1/50, 1000)
    else:
        return betaB(distance) * (1 - logistic(distance, 1/50, 2900))

def betaB(distance):
    return 2.5

def GammaR(distance):
    if distance < 2250:
        return betaR(distance) * logistic(distance, 1/35, 400)
    else:
        return betaR(distance) * (1 - logistic(distance, 1/200, 4100))

def betaR(distance):
    return -2.5

