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
        goal_point_mode: str = 'random',  # 'static', 'dynamic', 'random_dynamic', 'spiral', or 'random'
        random_init: bool = True,
    ):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        :param goal_point_mode: The movement mode of the goal point.
        :param random_init: whether to randomize initial conditions.
        """
        self.random_init = random_init
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
        self.current_goal_point_mode = goal_point_mode
        # spiral mode parameters
        self.spiral_center = np.array([0.0, 0.0, 0.0])
        self.spiral_radius = 5000.0
        self.spiral_angular_velocity = 0.01  # rad/s
        self.spiral_vertical_speed = 50.0  # ft/s
        self.spiral_time = 0.0
    
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
                        scaling_factor=2000,
                        cmp_scale=4.0,
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
                    rewards.UserDefinedComponent(
                        name="deck",
                        func=lambda h: -4 * (1-logistic(h, 1/20, 1300)),
                        props=(prp.altitude_sl_ft,),
                        state_variables=self.state_variables,
                        cmp_scale=1.0
                    ),
                    rewards.ScaledAsymptoticErrorComponent(
                        name="closure_rate_error",
                        prop=self.closure_rate,
                        state_variables=self.state_variables,
                        is_potential_based=False,
                        target=700.0,  # target altitude
                        scaling_factor=800.0,
                        cmp_scale=1.0,
                    ),
                    rewards.UserDefinedComponent(
                        name="small_aileron",
                        func=lambda cmd: -1 * abs(cmd),
                        props=(prp.aileron_cmd,),
                        state_variables=self.state_variables,
                        cmp_scale=1.0
                    ),
                )
                shaping_components = (
                    rewards.SmoothingComponent(
                        name="action_penalty",
                        props=[prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd],
                        state_variables=self.state_variables,
                        is_potential_based=True,
                        list_length=10,
                        cmp_scale=3.5,
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
    
    def task_step(
        self, sim: Simulation, action: Sequence[float], sim_steps: int, opponent_sim: Simulation=None
    ) -> Tuple[NamedTuple, float, bool, Dict]:
        if len(action) == len(self.action_variables):
            self_action = action
        elif len(action) == 2*len(self.action_variables):
            self_action = action[:len(self.action_variables)]
        else:
            raise ValueError(
                f"Action length {len(action)} does not match the expected length {len(self.action_variables)} or {2*len(self.action_variables)}"
            )

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
            elif sim[self.aircraft_HP] <= 0:
                # 坠机
                win = -1
            else:
                win = 0
            env_info = {"win": win, "steps_used": self.steps_left.max - sim[self.steps_left], "mode": self.current_goal_point_mode}
        if self.debug:
            self._validate_state(state, terminated, truncated, self_action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {"reward": reward_components, "env_info": env_info}
        observation = np.concatenate([np.array(state), np.array(opponent_state)])
        observation = self.observation_normalization(observation)

        return observation, reward.agent_reward(), terminated, False, info
    
    def _add_velocity_perturbation(self, base_velocity: np.ndarray, magnitude: float = 50.0) -> np.ndarray:
        """
        Adds a random perturbation to a velocity vector.

        :param base_velocity: The original velocity vector.
        :param magnitude: The magnitude of the perturbation.
        :return: The perturbed velocity vector.
        """
        perturbation = np.random.randn(3)
        perturbation = perturbation / np.linalg.norm(perturbation) * magnitude
        return base_velocity + perturbation

    def update_goal_point(self):
        """
        Update the goal point's position based on its movement mode.
        """
        dt = 1.0 / self.step_frequency_hz
        mode = self.current_goal_point_mode

        if mode == 'static':
            # Position remains constant, do nothing.
            pass
        elif mode == 'dynamic':
            # Constant velocity motion.
            self.goal_point_position += self.goal_point_velocity * dt
        elif mode == 'random_dynamic':
            # Motion with random perturbations in velocity.
            perturbed_velocity = self._add_velocity_perturbation(self.goal_point_velocity, magnitude=200.0)
            self.goal_point_position += perturbed_velocity * dt
        elif mode == 'spiral':
            # Spiral motion with perturbations.
            self.spiral_time += dt
            # Base spiral position
            x = self.spiral_center[0] + self.spiral_radius * math.cos(self.spiral_angular_velocity * self.spiral_time)
            y = self.spiral_center[1] + self.spiral_radius * math.sin(self.spiral_angular_velocity * self.spiral_time)
            z = self.spiral_center[2] + self.spiral_vertical_speed * self.spiral_time
            base_pos = np.array([x, y, z])
            # Add perturbation to the position for erratic movement
            self.goal_point_position = self._add_velocity_perturbation(base_pos, magnitude=50.0)
        else:
            raise ValueError(f"Unsupported goal point mode: {mode}")

        return self.goal_point_position
    
    def get_initial_conditions(self) -> Dict[Property, float]:
        """
        Get the initial conditions for the self aircraft.
        """
        # 初始化x和y的时候必须依靠NED坐标系，所以存成实例变量，在new_episode_init中调用
        if self.random_init:
            self.random_init_x = random.uniform(-8000.0, 8000.0)
            self.random_init_y = random.uniform(-8000.0, 8000.0)
            random_init_z = random.uniform(1500.0, 10000.0)
            random_init_heading = random.uniform(0.0, 360.0)
        else:
            self.random_init_x = 0.0
            self.random_init_y = 0.0
            random_init_z = 5000.0
            random_init_heading = 0.0
        base_initial_conditions = (
            types.MappingProxyType(  # MappingProxyType makes dict immutable
                {
                    prp.initial_altitude_ft: random_init_z,
                    prp.initial_terrain_altitude_ft: 0.00000001,
                    prp.initial_longitude_geoc_deg: -2.3273,
                    prp.initial_latitude_geod_deg: 51.4381,  # corresponds to UoBath
                }
            )
        )
        extra_conditions = {
            prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: random_init_heading,
        }
        return {**base_initial_conditions, **extra_conditions}

    def get_opponent_initial_conditions(self) -> Dict[Property, float]:
        """
        Get the initial conditions for the opponent aircraft.
        """
        if self.random_init:
            random_init_x = random.uniform(-8000.0, 8000.0)
            random_init_y = random.uniform(-8000.0, 8000.0)
            random_init_z = random.uniform(1500.0, 10000.0)
        else:
            random_init_x = 5000.0
            random_init_y = 0.0
            random_init_z = 5000.0
        base_oppo_initial_conditions = (
            types.MappingProxyType(  # MappingProxyType makes dict immutable
                {
                    prp.initial_altitude_ft: random_init_z,
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
        goal_point_conditions = {
            self.ned_Xposition_ft: random_init_x,
            self.ned_Yposition_ft: random_init_y,
        }
        return {**base_oppo_initial_conditions, **extra_conditions, **goal_point_conditions}

    def _update_custom_properties(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
        self._cal_self_position(sim, opponent_sim)
        self._update_extra_properties(sim, opponent_sim)
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

        self.goal_point_position = np.array([
            opponent_sim[self.ned_Xposition_ft],
            opponent_sim[self.ned_Yposition_ft],
            opponent_sim[prp.altitude_sl_ft]
        ])
        opponent_position = self.update_goal_point()
        sim[self.oppo_x_ft] = opponent_sim[self.ned_Xposition_ft] = opponent_position[0]
        sim[self.oppo_y_ft] = opponent_sim[self.ned_Yposition_ft] = opponent_position[1]
        sim[self.oppo_altitude_sl_ft] = opponent_sim[prp.altitude_sl_ft] = opponent_position[2]
        
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

        sim[self.bearing_accountingRollPitch_rad] = math.atan2(rby, rbx)
        sim[self.elevation_accountingRollPitch_rad] = math.atan2(rbz, math.sqrt(rbx**2+rby**2))

    def _update_HP(self, sim: Simulation, opponent_sim: Simulation) -> None:
        """
        Update the HP of the aircraft and opponent aircraft.
        """
        if sim[prp.altitude_sl_ft] <= 1:
            sim[self.aircraft_HP] = 0

    def _is_terminal(self, sim: Simulation, opponent_sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        HP_is_zero = self._is_hp_zero(sim, opponent_sim)
        reach_goal = sim[self.distance_oppo_ft] <= self.ARRIVE_RADIUS_FT
        return terminal_step or HP_is_zero or reach_goal
    
    def _is_hp_zero(self, sim: Simulation, opponent_sim: Simulation) -> bool:
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
        return reward

    def observe_first_state(self, sim: Simulation, opponent_sim: Simulation=None) -> np.ndarray:
        self._new_episode_init(sim, opponent_sim)
        self._update_custom_properties(sim, opponent_sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        opponent_state = np.zeros_like(state)
        observation = np.concatenate([np.array(state), np.array(opponent_state)])
        observation = self.observation_normalization(observation)
        return observation

    def _new_episode_init(self, sim: Simulation, opponent_sim: Simulation=None) -> None:
        super()._new_episode_init(sim, opponent_sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        # sim[self.steps_left] = self.steps_left.max
        
        if self.goal_point_mode == 'random':
            modes = ['static', 'dynamic', 'random_dynamic', 'spiral']
            self.current_goal_point_mode = random.choice(modes)
        else:
            self.current_goal_point_mode = self.goal_point_mode

        self.init_ecef_position = [sim[prp.ecef_x_ft], 
                                   sim[prp.ecef_y_ft], 
                                   sim[prp.ecef_z_ft]]
        lla_position = self.coordinate_transform.ecef2geo(*self.init_ecef_position)
        # 主机坐标随机初始化等价于NED坐标系原点随机初始化
        lla_position[0] += self.random_init_y * 0.3048 / 6378137.0 * (180.0 / math.pi)
        lla_position[1] += self.random_init_x * 0.3048 / (6378137.0 * math.cos(lla_position[0] * math.pi / 180.0)) * (180.0 / math.pi)
        self.coordinate_transform.setNEDorigin(*lla_position)

        # Initialize goal point
        if opponent_sim is not None:
            # Set opponent sim's initial state so it can be used for storage
            self.goal_point_position = np.array([
                opponent_sim[self.ned_Xposition_ft],
                opponent_sim[self.ned_Yposition_ft],
                opponent_sim[prp.altitude_sl_ft]
            ])
            
            mode = self.current_goal_point_mode
            cruise_speed_fps = self.aircraft.get_cruise_speed_fps()

            if mode == 'static':
                self.goal_point_velocity = np.array([0.0, 0.0, 0.0])
            
            elif mode in ['dynamic', 'random_dynamic']:
                # Random velocity similar to an aircraft's cruise speed
                speed = random.uniform(cruise_speed_fps * 0.5, cruise_speed_fps * 1.2)
                # Random direction in 3D
                phi = random.uniform(0, 2 * math.pi) # Azimuth
                theta = random.uniform(-math.pi/6, math.pi/6) # Elevation
                self.goal_point_velocity = np.array([
                    speed * math.cos(theta) * math.cos(phi),
                    speed * math.cos(theta) * math.sin(phi),
                    speed * math.sin(theta)
                ])
            
            elif mode == 'spiral':
                self.goal_point_velocity = np.array([0.0, 0.0, 0.0]) # Not used, but reset
                self.spiral_time = 0.0
                # Start spiral from the initial goal point position
                self.spiral_center = np.array(self.goal_point_position)
                # Adjust center so the spiral starts at goal_point_position
                self.spiral_center[0] -= self.spiral_radius
                # Randomize spiral parameters
                self.spiral_radius = random.uniform(5000.0, 9000.0)
                self.spiral_angular_velocity = random.uniform(0.01, 0.03) * random.choice([-1, 1])
                self.spiral_vertical_speed = random.uniform(-10.0, 100.0)

            else:
                raise ValueError(f"Unsupported goal point mode for initialization: {mode}")

            opponent_sim[self.steps_left] = self.steps_left.max
        
        if self.opponent == "jsbsim":
            if opponent_sim is None:
                raise ValueError("Opponent_sim is None. You should give it when restart a new episode.")

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

