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

class Opponent(object):
    """
    Base class for the opponent aircraft.

    Arbitrarily fly straightly towards an arbitrary direction.
    """

    def __init__(self):
        self._state = {
            "x_position_ft": 0,
            "y_position_ft": 0,
            "altidude_sl_ft": 0,
            "roll_rad": 0,
            "pitch_rad": 0,
            "heading_deg": 0,
            "u_fps": 0,
            "v_fps": 0,
            "w_fps": 0,
            "p_radps": 0,
            "q_radps": 0,
            "r_radps": 0,
            "alpha_deg": 0,
            "beta_deg": 0,
            "vtrue_fps": 0,
        }
        self.init_point = (0, 0, 0)
        self.init_direction = 0
        self.init_speed = 0
    
    def reset(self):
        """
        Reset the opponent aircraft.
        """
        self.init_point = self._randomly_choose_init_point()
        self.init_direction = self._randomly_choose_init_dirction()
        self.init_speed = self._randomly_choose_init_speed()
        self._state = {
            "x_position_ft": self.init_point[0],
            "y_position_ft": self.init_point[1],
            "altidude_sl_ft": self.init_point[2],
            "roll_rad": 0,
            "pitch_rad": 0,
            "heading_deg": math.degrees(self.init_direction),
            "u_fps": self.init_speed,
            "v_fps": 0,
            "w_fps": 0,
            "p_radps": 0,
            "q_radps": 0,
            "r_radps": 0,
            "alpha_deg": 0,
            "beta_deg": 0,
            "vtrue_fps": self.init_speed,
        }
    
    def _randomly_choose_init_point(self):
        """
        Randomly choose the initial point of the opponent aircraft.
        """
        x = np.random.uniform(-6000, -4000) if random.random() < 0.5 else np.random.uniform(4000, 6000)
        y = np.random.uniform(-6000, -4000) if random.random() < 0.5 else np.random.uniform(4000, 6000)
        # h = np.random.uniform(-400, -700)
        h = -500
        return (x, y, h)
    
    def _randomly_choose_init_dirction(self):
        """
        Randomly choose the initial direction of the opponent aircraft.
        """
        return np.random.uniform(0, 2 * math.pi)
    
    def _randomly_choose_init_speed(self):
        """
        Randomly choose the initial speed of the opponent aircraft.
        """
        return np.random.uniform(700, 1000)

    def step(self, frequency):
        """
        Move the opponent aircraft.
        """
        new_x, new_y = self._calculate_position(frequency)
        self._state["x_position_ft"] = new_x
        self._state["y_position_ft"] = new_y
        return self._state

    def _calculate_position(self, frequency):
        """
        Calculate the position of the opponent aircraft.
        """
        time = 1 / frequency
        x = self._state["x_position_ft"]
        y = self._state["y_position_ft"]
        heading_angle = math.radians(self._state["heading_deg"])
        speed = self._state["u_fps"]
        x += speed * time * math.cos(heading_angle)
        y += speed * time * math.sin(heading_angle)
        return x, y


    def get_state(self) -> np.ndarray:
        """
        Get the state of the opponent aircraft.

        :return: the state of the opponent aircraft
        """
        return self._state.copy()


class TrackingTask(FlightTask):
    """
    Base class for tracking tasks.
    
    The agent will track the target aircraft, while the targer aricraft
    will follow a predefined trajectory.
    This is a pre-task of 2 airplane tracking. It's observation space and 
    reward function will be similar to the LM LLP Control Zone model.
    """

    # self aircraft state variables that derived from raw data
    distance_oppo_ft = BoundedProperty(
        "target/distance-to-opponent", "distance to the opponent aircraft [ft]", 0, 50000
    )
    track_angle_rad = BoundedProperty(
        "target/track-angle", "track angle between the two aircraft [rad]", -math.pi, math.pi
    )
    bearing_accountingRollPitch_rad = BoundedProperty(
        "target/bearing-accountingRollPitch", "bearing accounting for roll and pitch [rad]", -math.pi, math.pi
    )
    elevation_accountingRollPitch_rad = BoundedProperty(
        "target/elevation-accountingRollPitch", "elevation accounting for roll and pitch [rad]", -math.pi, math.pi
    )
    bearing_pointMass_rad = BoundedProperty(
        "target/bearing-pointMass", "bearing accounting for point mass [rad]", -math.pi, math.pi
    )
    elevation_pointMass_rad = BoundedProperty(
        "target/elevation-pointMass", "elevation accounting for point mass [rad]", -math.pi, math.pi
    )
    ned_Xposition_ft = BoundedProperty(
        "position/positionX-ft",
        "current track [ft]",
        -10000,
        10000,
    )
    ned_Yposition_ft = BoundedProperty(
        "position/positionY-ft",
        "current track [ft]",
        -10000,
        10000,
    )
    # ned_Zposition_ft = BoundedProperty(
    #     "position/positionZ-ft",
    #     "current altitude [ft]",
    #     -10000,
    #     10000,
    # )
    adverse_angle_rad = prp.BoundedProperty(
        "target/adverse-angle", "adverse angle between the two aircraft [rad]", -math.pi, math.pi
    )
    closure_rate = prp.BoundedProperty(
        "target/closure-rate", "closure rate between the two aircraft [ft/s]", -10000, 10000
    )

    # opponent aircraft state variables
    oppo_x_ft = BoundedProperty(
        "oppo/position/x-ft", "opponent aircraft x position [ft]", -10000, 10000
    )
    oppo_y_ft = BoundedProperty(
        "oppo/position/y-ft", "opponent aircraft y position [ft]", -10000, 10000
    )
    oppo_altitude_sl_ft = BoundedProperty(
        "oppo/altitude-sl-ft", "altitude above mean sea level [ft]", -1400, 85000
    )
    oppo_roll_rad = BoundedProperty(
        "oppo/attitude/roll-rad", "roll [rad]", -math.pi, math.pi
    )
    oppo_pitch_rad = BoundedProperty(
        "oppo/attitude/pitch-rad", "pitch [rad]", -0.5 * math.pi, 0.5 * math.pi
    )
    oppo_heading_deg = BoundedProperty(
        "oppo/attitude/psi-deg", "heading [deg]", 0, 360
    )
    oppo_u_fps = BoundedProperty(
        "oppo/velocities/u-fps", "body frame x-axis velocity [ft/s]", -2200, 2200
    )
    oppo_v_fps = BoundedProperty(
        "oppo/velocities/v-fps", "body frame y-axis velocity [ft/s]", -2200, 2200
    )
    oppo_w_fps = BoundedProperty(
        "oppo/velocities/w-fps", "body frame z-axis velocity [ft/s]", 0, 2200
    )
    oppo_p_radps = BoundedProperty(
        "oppo/velocities/p-rad_sec", "roll rate [rad/s]", -2 * math.pi, 2 * math.pi
    )
    oppo_q_radps = BoundedProperty(
        "oppo/velocities/q-rad_sec", "pitch rate [rad/s]", -2 * math.pi, 2 * math.pi
    )
    oppo_r_radps = BoundedProperty(
        "oppo/velocities/r-rad_sec", "yaw rate [rad/s]", -2 * math.pi, 2 * math.pi
    )
    oppo_alpha_deg = BoundedProperty(
        "oppo/aero/alpha-deg", "angle of attack [deg]", -180, +180
    )
    oppo_beta_deg = BoundedProperty(
        "oppo/aero/beta-deg", "sideslip [deg]", -180, +180
    )
    oppo_track_angle_rad = BoundedProperty(
        "oppo/track-angle", "track angle [rad]", -math.pi, math.pi
    )
    oppo_vtrue_fps = BoundedProperty(
        "oppo/vvelocities/vtrue-fps", "true airspeed [ft/s]", 0, 2200
    )
    oppo_bearing_accountingRollPitch_rad = BoundedProperty(
        "oppo/bearing-accountingRollPitch", "bearing accounting for roll and pitch [rad]", -math.pi, math.pi
    )
    oppo_elevation_accountingRollPitch_rad = BoundedProperty(
        "oppo/elevation-accountingRollPitch", "elevation accounting for roll and pitch [rad]", -math.pi, math.pi
    )
    oppo_bearing_pointMass_rad = BoundedProperty(
        "oppo/bearing-pointMass", "bearing accounting for point mass [rad]", -math.pi, math.pi
    )
    oppo_elevation_pointMass_rad = BoundedProperty(
        "oppo/elevation-pointMass", "elevation accounting for point mass [rad]", -math.pi, math.pi
    )

    ## without Delta state variables
    # can be read from jsbsim
    tracking_state_variables = (
        prp.heading_deg,
        prp.ax_fps2,
        prp.ay_fps2,
        prp.az_fps2,
        prp.aroll_radps2,
        prp.apitch_radps2,
        prp.ayaw_radps2,
        prp.alpha_deg,
        prp.beta_deg,
        prp.aileron_left,
        prp.teflap_position_norm,
        prp.leflap_position_norm,
        prp.left_dht_rad,
        prp.right_dht_rad,
        prp.rudder,
        prp.throttle_Aug,
        prp.total_fuel,
        prp.engine_thrust_lbs,
        prp.vtrue_fps,
    )
    # must calculate in this file
    extra_state_variables = (
        distance_oppo_ft,
        track_angle_rad,
        bearing_accountingRollPitch_rad,
        elevation_accountingRollPitch_rad,
        bearing_pointMass_rad,
        elevation_pointMass_rad,
        # 为了reward计算必须加上这两个
        adverse_angle_rad,
        closure_rate,
    )
    # opponent aircraft state variables, must calculate in this file
    oppo_state_variables = (
        oppo_altitude_sl_ft,
        oppo_roll_rad,
        oppo_pitch_rad,
        oppo_heading_deg,
        oppo_u_fps,
        oppo_v_fps,
        oppo_w_fps,
        oppo_p_radps,
        oppo_q_radps,
        oppo_r_radps,
        oppo_alpha_deg,
        oppo_beta_deg,
        oppo_track_angle_rad,
        oppo_vtrue_fps,
        oppo_bearing_accountingRollPitch_rad,
        oppo_elevation_accountingRollPitch_rad,
        oppo_bearing_pointMass_rad,
        oppo_elevation_pointMass_rad,
    )
    action_variables = (
        prp.aileron_cmd,
        prp.elevator_cmd,
        prp.rudder_cmd,
        prp.throttle_cmd
    )

    # other variables
    DEFAULT_EPISODE_TIME_S = 60.0
    INITIAL_HEADING_DEG = 0
    THROTTLE_CMD = 0.6
    MIXTURE_CMD = 0.8

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
        self.step_frequency_hz = step_frequency_hz
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty(
            "info/steps_left", "steps remaining in episode", 0, episode_steps
        )
        self.aircraft = aircraft
        self.opponent = self._create_opponent()
        self.state_variables = (
            FlightTask.base_state_variables
            + self.tracking_state_variables
            + self.extra_state_variables
            + self.oppo_state_variables
        )
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        self.coordinate_transform = GPS_NED(unit='ft')
        # self.target_theta = 0
        super().__init__(assessor)
    
    def _create_opponent(self) -> Opponent:
        """
        Create the opponent aircraft.
        """
        return Opponent()
    
    def make_assessor(self, shaping_type: Shaping) -> assessors.AssessorImpl:
        """
        Create the assessor for the task.

        :param shaping_type: the type of shaping to use
        :return: the assessor
        """
        base_components = (
            rewards.ScaledAsymptoticErrorComponent(
                name="force_altitude_4500",
                prop=prp.altitude_sl_ft,
                state_variables=self.state_variables,
                target=4500,
                is_potential_based=False,
                scaling_factor=500,
                cmp_scale=1.2,
            ),
            rewards.UserDefinedComponent(
                name = "relative_position",
                func=lambda track, adverse: (track/(math.pi)-2)*logistic(adverse/(math.pi),18,0.5) - track/(math.pi) + 1,
                props=(self.track_angle_rad, self.adverse_angle_rad),
                state_variables=self.state_variables,
                cmp_scale=1.0
            ),
            rewards.UserDefinedComponent(
                name="closure_rate",
                func=lambda closure, adverse, distance:
                    closure/500 * (1-logistic(adverse/(math.pi),18,0.5)) * logistic(distance,1/500,2900),
                props=(self.closure_rate, self.adverse_angle_rad, self.distance_oppo_ft),
                state_variables=self.state_variables,
                cmp_scale=1.0
            ),
            rewards.UserDefinedComponent(
                name="gunsnap_blue",
                func=lambda distance, track:
                    GammaB(distance) * (1 - logistic(track/(math.pi), 1e5, 1/180)),
                props=(self.distance_oppo_ft, self.track_angle_rad),
                state_variables=self.state_variables,
                cmp_scale=1.0
            ),
            rewards.UserDefinedComponent(
                name="gunsnap_red",
                func=lambda distance, adverse:
                    -GammaR(distance) * logistic(adverse/(math.pi), 800, 178/180),
                props=(self.distance_oppo_ft, self.adverse_angle_rad),
                state_variables=self.state_variables,
                cmp_scale=1.0
            ),
            rewards.UserDefinedComponent(
                name="deck",
                func=lambda h: -4 * (1-logistic(h, 1/20, 1300)),
                props=(prp.altitude_sl_ft,),
                state_variables=self.state_variables,
                cmp_scale=1.0
            ),
            # rewards.UserDefinedComponent(
            #     name="too_close",
            #     func=lambda adverse, distance:
            #         -2 * (1-logistic(adverse/(math.pi), 18, 0.5)) * logistic(distance, 1/50, 800),
            #     props=(self.adverse_angle_rad, self.distance_oppo_ft),
            #     state_variables=self.state_variables,
            #     cmp_scale=1.0
            # )
        )
        # shaping_components = (rewards.SmoothingComponent(
        #         name="action_penalty",
        #         props=[prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd],
        #         state_variables=self.action_variables,
        #         is_potential_based=True,
        #         cmp_scale=0.3,
        #     ),)
        shaping_components = ()

        if shaping_type is Shaping.STANDARD:
            return assessors.AssessorImpl(
                base_components,
                shaping_components,
                positive_rewards=self.positive_rewards,
            )
        else:
            raise ValueError(f"Unsupported shaping type: {shaping_type}")

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
        self._cal_self_position(sim)
        self._cal_oppo_state(sim)
        self._update_extra_properties(sim)
        self._update_steps_left(sim)

    def _cal_self_position(self, sim: Simulation) -> None:
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
    
    def get_position(self, sim: Simulation) -> Tuple[float, float, float]:
        """
        Get the position of the self aircraft.
        """
        return (
            sim[self.ned_Xposition_ft],
            sim[self.ned_Yposition_ft],
            sim[prp.altitude_sl_ft],
            sim[self.oppo_x_ft],
            sim[self.oppo_y_ft],
            sim[self.oppo_altitude_sl_ft]
        )

    def _cal_oppo_state(self, sim: Simulation) -> None:
        """
        Calculate the state of the opponent aircraft.
        """
        # get raw data
        oppo_state = self.opponent.step(self.step_frequency_hz)
        sim[self.oppo_x_ft] = oppo_state["x_position_ft"]
        sim[self.oppo_y_ft] = oppo_state["y_position_ft"]
        sim[self.oppo_altitude_sl_ft] = oppo_state["altidude_sl_ft"]
        sim[self.oppo_roll_rad] = oppo_state["roll_rad"]
        sim[self.oppo_pitch_rad] = oppo_state["pitch_rad"]
        sim[self.oppo_heading_deg] = oppo_state["heading_deg"]
        sim[self.oppo_u_fps] = oppo_state["u_fps"]
        sim[self.oppo_v_fps] = oppo_state["v_fps"]
        sim[self.oppo_w_fps] = oppo_state["w_fps"]
        sim[self.oppo_p_radps] = oppo_state["p_radps"]
        sim[self.oppo_q_radps] = oppo_state["q_radps"]
        sim[self.oppo_r_radps] = oppo_state["r_radps"]
        sim[self.oppo_alpha_deg] = oppo_state["alpha_deg"]
        sim[self.oppo_beta_deg] = oppo_state["beta_deg"]
        sim[self.oppo_vtrue_fps] = oppo_state["vtrue_fps"]

        oppo_position = prp.Vector3(
            sim[self.oppo_x_ft],
            sim[self.oppo_y_ft],
            sim[self.oppo_altitude_sl_ft]
        )
        own_position = prp.Vector3(
            sim[self.ned_Xposition_ft],
            sim[self.ned_Yposition_ft],
            sim[prp.altitude_sl_ft]
        )

        sim[self.oppo_bearing_pointMass_rad] = prp.Vector3.cal_angle(
            (own_position-oppo_position).project_to_plane("xy"), 
            prp.Vector3(x=1, y=0, z=0)
        )

        dlt_x, dlt_y, dlt_z = (own_position-oppo_position).get_xyz()
        sim[self.oppo_elevation_pointMass_rad] = math.atan2(dlt_z, math.sqrt(dlt_x**2+dlt_y**2))
        R = Quaternion(0, dlt_x, dlt_y, -dlt_z)
        # Q = Quaternion(axis=[0,0,1], radians=math.radians(oppo_state["heading_deg"]))
        Q = prp.Eular2Quaternion(
            psi=math.radians(sim[self.oppo_heading_deg]),
            theta=sim[self.oppo_pitch_rad],
            phi=sim[self.oppo_roll_rad]
        )
        Rb = Q.inverse * R * Q
        rbx, rby, rbz = Rb.vector
        sim[self.oppo_track_angle_rad] = prp.Vector3.cal_angle(
            prp.Vector3(rbx, rby, rbz),
            prp.Vector3(1, 0, 0)
        )
        # sim[self.adverse_angle_rad] = prp.Vector3.cal_angle(
        #     prp.Vector3(rbx, rby, rbz),
        #     prp.Vector3(-1, 0, 0)
        # )

        sim[self.oppo_bearing_accountingRollPitch_rad] = math.atan2(rby, rbx)
        sim[self.oppo_elevation_accountingRollPitch_rad] = math.atan2(rbz, math.sqrt(rbx**2+rby**2))

        
    def _update_extra_properties(self, sim: Simulation) -> None:
        """
        Update the extra properties.
        """
        own_position = prp.Vector3(
            sim[self.ned_Xposition_ft],
            sim[self.ned_Yposition_ft],
            sim[prp.altitude_sl_ft]
        )
        oppo_position = prp.Vector3(
            sim[self.oppo_x_ft],
            sim[self.oppo_y_ft],
            sim[self.oppo_altitude_sl_ft]
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

        R_ = Quaternion(0, -dlt_x, -dlt_y, dlt_z)
        Q_ = prp.Eular2Quaternion(
            psi=math.radians(sim[self.oppo_heading_deg]),
            theta=sim[self.oppo_pitch_rad],
            phi=sim[self.oppo_roll_rad]
        )
        Rb_ = Q_.inverse * R_ * Q_
        rbx_, rby_, rbz_ = Rb_.vector
        sim[self.adverse_angle_rad] = prp.Vector3.cal_angle(
            prp.Vector3(rbx_, rby_, rbz_),
            prp.Vector3(-1, 0, 0)
        )

        sim[self.bearing_accountingRollPitch_rad] = math.atan2(rby, rbx)
        sim[self.elevation_accountingRollPitch_rad] = math.atan2(rbz, math.sqrt(rbx**2+rby**2))

    def _update_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        # TODO: issues if sequential?
        return terminal_step
    
    def _reward_terminal_override(
        self, reward: rewards.Reward, sim: Simulation
    ) -> rewards.Reward:
        return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        self.init_ecef_position = [sim[prp.ecef_x_ft], 
                                   sim[prp.ecef_y_ft], 
                                   sim[prp.ecef_z_ft]]
        lla_position = self.coordinate_transform.ecef2geo(*self.init_ecef_position)
        self.coordinate_transform.setNEDorigin(*lla_position)

        self.opponent.reset()

    def get_props_to_output(self) -> Tuple:
        return (
            prp.u_fps,
            prp.altitude_sl_ft,
            self.distance_oppo_ft,
            self.track_angle_rad,
            self.adverse_angle_rad,
            self.closure_rate,
            self.bearing_accountingRollPitch_rad,
            self.elevation_accountingRollPitch_rad,
            self.bearing_pointMass_rad,
            self.elevation_pointMass_rad,
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
    return 3

def GammaR(distance):
    if distance < 2250:
        return betaR(distance) * logistic(distance, 1/35, 400)
    else:
        return betaR(distance) * (1 - logistic(distance, 1/200, 4100))

def betaR(distance):
    return -3