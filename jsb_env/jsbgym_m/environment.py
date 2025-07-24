import gymnasium as gym
import numpy as np
from jsbgym_m.tasks import Shaping, HeadingControlTask
from jsbgym_m.task_tracking import TrackingTask
from jsbgym_m.simulation import Simulation
from jsbgym_m.visualiser import FigureVisualiser, FlightGearVisualiser, GraphVisualiser, MultiplayerFlightGearVisualiser
from jsbgym_m.aircraft import Aircraft, c172, f16
from typing import Optional, Type, Tuple, Dict
import warnings


class JsbSimEnv(gym.Env):
    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the Gymnasium Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the Gymnasium Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code then migrated to work with the gymnasium interface.
    """

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {
        "render_modes": ["human", "flightgear", "human_fg", "graph", "graph_fg"],
        "render_fps": 60,
    }

    def __init__(
        self,
        aircraft: Aircraft = c172,
        task_type: Type = HeadingControlTask,
        agent_interaction_freq: int = 10,
        shaping: Shaping = Shaping.STANDARD,
        render_mode: Optional[str] = None,
    ):
        """
        Constructor. Inits some internal state, but JsbSimEnv.reset() must be
        called first before interacting with environment.

        :param task_type: the Task subclass for the task agent is to perform
        :param aircraft: the JSBSim aircraft to be used
        :param agent_interaction_freq: int, how many times per second the agent
            should interact with environment.
        :param shaping: a HeadingControlTask.Shaping enum, what type of agent_reward
            shaping to use (see HeadingControlTask for options)
        """
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError(
                "agent interaction frequency must be less than "
                "or equal to JSBSim integration frequency of "
                f"{self.JSBSIM_DT_HZ} Hz."
            )
        self.sim: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.aircraft = aircraft
        self.task = task_type(shaping, agent_interaction_freq, aircraft)
        # set Space objects
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        # set visualisation objects
        self.figure_visualiser: FigureVisualiser = None
        self.flightgear_visualiser: FlightGearVisualiser = None
        self.graph_visualiser: GraphVisualiser = None
        self.step_delay = None
        self.render_mode = render_mode

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if (
            self.render_mode == "human"
            or self.render_mode == "graph"
            or self.render_mode == "human_fg"
            or self.render_mode == "graph_fg"
        ):
            self.render()

        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, terminated, False, info).

        :param action: the agent's action, with same length as action variables.
        :return:
            state: agent's observation of the current environment
            reward: amount of reward returned after previous action
            terminated: whether the episode has ended, in which case further step() calls are undefined
            False: Truncated
            info: auxiliary information, e.g. full reward shaping data
        """
        if action.shape != self.action_space.shape:
            raise ValueError("mismatch between action and action space size")

        state, reward, terminated, truncated, info = self.task.task_step(
            self.sim, action, self.sim_steps_per_agent_step
        )
        observation = np.array(state)

        # save reward components from info
        if self.render_mode == "human":
            if hasattr(self.task, "opponent"):
                x, y, z, oppoX, oppoY, oppoZ = self.task.get_position(self.sim)
                self.figure_visualiser.save_target(oppoX, oppoY, oppoZ)
            else:
                x, y, z = self.task.get_position(self.sim)
            self.figure_visualiser.save_position(x, y, z)
            self.figure_visualiser.save_reward_components(info["reward"])

        # plot trajectory
        if self.render_mode == "human" and terminated:
            self.render()
            if hasattr(self.task, "target_Xposition"):
                target = [self.task._get_target_position("x"), 
                          self.task._get_target_position("y"),
                          self.task._get_target_position("z")]
                self.figure_visualiser.plot_position(target)
                self.figure_visualiser.plot_reward_components()
            elif hasattr(self.task, "opponent"):
                self.figure_visualiser.plot_position("tracking")
                self.figure_visualiser.plot_reward_components()
            else:
                self.figure_visualiser.plot_position()
                self.figure_visualiser.plot_reward_components()

        return observation, reward, terminated, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.
        """
        super().reset(seed=seed)
        init_conditions = self.task.get_initial_conditions()
        if self.sim:
            self.sim.reinitialise(init_conditions)
        else:
            self.sim = self._init_new_sim(
                self.JSBSIM_DT_HZ, self.aircraft, init_conditions
            )

        state = self.task.observe_first_state(self.sim)

        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.sim)
        observation = np.array(state)
        info = {}
        if self.render_mode == "human":
            self.render()
        if self.render_mode == "graph":
            try:
                self.graph_visualiser.reset()
            except AttributeError:
                pass
        if "NoFG" not in str(self):
            warnings.warn(
                "If training, use NoFG instead of FG in the env_id. Using FG will cause errors while training after a while."
            )
        return observation, info

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        return Simulation(
            sim_frequency_hz=dt, aircraft=aircraft, init_conditions=initial_conditions
        )

    def render(self, flightgear_blocking=True):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        :param mode: str, the mode to render with
        :param flightgear_blocking: waits for FlightGear to load before
            returning if True, else returns immediately
        """

        if self.render_mode == "human":
            if not self.figure_visualiser:
                self.figure_visualiser = FigureVisualiser(
                    self.sim, self.task.get_props_to_output()
                )
            self.figure_visualiser.plot(self.sim)
        elif self.render_mode == "flightgear":
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearVisualiser(
                    self.sim, self.task.get_props_to_output(), flightgear_blocking
                )
        elif self.render_mode == "human_fg":
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearVisualiser(
                    self.sim, self.task.get_props_to_output(), flightgear_blocking
                )
            self.flightgear_visualiser.plot(self.sim)
        elif self.render_mode == "graph":
            if not self.graph_visualiser:
                self.graph_visualiser = GraphVisualiser(
                    self.sim, self.task.get_props_to_output()
                )
            self.graph_visualiser.plot(self.sim)
        elif self.render_mode == "graph_fg":
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearVisualiser(
                    self.sim, self.task.get_props_to_output(), flightgear_blocking
                )
            self.graph_visualiser.plot(self.sim)
        else:
            super().render()

    def close(self):
        """Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the
        program exits.
        """
        if self.sim:
            self.sim.close()
        if self.figure_visualiser:
            self.figure_visualiser.close()
        if self.flightgear_visualiser:
            self.flightgear_visualiser.close()
        if self.graph_visualiser:
            self.graph_visualiser.close()


class NoFGJsbSimEnv(JsbSimEnv):
    """
    An RL environment for JSBSim with rendering to FlightGear disabled.
    This class exists to be used for training agents where visualisation is not
    required. Otherwise, restrictions in JSBSim output initialisation cause it
    to open a new socket for every single episode, eventually leading to
    failure of the network.
    """

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {
        "render_modes": ["human", "graph"],
        "render_fps": 60,
    }

    def _init_new_sim(self, dt: float, aircraft: Aircraft, initial_conditions: Dict):
        return Simulation(
            sim_frequency_hz=dt,
            aircraft=aircraft,
            init_conditions=initial_conditions,
            allow_flightgear_output=False,
        )

    def render(self, flightgear_blocking=True):
        if (
            self.render_mode == "flightgear"
            or self.render_mode == "human_fg"
            or self.render_mode == "graph_fg"
        ):
            raise ValueError("FlightGear rendering is disabled for this class")
        else:
            super().render(flightgear_blocking)

class DoubleJsbSimEnv(JsbSimEnv):
    def __init__(
        self,
        aircraft: Aircraft = f16,
        task_type: Type = TrackingTask,
        agent_interaction_freq: int = 10,
        shaping: Shaping = Shaping.STANDARD,
        render_mode: Optional[str] = None,
        opponent_aircraft: Aircraft = f16,
    ):
        if not issubclass(task_type, TrackingTask):
            raise ValueError(
                "DoubleJsbSimEnv only supports TrackingTask for now."
            )
        super().__init__(
            aircraft=aircraft,
            task_type=task_type,
            agent_interaction_freq=agent_interaction_freq,
            shaping=shaping,
            render_mode=render_mode,
        )
        # 新增对手飞机参数
        self.opponent_aircraft = opponent_aircraft
        self.opponent_sim: Simulation = None
        self.flightgear_visualiser: MultiplayerFlightGearVisualiser = None

    def _init_new_sim(self, dt, aircraft, initial_conditions, output_file: str=None):
        if output_file is None:
            return super()._init_new_sim(dt, aircraft, initial_conditions)
        else:
            return Simulation(
                sim_frequency_hz=dt, 
                aircraft=aircraft, 
                init_conditions=initial_conditions, 
                output_file=output_file,
            )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        oppo_init_conditions = self.task.get_opponent_initial_conditions()
        if self.opponent_sim:
            self.opponent_sim.reinitialise(oppo_init_conditions)
        else:
            # use opponent_flightgear.xml to change the port
            self.opponent_sim = self._init_new_sim(
                dt=self.JSBSIM_DT_HZ,
                aircraft=self.opponent_aircraft, 
                initial_conditions=oppo_init_conditions,
                output_file="opponent_flightgear.xml"
            )
        super(JsbSimEnv,self).reset(seed=seed)
        init_conditions = self.task.get_initial_conditions()
        if self.sim:
            self.sim.reinitialise(init_conditions)
        else:
            self.sim = self._init_new_sim(
                self.JSBSIM_DT_HZ, self.aircraft, init_conditions
            )

        state = self.task.observe_first_state(self.sim, self.opponent_sim)


        if self.render_mode == "flightgear":
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = MultiplayerFlightGearVisualiser(
                    self.sim, self.task.get_props_to_output(), block_until_loaded=True,
                )
        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.sim)
            self.flightgear_visualiser.configure_simulation_output(self.opponent_sim)
        observation = np.array(state)
        info = {}
        if self.render_mode == "human":
            self.render()
        if self.render_mode == "graph":
            try:
                self.graph_visualiser.reset()
            except AttributeError:
                pass
        if "NoFG" not in str(self):
            warnings.warn(
                "If training, use NoFG instead of FG in the env_id. Using FG will cause errors while training after a while."
            )
        return observation, info

        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if (
            self.render_mode == "human"
            or self.render_mode == "graph"
            or self.render_mode == "human_fg"
            or self.render_mode == "graph_fg"
        ):
            self.render()
        
        if action.shape != self.action_space.shape:
            raise ValueError("mismatch between action and action space size")

        state, reward, terminated, truncated, info = self.task.task_step(
            self.sim, action, self.sim_steps_per_agent_step, self.opponent_sim
        )
        observation = np.array(state)

        # save reward components from info
        if self.render_mode == "human":
            if hasattr(self.task, "opponent"):
                x, y, z, oppoX, oppoY, oppoZ = self.task.get_position(self.sim)
                self.figure_visualiser.save_target(oppoX, oppoY, oppoZ)
            else:
                x, y, z = self.task.get_position(self.sim)
            self.figure_visualiser.save_position(x, y, z)
            self.figure_visualiser.save_reward_components(info["reward"])

        # plot trajectory
        if self.render_mode == "human" and terminated:
            self.render()
            if hasattr(self.task, "target_Xposition"):
                target = [self.task._get_target_position("x"), 
                          self.task._get_target_position("y"),
                          self.task._get_target_position("z")]
                self.figure_visualiser.plot_position(target)
                self.figure_visualiser.plot_reward_components()
            elif hasattr(self.task, "opponent"):
                self.figure_visualiser.plot_position("tracking")
                self.figure_visualiser.plot_reward_components()
            else:
                self.figure_visualiser.plot_position()
                self.figure_visualiser.plot_reward_components()

        return observation, reward, terminated, False, info

    def render(self, flightgear_blocking=True):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="human")'
            )
            return

        if self.render_mode == "human":
            if not self.figure_visualiser:
                self.figure_visualiser = FigureVisualiser(
                    self.sim, self.task.get_props_to_output()
                )
            self.figure_visualiser.plot(self.sim)
        elif self.render_mode == "flightgear":
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = MultiplayerFlightGearVisualiser(
                    self.sim, self.task.get_props_to_output(), block_until_loaded=flightgear_blocking,
                )

    def close(self):
        if self.opponent_sim:
            self.opponent_sim.close()
        super().close()




        # # 这部分之后要放到task里面
        # # 先运行对手飞机的控制逻辑
        # opponent_action = self._get_opponent_action()
        # self._apply_opponent_action(opponent_action)
        
        # # 再运行agent的动作
        # for prop, cmd in zip(self.task.action_variables, action):
        #     self.agent_sim[prop] = cmd
            
        # # 同步运行两个仿真
        # for _ in range(sim_steps):
        #     self.agent_sim.run()
        #     self.opponent_sim.run()


class NoFGDoubleJsbSimEnv(DoubleJsbSimEnv):
    """
    An RL environment for JSBSim with rendering to FlightGear disabled.
    This class exists to be used for training agents where visualisation is not
    required. Otherwise, restrictions in JSBSim output initialisation cause it
    to open a new socket for every single episode, eventually leading to
    failure of the network.
    """

    JSBSIM_DT_HZ: int = 60  # JSBSim integration frequency
    metadata = {
        "render_modes": ["human", "graph"],
        "render_fps": 60,
    }
    def _init_new_sim(self, dt: float, aircraft: Aircraft, initial_conditions: Dict, output_file: str=None):
        return Simulation(
            sim_frequency_hz=dt,
            aircraft=aircraft,
            init_conditions=initial_conditions,
            allow_flightgear_output=False,
            output_file=output_file if output_file else None
        )
    def render(self, flightgear_blocking=True):
        if (
            self.render_mode == "flightgear"
            or self.render_mode == "human_fg"
            or self.render_mode == "graph_fg"
        ):
            raise ValueError("FlightGear rendering is disabled for this class")
        else:
            super().render(flightgear_blocking)