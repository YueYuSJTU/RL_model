import gymnasium as gym
import subprocess
import time
import matplotlib as mpt
import matplotlib.pyplot as plt
import jsbgym_m.properties as prp
from jsbgym_m.aircraft import Aircraft
from jsbgym_m.simulation import Simulation
from typing import NamedTuple, Tuple, List, Dict


class AxesTuple(NamedTuple):
    """Holds references to figure subplots (axes)"""

    axes_state: plt.Axes
    axes_stick: plt.Axes
    axes_throttle: plt.Axes
    axes_rudder: plt.Axes


class FigureVisualiser(object):
    """Class for manging a matplotlib Figure displaying agent state and actions"""

    PLOT_PAUSE_SECONDS = 0.0001
    LABEL_TEXT_KWARGS = dict(
        fontsize=18, horizontalalignment="right", verticalalignment="baseline"
    )
    VALUE_TEXT_KWARGS = dict(
        fontsize=18, horizontalalignment="left", verticalalignment="baseline"
    )
    TEXT_X_POSN_LABEL = 0.8
    TEXT_X_POSN_VALUE = 0.9
    TEXT_Y_POSN_INITIAL = 1.0
    TEXT_Y_INCREMENT = -0.1

    def __init__(self, _: Simulation, print_props: Tuple[prp.Property]):
        """
        Constructor.
        Sets here is ft_per_deg_lon, which depends dynamically on aircraft's
        longitude (because of the conversion between geographic and Euclidean
        coordinate systems). We retrieve longitude from the simulation and
        assume it is constant thereafter.
        :param _: (unused) Simulation that will be plotted
        :param print_props: Propertys which will have their values printed to Figure.
            Must be retrievable from the plotted Simulation.
        """
        self.print_props = print_props
        self.figure: plt.Figure = None
        self.axes: AxesTuple = None
        self.value_texts: Tuple[plt.Text] = None
        self.positions: List[Tuple[float, float, float]] = []
        self.target: List[float, float, float] = []

    def plot(self, sim: Simulation) -> None:
        """
        Creates or updates a 3D plot of the episode.
        :param sim: Simulation that will be plotted
        """
        mpt.use("TkAgg")
        if not self.figure:
            self.figure, self.axes = self._plot_configure()

        # delete old control surface data points
        for subplot in self.axes[1:]:
            # pop and translate all data points
            while subplot.lines:
                data = subplot.lines[0].remove()
                del data

        self._print_state(sim)
        self._plot_control_states(sim, self.axes)
        self._plot_control_commands(sim, self.axes)
        # voodoo pause needed for figure to update
        plt.pause(self.PLOT_PAUSE_SECONDS)

    def close(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None

    def _plot_configure(self):
        """
        Creates a figure with subplots for states and actions.
        :return: (figure, axes) where:
            figure: a matplotlib Figure with subplots for state and controls
            axes: an AxesTuple object with references to all figure subplot axes
        """
        plt.ion()  # interactive mode allows dynamic updating of plot
        figure = plt.figure(figsize=(6, 11))

        spec = plt.GridSpec(
            nrows=3,
            ncols=2,
            width_ratios=[5, 1],  # second column very thin
            height_ratios=[6, 5, 1],  # bottom row very short
            wspace=0.3,
        )

        # create subplots
        axes_state = figure.add_subplot(spec[0, 0:])
        axes_stick = figure.add_subplot(spec[1, 0])
        axes_throttle = figure.add_subplot(spec[1, 1])
        axes_rudder = figure.add_subplot(spec[2, 0])

        # hide state subplot axes - text will be printed to it
        axes_state.axis("off")
        self._prepare_state_printing(axes_state)

        # config subplot for stick (aileron and elevator control in x/y axes)
        axes_stick.set_xlabel(
            "ailerons [-]",
        )
        axes_stick.set_ylabel("elevator [-]")
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
        axes_stick.xaxis.set_label_coords(0.5, 1.08)
        axes_stick.yaxis.set_label_coords(-0.05, 0.5)
        # make axes cross at origin
        axes_stick.spines["left"].set_position("zero")
        axes_stick.spines["bottom"].set_position("zero")
        # only show ticks at extremes of range
        axes_stick.set_xticks([-1, 1])
        axes_stick.xaxis.set_ticks_position("bottom")
        axes_stick.set_yticks([-1, 1])
        axes_stick.yaxis.set_ticks_position("left")
        axes_stick.tick_params(which="both", direction="inout")
        # show minor ticks throughout
        minor_locator = plt.MultipleLocator(0.2)
        axes_stick.xaxis.set_minor_locator(minor_locator)
        axes_stick.yaxis.set_minor_locator(minor_locator)
        # hide unneeded spines
        axes_stick.spines["right"].set_visible(False)
        axes_stick.spines["top"].set_visible(False)

        # config subplot for throttle: a 1D vertical plot
        axes_throttle.set_ylabel("throttle [-]")
        axes_throttle.set_ylim(bottom=0, top=1)
        axes_throttle.set_xlim(left=0, right=1)
        axes_throttle.spines["left"].set_position("zero")
        axes_throttle.yaxis.set_label_coords(0.5, 0.5)
        axes_throttle.set_yticks([0, 0.5, 1])
        axes_throttle.yaxis.set_minor_locator(minor_locator)
        axes_throttle.tick_params(axis="y", which="both", direction="inout")
        # hide horizontal x-axis and related spines
        axes_throttle.xaxis.set_visible(False)
        for spine in ["right", "bottom", "top"]:
            axes_throttle.spines[spine].set_visible(False)

        # config rudder subplot: 1D horizontal plot
        axes_rudder.set_xlabel("rudder [-]")
        axes_rudder.set_xlim(left=-1, right=1)
        axes_rudder.set_ylim(bottom=0, top=1)
        axes_rudder.xaxis.set_label_coords(0.5, -0.5)
        axes_stick.spines["bottom"].set_position("zero")
        axes_rudder.set_xticks([-1, 0, 1])
        axes_rudder.xaxis.set_minor_locator(minor_locator)
        axes_rudder.tick_params(axis="x", which="both", direction="inout")
        axes_rudder.get_yaxis().set_visible(False)  # only want a 1D subplot
        for spine in ["left", "right", "top"]:
            axes_rudder.spines[spine].set_visible(False)

        all_axes = AxesTuple(
            axes_state=axes_state,
            axes_stick=axes_stick,
            axes_throttle=axes_throttle,
            axes_rudder=axes_rudder,
        )

        # create figure-wide legend
        cmd_entry = (
            plt.Line2D(
                [], [], color="b", marker="o", ms=10, linestyle="", fillstyle="none"
            ),
            "Commanded Position, normalised",
        )
        pos_entry = (
            plt.Line2D([], [], color="r", marker="+", ms=10, linestyle=""),
            "Current Position, normalised",
        )
        figure.legend(
            (cmd_entry[0], pos_entry[0]),
            (cmd_entry[1], pos_entry[1]),
            loc="lower center",
        )

        plt.show()
        # voodoo pause needed for figure to appear
        plt.pause(self.PLOT_PAUSE_SECONDS)

        return figure, all_axes

    def _prepare_state_printing(self, ax: plt.Axes):
        ys = [
            self.TEXT_Y_POSN_INITIAL + i * self.TEXT_Y_INCREMENT
            for i in range(len(self.print_props))
        ]

        for prop, y in zip(self.print_props, ys):
            label = str(prop.name)
            ax.text(
                self.TEXT_X_POSN_LABEL,
                y,
                label,
                transform=ax.transAxes,
                **(self.LABEL_TEXT_KWARGS),
            )

        # print and store empty Text objects which we will rewrite each plot call
        value_texts = []
        dummy_msg = ""
        for y in ys:
            text = ax.text(
                self.TEXT_X_POSN_VALUE,
                y,
                dummy_msg,
                transform=ax.transAxes,
                **(self.VALUE_TEXT_KWARGS),
            )
            value_texts.append(text)
        self.value_texts = tuple(value_texts)

    def _print_state(self, sim: Simulation):
        # update each Text object with latest value
        for prop, text in zip(self.print_props, self.value_texts):
            text.set_text(f"{sim[prop]:.4g}")


    def _plot_control_states(self, sim: Simulation, all_axes: AxesTuple):
        control_surfaces = [prp.aileron_left, prp.elevator, prp.throttle, prp.rudder]
        ail, ele, thr, rud = [sim[control] for control in control_surfaces]
        # plot aircraft control surface positions
        all_axes.axes_stick.plot(
            [ail], [ele], "r+", mfc="none", markersize=10, clip_on=False
        )
        all_axes.axes_throttle.plot(
            [0], [thr], "r+", mfc="none", markersize=10, clip_on=False
        )
        all_axes.axes_rudder.plot(
            [rud], [0], "r+", mfc="none", markersize=10, clip_on=False
        )

    def _plot_control_commands(self, sim: Simulation, all_axes: AxesTuple):
        """
        Plots agent-commanded actions on the environment figure.
        :param sim: Simulation to plot control commands from
        :param all_axes: AxesTuple, collection of axes of subplots to plot on
        """
        ail_cmd = sim[prp.aileron_cmd]
        ele_cmd = sim[prp.elevator_cmd]
        thr_cmd = sim[prp.throttle_cmd]
        rud_cmd = sim[prp.rudder_cmd]

        all_axes.axes_stick.plot(
            [ail_cmd], [ele_cmd], "bo", mfc="none", markersize=10, clip_on=False
        )
        all_axes.axes_throttle.plot(
            [0], [thr_cmd], "bo", mfc="none", markersize=10, clip_on=False
        )
        all_axes.axes_rudder.plot(
            [rud_cmd], [0], "bo", mfc="none", markersize=10, clip_on=False
        )

    def save_position(self, x, y, z):
        """
        save position to self.positions
        """
        self.positions.append((x, y, z))
    
    def save_target(self, x, y, z):
        """
        save target to self.positions
        """
        self.target.append([x, y, z])
    
    def plot_position(self, target = None):
        """
        Plots the positions recorded during the simulation.
        """
        if not self.positions:
            print("No positions to plot.")
            return

        lats, longs, alts = zip(*self.positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(lats, longs, alts, label='Position')
        ax.scatter(lats[0], longs[0], alts[0], label='Start', color='r')
        ax.scatter(lats[-1], longs[-1], alts[-1], label='End', color='b')
        if target == "tracking":
            targetlats, targetlongs, targetalts = zip(*self.target)
            targetlats = list(targetlats)
            targetlongs = list(targetlongs)
            targetalts = list(targetalts)
            for i in range(len(targetlats)):
                targetlats[i] = targetlats[i] + lats[0]
                targetlongs[i] = targetlongs[i] + longs[0]
                targetalts[i] = targetalts[i]
            ax.plot(targetlats, targetlongs, targetalts, label='Target')
            ax.scatter(targetlats[0], targetlongs[0], targetalts[0], color='r')
            ax.scatter(targetlats[-1], targetlongs[-1], targetalts[-1], color='b')
        elif target:
            ax.scatter(target[0], target[1], target[2], label='Target', color='g')
        ax.set_xlabel('x-Position')
        ax.set_ylabel('y-Position')
        ax.set_zlabel('Altitude')
        ax.legend()
        plt.show(block = False)

    def save_reward_components(self, reward_cmps: Dict[str, float]):
        """
        save reward components to self.reward_components
        """
        if not hasattr(self, "reward_components"):
            self.reward_components = {}
        for key, value in reward_cmps.items():
            if key not in self.reward_components:
                self.reward_components[key] = [value]
            else:
                self.reward_components[key].append(value)
    
    def plot_reward_components(self):
        """
        Plots the reward components recorded during the simulation.
        """
        if not hasattr(self, "reward_components"):
            print("No reward components to plot.")
            return

        num_components = len(self.reward_components)
        fig, axes = plt.subplots(num_components, 1, figsize=(10, 5 * num_components))
        if num_components == 1:
            axes = [axes]

        for ax, (key, value) in zip(axes, self.reward_components.items()):
            ax.plot(value, label=key)
            ax.set_xlabel('Time')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.set_xlim(0, len(value) - 1)
            ax.set_ylim(min(min(v) for v in self.reward_components.values()), 
                        max(max(v) for v in self.reward_components.values()))

        plt.tight_layout()
        plt.show(block=True)


class GraphVisualiser(object):
    PLOT_PAUSE_SECONDS = 0.0001

    def __init__(self, _: Simulation, print_props: Tuple[prp.Property]):
        self.print_props = print_props
        self.figure: plt.Figure = None

    def plot(self, sim: Simulation):
        mpt.use("TkAgg")
        plt.ion()
        if not self.figure:
            self.figure = plt.figure(figsize=(6, 11))
            plt.show()
            self.ax = self.figure.add_subplot(111, projection="3d")

        self.alt = sim[prp.altitude_sl_ft]
        self.long = sim[prp.lng_geoc_deg]
        self.lat = sim[prp.lat_geod_deg]

        self.ax.set_xlabel("Lattitude")
        self.ax.set_ylabel("Longitude")
        self.ax.set_zlabel("Altitude")

        self.ax.scatter(self.lat, self.long, self.alt)

        # voodoo pause needed for figure to appear
        plt.pause(self.PLOT_PAUSE_SECONDS)

    def reset(self):
        plt.cla()
        if not self.figure:
            self.figure = plt.figure(figsize=(6, 11))
            plt.show()
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_xlabel("Lattitude")
            self.ax.set_ylabel("Longitude")
            self.ax.set_zlabel("Altitude")

    def close(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None


class FlightGearVisualiser(object):
    """
    Class for visualising aircraft using the FlightGear simulator.

    This visualiser launches FlightGear and (by default) waits for it to
    launch. A Figure is also displayed (by creating its own FigureVisualiser)
    which is used to display the agent's actions.
    """

    TYPE = "socket"
    DIRECTION = "in"
    RATE = 60
    SERVER = "127.0.0.1"
    PORT = 5550
    PROTOCOL = "udp"
    LOADED_MESSAGE = "loading cities done"
    LOADED_MESSAGE1 = "Starting hard-coded terrain presampling"
    LOADED_MESSAGE2 = "PNG lib warning : Malformed iTXt chunk"
    FLIGHTGEAR_TIME_FACTOR = 1  # sim speed relative to realtime, higher is faster

    def __init__(
        self, sim: Simulation, print_props: Tuple[prp.Property], block_until_loaded=True
    ):
        """
        Launches FlightGear in subprocess and starts figure for plotting actions.

        :param sim: Simulation that will be visualised
        :param aircraft: Aircraft to be loaded in FlightGear for visualisation
        :param print_props: collection of Propertys to be printed to Figure
        :param block_until_loaded: visualiser will block until it detects that
            FlightGear has loaded if True.
        """
        self.configure_simulation_output(sim)
        self.print_props = print_props
        self.flightgear_process = self._launch_flightgear(sim.get_aircraft())
        self.figure = FigureVisualiser(sim, print_props)
        if block_until_loaded:
            self._block_until_flightgear_loaded()

    def plot(self, sim: Simulation) -> None:
        """
        Updates a 3D plot of agent actions.
        """
        self.figure.plot(sim)

    @staticmethod
    def _launch_flightgear(aircraft: Aircraft):
        cmd_line_args = FlightGearVisualiser._create_cmd_line_args(
            aircraft.flightgear_id
        )
        gym.logger.info(f'Subprocess: "{cmd_line_args}"')
        flightgear_process = subprocess.Popen(
            cmd_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        gym.logger.info("Started FlightGear")
        return flightgear_process

    def configure_simulation_output(self, sim: Simulation):
        sim.enable_flightgear_output()
        sim.set_simulation_time_factor(self.FLIGHTGEAR_TIME_FACTOR)

    @staticmethod
    def _create_cmd_line_args(aircraft_id: str):
        flightgear_cmd = "fgfs"
        aircraft_arg = f"--aircraft={aircraft_id}"
        flight_model_arg = (
            "--native-fdm=" + f"{FlightGearVisualiser.TYPE},"
            f"{FlightGearVisualiser.DIRECTION},"
            f"{FlightGearVisualiser.RATE},"
            f"{FlightGearVisualiser.SERVER},"
            f"{FlightGearVisualiser.PORT},"
            f"{FlightGearVisualiser.PROTOCOL}"
        )
        flight_model_type_arg = "--fdm=" + "null"
        disable_ai_arg = "--disable-ai-traffic"
        disable_live_weather_arg = "--disable-real-weather-fetch"
        time_of_day_arg = "--timeofday=" + "morning"
        return (
            flightgear_cmd,
            aircraft_arg,
            flight_model_arg,
            flight_model_type_arg,  # Aircraft starts from inside ground if used - ensures no external fdm takes effect.
            disable_ai_arg,
            disable_live_weather_arg,
            time_of_day_arg,
        )

    def _block_until_flightgear_loaded(self):
        while True:
            msg_out = self.flightgear_process.stdout.readline().decode()
            if self.LOADED_MESSAGE in msg_out or self.LOADED_MESSAGE1 in msg_out or self.LOADED_MESSAGE2 in msg_out:
                time.sleep(5)
                print("FlightGear Loading Complete")
                break
            else:
                time.sleep(0.1)

    def close(self):
        if self.flightgear_process:
            self.flightgear_process.kill()


class MultiplayerFlightGearVisualiser(FlightGearVisualiser):
    def __init__(
        self,
        sim: Simulation,
        print_props: Tuple[prp.Property],
        multiplayer_ports: List[int] = [5552],  # 默认添加5552端口实例
        block_until_loaded=True,
    ):
        self.multiplayer_ports = [5550] + multiplayer_ports  # 主端口5550 + 附加端口
        self.flightgear_processes = []  # 存储多个进程
        
        # 手动执行父类初始化中的关键步骤
        self.configure_simulation_output(sim)
        self.print_props = print_props
        self.figure = FigureVisualiser(sim, print_props)
        
        # 启动所有FlightGear实例
        self._launch_all_instances(sim.get_aircraft())
        
        if block_until_loaded:
            self._block_until_all_loaded()

    def _launch_all_instances(self, aircraft: Aircraft):
        """启动所有FlightGear实例"""
        for port in self.multiplayer_ports:
            cmd = self._create_cmd_line_args(
                aircraft.flightgear_id, 
                port,
                is_main=(port == 5550)  # 标记主实例
            )
            gym.logger.info(f'Launching FG on port {port}: {" ".join(cmd)}')
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.flightgear_processes.append(process)
        
        gym.logger.info(f"Started {len(self.flightgear_processes)} FlightGear instances")

    @staticmethod
    def _create_cmd_line_args(aircraft_id: str, port: int, is_main: bool):
        """为指定端口生成命令行参数"""
        base_args = list(FlightGearVisualiser._create_cmd_line_args(aircraft_id))
        
        # 修改主连接端口
        base_args[2] = f"--native-fdm=socket,in,60,127.0.0.1,{port},udp"
        
        # 移除禁用AI的参数
        if '--disable-ai-traffic' in base_args:
            base_args.remove('--disable-ai-traffic')
        
        # 添加多人游戏支持
        base_args.extend([
            "--enable-ai-traffic",
            "--allow-nasal-from-sockets",
            "--enable-ai-models",
            # f"--callsign=Player{port}"
        ])
        
        # 配置多人游戏端口
        if is_main:
            base_args.extend([
                "--multiplay=out,10,127.0.0.1,5000",
                "--multiplay=in,10,127.0.0.1,5001"
            ])
        else:
            # 其他实例连接到主实例的5000端口
            base_args.extend([
                "--multiplay=out,10,127.0.0.1,5001",  # 添加这行输出配置
                "--multiplay=in,10,127.0.0.1,5000"
            ])
        base_args.append(f"--callsign=Player{port}")

        return tuple(base_args)

    def _block_until_all_loaded(self):
        """等待所有实例加载完成"""
        print("Waiting for FlightGear instances to load...")
        loaded = [False] * len(self.flightgear_processes)
        
        while not all(loaded):
            for i, process in enumerate(self.flightgear_processes):
                if loaded[i]:
                    continue
                    
                # 非阻塞读取输出
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                # 检查加载完成标志
                if (self.LOADED_MESSAGE in line or 
                    self.LOADED_MESSAGE1 in line or 
                    self.LOADED_MESSAGE2 in line):
                    loaded[i] = True
                    print(f"Instance on port {self.multiplayer_ports[i]} loaded")
            
            time.sleep(0.1)
        
        print("All FlightGear instances ready")
        time.sleep(5)  # 额外缓冲时间

    def close(self):
        """关闭所有FlightGear实例"""
        for process in self.flightgear_processes:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("All FlightGear instances closed")