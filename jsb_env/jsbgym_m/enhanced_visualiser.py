import gymnasium as gym
import subprocess
import time
import math
import matplotlib as mpt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
from typing import NamedTuple, Tuple, List, Dict, Optional
import os

import jsbgym_m.properties as prp
from jsbgym_m.aircraft import Aircraft
from jsbgym_m.simulation import Simulation
from jsbgym_m.visualiser import FigureVisualiser, AxesTuple


class EnhancedAxesTuple(NamedTuple):
    """扩展的子图引用，包含3D和控制量显示"""
    axes_3d: plt.Axes        # 3D飞机位置和姿态显示
    axes_state: plt.Axes     # 状态信息文本显示
    axes_stick: plt.Axes     # 操纵杆控制
    axes_throttle: plt.Axes  # 油门控制
    axes_rudder: plt.Axes    # 方向舵控制


class Enhanced3DVisualiser(object):
    """增强的3D可视化器，结合飞机姿态和控制量显示"""
    
    PLOT_PAUSE_SECONDS = 0.0001
    LABEL_TEXT_KWARGS = dict(
        fontsize=14, horizontalalignment="right", verticalalignment="baseline"
    )
    VALUE_TEXT_KWARGS = dict(
        fontsize=14, horizontalalignment="left", verticalalignment="baseline"
    )
    TEXT_X_POSN_LABEL = 0.7
    TEXT_X_POSN_VALUE = 0.8
    TEXT_Y_POSN_INITIAL = 0.95
    TEXT_Y_INCREMENT = -0.08
    
    # 飞机模型缩放和显示参数
    AIRCRAFT_SCALE = 100.0  # 飞机模型大小
    TRAIL_LENGTH = 10      # 轨迹长度
    VIEW_SIZE = 5000        # 视图大小（英尺）
    
    def __init__(self, _: Simulation, print_props: Tuple[prp.Property]):
        """
        初始化增强的3D可视化器
        
        :param _: 模拟对象（未使用）
        :param print_props: 需要显示的属性列表
        """
        self.print_props = print_props
        self.figure: plt.Figure = None
        self.axes: EnhancedAxesTuple = None
        self.value_texts: Tuple[plt.Text] = None
        
        # 存储轨迹数据
        self.positions: List[Tuple[float, float, float]] = []
        self.attitudes: List[Tuple[float, float, float]] = []  # roll, pitch, yaw
        
        # 创建简化的飞机模型点
        self._create_aircraft_model()
        
    def _create_aircraft_model(self):
        """创建简化的飞机模型几何体"""
        # 简化的飞机模型 - 用线段表示
        # 机身
        self.aircraft_lines = {
            'fuselage': np.array([[-1, 0, 0], [1, 0, 0]]),  # 机身主轴
            'wing': np.array([[-0.2, -0.8, 0], [-0.2, 0.8, 0]]),  # 主翼
            'tail_h': np.array([[0.8, -0.3, 0], [0.8, 0.3, 0]]),  # 水平尾翼
            'tail_v': np.array([[0.8, 0, -0.3], [0.8, 0, 0.3]]),  # 垂直尾翼
        }
        
    def plot(self, sim: Simulation) -> None:
        """
        创建或更新飞机状态的3D可视化
        
        :param sim: 要可视化的模拟对象
        """
        mpt.use("TkAgg")
        if not self.figure:
            self.figure, self.axes = self._plot_configure()
            
        # 获取当前飞机状态
        current_pos = self._get_aircraft_position(sim)
        current_att = self._get_aircraft_attitude(sim)
        
        # 存储轨迹数据
        self.positions.append(current_pos)
        self.attitudes.append(current_att)
        
        # 限制轨迹长度
        if len(self.positions) > self.TRAIL_LENGTH:
            self.positions.pop(0)
            self.attitudes.pop(0)
            
        # 更新3D显示
        self._update_3d_display(current_pos, current_att)
        
        # 更新状态文本
        self._print_state(sim)
        
        # 更新控制量显示
        self._plot_control_states(sim, self.axes)
        self._plot_control_commands(sim, self.axes)
        
        plt.pause(self.PLOT_PAUSE_SECONDS)
        
    def _get_aircraft_position(self, sim: Simulation) -> Tuple[float, float, float]:
        """获取飞机位置"""
        # 尝试使用NED坐标，如果不可用则使用经纬度
        try:
            # 假设使用NED坐标系
            x = sim[prp.lng_geoc_deg] * 111320 * math.cos(math.radians(sim[prp.lat_geod_deg]))  # 近似转换为米
            y = sim[prp.lat_geod_deg] * 111320  # 近似转换为米
            z = sim[prp.altitude_sl_ft] * 0.3048  # 转换为米
            return (x, y, z)
        except:
            # 使用相对位置
            return (0, 0, sim[prp.altitude_sl_ft] * 0.3048)
            
    def _get_aircraft_attitude(self, sim: Simulation) -> Tuple[float, float, float]:
        """获取飞机姿态（roll, pitch, yaw）"""
        roll = sim[prp.roll_rad]
        pitch = sim[prp.pitch_rad] 
        yaw = sim[prp.psi_rad]
        return (roll, pitch, yaw)
        
    def _update_3d_display(self, position: Tuple[float, float, float], 
                          attitude: Tuple[float, float, float]):
        """更新3D显示"""
        ax = self.axes.axes_3d
        
        # 清除之前的飞机模型
        for artist in ax.collections[:]:
            artist.remove()
        for line in ax.lines[:]:
            if hasattr(line, '_aircraft_model'):
                line.remove()
                
        # 绘制轨迹
        if len(self.positions) > 1:
            trail_x, trail_y, trail_z = zip(*self.positions)
            ax.plot(trail_x, trail_y, trail_z, 'b-', alpha=0.7, linewidth=2, label='Flight Path')
            
        # 绘制当前飞机模型
        self._draw_aircraft_model(ax, position, attitude)
        
        # 设置视图范围
        x, y, z = position
        size = self.VIEW_SIZE * 0.3048  # 转换为米
        ax.set_xlim([x - size, x + size])
        ax.set_ylim([y - size, y + size]) 
        ax.set_zlim([max(0, z - size/2), z + size/2])
        
        # 绘制地面网格
        self._draw_ground_grid(ax, x, y, size)
        
    def _draw_aircraft_model(self, ax: plt.Axes, position: Tuple[float, float, float],
                           attitude: Tuple[float, float, float]):
        """绘制飞机模型"""
        x, y, z = position
        roll, pitch, yaw = attitude
        
        # 创建旋转矩阵
        rotation_matrix = self._create_rotation_matrix(roll, pitch, yaw)
        
        # 绘制飞机各部分
        colors = {'fuselage': 'red', 'wing': 'blue', 'tail_h': 'green', 'tail_v': 'orange'}
        
        for part_name, line_points in self.aircraft_lines.items():
            # 缩放和旋转
            scaled_points = line_points * self.AIRCRAFT_SCALE
            rotated_points = np.dot(scaled_points, rotation_matrix.T)
            
            # 平移到飞机位置
            world_points = rotated_points + np.array([x, y, z])
            
            # 绘制线段
            line = ax.plot3D(world_points[:, 0], world_points[:, 1], world_points[:, 2],
                           color=colors[part_name], linewidth=3, label=part_name)[0]
            line._aircraft_model = True  # 标记为飞机模型
            
    def _create_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """创建欧拉角旋转矩阵（ZYX顺序）"""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch) 
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        # ZYX欧拉角旋转矩阵
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R
        
    def _draw_ground_grid(self, ax: plt.Axes, center_x: float, center_y: float, size: float):
        """绘制地面网格"""
        grid_size = size / 10
        x_range = np.arange(center_x - size, center_x + size, grid_size)
        y_range = np.arange(center_y - size, center_y + size, grid_size)
        
        # 绘制地面网格线
        for x in x_range:
            ax.plot3D([x, x], [center_y - size, center_y + size], [0, 0], 
                     'k-', alpha=0.3, linewidth=0.5)
        for y in y_range:
            ax.plot3D([center_x - size, center_x + size], [y, y], [0, 0],
                     'k-', alpha=0.3, linewidth=0.5)
                     
    def _plot_configure(self):
        """配置图形布局"""
        plt.ion()
        figure = plt.figure(figsize=(16, 10))
        
        # 创建网格布局
        gs = plt.GridSpec(2, 4, width_ratios=[3, 1, 1, 1], height_ratios=[3, 1], 
                         hspace=0.3, wspace=0.3)
        
        # 3D主显示区域
        axes_3d = figure.add_subplot(gs[0, :3], projection='3d')
        axes_3d.set_xlabel('East [m]', fontsize=12)
        axes_3d.set_ylabel('North [m]', fontsize=12) 
        axes_3d.set_zlabel('Altitude [m]', fontsize=12)
        axes_3d.set_title('Aircraft 3D Position and Attitude', fontsize=14)
        
        # 状态信息显示区域
        axes_state = figure.add_subplot(gs[0, 3])
        axes_state.axis('off')
        self._prepare_state_printing(axes_state)
        
        # 控制量显示区域（底部）
        axes_stick = figure.add_subplot(gs[1, 0])
        axes_throttle = figure.add_subplot(gs[1, 1]) 
        axes_rudder = figure.add_subplot(gs[1, 2])
        
        # 配置控制量显示
        self._configure_control_axes(axes_stick, axes_throttle, axes_rudder)
        
        all_axes = EnhancedAxesTuple(
            axes_3d=axes_3d,
            axes_state=axes_state,
            axes_stick=axes_stick,
            axes_throttle=axes_throttle,
            axes_rudder=axes_rudder
        )
        
        # 创建图例
        figure.legend(
            [plt.Line2D([], [], color='b', marker='o', ms=10, linestyle='', fillstyle='none'),
             plt.Line2D([], [], color='r', marker='+', ms=10, linestyle='')],
            ['Commanded Position', 'Current Position'],
            loc='lower right'
        )
        
        plt.show()
        plt.pause(self.PLOT_PAUSE_SECONDS)
        
        return figure, all_axes
        
    def _configure_control_axes(self, axes_stick: plt.Axes, axes_throttle: plt.Axes, 
                               axes_rudder: plt.Axes):
        """配置控制量显示子图"""
        # 操纵杆显示配置
        axes_stick.set_xlabel('Ailerons [-]', fontsize=10)
        axes_stick.set_ylabel('Elevator [-]', fontsize=10)
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
        axes_stick.spines['left'].set_position('zero')
        axes_stick.spines['bottom'].set_position('zero')
        axes_stick.set_xticks([-1, 0, 1])
        axes_stick.set_yticks([-1, 0, 1])
        axes_stick.spines['right'].set_visible(False)
        axes_stick.spines['top'].set_visible(False)
        axes_stick.grid(True, alpha=0.3)
        
        # 油门显示配置
        axes_throttle.set_ylabel('Throttle [-]', fontsize=10)
        axes_throttle.set_ylim(bottom=0, top=1)
        axes_throttle.set_xlim(left=0, right=1)
        axes_throttle.set_yticks([0, 0.5, 1])
        axes_throttle.xaxis.set_visible(False)
        for spine in ['right', 'bottom', 'top']:
            axes_throttle.spines[spine].set_visible(False)
        axes_throttle.grid(True, alpha=0.3)
        
        # 方向舵显示配置
        axes_rudder.set_xlabel('Rudder [-]', fontsize=10)
        axes_rudder.set_xlim(left=-1, right=1)
        axes_rudder.set_ylim(bottom=0, top=1)
        axes_rudder.set_xticks([-1, 0, 1])
        axes_rudder.yaxis.set_visible(False)
        for spine in ['left', 'right', 'top']:
            axes_rudder.spines[spine].set_visible(False)
        axes_rudder.grid(True, alpha=0.3)
        
    def _prepare_state_printing(self, ax: plt.Axes):
        """准备状态信息显示"""
        ys = [
            self.TEXT_Y_POSN_INITIAL + i * self.TEXT_Y_INCREMENT
            for i in range(len(self.print_props))
        ]
        
        for prop, y in zip(self.print_props, ys):
            label = str(prop.name).split('/')[-1]  # 简化属性名显示
            ax.text(
                self.TEXT_X_POSN_LABEL, y, f"{label}:",
                transform=ax.transAxes, **self.LABEL_TEXT_KWARGS
            )
            
        # 创建值文本对象
        value_texts = []
        for y in ys:
            text = ax.text(
                self.TEXT_X_POSN_VALUE, y, "",
                transform=ax.transAxes, **self.VALUE_TEXT_KWARGS
            )
            value_texts.append(text)
        self.value_texts = tuple(value_texts)
        
    def _print_state(self, sim: Simulation):
        """更新状态信息显示"""
        for prop, text in zip(self.print_props, self.value_texts):
            try:
                value = sim[prop]
                if isinstance(value, float):
                    text.set_text(f"{value:.3f}")
                else:
                    text.set_text(f"{value}")
            except:
                text.set_text("N/A")
                
    def _plot_control_states(self, sim: Simulation, all_axes: EnhancedAxesTuple):
        """绘制控制面当前位置"""
        try:
            # 清除之前的标记
            for ax in [all_axes.axes_stick, all_axes.axes_throttle, all_axes.axes_rudder]:
                for line in ax.lines[:]:
                    if hasattr(line, '_control_state'):
                        line.remove()
                        
            ail = sim[prp.aileron_left]
            ele = sim[prp.elevator] 
            thr = sim[prp.throttle]
            rud = sim[prp.rudder]
            
            # 绘制当前控制面位置
            line1 = all_axes.axes_stick.plot([ail], [ele], 'r+', 
                                           mfc='none', markersize=12, markeredgewidth=2)[0]
            line1._control_state = True
            
            line2 = all_axes.axes_throttle.plot([0.5], [thr], 'r+',
                                              mfc='none', markersize=12, markeredgewidth=2)[0] 
            line2._control_state = True
            
            line3 = all_axes.axes_rudder.plot([rud], [0.5], 'r+',
                                            mfc='none', markersize=12, markeredgewidth=2)[0]
            line3._control_state = True
        except Exception as e:
            pass  # 忽略属性不存在的错误
            
    def _plot_control_commands(self, sim: Simulation, all_axes: EnhancedAxesTuple):
        """绘制控制指令"""
        try:
            # 清除之前的指令标记
            for ax in [all_axes.axes_stick, all_axes.axes_throttle, all_axes.axes_rudder]:
                for line in ax.lines[:]:
                    if hasattr(line, '_control_cmd'):
                        line.remove()
                        
            ail_cmd = sim[prp.aileron_cmd]
            ele_cmd = sim[prp.elevator_cmd]
            thr_cmd = sim[prp.throttle_cmd] 
            rud_cmd = sim[prp.rudder_cmd]
            
            # 绘制控制指令
            line1 = all_axes.axes_stick.plot([ail_cmd], [ele_cmd], 'bo',
                                           mfc='none', markersize=10, markeredgewidth=2)[0]
            line1._control_cmd = True
            
            line2 = all_axes.axes_throttle.plot([0.5], [thr_cmd], 'bo',
                                              mfc='none', markersize=10, markeredgewidth=2)[0]
            line2._control_cmd = True
            
            line3 = all_axes.axes_rudder.plot([rud_cmd], [0.5], 'bo', 
                                            mfc='none', markersize=10, markeredgewidth=2)[0]
            line3._control_cmd = True
        except Exception as e:
            pass  # 忽略属性不存在的错误
            
    def close(self):
        """关闭可视化器"""
        if self.figure:
            plt.close(self.figure)
            self.figure = None
            self.axes = None
            
    def reset(self):
        """重置轨迹数据"""
        self.positions = []
        self.attitudes = []
        if self.axes and self.axes.axes_3d:
            # 清除3D显示中的轨迹
            for line in self.axes.axes_3d.lines[:]:
                if not hasattr(line, '_aircraft_model'):
                    line.remove()


class AnimatedEnhancedVisualiser(Enhanced3DVisualiser):
    """带动画功能的增强可视化器"""
    
    def __init__(self, simulation: Simulation, print_props: Tuple[prp.Property]):
        super().__init__(simulation, print_props)
        self.animation_data = []  # 存储动画数据
        self.is_recording = False
        
    def start_recording(self):
        """开始记录动画数据"""
        self.is_recording = True
        self.animation_data = []
        
    def stop_recording(self):
        """停止记录动画数据"""
        self.is_recording = False
        
    def plot(self, sim: Simulation) -> None:
        """重写plot方法以支持动画记录"""
        if self.is_recording:
            # 记录当前帧数据
            frame_data = {
                'position': self._get_aircraft_position(sim),
                'attitude': self._get_aircraft_attitude(sim),
                'controls': self._get_control_states(sim),
                'state_data': {prop: sim[prop] for prop in self.print_props}
            }
            self.animation_data.append(frame_data)
            
        # 调用父类的plot方法
        super().plot(sim)
        
    def _get_control_states(self, sim: Simulation) -> Dict:
        """获取控制状态数据"""
        try:
            return {
                'aileron': sim[prp.aileron_left],
                'elevator': sim[prp.elevator],
                'throttle': sim[prp.throttle], 
                'rudder': sim[prp.rudder],
                'aileron_cmd': sim[prp.aileron_cmd],
                'elevator_cmd': sim[prp.elevator_cmd],
                'throttle_cmd': sim[prp.throttle_cmd],
                'rudder_cmd': sim[prp.rudder_cmd]
            }
        except:
            return {}
            
    def create_animation(self, filename: str = None, fps: int = 30):
        """创建动画"""
        if not self.animation_data:
            print("No animation data recorded!")
            return
            
        fig, axes = self._plot_configure()
        
        def animate(frame_idx):
            if frame_idx >= len(self.animation_data):
                return
                
            frame_data = self.animation_data[frame_idx]
            
            # 更新3D显示
            self._update_3d_display(frame_data['position'], frame_data['attitude'])
            
            # 更新状态文本
            for prop, text in zip(self.print_props, self.value_texts):
                try:
                    value = frame_data['state_data'][prop]
                    if isinstance(value, float):
                        text.set_text(f"{value:.3f}")
                    else:
                        text.set_text(f"{value}")
                except:
                    text.set_text("N/A")
                    
            # 更新控制量显示（简化版）
            controls = frame_data['controls']
            if controls:
                # 这里可以添加控制量的动画更新
                pass
                
        anim = animation.FuncAnimation(fig, animate, frames=len(self.animation_data),
                                     interval=1000//fps, blit=False, repeat=True)
        
        if filename:
            if filename.endswith('.gif'):
                anim.save(filename, writer='pillow', fps=fps)
            else:
                anim.save(filename, writer='ffmpeg', fps=fps)
            print(f"Animation saved to {filename}")
        else:
            plt.show()
            
        return anim