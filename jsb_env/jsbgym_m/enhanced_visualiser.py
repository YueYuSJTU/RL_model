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
from scipy.io import loadmat

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
    AIRCRAFT_SCALE = 70.0   # F-16 模型缩放因子
    TRAIL_LENGTH = 100      # 轨迹长度
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
        self.grid_lines = []  # 存储地面网格线引用
        self.last_grid_center = None  # 记录上次网格中心位置

        # 存储轨迹数据 - 主机
        self.positions: List[Tuple[float, float, float]] = []
        self.attitudes: List[Tuple[float, float, float]] = []  # roll, pitch, yaw
        
        # 存储敌机轨迹数据
        self.opponent_positions: List[Tuple[float, float, float]] = []
        self.opponent_attitudes: List[Tuple[float, float, float]] = []  # roll, pitch, yaw
        
        # 存储飞机 3D 模型对象
        self.aircraft_polys = []    # 存储主机 3D 模型
        self.opponent_polys = []    # 存储敌机 3D 模型
        
        # 加载 F-16 3D 模型
        self._load_f16_model()
        
    def _load_f16_model(self):
        """加载 F-16 3D 模型数据"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                '/home/ubuntu/Workfile/RL/RL_model/src/multiAgent/aerobench/visualize/f-16.mat',
                os.path.join(os.path.dirname(__file__), '../../src/multiAgent/aerobench/visualize/f-16.mat'),
                os.path.join(os.path.dirname(__file__), 'f-16.mat'),
                'f-16.mat'
            ]
            
            self.f16_pts = None
            self.f16_faces = None
            
            for path in possible_paths:
                if os.path.exists(path):
                    data = loadmat(path)
                    self.f16_pts = data['V']    # 顶点数据
                    self.f16_faces = data['F']  # 面数据
                    print(f"Loaded F-16 model from: {path}")
                    break
            
            if self.f16_pts is None:
                print("Warning: f-16.mat not found, falling back to simple model")
                self._create_simple_aircraft_model()
        except Exception as e:
            print(f"Error loading F-16 model: {e}")
            self._create_simple_aircraft_model()
        
    def _create_simple_aircraft_model(self):
        """创建简化的飞机模型几何体（备用方案）"""
        # 简化的飞机模型 - 用线段表示
        self.aircraft_lines = {
            'fuselage': np.array([[-1, 0, 0], [1, 0, 0]]),  # 机身主轴
            'wing': np.array([[-0.2, -0.8, 0], [-0.2, 0.8, 0]]),  # 主翼
            'tail_h': np.array([[0.8, -0.3, 0], [0.8, 0.3, 0]]),  # 水平尾翼
            'tail_v': np.array([[0.8, 0, -0.3], [0.8, 0, 0.3]]),  # 垂直尾翼
        }
        
    def plot(self, sim: Simulation, opponent_sim: Simulation = None) -> None:
        """
        创建或更新飞机状态的3D可视化
        
        :param sim: 主机的模拟对象
        :param opponent_sim: 敌机的模拟对象（可选）
        """
        mpt.use("TkAgg")
        if not self.figure:
            self.figure, self.axes = self._plot_configure()
            
        # 获取当前主机状态
        current_pos = self._get_aircraft_position(sim)
        current_att = self._get_aircraft_attitude(sim)
        
        # 存储主机轨迹数据
        self.positions.append(current_pos)
        self.attitudes.append(current_att)
        
        # 限制主机轨迹长度
        if len(self.positions) > self.TRAIL_LENGTH:
            self.positions.pop(0)
            self.attitudes.pop(0)
            
        # 处理敌机数据（如果提供）
        opponent_pos = None
        opponent_att = None
        if opponent_sim is not None:
            opponent_pos = self._get_aircraft_position(opponent_sim)
            opponent_att = self._get_aircraft_attitude(opponent_sim)
            
            # 存储敌机轨迹数据
            self.opponent_positions.append(opponent_pos)
            self.opponent_attitudes.append(opponent_att)
            
            # 限制敌机轨迹长度
            if len(self.opponent_positions) > self.TRAIL_LENGTH:
                self.opponent_positions.pop(0)
                self.opponent_attitudes.pop(0)
            
        # 更新3D显示（包含主机和敌机）
        self._update_3d_display(current_pos, current_att, opponent_pos, opponent_att)
        
        # 更新状态文本（仅主机）
        self._print_state(sim)
        
        # 更新控制量显示（仅主机）
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
                        attitude: Tuple[float, float, float],
                        opponent_position: Tuple[float, float, float] = None,
                        opponent_attitude: Tuple[float, float, float] = None):
        """更新3D显示，包含主机和敌机"""
        ax = self.axes.axes_3d
        
        # 清除之前的飞机模型和轨迹
        for artist in ax.collections[:]:
            if hasattr(artist, '_aircraft_poly') or hasattr(artist, '_opponent_poly'):
                artist.remove()
        for line in ax.lines[:]:
            if (hasattr(line, '_aircraft_model') or hasattr(line, '_trail') or 
                hasattr(line, '_opponent_model') or hasattr(line, '_opponent_trail')):
                line.remove()
        
        # 清除之前的文字对象
        for text in ax.texts[:]:
            if hasattr(text, '_aircraft_label') or hasattr(text, '_opponent_label'):
                text.remove()
        
        # 清除存储的多边形和文字对象
        for obj in self.aircraft_polys + self.opponent_polys:
            try:
                obj.remove()
            except:
                pass
        self.aircraft_polys.clear()
        self.opponent_polys.clear()
                
        # 绘制主机轨迹
        if len(self.positions) > 1:
            trail_x, trail_y, trail_z = zip(*self.positions[-self.TRAIL_LENGTH:])
            trail_line = ax.plot(trail_x, trail_y, trail_z, 'b-', alpha=0.7, linewidth=2, label='Own Aircraft Path')[0]
            trail_line._trail = True
            
        # 绘制敌机轨迹（如果存在）
        if opponent_position is not None and len(self.opponent_positions) > 1:
            opp_trail_x, opp_trail_y, opp_trail_z = zip(*self.opponent_positions[-self.TRAIL_LENGTH:])
            opp_trail_line = ax.plot(opp_trail_x, opp_trail_y, opp_trail_z, 'r--', alpha=0.7, linewidth=2, label='Opponent Aircraft Path')[0]
            opp_trail_line._opponent_trail = True
            
        # 绘制当前主机模型
        self._draw_aircraft_model(ax, position, attitude, is_opponent=False)
        
        # 绘制当前敌机模型（如果存在）
        if opponent_position is not None and opponent_attitude is not None:
            self._draw_aircraft_model(ax, opponent_position, opponent_attitude, is_opponent=True)
        
        # 设置视图范围（以主机为中心）
        x, y, z = position
        size = self.VIEW_SIZE * 0.3048  # 转换为米
        ax.set_xlim([x - size, x + size])
        ax.set_ylim([y - size, y + size]) 
        ax.set_zlim([max(0, z - size/2), z + size/2])
        
        # 绘制地面网格
        self._draw_ground_grid(ax, x, y, size)

    def _draw_aircraft_model(self, ax: plt.Axes, position: Tuple[float, float, float],
                           attitude: Tuple[float, float, float], is_opponent: bool = False):
        """绘制飞机模型 - 使用 F-16 3D 模型或线条"""
        x, y, z = position
        roll, pitch, yaw = attitude
        
        print(f"Drawing aircraft model at: ({x:.1f}, {y:.1f}, {z:.1f}), is_opponent: {is_opponent}")
        
        # 优先使用 F-16 3D 模型
        if hasattr(self, 'f16_pts') and self.f16_pts is not None:
            try:
                self._draw_f16_model(ax, position, attitude, is_opponent)
                print("Successfully drew F-16 3D model")
                return
            except Exception as e:
                print(f"Failed to draw F-16 3D model: {e}")
        
        # 如果 F-16 模型失败，使用线条模型
        try:
            self._draw_aircraft_lines(ax, position, attitude, is_opponent)
            print("Successfully drew aircraft lines")
        except Exception as e2:
            print(f"Failed to draw aircraft lines: {e2}")
            # 最后的备用方案：简单点标记
            color = 'red' if is_opponent else 'blue'
            marker = 'x' if is_opponent else 'o'
            scatter = ax.scatter([x], [y], [z], c=color, s=500, marker=marker, 
                               edgecolors='black', linewidth=3, alpha=1.0)
            scatter._aircraft_model = True if not is_opponent else None
            scatter._opponent_model = True if is_opponent else None
            print("Drew fallback scatter point")

    def _draw_f16_model(self, ax: plt.Axes, position: Tuple[float, float, float],
                       attitude: Tuple[float, float, float], is_opponent: bool = False):
        """绘制 F-16 3D 模型 - 参考 anim3d.py"""
        x, y, z = position
        roll, pitch, yaw = attitude
        
        # 缩放 F-16 模型（参考 anim3d.py 的 scale3d 函数）
        scale_factor = self.AIRCRAFT_SCALE * 0.3048  # 转换为米
        pts = self._scale3d(self.f16_pts, [-scale_factor, scale_factor, scale_factor])
        
        # 旋转 F-16 模型（参考 anim3d.py 的 rotate3d 函数）
        # 注意：anim3d 中使用的是 (theta, psi - pi/2, -phi) 的顺序
        theta = attitude[1]  # pitch
        psi = attitude[2] - math.pi/2  # yaw，减去 pi/2 是因为模型的默认朝向
        phi = -attitude[0]   # roll，取负号
        
        pts = self._rotate3d(pts, theta, psi, phi)
        
        # 平移到飞机位置
        pts = pts + np.array([x, y, z])
        
        # 准备绘制多边形面
        verts = []
        face_colors = []
        edge_colors = []
        
        # 根据是否为敌机选择颜色
        if is_opponent:
            face_color = '0.7'  # 浅灰色
            edge_color = 'darkred'
        else:
            face_color = '0.2'  # 深灰色  
            edge_color = 'darkblue'
        
        # 构建面片（参考 anim3d.py 的面片构建）
        count = 0
        for face in self.f16_faces:
            count += 1
            
            # 可以跳过一些面片来提高性能
            if count % 5 != 0:  # 只绘制每第5个面
                continue
                
            face_pts = []
            for findex in face:
                if findex-1 < len(pts):  # 确保索引有效
                    face_pts.append(tuple(pts[findex-1]))
            
            if len(face_pts) >= 3:  # 确保至少有3个点形成面
                verts.append(face_pts)
                face_colors.append(face_color)
                edge_colors.append(edge_color)
        
        # 创建并添加 Poly3DCollection
        if verts:
            poly_collection = Poly3DCollection(verts, 
                                             facecolors=face_colors,
                                             edgecolors=edge_colors,
                                             alpha=0.8,
                                             linewidths=0.5)
            
            # 标记多边形对象以便后续清理
            poly_collection._aircraft_poly = True if not is_opponent else None
            poly_collection._opponent_poly = True if is_opponent else None
            
            ax.add_collection3d(poly_collection)
            
            # 保存多边形对象引用
            if is_opponent:
                self.opponent_polys.append(poly_collection)
            else:
                self.aircraft_polys.append(poly_collection)
        
        # 添加飞机标签 - 修正标记属性
        label = "Enemy" if is_opponent else "Own"
        color = 'red' if is_opponent else 'blue'
        text = ax.text(x, y, z + 100, label, fontsize=12, color=color, 
                      ha='center', va='bottom', weight='bold')
        
        # 正确标记文字对象
        if is_opponent:
            text._opponent_label = True
            self.opponent_polys.append(text)
        else:
            text._aircraft_label = True
            self.aircraft_polys.append(text)

    def _draw_aircraft_lines(self, ax: plt.Axes, position: Tuple[float, float, float],
                           attitude: Tuple[float, float, float], is_opponent: bool = False):
        """使用线条绘制飞机模型（备用方案） - 添加调试信息"""
        x, y, z = position
        roll, pitch, yaw = attitude
        
        print(f"Drawing aircraft lines at position: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        # 检查是否有线条模型数据
        if not hasattr(self, 'aircraft_lines') or not self.aircraft_lines:
            print("Warning: No aircraft_lines data available")
            # 创建基本的标记点
            color = 'red' if is_opponent else 'blue'
            marker = '^' if not is_opponent else 'v'
            scatter = ax.scatter([x], [y], [z], c=color, s=300, marker=marker, 
                               edgecolors='black', linewidth=3, alpha=0.9)
            scatter._aircraft_model = True if not is_opponent else None
            scatter._opponent_model = True if is_opponent else None
            
            # 添加标签 - 确保正确标记
            label = "Enemy" if is_opponent else "Own"
            text = ax.text(x, y, z + 100, label, fontsize=12, color=color, 
                          ha='center', va='bottom', weight='bold')
            if is_opponent:
                text._opponent_label = True
            else:
                text._aircraft_label = True
            return
        
        # 创建旋转矩阵
        rotation_matrix = self._create_rotation_matrix(roll, pitch, yaw)
        
        # 根据是否为敌机选择不同的颜色方案
        if is_opponent:
            colors = {'fuselage': 'darkred', 'wing': 'darkblue', 'tail_h': 'darkgreen', 'tail_v': 'darkorange'}
            line_style = '--'
            linewidth = 3
            marker_attr = '_opponent_model'
        else:
            colors = {'fuselage': 'red', 'wing': 'blue', 'tail_h': 'green', 'tail_v': 'orange'}
            line_style = '-'
            linewidth = 4
            marker_attr = '_aircraft_model'
        
        # 绘制飞机各部分
        for part_name, line_points in self.aircraft_lines.items():
            # 缩放和旋转
            scaled_points = line_points * self.AIRCRAFT_SCALE
            rotated_points = np.dot(scaled_points, rotation_matrix.T)
            
            # 平移到飞机位置
            world_points = rotated_points + np.array([x, y, z])
            
            print(f"Drawing {part_name}: {world_points}")
            
            # 绘制线段
            line = ax.plot3D(world_points[:, 0], world_points[:, 1], world_points[:, 2],
                           color=colors[part_name], linewidth=linewidth, linestyle=line_style)[0]
            setattr(line, marker_attr, True)  # 动态设置标记属性
        
        # 添加标签 - 确保正确标记
        label = "Enemy" if is_opponent else "Own"
        color = 'red' if is_opponent else 'blue'
        text = ax.text(x, y, z + 100, label, fontsize=12, color=color, 
                      ha='center', va='bottom', weight='bold')
        if is_opponent:
            text._opponent_label = True
        else:
            text._aircraft_label = True

    def _scale3d(self, pts: np.ndarray, scale_list: List[float]) -> np.ndarray:
        """缩放 3D 点云 - 参考 anim3d.py"""
        assert len(scale_list) == 3
        
        rv = np.zeros(pts.shape)
        
        for i in range(pts.shape[0]):
            for d in range(3):
                rv[i, d] = scale_list[d] * pts[i, d]
                
        return rv

    def _rotate3d(self, pts: np.ndarray, theta: float, psi: float, phi: float) -> np.ndarray:
        """旋转 3D 点云 - 参考 anim3d.py"""
        sinTheta = math.sin(theta)
        cosTheta = math.cos(theta)
        sinPsi = math.sin(psi)
        cosPsi = math.cos(psi)
        sinPhi = math.sin(phi)
        cosPhi = math.cos(phi)

        transform_matrix = np.array([
            [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta],
            [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi,
             -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi,
             -cosTheta * sinPhi],
            [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi,
             sinPsi * sinTheta * cosPhi + cosPsi * sinPhi,
             cosTheta * cosPhi]], dtype=float)

        rv = np.zeros(pts.shape)

        for i in range(pts.shape[0]):
            rv[i] = np.dot(pts[i], transform_matrix)

        return rv
        
    def _draw_ground_grid(self, ax: plt.Axes, center_x: float, center_y: float, size: float):
        """绘制地面网格（优化版）"""
        # 只有当飞机移动较远时才重新绘制网格
        if (self.last_grid_center is None or 
            abs(center_x - self.last_grid_center[0]) > size/4 or 
            abs(center_y - self.last_grid_center[1]) > size/4):
            
            # 清除旧的网格线
            for line in self.grid_lines:
                line.remove()
            self.grid_lines.clear()
            
            # 绘制新的网格线
            grid_size = size / 5  # 减少网格密度
            x_range = np.arange(center_x - size, center_x + size, grid_size)
            y_range = np.arange(center_y - size, center_y + size, grid_size)
            
            for x in x_range:
                line = ax.plot3D([x, x], [center_y - size, center_y + size], [0, 0], 
                            'k-', alpha=0.3, linewidth=0.5)[0]
                self.grid_lines.append(line)
                
            for y in y_range:
                line = ax.plot3D([center_x - size, center_x + size], [y, y], [0, 0],
                            'k-', alpha=0.3, linewidth=0.5)[0]
                self.grid_lines.append(line)
                
            self.last_grid_center = (center_x, center_y)
                     
    def _plot_configure(self):
        """配置图形布局"""
        plt.ion()
        figure = plt.figure(figsize=(18, 10))  # 稍微减小宽度，增加高度比例
        
        # 创建更复杂的网格布局 - 减少左侧留白
        gs = plt.GridSpec(3, 5, width_ratios=[5, 1.2, 1.2, 1, 1], height_ratios=[2, 1, 0.5], 
                         hspace=0.3, wspace=0.4, left=0.05, right=0.98, top=0.95, bottom=0.08)
        
        # 3D主显示区域 - 占据左侧大部分空间
        axes_3d = figure.add_subplot(gs[:, :1], projection='3d')  # 占据所有行，第一列
        axes_3d.set_xlabel('East [m]', fontsize=14)
        axes_3d.set_ylabel('North [m]', fontsize=14) 
        axes_3d.set_zlabel('Altitude [m]', fontsize=14)
        axes_3d.set_title('Aircraft 3D Position and Attitude', fontsize=16, pad=20)
        
        # 副翼+升降舵复合图 - 增大尺寸，确保正方形
        axes_stick = figure.add_subplot(gs[0:2, 1:3])  # 占据前两行，第2,3列
        
        # 油门图 - 右侧
        axes_throttle = figure.add_subplot(gs[0:2, 3])  # 占据前两行，第4列
        
        # 方向舵图 - 下方，增大尺寸
        axes_rudder = figure.add_subplot(gs[2, 1:4])  # 第三行，第二、三列
        
        # 状态信息显示区域 - 右侧
        axes_state = figure.add_subplot(gs[:, 4])  # 所有行，第5列
        axes_state.axis('off')
        self._prepare_state_printing(axes_state)
        
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
        # 操纵杆显示配置 - 确保正方形比例
        axes_stick.set_xlabel('Ailerons [-]', fontsize=12)
        axes_stick.set_ylabel('Elevator [-]', fontsize=12)
        axes_stick.set_title('Control Stick', fontsize=14, pad=15)
        axes_stick.set_xlim(left=-1, right=1)
        axes_stick.set_ylim(bottom=-1, top=1)
        axes_stick.set_aspect('equal')  # 确保正方形显示
        axes_stick.spines['left'].set_position('zero')
        axes_stick.spines['bottom'].set_position('zero')
        axes_stick.set_xticks([-1, -0.5, 0, 0.5, 1])
        axes_stick.set_yticks([-1, -0.5, 0, 0.5, 1])
        axes_stick.spines['right'].set_visible(False)
        axes_stick.spines['top'].set_visible(False)
        axes_stick.grid(True, alpha=0.3)
        axes_stick.tick_params(labelsize=10)
        
        # 油门显示配置 - 修正显示范围为0-1
        axes_throttle.set_ylabel('Throttle [-]', fontsize=12)
        axes_throttle.set_title('Throttle', fontsize=14, pad=15)
        axes_throttle.set_ylim(bottom=0, top=2)
        axes_throttle.set_xlim(left=-0.5, right=0.5)  # 调整x轴范围以更好显示条形图
        axes_throttle.set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes_throttle.set_xticks([])  # 隐藏x轴刻度
        for spine in ['right', 'bottom', 'top']:
            axes_throttle.spines[spine].set_visible(False)
        axes_throttle.grid(True, alpha=0.3, axis='y')
        axes_throttle.tick_params(labelsize=10)
        
        # 方向舵显示配置 - 增大显示区域
        axes_rudder.set_xlabel('Rudder [-]', fontsize=12)
        axes_rudder.set_title('Rudder', fontsize=14, pad=15)
        axes_rudder.set_xlim(left=-1, right=1)
        axes_rudder.set_ylim(bottom=-0.3, top=0.3)  # 调整y轴范围以更好显示条形图
        axes_rudder.set_xticks([-1, -0.5, 0, 0.5, 1])
        axes_rudder.set_yticks([])  # 隐藏y轴刻度
        for spine in ['left', 'right', 'top']:
            axes_rudder.spines[spine].set_visible(False)
        axes_rudder.grid(True, alpha=0.3, axis='x')
        axes_rudder.tick_params(labelsize=10)
        
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
                for patch in ax.patches[:]:
                    if hasattr(patch, '_control_state'):
                        patch.remove()
                        
            ail = sim[prp.aileron_left]
            ele = sim[prp.elevator] 
            thr = sim[prp.throttle]
            rud = sim[prp.rudder]
            
            # 绘制当前控制面位置
            line1 = all_axes.axes_stick.plot([ail], [ele], 'r+', 
                                           mfc='none', markersize=18, markeredgewidth=4)[0]
            line1._control_state = True
            
            # 油门用矩形条显示 - 调整宽度和位置
            throttle_bar = all_axes.axes_throttle.bar([0], [thr], width=0.4, 
                                                    bottom=[0], color='red', alpha=0.7)[0]
            throttle_bar._control_state = True
            
            # 方向舵用矩形条显示 - 调整高度和位置
            rudder_bar = all_axes.axes_rudder.bar([rud], [0.2], width=0.08, 
                                                bottom=[-0.1], color='red', alpha=0.7)[0]
            rudder_bar._control_state = True
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
                for patch in ax.patches[:]:
                    if hasattr(patch, '_control_cmd'):
                        patch.remove()
                        
            ail_cmd = sim[prp.aileron_cmd]
            ele_cmd = sim[prp.elevator_cmd]
            thr_cmd = sim[prp.throttle_cmd] 
            rud_cmd = sim[prp.rudder_cmd]
            
            # 绘制控制指令
            line1 = all_axes.axes_stick.plot([ail_cmd], [ele_cmd], 'bo',
                                           mfc='none', markersize=15, markeredgewidth=3)[0]
            line1._control_cmd = True
            
            # 油门指令用条形图显示 - 调整样式
            throttle_cmd_bar = all_axes.axes_throttle.bar([0], [thr_cmd], width=0.25, 
                                                        bottom=[0], color='blue', alpha=0.5)[0]
            throttle_cmd_bar._control_cmd = True
            
            # 方向舵指令用条形图显示 - 调整样式
            rudder_cmd_bar = all_axes.axes_rudder.bar([rud_cmd], [0.15], width=0.06, 
                                                    bottom=[-0.075], color='blue', alpha=0.5)[0]
            rudder_cmd_bar._control_cmd = True
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
        self.opponent_positions = []
        self.opponent_attitudes = []
        if self.axes and self.axes.axes_3d:
            # 清除3D显示中的轨迹
            for line in self.axes.axes_3d.lines[:]:
                if not (hasattr(line, '_aircraft_model') or hasattr(line, '_opponent_model')):
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
        
    def plot(self, sim: Simulation, opponent_sim: Simulation = None) -> None:
        """重写plot方法以支持动画记录和双机显示"""
        if self.is_recording:
            # 记录当前帧数据
            frame_data = {
                'position': self._get_aircraft_position(sim),
                'attitude': self._get_aircraft_attitude(sim),
                'controls': self._get_control_states(sim),
                'state_data': {prop: sim[prop] for prop in self.print_props}
            }
            
            # 记录敌机数据（如果存在）
            if opponent_sim is not None:
                frame_data['opponent_position'] = self._get_aircraft_position(opponent_sim)
                frame_data['opponent_attitude'] = self._get_aircraft_attitude(opponent_sim)
            else:
                frame_data['opponent_position'] = None
                frame_data['opponent_attitude'] = None
                
            self.animation_data.append(frame_data)
            
        # 调用父类的plot方法
        super().plot(sim, opponent_sim)
        
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
            self._update_3d_display(frame_data['position'], frame_data['attitude'],
                                    frame_data['opponent_position'], frame_data['opponent_attitude'])
            
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