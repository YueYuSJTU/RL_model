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

# 定义新的 AxesTuple，不需要包含 control axes，因为它们在另一个对象里管理
class CombatAxesTuple(NamedTuple):
    axes_3d: plt.Axes

class ControlAxesTuple(NamedTuple):
    axes_stick: plt.Axes
    axes_throttle: plt.Axes
    axes_rudder: plt.Axes

class Enhanced3DVisualiser(object):
    """
    重构后的可视化器：
    1. 双窗口显示 (3D对战 / 仪表盘)
    2. 动态相机追踪
    3. 姿态坐标轴辅助
    """
    
    PLOT_PAUSE_SECONDS = 0.001

    GUN_RANGE_FT = 4500.0  # 攻击距离 (英尺)
    GUN_ANGLE_RAD = math.radians(15) # 攻击角度 (弧度)
    
    # === 视觉参数 ===
    AIRCRAFT_SCALE = 100.0     # 稍微加大一点模型
    TRAIL_LENGTH = 150        # 延长轨迹
    MIN_VIEW_SIZE = 1000      # 最小视场半径 (英尺)
    MAX_VIEW_SIZE = 6000     # 最大视场半径 (防止远距离丢失)
    AXIS_LENGTH = 60.0        # 姿态辅助轴长度
    
    def __init__(self, _: Simulation, print_props: Tuple[prp.Property]):
        self.print_props = print_props
        
        # 窗口句柄
        self.fig_combat: plt.Figure = None
        self.fig_control: plt.Figure = None
        self.ax_combat: plt.Axes = None
        self.axes_control: ControlAxesTuple = None
        
        # HUD 文本对象
        self.hud_texts_left = []
        self.hud_texts_right = []
        
        self.grid_lines = []
        self.last_grid_center = None

        # 轨迹数据
        self.positions: List[Tuple[float, float, float]] = []
        self.attitudes: List[Tuple[float, float, float]] = []
        self.opponent_positions: List[Tuple[float, float, float]] = []
        self.opponent_attitudes: List[Tuple[float, float, float]] = []
        
        # 3D 对象缓存
        self.aircraft_polys = []
        self.opponent_polys = []
        self.attitude_axes = [] # 存储姿态坐标轴
        
        self._load_f16_model()
        
    def _load_f16_model(self):
        """加载 F-16 3D 模型数据"""
        try:
            possible_paths = [
                '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat',
                os.path.join(os.path.dirname(__file__), '../../archive/aerobench/visualize/f-16.mat'),
                os.path.join(os.path.dirname(__file__), 'f-16.mat'),
                'f-16.mat'
            ]
            self.f16_pts = None
            self.f16_faces = None
            for path in possible_paths:
                if os.path.exists(path):
                    data = loadmat(path)
                    self.f16_pts = data['V']
                    self.f16_faces = data['F']
                    print(f"Loaded F-16 model from: {path}")
                    break
            if self.f16_pts is None:
                self._create_simple_aircraft_model()
        except Exception as e:
            print(f"Error loading F-16 model: {e}")
            self._create_simple_aircraft_model()
        
    def _create_simple_aircraft_model(self):
        self.aircraft_lines = {
            'fuselage': np.array([[-1, 0, 0], [1, 0, 0]]),
            'wing': np.array([[-0.2, -0.8, 0], [-0.2, 0.8, 0]]),
            'tail_h': np.array([[0.8, -0.3, 0], [0.8, 0.3, 0]]),
            'tail_v': np.array([[0.8, 0, -0.3], [0.8, 0, 0.3]]),
        }
        
    def plot(self, sim: Simulation, opponent_sim: Simulation = None) -> None:
        mpt.use("TkAgg")
        
        # 初始化窗口（如果未初始化）
        if not self.fig_combat or not plt.fignum_exists(self.fig_combat.number):
            self._init_combat_window()
        if not self.fig_control or not plt.fignum_exists(self.fig_control.number):
            self._init_control_window()
            
        # 获取数据
        current_pos = self._get_aircraft_position(sim)
        current_att = self._get_aircraft_attitude(sim)
        
        self.positions.append(current_pos)
        self.attitudes.append(current_att)
        if len(self.positions) > self.TRAIL_LENGTH:
            self.positions.pop(0)
            self.attitudes.pop(0)
            
        opp_pos = None
        opp_att = None
        if opponent_sim:
            opp_pos = self._get_aircraft_position(opponent_sim)
            opp_att = self._get_aircraft_attitude(opponent_sim)
            self.opponent_positions.append(opp_pos)
            self.opponent_attitudes.append(opp_att)
            if len(self.opponent_positions) > self.TRAIL_LENGTH:
                self.opponent_positions.pop(0)
                self.opponent_attitudes.pop(0)
            
        # === 更新 3D 视图 ===
        self._update_3d_display(current_pos, current_att, opp_pos, opp_att)
        
        # === 更新 HUD 信息 ===
        self._update_hud_info(sim, opponent_sim)
        
        # === 更新 控制面板 ===
        self._plot_control_states(sim)
        self._plot_control_commands(sim)
        
        plt.pause(self.PLOT_PAUSE_SECONDS)

    def _init_combat_window(self):
        """初始化 3D 对战主窗口"""
        self.fig_combat = plt.figure("Air Combat Arena", figsize=(12, 9))
        self.ax_combat = self.fig_combat.add_subplot(111, projection='3d')
        
        # 尝试最大化窗口 (依赖后端)
        try:
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed') # Windows
        except:
            try:
                manager.resize(*manager.window.maxsize()) # TkAgg Linux
            except:
                pass

        # 设置黑色背景更有空战氛围
        # self.ax_combat.set_facecolor('black') 
        # self.fig_combat.patch.set_facecolor('black')
        
        self.ax_combat.set_xlabel('East [m]')
        self.ax_combat.set_ylabel('North [m]')
        self.ax_combat.set_zlabel('Altitude [m]')
        
        # 初始化 HUD 文本占位符 (左上角 - 我方信息)
        start_y = 0.95
        for _ in range(len(self.print_props) + 2): # +2 预留给标题
            t = self.fig_combat.text(0.02, start_y, "", color='blue', 
                                   fontfamily='monospace', fontsize=10, weight='bold')
            self.hud_texts_left.append(t)
            start_y -= 0.025
            
        # 初始化 HUD 文本占位符 (右上角 - 敌方信息)
        start_y = 0.95
        for _ in range(5): # 简略显示敌方信息
            t = self.fig_combat.text(0.85, start_y, "", color='red', 
                                   fontfamily='monospace', fontsize=10, weight='bold')
            self.hud_texts_right.append(t)
            start_y -= 0.025

    def _init_control_window(self):
        """初始化 控制输入副窗口"""
        self.fig_control = plt.figure("Flight Controls", figsize=(5, 8))
        
        # 布局: 上面是 Stick (正方形), 下面并排 Throttle 和 Rudder
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 0.2], wspace=0.4, hspace=0.3)
        
        ax_stick = self.fig_control.add_subplot(gs[0, :])
        ax_thr = self.fig_control.add_subplot(gs[1, 0])
        ax_rud = self.fig_control.add_subplot(gs[1, 1])
        
        # Stick 配置
        ax_stick.set_title("Stick Input", fontsize=10)
        ax_stick.set_xlim(-1, 1)
        ax_stick.set_ylim(-1, 1)
        ax_stick.set_aspect('equal')
        ax_stick.grid(True, linestyle='--')
        ax_stick.spines['left'].set_position('center')
        ax_stick.spines['bottom'].set_position('center')
        ax_stick.spines['right'].set_visible(False)
        ax_stick.spines['top'].set_visible(False)
        
        # Throttle 配置
        ax_thr.set_title("Throttle", fontsize=10)
        ax_thr.set_ylim(0, 1) # 修改为0-1通常范围
        ax_thr.set_xlim(-0.5, 0.5)
        ax_thr.set_xticks([])
        ax_thr.grid(True, axis='y')
        
        # Rudder 配置
        ax_rud.set_title("Rudder", fontsize=10)
        ax_rud.set_xlim(-1, 1)
        ax_rud.set_ylim(0, 1)
        ax_rud.set_yticks([])
        ax_rud.grid(True, axis='x')
        
        self.axes_control = ControlAxesTuple(ax_stick, ax_thr, ax_rud)

    def _update_3d_display(self, pos1, att1, pos2=None, att2=None):
        ax = self.ax_combat
        
        # 清理旧对象 (Performance critical)
        self._clean_axes(ax)
        
        # 1. 绘制轨迹
        self._draw_trail(ax, self.positions, 'b', '-')
        if pos2:
            self._draw_trail(ax, self.opponent_positions, 'r', '--')
            
        # 2. 绘制模型
        self._draw_aircraft_model(ax, pos1, att1, is_opponent=False)
        if pos2 and att2:
            self._draw_aircraft_model(ax, pos2, att2, is_opponent=True)
            
        # 3. 动态相机逻辑 (核心修改)
        if pos2:
            # 计算中点
            center_x = (pos1[0] + pos2[0]) / 2
            center_y = (pos1[1] + pos2[1]) / 2
            center_z = (pos1[2] + pos2[2]) / 2
            
            # 计算需要的半径 (距离的一半 + 余量)
            dist = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)
            view_radius = max(self.MIN_VIEW_SIZE, min(dist * 0.8, self.MAX_VIEW_SIZE))
        else:
            center_x, center_y, center_z = pos1
            view_radius = self.MIN_VIEW_SIZE

        # 设置轴范围
        ax.set_xlim([center_x - view_radius, center_x + view_radius])
        ax.set_ylim([center_y - view_radius, center_y + view_radius])
        ax.set_zlim([0, max(center_z + view_radius, 2000)]) # 地面是0
        
        # 4. 绘制地面投影线 (辅助定位)
        ax.plot([pos1[0], pos1[0]], [pos1[1], pos1[1]], [0, pos1[2]], 'b--', linewidth=0.5, alpha=0.5)
        if pos2:
            ax.plot([pos2[0], pos2[0]], [pos2[1], pos2[1]], [0, pos2[2]], 'r--', linewidth=0.5, alpha=0.5)

        # 5. 绘制地面网格 (基于当前视野中心)
        self._draw_ground_grid(ax, center_x, center_y, view_radius * 2)

    def _clean_axes(self, ax):
        # 移除集合 (Poly3DCollection)
        for artist in list(ax.collections): # copy list to safely remove
            if hasattr(artist, '_aircraft_poly') or hasattr(artist, '_opponent_poly'):
                artist.remove()
        
        # 移除线条 (Line3D)
        for line in list(ax.lines):
            # 保留网格线，移除其他动态对象
            if line not in self.grid_lines:
                line.remove()
                
        # 移除文本
        for text in list(ax.texts):
            text.remove()

    def _draw_trail(self, ax, positions, color, style):
        if len(positions) > 1:
            xs, ys, zs = zip(*positions)
            ax.plot(xs, ys, zs, c=color, linestyle=style, alpha=0.6, linewidth=1.5)

    def _draw_aircraft_model(self, ax: plt.Axes, pos, att, is_opponent=False):
        """绘制飞机模型及增强的姿态指示"""
        # 1. 绘制机体 (Mesh 或 Scatter)
        if hasattr(self, 'f16_pts') and self.f16_pts is not None:
            self._draw_f16_mesh(ax, pos, att, is_opponent)
        else:
            # 备用点
            c = 'red' if is_opponent else 'blue'
            ax.scatter([pos[0]], [pos[1]], [pos[2]], c=c, s=200, marker='^' if not is_opponent else 'v')

        # 2. 绘制姿态坐标轴 (核心增强: 红X-机头, 绿Y-右翼, 蓝Z-机腹)
        self._draw_local_axes(ax, pos, att)

        # 3. === 新增：绘制攻击范围锥 ===
        self._draw_attack_cone(ax, pos, att, is_opponent)
        
        # 3. 标签
        label = f"Enemy\nAlt: {pos[2]:.0f}" if is_opponent else f"OWN\nAlt: {pos[2]:.0f}"
        color = 'darkred' if is_opponent else 'darkblue'
        ax.text(pos[0], pos[1], pos[2] + self.AIRCRAFT_SCALE*2, label, 
               color=color, ha='center', fontsize=9, weight='bold')

    def _draw_attack_cone(self, ax, pos, att, is_opponent):
        """绘制攻击圆锥"""
        # 参数转换：英尺 -> 米
        range_m = self.GUN_RANGE_FT * 0.3048
        # 计算圆锥底面半径
        radius_m = range_m * math.tan(self.GUN_ANGLE_RAD)
        
        # 生成圆锥底面的圆周点 (Body Frame: X轴为中心轴)
        # 这里的圆是在 Y-Z 平面上
        theta = np.linspace(0, 2*np.pi, 16) # 16边形近似圆形
        
        # Body Frame 坐标:
        # X: 向前延伸到 range_m
        # Y: 半径 * cos(theta)
        # Z: 半径 * sin(theta)
        x_circle = np.full_like(theta, range_m)
        y_circle = radius_m * np.cos(theta)
        z_circle = radius_m * np.sin(theta)
        
        # 组合成点集 (N, 3)
        circle_pts = np.stack([x_circle, y_circle, z_circle], axis=1)
        
        # 加上顶点 (0, 0, 0)
        apex = np.array([0.0, 0.0, 0.0])
        
        # === 旋转 ===
        # 使用与飞机模型相同的旋转逻辑
        roll, pitch, yaw = att
        
        # 旋转圆周点
        rotated_circle = self._rotate3d(circle_pts, pitch, yaw - math.pi/2, -roll)
        # 注意：这里的欧拉角参数顺序和符号是为了匹配 f16_model 的加载逻辑
        # 如果你发现锥体方向和红色的机头线不重合，请调整这里的参数，例如:
        # rotated_circle = self._rotate3d_simple(circle_pts, roll, pitch, yaw) 
        
        # 平移到飞机位置
        rotated_circle += np.array(pos)
        apex_world = np.array(pos)
        
        # === 构建绘图面 ===
        verts = []
        # 构建侧面三角形：顶点 -> 圆周点i -> 圆周点i+1
        for i in range(len(rotated_circle) - 1):
            verts.append([apex_world, rotated_circle[i], rotated_circle[i+1]])
        
        # 颜色设置
        color = 'red' if is_opponent else 'cyan'
        
        # 绘制半透明锥体
        poly = Poly3DCollection(verts, facecolors=color, edgecolors=color, 
                                alpha=0.1, linewidths=0.5)
        
        # 标记以便清理
        if is_opponent:
            poly._opponent_poly = True
        else:
            poly._aircraft_poly = True
            
        ax.add_collection3d(poly)
        
        # 额外画一条中心线表示射击轴线
        # 取圆心的旋转点
        center_pt = np.array([[range_m, 0, 0]])
        rot_center = self._rotate3d(center_pt, pitch, yaw - math.pi/2, -roll)[0] + np.array(pos)
        
        ax.plot([pos[0], rot_center[0]], [pos[1], rot_center[1]], [pos[2], rot_center[2]], 
                color=color, linestyle='--', linewidth=1, alpha=0.3)

    def _draw_local_axes(self, ax, pos, att):
        """绘制局部坐标轴以清晰显示姿态"""
        roll, pitch, yaw = att
        # 构建旋转矩阵 (NED系: X北, Y东, Z下 -> 转换到绘图系)
        # 注意：anim3d 和 jsbgym 的欧拉角定义可能导致需要微调
        # 这里使用标准航空旋转矩阵构造基向量
        
        # Body frame basis vectors (Unit length)
        # X axis (Nose)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)
        
        # 旋转矩阵 R_body_to_nav
        # x_body in nav
        xb = np.array([cp*cy, cp*sy, -sp])
        # y_body in nav
        yb = np.array([sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp])
        # z_body in nav
        zb = np.array([cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp])
        
        # 绘图坐标系调整 (通常 matplotlib z是向上，而航空 z是向下)
        # 我们需要把 NED 的 Z 反转适配 Plot
        # Plot Frame: X=East, Y=North, Z=Up (假设之前代码是这样映射的)
        # 查看 _get_aircraft_position: x ~ long(East?), y ~ lat(North?)
        # 之前的代码: x=long*cos(lat), y=lat. 
        # 假设 Plot X=East, Y=North, Z=Up.
        
        # 修正映射以匹配视觉直觉:
        # 红色(机头)
        scale = self.AXIS_LENGTH
        
        # 简单的旋转应用到基向量
        def rot(vec):
            # 使用之前的 _rotate3d 逻辑保持一致性，或者重新计算
            # 之前的 _rotate3d 比较复杂，这里简化处理，直接用姿态方向画线
            return self._rotate_vector(vec, roll, pitch, yaw)

        v_nose = rot(np.array([1, 0, 0])) * scale
        v_wing = rot(np.array([0, 1, 0])) * scale
        v_down = rot(np.array([0, 0, 1])) * scale # 实际上是机腹方向
        
        p = np.array(pos)
        
        # X轴 - 红 (机头)
        ax.plot([p[0], p[0]+v_nose[0]], [p[1], p[1]+v_nose[1]], [p[2], p[2]+v_nose[2]], 'r-', lw=2)
        # Y轴 - 绿 (右翼)
        ax.plot([p[0], p[0]+v_wing[0]], [p[1], p[1]+v_wing[1]], [p[2], p[2]+v_wing[2]], 'g-', lw=2)
        # Z轴 - 蓝 (垂直机身向下) - 注意：为了看清翻滚，这个轴很重要
        ax.plot([p[0], p[0]+v_down[0]], [p[1], p[1]+v_down[1]], [p[2], p[2]+v_down[2]], 'b-', lw=2)

    def _rotate_vector(self, vec, roll, pitch, yaw):
        # 简易旋转: Yaw -> Pitch -> Roll
        # 这是一个简化的旋转，具体取决于你的环境坐标系定义
        # 假设: X向前, Y向右, Z向下
        
        # Rotate Z (Yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Rotate Y (Pitch)
        c, s = math.cos(pitch), math.sin(pitch)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        # Rotate X (Roll)
        c, s = math.cos(roll), math.sin(roll)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        
        # Order: Rz * Ry * Rx * vec
        return Rz @ Ry @ Rx @ vec

    def _draw_f16_mesh(self, ax, pos, att, is_opponent):
        # ... (保留原有的 F16 绘制逻辑，微调颜色和透明度) ...
        scale_factor = self.AIRCRAFT_SCALE * 0.3048
        pts = self._scale3d(self.f16_pts, [-scale_factor, scale_factor, scale_factor])
        
        theta = att[1]
        psi = att[2] - math.pi/2
        phi = -att[0]
        
        pts = self._rotate3d(pts, theta, psi, phi)
        pts = pts + np.array(pos)
        
        verts = []
        # 优化：只画每第3个面，提高帧率
        for face in self.f16_faces[::3]: 
            face_pts = []
            for findex in face:
                if findex-1 < len(pts):
                    face_pts.append(tuple(pts[findex-1]))
            if len(face_pts) >= 3:
                verts.append(face_pts)
        
        if is_opponent:
            fc, ec = 'salmon', 'darkred'
        else:
            fc, ec = 'skyblue', 'darkblue'
            
        poly = Poly3DCollection(verts, facecolors=fc, edgecolors=ec, alpha=0.9, linewidths=0.1)
        if is_opponent: poly._opponent_poly = True
        else: poly._aircraft_poly = True
        ax.add_collection3d(poly)

    def _update_hud_info(self, sim, opp_sim):
        # 更新左侧 HUD (我方)
        self.hud_texts_left[0].set_text(">>> OWN AIRCRAFT <<<")
        for i, prop in enumerate(self.print_props):
            try:
                val = sim[prop]
                name = str(prop.name).split('/')[-1]
                self.hud_texts_left[i+1].set_text(f"{name}: {val:.2f}")
            except:
                pass
                
        # 更新右侧 HUD (敌方 - 仅显示基本信息)
        if opp_sim:
            self.hud_texts_right[0].set_text(">>> TARGET <<<")
            try:
                # 获取相对距离等信息需要计算，这里暂时只显示属性
                alt = opp_sim[prp.altitude_sl_ft]
                self.hud_texts_right[1].set_text(f"Altitude: {alt:.0f} ft")
                vel = opp_sim[prp.velocities_vc_fps]
                self.hud_texts_right[2].set_text(f"Speed: {vel:.0f} fps")
                # 计算距离
                p1 = np.array(self._get_aircraft_position(sim))
                p2 = np.array(self._get_aircraft_position(opp_sim))
                dist = np.linalg.norm(p1 - p2)
                self.hud_texts_right[3].set_text(f"Distance: {dist:.0f} m")
            except:
                pass

    def _plot_control_states(self, sim):
        ax_stick = self.axes_control.axes_stick
        ax_thr = self.axes_control.axes_throttle
        ax_rud = self.axes_control.axes_rudder
        
        # === 修改开始：使用兼容性更好的清理方式 ===
        for ax in [ax_stick, ax_thr, ax_rud]:
            # 清除线条 (Line2D)
            # 必须使用 list() 创建副本，因为我们在遍历时修改了原列表
            for line in list(ax.lines):
                line.remove()
            
            # 清除形状/柱状图 (Patch)
            for patch in list(ax.patches):
                patch.remove()
                
            # 清除文字 (Text) - 比如油门的百分比文字
            for text in list(ax.texts):
                text.remove()
                
            # 重新画 Stick 的网格和中心线 (因为刚才被全清除了)
            if ax == ax_stick:
                ax.grid(True, linestyle='--')
                ax.axhline(0, color='black', lw=0.5)
                ax.axvline(0, color='black', lw=0.5)
        # === 修改结束 ===
        
        # 获取值
        try:
            ail = sim[prp.aileron_left]
            ele = sim[prp.elevator]
            thr = sim[prp.throttle]
            rud = sim[prp.rudder]
            
            # 1. Stick: 十字光标
            ax_stick.plot([ail], [ele], 'r+', markersize=20, markeredgewidth=3, label='Current')
            
            # 2. Throttle: 柱状图
            # 背景槽
            ax_thr.bar([0], [1.0], width=0.5, color='gray', alpha=0.2)
            # 当前值
            col = 'red' if thr > 0.9 else 'green'
            ax_thr.bar([0], [thr], width=0.5, color=col, alpha=0.8)
            ax_thr.text(0, 0.5, f"{thr*100:.0f}%", ha='center', va='center', rotation=90, color='black')
            
            # 3. Rudder: 水平条
            ax_rud.barh([0], [2], height=0.5, left=-1, color='gray', alpha=0.2)
            ax_rud.barh([0], [0.1], height=0.8, left=rud-0.05, color='blue') # 指示块
            
        except Exception as e:
            # 打印错误以防万一，但不中断程序
            print(f"Error plotting controls: {e}")
            pass

    def _plot_control_commands(self, sim):
        # 可以在 _plot_control_states 中合并绘制，用蓝色圆圈表示指令
        ax_stick = self.axes_control.axes_stick
        try:
            ail_cmd = sim[prp.aileron_cmd]
            ele_cmd = sim[prp.elevator_cmd]
            ax_stick.plot([ail_cmd], [ele_cmd], 'bo', markersize=10, fillstyle='none', markeredgewidth=2, label='Cmd')
            # ax_stick.legend(loc='upper right', fontsize='small')
        except:
            pass

    # ... (保留原有的 helper functions: _get_aircraft_position, _get_aircraft_attitude, _scale3d, _rotate3d, _draw_ground_grid) ...
    
    def _get_aircraft_position(self, sim: Simulation) -> Tuple[float, float, float]:
        try:
            x = sim[prp.lng_geoc_deg] * 111320 * math.cos(math.radians(sim[prp.lat_geod_deg]))
            y = sim[prp.lat_geod_deg] * 111320
            z = sim[prp.altitude_sl_ft] * 0.3048
            return (x, y, z)
        except:
            return (0, 0, sim[prp.altitude_sl_ft] * 0.3048)
            
    def _get_aircraft_attitude(self, sim: Simulation) -> Tuple[float, float, float]:
        return (sim[prp.roll_rad], sim[prp.pitch_rad], sim[prp.psi_rad])

    def _scale3d(self, pts: np.ndarray, scale_list: List[float]) -> np.ndarray:
        rv = np.zeros(pts.shape)
        for i in range(pts.shape[0]):
            for d in range(3):
                rv[i, d] = scale_list[d] * pts[i, d]
        return rv

    def _rotate3d(self, pts: np.ndarray, theta: float, psi: float, phi: float) -> np.ndarray:
        sinTheta, cosTheta = math.sin(theta), math.cos(theta)
        sinPsi, cosPsi = math.sin(psi), math.cos(psi)
        sinPhi, cosPhi = math.sin(phi), math.cos(phi)

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
        if (self.last_grid_center is None or 
            abs(center_x - self.last_grid_center[0]) > size/4 or 
            abs(center_y - self.last_grid_center[1]) > size/4):
            
            for line in self.grid_lines: line.remove()
            self.grid_lines.clear()
            
            grid_size = size / 6 
            x_start = center_x - size
            y_start = center_y - size
            
            # 使用简单的 grid 逻辑
            for i in range(13):
                x = x_start + i * grid_size
                line = ax.plot([x, x], [y_start, y_start + size*2], [0, 0], 'gray', alpha=0.3, lw=0.5)[0]
                self.grid_lines.append(line)
                
            for i in range(13):
                y = y_start + i * grid_size
                line = ax.plot([x_start, x_start + size*2], [y, y], [0, 0], 'gray', alpha=0.3, lw=0.5)[0]
                self.grid_lines.append(line)
                
            self.last_grid_center = (center_x, center_y)

    def close(self):
        if self.fig_combat:
            plt.close(self.fig_combat)
        if self.fig_control:
            plt.close(self.fig_control)
        self.fig_combat = None
        self.fig_control = None

    def reset(self):
        self.positions = []
        self.attitudes = []
        self.opponent_positions = []
        self.opponent_attitudes = []
        if self.ax_combat:
            self._clean_axes(self.ax_combat)

# Animation 类可以直接继承上面的 Enhanced3DVisualiser，逻辑无需大改
class AnimatedEnhancedVisualiser(Enhanced3DVisualiser):
    def __init__(self, simulation: Simulation, print_props: Tuple[prp.Property]):
        super().__init__(simulation, print_props)
        self.animation_data = []
        self.is_recording = False
        
    def start_recording(self):
        self.is_recording = True
        self.animation_data = []
        
    def stop_recording(self):
        self.is_recording = False
        
    def plot(self, sim: Simulation, opponent_sim: Simulation = None) -> None:
        if self.is_recording:
            # 简单记录数据
            frame_data = {
                'pos1': self._get_aircraft_position(sim),
                'att1': self._get_aircraft_attitude(sim),
                'state_data': {prop: sim[prop] for prop in self.print_props}
            }
            if opponent_sim:
                frame_data['pos2'] = self._get_aircraft_position(opponent_sim)
                frame_data['att2'] = self._get_aircraft_attitude(opponent_sim)
            self.animation_data.append(frame_data)
            
        super().plot(sim, opponent_sim)