import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================
# 几何变换函数
# =========================
def scale3d(pts: np.ndarray, scale_list) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    scale_arr = np.array(scale_list, dtype=float).reshape(1, 3)
    return pts * scale_arr

def rotate3d(pts: np.ndarray, theta: float, psi: float, phi: float) -> np.ndarray:
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

    return pts @ transform_matrix

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


# =========================
# 飞机模型处理
# =========================
def transform_f16_model(f16_pts, pos, att, scale=1.0):
    roll, pitch, yaw = att
    pts = scale3d(f16_pts, [-scale, scale, scale])
    theta, psi, phi = pitch, -yaw, -roll
    pts = rotate3d(pts, theta, psi, phi)
    pts = pts + np.array(pos, dtype=float)
    return pts

def build_faces_vertices(pts, f16_faces):
    verts = []
    for face in f16_faces:
        face_pts = []
        for idx in np.ravel(face):
            ii = int(idx) - 1
            if 0 <= ii < len(pts):
                face_pts.append(pts[ii])
        if len(face_pts) >= 3:
            verts.append(face_pts)
    return verts

def draw_f16_solid(ax, f16_pts, f16_faces, pos, att, body_color, scale=1.0, alpha=1.0):
    pts = transform_f16_model(f16_pts, pos, att, scale=scale)
    verts = build_faces_vertices(pts, f16_faces)
    poly = Poly3DCollection(verts, facecolors=body_color, edgecolors='none', linewidths=0.0, alpha=alpha)
    ax.add_collection3d(poly)
    return pts


# =========================
# 轨迹与箭头辅助
# =========================
def get_cubic_bezier_curve(p0, p3, offset1, offset2, n_points=50):
    """生成 3D 三次贝塞尔曲线，使用两个控制点让曲线出现两个弯折"""
    p0, p3 = np.array(p0, dtype=float), np.array(p3, dtype=float)
    
    # 按照起点到终点的方向等分，加入不同的偏置形成两个控制点
    v = p3 - p0
    p1 = p0 + v * (1.0 / 3.0) + np.array(offset1, dtype=float)
    p2 = p0 + v * (2.0 / 3.0) + np.array(offset2, dtype=float)
    
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 3))
    for i, tv in enumerate(t):
        u = 1 - tv
        curve[i] = (u**3 * p0 
                    + 3 * u**2 * tv * p1 
                    + 3 * u * tv**2 * p2 
                    + tv**3 * p3)
    return curve

def draw_cone_arrow(ax, tip, direction, color, radius=30.0, height=80.0, resolution=16):
    """绘制 3D 圆锥箭头"""
    tip = np.array(tip, dtype=float)
    dir_vec = normalize(direction)
    
    # 构造局部坐标系
    up = np.array([0, 0, 1])
    if np.abs(np.dot(dir_vec, up)) > 0.99:
        up = np.array([1, 0, 0])
    x_axis = normalize(np.cross(up, dir_vec))
    y_axis = normalize(np.cross(dir_vec, x_axis))
    
    # 底面中心 (圆锥底面在后方)
    base_center = tip - dir_vec * height
    
    # 计算底面圆周点
    theta = np.linspace(0, 2 * np.pi, resolution)
    verts = []
    
    circle_pts = [base_center + radius * (np.cos(t) * x_axis + np.sin(t) * y_axis) for t in theta]
    
    # 连接到底面和顶点形成三角形面
    for i in range(resolution - 1):
        verts.append([tip, circle_pts[i], circle_pts[i+1]])
    verts.append([tip, circle_pts[-1], circle_pts[0]])
    
    poly = Poly3DCollection(verts, facecolors=color, edgecolors='none', alpha=0.9)
    ax.add_collection3d(poly)


def set_axes_equal(ax, points):
    points = np.asarray(points)
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2
    max_range = max(max_range, 1.0)

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)


# =========================
# 绘图逻辑封装
# =========================
def plot_scenario(f16_pts, f16_faces, blue_pos, blue_att, red_pos, red_att, green_target_pos, aircraft_scale=15, title=""):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if title:
        ax.set_title(title, fontsize=16)

    # 绘制蓝色飞机及轨迹
    blue_pts_world = draw_f16_solid(ax, f16_pts, f16_faces, blue_pos, blue_att, 'royalblue', scale=aircraft_scale)
    ax.plot([], [], [], marker='s', color='none', markerfacecolor='royalblue', markersize=10, linestyle='None', label='ego aircraft')

    curve_b2r = get_cubic_bezier_curve(blue_pos, red_pos, offset1=[-150, 200, 100], offset2=[150, -100, -50], n_points=80)
    ax.plot(curve_b2r[:, 0], curve_b2r[:, 1], curve_b2r[:, 2], linestyle='--', color='royalblue', linewidth=2.5)
    
    # # 蓝色箭头
    # b2r_dir = curve_b2r[-1] - curve_b2r[-2]
    # arrow_tip_b2r = red_pos - normalize(b2r_dir) * 40.0
    # draw_cone_arrow(ax, arrow_tip_b2r, b2r_dir, color='royalblue')

    # 绘制红色飞机及轨迹
    red_pts_world = draw_f16_solid(ax, f16_pts, f16_faces, red_pos, red_att, 'crimson', scale=aircraft_scale)
    ax.plot([], [], [], marker='s', color='none', markerfacecolor='crimson', markersize=10, linestyle='None', label='opponent aircraft')

    curve_r2g = get_cubic_bezier_curve(red_pos, green_target_pos, offset1=[-100, 150, 50], offset2=[100, -150, -100], n_points=80)
    ax.plot(curve_r2g[:, 0], curve_r2g[:, 1], curve_r2g[:, 2], linestyle='--', color='crimson', linewidth=2.5)
    
    # # 红色箭头
    # r2g_dir = curve_r2g[-1] - curve_r2g[-2]
    # arrow_tip_r2g = green_target_pos - normalize(r2g_dir) * 10.0
    # draw_cone_arrow(ax, arrow_tip_r2g, r2g_dir, color='crimson')

    # 绘制绿色目标点，并将其设在最上层
    ax.scatter(*green_target_pos, color='limegreen', s=500, marker='*', label='goal point', zorder=100)

    # 图例设置
    ax.legend(loc='upper right', fontsize=12, frameon=True, facecolor='white', edgecolor='black')

    # 视角设置及美化
    ax.view_init(elev=25, azim=-55)
    ax.grid(False)
    ax.set_axis_off()

    all_points = np.vstack([
        blue_pts_world,
        red_pts_world,
        green_target_pos.reshape(1, 3),
        curve_b2r,
        curve_r2g
    ])
    set_axes_equal(ax, all_points)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()


# =========================
# 主函数
# =========================
def main():
    mat_path = '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat'
    try:
        data = loadmat(mat_path)
        f16_pts = data['V']
        f16_faces = data['F']
    except FileNotFoundError:
        # 如果没有模型文件，提供备用的空占位逻辑以防报错
        f16_pts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
        f16_faces = [[1,2,3], [1,2,4]]

    aircraft_scale = 15

    # -------------------------
    # 场景一：正常追逐
    # -------------------------
    blue_pos_1 = np.array([0.0, 0.0, 0.0])
    blue_att_1 = (math.radians(0), math.radians(20), math.radians(45))
    
    red_pos_1 = np.array([400.0, 500.0, 300.0])
    red_att_1 = (math.radians(10), math.radians(-10), math.radians(60))
    
    green_target_pos_1 = np.array([900.0, 800.0, 100.0])

    plot_scenario(f16_pts, f16_faces, blue_pos_1, blue_att_1, red_pos_1, red_att_1, green_target_pos_1, aircraft_scale, "Scenario 1: Chase")

    # -------------------------
    # 场景二：互相追逐，目标点与蓝机重合
    # -------------------------
    blue_pos_2 = np.array([0.0, 0.0, 0.0])
    blue_att_2 = (math.radians(0), math.radians(10), math.radians(45))
    
    red_pos_2 = np.array([600.0, 600.0, 200.0])
    # 让红机朝向蓝机的方向，形成互相追逐
    red_att_2 = (math.radians(-20), math.radians(-15), math.radians(225))
    
    # 目标点与蓝机重合：将其在Z轴稍微抬高一段距离，以便其能凸出机顶显示，而不被掩埋在飞机内部
    green_target_pos_2 = blue_pos_2.copy() + np.array([0.0, 0.0, 20.0])

    plot_scenario(f16_pts, f16_faces, blue_pos_2, blue_att_2, red_pos_2, red_att_2, green_target_pos_2, aircraft_scale, "Scenario 2: Mutual Chase")

    plt.show()

if __name__ == "__main__":
    main()
