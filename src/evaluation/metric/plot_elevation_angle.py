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
    """
    与原代码保持一致
    theta: pitch
    psi: yaw相关变换
    phi: roll相关变换
    """
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
    """
    输入:
        f16_pts: 原始顶点
        pos: (x, y, z)
        att: (roll, pitch, yaw)，单位: 弧度
    输出:
        变换后的顶点
    """
    roll, pitch, yaw = att

    pts = scale3d(f16_pts, [-scale, scale, scale])

    theta = pitch
    psi = -yaw
    phi = -roll

    pts = rotate3d(pts, theta, psi, phi)
    pts = pts + np.array(pos, dtype=float)
    return pts


def build_faces_vertices(pts, f16_faces):
    """
    将 matlab 中的面索引转换成 Poly3DCollection 所需顶点列表
    假定 f16_faces 是 1-based 索引
    """
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


def draw_f16_solid(ax, f16_pts, f16_faces, pos, att,
                   body_color='royalblue', scale=1.0,
                   alpha=1.0, edge=False):
    pts = transform_f16_model(f16_pts, pos, att, scale=scale)
    verts = build_faces_vertices(pts, f16_faces)

    if edge:
        poly = Poly3DCollection(
            verts,
            facecolors=body_color,
            edgecolors='k',
            linewidths=0.15,
            alpha=alpha
        )
    else:
        poly = Poly3DCollection(
            verts,
            facecolors=body_color,
            edgecolors='none',
            linewidths=0.0,
            alpha=alpha
        )

    ax.add_collection3d(poly)
    return pts


# =========================
# 姿态辅助
# =========================
def euler_to_forward_vector(roll, pitch, yaw):
    """
    假设模型局部机头方向为 +x
    """
    local_forward = np.array([[1.0, 0.0, 0.0]])
    world_forward = rotate3d(local_forward, pitch, -yaw, -roll)[0]
    return normalize(world_forward)


# =========================
# 半球构造
# =========================
def sample_point_on_upper_hemisphere(center, radius, az_deg=40, el_deg=35):
    """
    在以上半球(z>=center_z)上采样一个点
    az_deg: 水平角（绕z轴）
    el_deg: 仰角，0表示在赤道圆上，90表示在最顶点
    """
    az = math.radians(az_deg)
    el = math.radians(el_deg)

    x = radius * math.cos(el) * math.cos(az)
    y = radius * math.cos(el) * math.sin(az)
    z = radius * math.sin(el)

    return np.array(center, dtype=float) + np.array([x, y, z], dtype=float)


def draw_upper_hemisphere(ax, center, radius,
                          color='deepskyblue',
                          alpha=0.18,
                          resolution_u=80,
                          resolution_v=40,
                          wire=False):
    """
    绘制以上半球
    """
    center = np.asarray(center, dtype=float)

    phi = np.linspace(0, 2 * np.pi, resolution_u)
    theta = np.linspace(0, np.pi / 2, resolution_v)

    Phi, Theta = np.meshgrid(phi, theta)

    X = radius * np.sin(Theta) * np.cos(Phi) + center[0]
    Y = radius * np.sin(Theta) * np.sin(Phi) + center[1]
    Z = radius * np.cos(Theta) + center[2]

    if wire:
        ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.6, alpha=alpha)
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=True)

    return X, Y, Z


def draw_base_circle(ax, center, radius, color='deepskyblue', lw=1.5, alpha=0.8):
    t = np.linspace(0, 2 * np.pi, 300)
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    z = np.full_like(t, center[2])
    ax.plot(x, y, z, color=color, linewidth=lw, alpha=alpha)


# =========================
# 几何关系辅助
# =========================
def line_sphere_intersection_from_center(center, radius, direction):
    """
    直线: p = center + t * direction
    因为直线从球心出发，与球面的正向交点就是:
        center + radius * normalize(direction)
    """
    direction = normalize(direction)
    return np.asarray(center, dtype=float) + radius * direction


def vertical_foot_on_base_plane(point, base_z):
    """
    从点沿 -z 方向向下投影到平面 z = base_z
    """
    point = np.asarray(point, dtype=float)
    return np.array([point[0], point[1], base_z], dtype=float)


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

def draw_angle_arc_3d(ax, center, v1, v2, radius=80.0, color='k', lw=2.0,
                      n=100, label=None, label_offset=(0, 0, 0)):
    """
    在3D中，以 center 为顶点，在 v1 和 v2 之间画一个角度弧线。
    弧线位于由 v1、v2 张成的平面内。

    参数:
        center: 角顶点
        v1, v2: 两个方向向量
        radius: 弧线半径
        color: 颜色
        lw: 线宽
        n: 弧线采样点数
        label: 角度文字，如 r'$\lambda$'
        label_offset: 文本偏移
    """
    center = np.array(center, dtype=float)
    a = normalize(v1)
    b = normalize(v2)

    # 防止退化
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return

    # 法向量
    normal = np.cross(a, b)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1e-12:
        return
    normal = normal / normal_norm

    # 构造平面内正交基
    e1 = a
    e2 = np.cross(normal, e1)
    e2 = normalize(e2)

    # 计算从 e1 转到 b 的夹角
    x = np.dot(b, e1)
    y = np.dot(b, e2)
    angle = math.atan2(y, x)

    # 为了总是画较小夹角
    if angle < 0:
        angle += 2 * math.pi
    if angle > math.pi:
        angle = angle - 2 * math.pi

    ts = np.linspace(0, angle, n)

    arc = np.array([
        center + radius * (math.cos(t) * e1 + math.sin(t) * e2)
        for t in ts
    ])

    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color, linewidth=lw)

    if label is not None:
        tm = angle / 2
        p_text = center + radius * 1.18 * (math.cos(tm) * e1 + math.sin(tm) * e2)
        ax.text(
            p_text[0] + label_offset[0],
            p_text[1] + label_offset[1],
            p_text[2] + label_offset[2],
            label,
            color=color,
            fontsize=13
        )

# =========================
# 主函数
# =========================
def main():
    mat_path = '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat'

    data = loadmat(mat_path)
    f16_pts = data['V']
    f16_faces = data['F']

    # -------------------------
    # 场景定义
    # -------------------------
    # 蓝色飞机：球心
    blue_pos = np.array([0.0, 0.0, 0.0])

    # 蓝色飞机与底面平行：roll=0, pitch=0
    # yaw 可自由设置，表示它在底面内朝向哪个方向
    blue_att = (
        math.radians(0),     # roll
        math.radians(0),     # pitch
        math.radians(25)     # yaw
    )

    hemisphere_radius = 1000.0

    # 红色飞机放在上半球面上
    red_pos = sample_point_on_upper_hemisphere(
        center=blue_pos,
        radius=hemisphere_radius,
        az_deg=60,
        el_deg=38
    )

    red_att = (
        math.radians(20),
        math.radians(10),
        math.radians(70)
    )

    aircraft_scale = 20

    # -------------------------
    # 几何关系
    # -------------------------
    # 蓝色飞机 nose direction 与半球面交点
    blue_forward = euler_to_forward_vector(*blue_att)
    blue_nose_hit = line_sphere_intersection_from_center(
        center=blue_pos,
        radius=hemisphere_radius,
        direction=blue_forward
    )

    # 红色飞机向下做垂线，垂足在底面 z = blue_pos[2]
    foot_pos = vertical_foot_on_base_plane(red_pos, blue_pos[2])

    # a: 蓝色飞机到垂足的连线方向
    a_vec = foot_pos - blue_pos

    # 两飞机连线方向（从蓝机指向红机）
    los_vec = red_pos - blue_pos

    # -------------------------
    # 创建画布
    # -------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------
    # 绘制半球与底面圆
    # -------------------------
    draw_upper_hemisphere(
        ax,
        center=blue_pos,
        radius=hemisphere_radius,
        color='lightskyblue',
        alpha=0.16,
        resolution_u=90,
        resolution_v=45,
        wire=False
    )

    draw_base_circle(
        ax,
        center=blue_pos,
        radius=hemisphere_radius,
        color='deepskyblue',
        lw=1.5,
        alpha=0.9
    )

    # -------------------------
    # 绘制飞机
    # -------------------------
    blue_pts_world = draw_f16_solid(
        ax, f16_pts, f16_faces,
        pos=blue_pos,
        att=blue_att,
        body_color='royalblue',
        scale=aircraft_scale,
        alpha=1.0,
        edge=False
    )

    red_pts_world = draw_f16_solid(
        ax, f16_pts, f16_faces,
        pos=red_pos,
        att=red_att,
        body_color='crimson',
        scale=aircraft_scale,
        alpha=1.0,
        edge=False
    )

    ax.plot(
        [blue_pos[0], red_pos[0]],
        [blue_pos[1], red_pos[1]],
        [blue_pos[2], red_pos[2]],
        linestyle='--',
        color='black',
        linewidth=1.8
    )

    # -------------------------
    # 蓝色飞机 nose direction:
    # 改为虚线并延长到球面
    # -------------------------
    ax.plot(
        [blue_pos[0], blue_nose_hit[0]],
        [blue_pos[1], blue_nose_hit[1]],
        [blue_pos[2], blue_nose_hit[2]],
        linestyle='--',
        color='black',
        linewidth=2.0
    )

    # -------------------------
    # 红色飞机向下垂线
    # -------------------------
    ax.plot(
        [red_pos[0], foot_pos[0]],
        [red_pos[1], foot_pos[1]],
        [red_pos[2], foot_pos[2]],
        linestyle='--',
        color='black',
        linewidth=1.8
    )

    # -------------------------
    # 连接蓝色飞机和垂足
    # -------------------------
    ax.plot(
        [blue_pos[0], foot_pos[0]],
        [blue_pos[1], foot_pos[1]],
        [blue_pos[2], foot_pos[2]],
        linestyle='-',
        color='black',
        linewidth=1.8
    )

    # -------------------------
    # 角度弧线
    # 1) a 与 blue nose direction
    # 2) a 与 两飞机连线
    # -------------------------
    draw_angle_arc_3d(
        ax,
        center=blue_pos,
        v1=a_vec,
        v2=blue_forward,
        radius=255.0,
        color='black',
        lw=2.0,
        n=240,
        label='',
        label_offset=(0, 0, 0)
    )

    draw_angle_arc_3d(
        ax,
        center=blue_pos,
        v1=a_vec,
        v2=los_vec,
        radius=185.0,
        color='black',
        lw=2.0,
        n=240,
        label='',
        label_offset=(0, 0, 0)
    )

    # 可选：标记关键点
    ax.scatter(*blue_pos, color='royalblue', s=35)
    ax.scatter(*red_pos, color='crimson', s=35)
    ax.scatter(*foot_pos, color='black', s=20)

    # -------------------------
    # 视角设置
    # -------------------------
    ax.view_init(elev=18, azim=-65)

    # -------------------------
    # 美化
    # -------------------------
    ax.grid(False)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_axis_off()

    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    # -------------------------
    # 等比例范围
    # -------------------------
    all_points = np.vstack([
        blue_pts_world,
        red_pts_world,
        blue_pos.reshape(1, 3),
        red_pos.reshape(1, 3),
        foot_pos.reshape(1, 3),
        blue_nose_hit.reshape(1, 3),
        np.array([
            blue_pos + [hemisphere_radius, 0, 0],
            blue_pos + [-hemisphere_radius, 0, 0],
            blue_pos + [0, hemisphere_radius, 0],
            blue_pos + [0, -hemisphere_radius, 0],
            blue_pos + [0, 0, hemisphere_radius]
        ])
    ])
    set_axes_equal(ax, all_points)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()