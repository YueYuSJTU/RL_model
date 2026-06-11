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
    与你给出的代码保持一致
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
        att: (roll, pitch, yaw)  单位: 弧度
    输出:
        变换后的顶点
    """
    roll, pitch, yaw = att

    # 与你原代码一致的缩放约定
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
        # face 可能是 [i,j,k] 或更长
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
    """
    实心填充绘制飞机
    edge=False 时基本不显示面边界，避免方块感
    """
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
# 方向向量辅助
# =========================
def euler_to_forward_vector(roll, pitch, yaw):
    """
    通过与模型一致的变换方式，求机头方向在世界坐标中的指向。
    假设模型局部机头方向为 +x。
    """
    local_forward = np.array([[1.0, 0.0, 0.0]])
    world_forward = rotate3d(local_forward, pitch, -yaw, -roll)[0]
    return normalize(world_forward)

def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n
def rotation_matrix_from_z(direction):
    """
    构造一个旋转矩阵，将局部 z 轴 [0,0,1] 旋转到目标 direction
    """
    direction = normalize(direction)
    z_axis = np.array([0.0, 0.0, 1.0])
    # 平行
    if np.allclose(direction, z_axis):
        return np.eye(3)
    # 反平行
    if np.allclose(direction, -z_axis):
        return np.array([
            [1, 0,  0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=float)
    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)
    vx = np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ], dtype=float)
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R
def transform_mesh(X, Y, Z, R, t):
    """
    对网格点做旋转和平移
    """
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts = pts @ R.T + t
    X2 = pts[:, 0].reshape(X.shape)
    Y2 = pts[:, 1].reshape(Y.shape)
    Z2 = pts[:, 2].reshape(Z.shape)
    return X2, Y2, Z2
def draw_3d_arrow(
    ax,
    start,
    direction,
    length=1.0,
    shaft_radius=0.03,
    head_radius=0.07,
    head_length_ratio=0.25,
    color='k',
    alpha=1.0,
    resolution=24,
    label=None,
    text_offset=(0, 0, 0)
):
    """
    绘制一个真正的三维实体箭头（圆柱 + 圆锥）
    局部坐标默认沿 +z 方向生成，再旋转到目标 direction
    """
    start = np.array(start, dtype=float)
    direction = normalize(direction)
    head_length = length * head_length_ratio
    shaft_length = length - head_length
    # -------------------------
    # 1. 圆柱箭杆（局部沿 z）
    # -------------------------
    theta = np.linspace(0, 2 * np.pi, resolution)
    z_shaft = np.linspace(0, shaft_length, 2)
    Theta_shaft, Z_shaft = np.meshgrid(theta, z_shaft)
    X_shaft = shaft_radius * np.cos(Theta_shaft)
    Y_shaft = shaft_radius * np.sin(Theta_shaft)
    # -------------------------
    # 2. 圆锥箭头（局部沿 z）
    # -------------------------
    z_head = np.linspace(shaft_length, length, 2)
    Theta_head, Z_head = np.meshgrid(theta, z_head)
    # 半径从 head_radius 线性收缩到 0
    R_head = head_radius * (length - Z_head) / head_length
    X_head = R_head * np.cos(Theta_head)
    Y_head = R_head * np.sin(Theta_head)
    # -------------------------
    # 3. 旋转到目标方向并平移
    # -------------------------
    R = rotation_matrix_from_z(direction)
    Xs, Ys, Zs = transform_mesh(X_shaft, Y_shaft, Z_shaft, R, start)
    Xh, Yh, Zh = transform_mesh(X_head, Y_head, Z_head, R, start)
    # -------------------------
    # 4. 绘制表面
    # -------------------------
    ax.plot_surface(Xs, Ys, Zs, color=color, alpha=alpha, linewidth=0, shade=True)
    ax.plot_surface(Xh, Yh, Zh, color=color, alpha=alpha, linewidth=0, shade=True)
    # -------------------------
    # 5. 加底盖（可选，美观）
    # -------------------------
    # 箭杆底盖
    base_circle = np.stack([
        shaft_radius * np.cos(theta),
        shaft_radius * np.sin(theta),
        np.zeros_like(theta)
    ], axis=1)
    base_circle = base_circle @ R.T + start
    base_center = start
    base_faces = []
    for i in range(len(theta) - 1):
        tri = [base_center, base_circle[i], base_circle[i + 1]]
        base_faces.append(tri)
    poly_base = Poly3DCollection(base_faces, facecolors=color, edgecolors='none', alpha=alpha)
    ax.add_collection3d(poly_base)
    # 箭头底盖（圆锥底）
    head_base_local = np.stack([
        head_radius * np.cos(theta),
        head_radius * np.sin(theta),
        np.full_like(theta, shaft_length)
    ], axis=1)
    head_base_circle = head_base_local @ R.T + start
    head_base_center = np.array([0.0, 0.0, shaft_length]) @ R.T + start
    head_base_faces = []
    for i in range(len(theta) - 1):
        tri = [head_base_center, head_base_circle[i + 1], head_base_circle[i]]
        head_base_faces.append(tri)
    poly_head_base = Poly3DCollection(head_base_faces, facecolors=color, edgecolors='none', alpha=alpha)
    ax.add_collection3d(poly_head_base)
    # -------------------------
    # 6. 文字标注
    # -------------------------
    end = start + direction * length
    if label is not None:
        ax.text(
            end[0] + text_offset[0],
            end[1] + text_offset[1],
            end[2] + text_offset[2],
            label,
            color=color,
            fontsize=11
        )

def draw_arrow(ax, start, direction, length=1.0, color='k', lw=2.0, label=None, text_offset=(0, 0, 0)):
    start = np.array(start, dtype=float)
    direction = normalize(direction) * length

    ax.quiver(
        start[0], start[1], start[2],
        direction[0], direction[1], direction[2],
        color=color,
        linewidth=lw,
        arrow_length_ratio=0.12
    )

    end = start + direction
    if label is not None:
        ax.text(
            end[0] + text_offset[0],
            end[1] + text_offset[1],
            end[2] + text_offset[2],
            label,
            color=color,
            fontsize=11
        )


def set_axes_equal(ax, points):
    """
    让 3D 坐标轴尽量等比例
    """
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
    # 修改为你的 mat 文件路径
    mat_path = '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat'

    data = loadmat(mat_path)
    f16_pts = data['V']
    f16_faces = data['F']

    # -------------------------
    # 场景定义
    # -------------------------
    # 蓝色飞机：追踪者
    blue_pos = np.array([0.0, 0.0, 0.0])
    blue_att = (
        math.radians(5),    # roll
        math.radians(-8),   # pitch
        math.radians(25)    # yaw
    )

    # 红色飞机：规避者
    red_pos = np.array([900.0, 500.0, 220.0])
    red_att = (
        math.radians(20),   # roll
        math.radians(10),   # pitch
        math.radians(70)    # yaw
    )

    # 飞机尺寸
    aircraft_scale = 20

    # -------------------------
    # 创建画布
    # -------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制飞机（实心）
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

    # -------------------------
    # 两机连线
    # -------------------------
    ax.plot(
        [blue_pos[0], red_pos[0]],
        [blue_pos[1], red_pos[1]],
        [blue_pos[2], red_pos[2]],
        linestyle='--',
        color='black',
        linewidth=1.8
    )

    los_blue = normalize(red_pos - blue_pos)   # 蓝机看向红机
    los_red = normalize(blue_pos - red_pos)    # 红机看向蓝机

    mid = (blue_pos + red_pos) / 2
    # ax.text(mid[0], mid[1], mid[2], 'LOS', color='black', fontsize=12)

    # -------------------------
    # 蓝色飞机机头朝向
    # -------------------------
    blue_forward = euler_to_forward_vector(*blue_att)
    draw_3d_arrow(
        ax,
        start=blue_pos,
        direction=blue_forward,
        length=420.0,
        shaft_radius=4.0,
        head_radius=18.0,
        head_length_ratio=0.08,
        color='black',
        alpha=1.0,
        resolution=32,
        label='',
        text_offset=(10, 10, 10)
    )

    draw_angle_arc_3d(
        ax,
        center=blue_pos,
        v1=blue_forward,
        v2=los_blue,
        radius=195.0,
        color='black',
        lw=2.2,
        n=240,
        label='',
        label_offset=(0, 0, 0)
    )

    # -------------------------
    # 红色飞机尾巴朝向
    # 尾巴方向 = 机头方向反向
    # -------------------------
    red_forward = euler_to_forward_vector(*red_att)
    red_tail = -red_forward
    draw_3d_arrow(
        ax,
        start=red_pos,
        direction=red_tail,
        length=420.0,
        shaft_radius=4.0,
        head_radius=18.0,
        head_length_ratio=0.08,
        color='black',
        alpha=1.0,
        resolution=32,
        label='',
        text_offset=(10, 10, 10)
    )

    draw_angle_arc_3d(
        ax,
        center=red_pos,
        v1=red_tail,
        v2=los_red,
        radius=195.0,
        color='black',
        lw=2.2,
        n=240,
        label='',
        label_offset=(0, 0, 0)
    )

    # -------------------------
    # 坐标轴
    # x 正右, y 右上45°, z 正下
    # 通过view_init尽量实现视觉效果
    # -------------------------
    axis_origin = np.array([-250.0, -180.0, -120.0])
    # axis_len = 260.0

    # draw_arrow(ax, axis_origin, [1, 0, 0], length=axis_len, color='black', lw=1.8, label='x')
    # draw_arrow(ax, axis_origin, [0, 1, 0], length=axis_len, color='black', lw=1.8, label='y')
    # draw_arrow(ax, axis_origin, [0, 0, 1], length=axis_len, color='black', lw=1.8, label='z')

    # # -------------------------
    # # 标注飞机位置
    # # -------------------------
    # ax.scatter(*blue_pos, color='royalblue', s=28)
    # ax.scatter(*red_pos, color='crimson', s=28)

    # ax.text(blue_pos[0] - 80, blue_pos[1] - 40, blue_pos[2] - 20, 'Blue aircraft', color='royalblue', fontsize=11)
    # ax.text(red_pos[0] + 20, red_pos[1] + 20, red_pos[2] + 20, 'Red aircraft', color='crimson', fontsize=11)

    # # -------------------------
    # # 坐标轴标签与标题
    # # -------------------------
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Air Combat Geometry')

    # -------------------------
    # 视角设置
    # 尽量让:
    # x -> 向右
    # y -> 右上45度
    # z -> 向下
    # -------------------------
    ax.view_init(elev=18, azim=-65)

        # 去网格
    ax.grid(False)

    # 去掉pane背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # 去掉pane边框
    ax.xaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_edgecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_edgecolor((1, 1, 1, 0))

    # 去掉坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 去掉坐标轴线
    ax.xaxis.line.set_color((1, 1, 1, 0))
    ax.yaxis.line.set_color((1, 1, 1, 0))
    ax.zaxis.line.set_color((1, 1, 1, 0))

    # 去掉坐标轴标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # 整个坐标轴关闭
    ax.set_axis_off()

    # figure背景设为白色/透明都可以
    fig.patch.set_alpha(0.0)   # 透明背景
    ax.set_facecolor((1, 1, 1, 0))

    # 等比例范围
    all_points = np.vstack([
        blue_pts_world,
        red_pts_world,
        blue_pos.reshape(1, 3),
        red_pos.reshape(1, 3),
        axis_origin.reshape(1, 3)
    ])
    set_axes_equal(ax, all_points)

    # 尝试等比例显示
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()

    # 如需保存高分辨率图片，取消下面注释
    # plt.savefig('src/evaluation/plot/air_combat_geometry.png', dpi=600, bbox_inches='tight')
    # plt.savefig('air_combat_geometry.pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()