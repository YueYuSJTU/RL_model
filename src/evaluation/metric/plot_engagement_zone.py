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


# =========================
# 飞机模型处理
# =========================
def transform_f16_model(f16_pts, pos, att, scale=1.0):
    roll, pitch, yaw = att
    pts = scale3d(f16_pts, [-scale, scale, scale])
    pts = rotate3d(pts, pitch, -yaw, -roll)
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


def draw_f16_solid(ax, f16_pts, f16_faces, pos, att,
                   body_color='royalblue', scale=1.0,
                   alpha=1.0, edge=False):
    pts = transform_f16_model(f16_pts, pos, att, scale=scale)
    verts = build_faces_vertices(pts, f16_faces)

    poly = Poly3DCollection(
        verts,
        facecolors=body_color,
        edgecolors='k' if edge else 'none',
        linewidths=0.15 if edge else 0.0,
        alpha=alpha
    )

    ax.add_collection3d(poly)
    return pts


# =========================
# 区域绘制及标注辅助
# =========================
def draw_engagement_zone(ax, pos, att, min_dist=1500.0, max_dist=5000.0, half_angle_deg=15.0, color='red', alpha=0.3):
    roll, pitch, yaw = att
    res_x = 2
    res_theta = 40
    
    x = np.linspace(min_dist, max_dist, res_x)
    theta = np.linspace(0, 2 * np.pi, res_theta)
    X, Theta = np.meshgrid(x, theta)
    
    # 局部坐标系截头圆锥
    R = X * math.tan(math.radians(half_angle_deg))
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)

    # 将局部坐标拼成 Nx3 并变换到世界坐标
    local_pts = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    world_pts = rotate3d(local_pts, pitch, -yaw, -roll) + np.array(pos, dtype=float)

    X_w = world_pts[:, 0].reshape(res_theta, res_x)
    Y_w = world_pts[:, 1].reshape(res_theta, res_x)
    Z_w = world_pts[:, 2].reshape(res_theta, res_x)

    # 画圆锥侧面
    ax.plot_surface(X_w, Y_w, Z_w, color=color, alpha=alpha, edgecolors='none', shade=True)
    
    # 画截头圆面和边缘线
    for i in range(res_x):
        ax.plot(X_w[:, i], Y_w[:, i], Z_w[:, i], color=color, alpha=max(alpha, 0.5), linewidth=1.5)
        
    # 用虚线绘制从 0 到 min_dist 的圆锥部分
    num_dashed_lines = 12
    dash_theta = np.linspace(0, 2 * np.pi, num_dashed_lines, endpoint=False)
    r_min = min_dist * math.tan(math.radians(half_angle_deg))
    
    for t in dash_theta:
        lx = [0, min_dist]
        ly = [0, r_min * math.cos(t)]
        lz = [0, r_min * math.sin(t)]
        l_pts = np.vstack([lx, ly, lz]).T
        w_pts = rotate3d(l_pts, pitch, -yaw, -roll) + np.array(pos, dtype=float)
        ax.plot(w_pts[:, 0], w_pts[:, 1], w_pts[:, 2], color=color, linestyle='--', linewidth=1.2, alpha=0.6)
        
    return world_pts


def draw_annotations(ax, pos, att, min_dist, max_dist, half_angle_deg):
    roll, pitch, yaw = att
    
    # 辅助转换函数，将局部坐标转为世界坐标
    def transform(l_pts):
        return rotate3d(np.asarray(l_pts), pitch, -yaw, -roll) + np.array(pos, dtype=float)
        
    # 1. 中心轴线
    axis_local = [[0, 0, 0], [max_dist * 1.1, 0, 0]]
    axis_w = transform(axis_local)
    ax.plot(axis_w[:,0], axis_w[:,1], axis_w[:,2], color='black', linestyle='-.', linewidth=1)
    
    # 2. 角度标注：轴线与圆锥面上母线的夹角
    r_max = max_dist * math.tan(math.radians(half_angle_deg))
    # 在局部X-Z平面(+Z方向)画一条母线
    surf_local = [[0, 0, 0], [max_dist, 0, r_max]]
    surf_w = transform(surf_local)
    ax.plot(surf_w[:,0], surf_w[:,1], surf_w[:,2], color='black', linestyle=':', linewidth=1.5)
    
    # 画夹角弧线
    arc_radius = max_dist * 0.4
    alphas = np.linspace(0, math.radians(half_angle_deg), 20)
    arc_local = np.zeros((20, 3))
    arc_local[:, 0] = arc_radius * np.cos(alphas)
    arc_local[:, 2] = arc_radius * np.sin(alphas)
    arc_w = transform(arc_local)
    ax.plot(arc_w[:,0], arc_w[:,1], arc_w[:,2], color='black', linewidth=1.5)
    
    # 3. 距离标注 (使用引出线)
    # 向局部坐标的 -Z 方向（机腹下方）引出
    offset_z = -max_dist * 0.25 
    
    # 垂直引出线
    ext_local = [
        [0, 0, 0], [0, 0, offset_z * 1.1],
        [min_dist, 0, 0], [min_dist, 0, offset_z * 0.6],
        [max_dist, 0, 0], [max_dist, 0, offset_z * 1.1]
    ]
    ext_w = transform(ext_local)
    ax.plot(ext_w[0:2,0], ext_w[0:2,1], ext_w[0:2,2], color='black', linestyle='--', linewidth=1)
    ax.plot(ext_w[2:4,0], ext_w[2:4,1], ext_w[2:4,2], color='black', linestyle='--', linewidth=1)
    ax.plot(ext_w[4:6,0], ext_w[4:6,1], ext_w[4:6,2], color='black', linestyle='--', linewidth=1)
    
    # min_dist 横向尺寸线
    min_dim_local = [[0, 0, offset_z * 0.5], [min_dist, 0, offset_z * 0.5]]
    min_dim_w = transform(min_dim_local)
    ax.plot(min_dim_w[:,0], min_dim_w[:,1], min_dim_w[:,2], color='black', linewidth=1.5)
    
    # max_dist 横向尺寸线
    max_dim_local = [[0, 0, offset_z], [max_dist, 0, offset_z]]
    max_dim_w = transform(max_dim_local)
    ax.plot(max_dim_w[:,0], max_dim_w[:,1], max_dim_w[:,2], color='black', linewidth=1.5)


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
# 主函数
# =========================
def main():
    mat_path = '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat'

    data = loadmat(mat_path)
    f16_pts = data['V']
    f16_faces = data['F']

    # 飞机姿态定义
    blue_pos = np.array([0.0, 0.0, 0.0])
    blue_att = (math.radians(0), math.radians(20), math.radians(45))
    aircraft_scale = 100 # 为配合3000距离放大显示
    
    min_dist = 1500.0
    max_dist = 5000.0
    half_angle_deg = 15.0

    # 创建画布
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制蓝色飞机
    blue_pts_world = draw_f16_solid(
        ax, f16_pts, f16_faces,
        pos=blue_pos, att=blue_att,
        body_color='royalblue', scale=aircraft_scale, alpha=1.0
    )

    # 绘制交战区域
    zone_pts_world = draw_engagement_zone(
        ax, 
        pos=blue_pos, 
        att=blue_att, 
        min_dist=min_dist, 
        max_dist=max_dist, 
        half_angle_deg=half_angle_deg, 
        color='crimson', 
        alpha=0.3
    )
    
    # 绘制无字黑色辅助线(夹角、距离)
    draw_annotations(
        ax, 
        pos=blue_pos, 
        att=blue_att, 
        min_dist=min_dist, 
        max_dist=max_dist, 
        half_angle_deg=half_angle_deg
    )

    # 视角与美化设置
    ax.view_init(elev=20, azim=-50)
    ax.grid(False)
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

    all_points = np.vstack([blue_pts_world, zone_pts_world, blue_pos.reshape(1, 3)])
    set_axes_equal(ax, all_points)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()