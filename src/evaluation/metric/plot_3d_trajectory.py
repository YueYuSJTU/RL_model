import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat

sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")

from stable_baselines3.common.vec_env import DummyVecEnv
from src.agents.make_agent import load_agent
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from src.agents.model_wrapper import ObsAdaptingModel

def scale3d(pts: np.ndarray, scale_list: list) -> np.ndarray:
    rv = np.zeros(pts.shape)
    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]
    return rv

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

    rv = np.zeros(pts.shape)
    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)
    return rv

def draw_f16(ax, f16_pts, f16_faces, pos, att, is_opponent=False, scale=30.0):
    """在指定位置绘制F-16实体姿态模型"""
    roll, pitch, yaw = att
    
    # 缩放 (原模型可能较大，根据需要缩放以适应轨迹尺度)
    pts = scale3d(f16_pts, [-scale, scale, scale])
    
    # 根据 enhanced_visualiser.py 中的规则转换欧拉角
    theta = pitch
    # psi = yaw - math.pi / 2
    psi = -yaw
    phi = -roll
    
    # 旋转与平移
    pts = rotate3d(pts, theta, psi, phi)
    pts = pts + np.array(pos)
    
    verts = []
    # 可以通过切片 [::2] 或 [::3] 来降低多边形数量加快渲染，如果不卡则全部渲染
    for face in f16_faces[::10]: 
        face_pts = []
        for findex in face:
            if findex - 1 < len(pts):
                face_pts.append(tuple(pts[findex - 1]))
        if len(face_pts) >= 3:
            verts.append(face_pts)
            
    fc, ec = ('salmon', 'darkred') if is_opponent else ('skyblue', 'darkblue')
    poly = Poly3DCollection(verts, facecolors=fc, edgecolors=ec, alpha=0.9, linewidths=0.1)
    ax.add_collection3d(poly)

def main():
    # 定义待评估的两个模型路径
    model1_path = "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35"
    # model2_path = "experiments/20250616_221656/stage2/20250616_221824_TrackingTask_ppo_1layer1"
    model2_path = "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30"
    
    # 获取并加载配置
    env1_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))
    env1_cfg["render_mode"] = None
    env1_cfg["use_vec_normalize"] = False
    
    agent1_cfg = load_config(os.path.join(model1_path, "agent_config.yaml"))
    agent2_cfg = load_config(os.path.join(model2_path, "agent_config.yaml"))
    
    # 建立评测环境
    vec_env = create_env(env1_cfg, training=False, vec_env_cls=DummyVecEnv)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.env_method("update_task_parameters", goal_point_prob=0.0)
    
    # 加载两个模型
    fake_env1 = create_env(env1_cfg, training=False, vec_env_kwargs=None)
    model1 = load_agent(fake_env1, agent1_cfg.get("algorithm", "PPO"), os.path.join(model1_path, "best_model"), agent1_cfg["device"], agent1_cfg)
    model1 = ObsAdaptingModel(model1, env1_cfg)
    
    env2_cfg = load_config(os.path.join(model2_path, "env_config.yaml"))
    fake_env2 = create_env(env2_cfg, training=False, vec_env_kwargs=None)
    model2 = load_agent(fake_env2, agent2_cfg.get("algorithm", "PPO"), os.path.join(model2_path, "best_model"), agent2_cfg["device"], agent2_cfg)
    model2 = ObsAdaptingModel(model2, env2_cfg)

    print("开始执行对战评测并记录历史轨迹...")
    obs = vec_env.reset()
    obs_length = obs.shape[1]
    
    trajectory_self = []
    trajectory_oppo = []
    
    episode_done = False
    state1 = None
    episode_start1 = np.ones((vec_env.num_envs,), dtype=bool)
    
    while not episode_done:
        try:
            action1, state1 = model1.predict(obs[:, :obs_length//2], state=state1, episode_start=episode_start1, deterministic=True)
        except TypeError:
            action1, _ = model1.predict(obs[:, :obs_length//2], deterministic=True)
            
        action2, _ = model2.predict(obs[:, obs_length//2:], deterministic=True)
        combined_action = np.concatenate([action1, action2], axis=-1)
        
        obs, reward, dones, info_list = vec_env.step(combined_action)
        episode_start1 = dones
        
        # 提取轨迹数据
        info = info_list[0]
        if "trajectory" in info:
            trajectory_self.append(info["trajectory"]["self"])
            trajectory_oppo.append(info["trajectory"]["oppo"])
        
        if dones[0]:
            episode_done = True
            
    vec_env.close()
    print(f"数据记录完成，总计记录了 {len(trajectory_self)} 步，开始绘制 3D 轨迹...")
    
    # 加载 F-16 3D 模型
    mat_path = '/home/ubuntu/Workfile/RL/RL_model/archive/aerobench/visualize/f-16.mat'
    try:
        data = loadmat(mat_path)
        f16_pts = data['V']
        f16_faces = data['F']
    except Exception as e:
        print(f"无法加载模型文件 {mat_path}: {e}")
        return

    # === 绘制过程 ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 抽取坐标系数据
    xs_self = [p["x_ft"] for p in trajectory_self]
    ys_self = [p["y_ft"] for p in trajectory_self]
    zs_self = [p["z_ft"] for p in trajectory_self]
    
    xs_oppo = [p["x_ft"] for p in trajectory_oppo]
    ys_oppo = [p["y_ft"] for p in trajectory_oppo]
    zs_oppo = [p["z_ft"] for p in trajectory_oppo]
    
    # 绘制连续轨迹 (主控蓝色，对手红色)
    ax.plot(xs_self, ys_self, zs_self, 'b-', label='Ours-LSTM', alpha=0.6)
    ax.plot(xs_oppo, ys_oppo, zs_oppo, 'r-', label='Ours-MLP', alpha=0.6)

    # 绘制初始点
    if len(trajectory_self) > 0:
        p_s = trajectory_self[0]
        pos_s = (p_s["x_ft"], p_s["y_ft"], p_s["z_ft"])
        p_o = trajectory_oppo[0]
        pos_o = (p_o["x_ft"], p_o["y_ft"], p_o["z_ft"])
        ax.scatter(*pos_s, color='blue', marker='o', s=50, label="LSTM Start")
        ax.scatter(*pos_o, color='red', marker='o', s=50, label="MLP Start")
    
    # 每隔一段距离或固定步长，绘制一次飞机姿态 (例如每50步)
    draw_interval = 70
    for idx in range(0, len(trajectory_self), draw_interval):
        p_s = trajectory_self[idx]
        pos_s = (p_s["x_ft"], p_s["y_ft"], p_s["z_ft"])
        att_s = (p_s["roll_rad"], p_s["pitch_rad"], math.radians(p_s["yaw_deg"]))
        draw_f16(ax, f16_pts, f16_faces, pos_s, att_s, is_opponent=False, scale=60.0)
        
        p_o = trajectory_oppo[idx]
        pos_o = (p_o["x_ft"], p_o["y_ft"], p_o["z_ft"])
        att_o = (p_o["roll_rad"], p_o["pitch_rad"], math.radians(p_o["yaw_deg"]))
        draw_f16(ax, f16_pts, f16_faces, pos_o, att_o, is_opponent=True, scale=60.0)

    # # 绘制始末点标志
    # if len(trajectory_self) > 0:
    #     ax.scatter(*pos_s, color='blue', marker='o', s=50, label="Self End")
    #     ax.scatter(*pos_o, color='red', marker='o', s=50, label="Oppo End")

    ax.set_xlabel('X Position (ft)')
    ax.set_ylabel('Y Position (ft)')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title("3D Combat Trajectory with Attitudes")
    ax.legend()
    
    # 强制等比例显示
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass 
        
    plt.show()

if __name__ == "__main__":
    main()
