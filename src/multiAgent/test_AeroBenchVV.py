import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import aerobench            # type: ignore
sys.path.append("/home/ubuntu/Workfile/RL/jsbgym")
import jsbgym_m             # type: ignore
import gymnasium as gym
import numpy as np

# make waypoint list
waypoints = [[-5000, -7500, 5000],
                [-15000, -7500, 5000-500],
                [-15000, 5000, 5000-200]]
ap = aerobench.waypoint_autopilot.WaypointAutopilot(waypoints)

# 适配器函数
def adapt_autopilot_to_new_simulator(autopilot, simulator_state):
    # 转换状态格式
    x_f16 = convert_state_format(simulator_state)
    
    # 获取高级控制命令
    Nz, ps, Ny_r, throttle = autopilot.get_u_ref(0, x_f16)
    
    # 使用低级控制器转换为舵面命令
    llc = autopilot.llc
    _, u_deg = llc.get_u_deg([Nz, ps, Ny_r, throttle], x_f16)
    
    # 返回舵面命令
    return u_deg[2], u_deg[1], u_deg[3], u_deg[0]  # 副翼，升降舵,方向舵,油门

def convert_state_format(simulator_state):
    VT = 0
    
    ALPHA = 1
    BETA = 2
    PHI = 3 # roll angle
    THETA = 4 # pitch angle
    PSI = 5 # yaw angle
    
    P = 6
    Q = 7
    R = 8
    
    POSN = 9
    POS_N = 9
    
    POSE = 10
    POS_E = 10
    
    ALT = 11
    H = 11
    
    POW = 12
    
    # 新增积分器状态 (索引13, 14, 15)
    INT_E_NZ = 13    # 垂直加速度积分误差
    INT_E_PS = 14    # 稳定轴滚转率积分误差
    INT_E_NY_R = 15  # 侧向加速度积分误差

    # 创建一个16元素的状态向量
    x_f16 = np.zeros(16)
    x_f16[VT] = simulator_state[27]
    x_f16[ALPHA] = np.deg2rad(simulator_state[16])
    x_f16[BETA] = np.deg2rad(simulator_state[17])
    x_f16[PHI] = simulator_state[2]
    x_f16[THETA] = simulator_state[1]
    x_f16[PSI] = np.deg2rad(simulator_state[9])
    x_f16[P] = simulator_state[6]
    x_f16[Q] = simulator_state[7]
    x_f16[R] = simulator_state[8]
    x_f16[POS_N] = simulator_state[58]
    x_f16[POS_E] = simulator_state[59]
    x_f16[H] = simulator_state[0]
    x_f16[POW] = int(simulator_state[24] * 5)
    
    # 积分器状态初始化为0
    x_f16[INT_E_NZ] = 0.0
    x_f16[INT_E_PS] = 0.0
    x_f16[INT_E_NY_R] = 0.0
    
    return x_f16

# plane = "C172"
plane = "F16"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
# task = "TrajectoryTask"
task = "TrackingTask"

# shape = "Shaping.STANDARD"
# shape = "Shaping.EXTRA"
shape = "stage1"

# render_mode = "flightgear"
# render_mode = None
# render_mode = "graph"
render_mode = "human"


ENV_ID = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
# ENV_ID = "C172-HeadingControlTask-Shaping.EXTRA-FG-v0"
env = gym.make(ENV_ID, render_mode=render_mode)
# env = gym.make(ENV_ID)

action1 = np.array([0.5, 0.06, 0.0, 0.4])
action2 = np.array([-0.5, 0.06, 0.0, 0.4])

env.reset()
# action = env.action_space.sample()
action = np.concatenate((action1, action2))
observation, reward, terminated, truncated, info = env.step(action)
while not (terminated or truncated):
    if render_mode is not None:
        env.render()   
    observation, reward, terminated, truncated, info = env.step(action)
    action1 = adapt_autopilot_to_new_simulator(ap, observation)
    # action = env.action_space.sample()
    action = np.concatenate((action1, action2))
    # print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()