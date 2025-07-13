import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import aerobench            # type: ignore
# sys.path.append("/home/ubuntu/Workfile/RL/jsbgym")
import jsb_env.jsbgym_m             # type: ignore
import gymnasium as gym
import numpy as np

# make waypoint list
waypoints = [[-5000, -7500, 5000],
                [-15000, -7500, 5000-500],
                [-15000, 5000, 5000-200]]
ap = aerobench.waypoint_autopilot.WaypointAutopilot(waypoints)

# # 适配器函数
# def adapt_autopilot_to_new_simulator(autopilot, simulator_state):
#     # 转换状态格式
#     x_f16 = convert_state_format(simulator_state)
    
#     # 获取高级控制命令
#     Nz, ps, Ny_r, throttle = autopilot.get_u_ref(0, x_f16)
    
#     # 使用低级控制器转换为舵面命令
#     llc = autopilot.llc
#     _, u_deg = llc.get_u_deg([Nz, ps, Ny_r, throttle], x_f16)
    
#     # 返回舵面命令
#     return u_deg[2], u_deg[1], u_deg[3], u_deg[0]  # 副翼，升降舵,方向舵,油门

def convert_basic_state_format(simulator_state):
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
    x_f16 = np.zeros(13)
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
    
    # # 积分器状态初始化为0
    # x_f16[INT_E_NZ] = 0.0
    # x_f16[INT_E_PS] = 0.0
    # x_f16[INT_E_NY_R] = 0.0
    
    return x_f16

class IntegratorStateTracker:
    def __init__(self):
        # 初始化积分器状态
        self.int_e_nz = 0.0
        self.int_e_ps = 0.0
        self.int_e_ny_r = 0.0
        self.prev_time = None
        
    def update(self, t, x_f16, u_ref, Nz, ps, Ny_r):
        """更新积分器状态"""
        # 首次调用时初始化时间
        if self.prev_time is None:
            self.prev_time = t
            return
            
        # 计算时间步长
        dt = t - self.prev_time
        self.prev_time = t
        
        # 计算误差导数
        derivatives = [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]
        
        # 使用欧拉法更新积分器状态
        self.int_e_nz += derivatives[0] * dt
        self.int_e_ps += derivatives[1] * dt
        self.int_e_ny_r += derivatives[2] * dt
        
    def get_states(self):
        """获取当前积分器状态"""
        return self.int_e_nz, self.int_e_ps, self.int_e_ny_r

# 创建积分器状态追踪器实例
integrator_tracker = IntegratorStateTracker()
current_time = 0.0  # 模拟时间，每步增加仿真步长

def adapt_autopilot_to_new_simulator(autopilot, simulator_state):
    global current_time
    
    # 转换状态格式，但不包括积分器状态
    x_f16_basic = convert_basic_state_format(simulator_state)
    
    # 获取高级控制命令
    Nz, ps, Ny_r, throttle = autopilot.get_u_ref(current_time, x_f16_basic)
    
    # 更新积分器状态
    integrator_tracker.update(current_time, x_f16_basic, [Nz, ps, Ny_r, throttle], Nz, ps, Ny_r)
    
    # 从追踪器获取积分器状态
    int_e_nz, int_e_ps, int_e_ny_r = integrator_tracker.get_states()
    
    # 完成状态向量，添加积分器状态
    x_f16_full = np.append(x_f16_basic, [int_e_nz, int_e_ps, int_e_ny_r])
    
    # 使用低级控制器转换为舵面命令
    llc = autopilot.llc
    _, u_deg = llc.get_u_deg([Nz, ps, Ny_r, throttle], x_f16_full)
    
    # 更新模拟时间
    # 这里积分时间为0.1s的原因是jsb_env.jsbgym_m.enviroment.JsbSimEnv中定义了:
    # agent_interaction_freq: int = 10,
    current_time += 0.1

    # 缩放舵面角度到[-1, 1]范围
    aileron_scaled = np.clip(u_deg[2] / 21.5, -1.0, 1.0)  # 副翼
    elevator_scaled = np.clip(u_deg[1] / 25.0, -1.0, 1.0)  # 升降舵
    rudder_scaled = np.clip(u_deg[3] / 30.0, -1.0, 1.0)    # 方向舵
    throttle = u_deg[0]  # 油门通常已经是[0,1]范围，保持不变
    
    # 返回舵面命令
    return aileron_scaled, elevator_scaled, rudder_scaled, throttle  # 副翼，升降舵，方向舵，油门

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
    action = np.concatenate((action1, action2))
    print(f"posotion N: {observation[58]}, position E: {observation[59]}, altitude: {observation[0]}")
    # print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()