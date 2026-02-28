import sys
import time
import math
import logging
import threading
import numpy as np

# 导入 dogfight_client
try:
    from src.visualization import dogfight_client as df
except ImportError:
    sys.path.append("./src")
    from visualization import dogfight_client as df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SmoothConnector:
    """
    负责处理与 Sandbox 的通信，并自动进行帧插值平滑。
    """
    def __init__(self, host="127.0.1.1", port=50888, update_rate=60):
        self.host = host
        self.port = port
        self.update_rate = update_rate # 发送给 Sandbox 的频率 (FPS)
        self.plane_id = None
        self.running = False
        
        # === 状态变量 ===
        # Use simple locking for thread safety
        self.lock = threading.Lock()
        
        # 当前显示的状态 (用于发送给 Sandbox)
        self.render_pos = np.array([0.0, 500.0, 0.0])
        self.render_rot = np.array([0.0, 0.0, 0.0]) # Euler: Yaw, Pitch, Roll
        
        # 目标状态 (由 RL/物理逻辑 更新)
        self.target_pos = np.array([0.0, 500.0, 0.0])
        self.target_rot = np.array([0.0, 0.0, 0.0])
        
        # 上一次逻辑更新的时间
        self.last_logic_time = time.time()
        # 平滑因子 (0.0 - 1.0)，越小越平滑但延迟越高，越大响应越快但越抖
        # 对于 RL 展示，建议 0.1 - 0.3
        self.smoothing_factor = 0.2 

    def connect(self):
        try:
            df.connect(self.host, self.port)
            time.sleep(0.5)
            planes = df.get_planes_list()
            if not planes:
                raise Exception("No planes found")
            self.plane_id = planes[1]
            
            # 开启自定义物理
            df.set_machine_custom_physics_mode(self.plane_id, True)
            
            # 【关键修改】关闭客户端更新模式。
            # 让 Sandbox 自己全速渲染，我们只负责不断更新飞机的物理坐标。
            # 这样即使 Python 卡顿，UI 菜单和背景也不会卡死。
            df.set_client_update_mode(False) 
            
            logging.info(f"Connected to plane {self.plane_id}")
            self.running = True
            
            # 启动发送线程
            self.thread = threading.Thread(target=self._render_loop)
            self.thread.daemon = True
            self.thread.start()
            
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            sys.exit(1)

    def update_target(self, pos, rot_euler):
        """
        主线程调用此方法更新目标位置（来自 RL 或 计算逻辑）
        pos: [x, y, z]
        rot_euler: [yaw, pitch, roll]
        """
        with self.lock:
            self.target_pos = np.array(pos, dtype=np.float64)
            self.target_rot = np.array(rot_euler, dtype=np.float64)

    def _render_loop(self):
        """
        后台线程：高频插值并发送数据
        """
        dt = 1.0 / self.update_rate
        
        while self.running:
            loop_start = time.time()
            
            with self.lock:
                # === 插值计算 (Lerp) ===
                # 每一帧只向目标移动一小步，消除突变
                self.render_pos += (self.target_pos - self.render_pos) * self.smoothing_factor
                
                # 角度插值需要特殊处理 (处理 -PI 到 PI 的跳变)，这里简化处理
                # 如果转圈出现抽搐，需要加 wrap_angle 逻辑
                diff_rot = self.target_rot - self.render_rot
                self.render_rot += diff_rot * self.smoothing_factor

                # 复制数据用于发送
                current_pos = self.render_pos.copy()
                current_rot = self.render_rot.copy()

            # === 构造矩阵并发送 ===
            matrix = self._get_matrix(current_rot[0], current_rot[1], current_rot[2], current_pos)
            
            # 这里的 velocity 我们发 0 或者发一个近似值，
            # 因为位置已经是我们强行指定的了，velocity 仅用于 Sandbox 内部的特效（如尾迹）
            df.update_machine_kinetics(self.plane_id, matrix, [0, 0, 0])
            
            # 保持帧率
            elapsed = time.time() - loop_start
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)

    def _get_matrix(self, yaw, pitch, roll, pos):
        # 简化的欧拉角转矩阵 (与之前相同)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        
        r00 = cy * cr + sy * sp * sr
        r01 = cp * sr
        r02 = -sy * cr + cy * sp * sr
        r10 = -cy * sr + sy * sp * cr
        r11 = cp * cr
        r12 = sy * sr + cy * sp * cr
        r20 = sy * cp
        r21 = -sp
        r22 = cy * cp
        
        return [
            r00, r01, r02,
            r10, r11, r12,
            r20, r21, r22,
            pos[0], pos[1], pos[2]
        ]

# === 主逻辑 (模拟 RL 环境) ===
def run_smooth_test():
    connector = SmoothConnector()
    connector.connect()

    # 模拟物理状态
    sim_pos = np.array([0.0, 500.0, 0.0])
    sim_yaw = 0.0
    velocity = 200.0
    
    # 模拟 RL 的低频更新 (比如 20Hz)
    logic_dt = 1.0 / 20.0 
    
    logging.info("开始平滑飞行测试...")
    logging.info("逻辑更新率: 20Hz (模拟RL) | 渲染更新率: 60Hz (插值)")
    
    try:
        while True:
            start_t = time.time()

            # --- 1. 物理计算 (模拟 step) ---
            sim_yaw += 0.03 # 每次转一点
            vx = math.sin(sim_yaw) * velocity
            vz = math.cos(sim_yaw) * velocity
            
            sim_pos[0] += vx * logic_dt
            sim_pos[2] += vz * logic_dt
            
            # --- 2. 将计算结果推送给 Connector ---
            # Connector 会在后台线程自动把这 20Hz 的数据平滑成 60Hz 发送给 Sandbox
            connector.update_target(sim_pos, [sim_yaw, 0.0, 0.0])

            # --- 3. 维持逻辑帧率 ---
            elapsed = time.time() - start_t
            time.sleep(max(0, logic_dt - elapsed))
            
    except KeyboardInterrupt:
        logging.info("Stop.")
        connector.running = False

if __name__ == "__main__":
    run_smooth_test()