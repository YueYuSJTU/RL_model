import numpy as np
from pynput import keyboard
import threading

class KeyboardController:
    """
    通过键盘监听控制飞机的动作输出。
    假设动作空间为4维 (1, 4)，范围通常在 [-1, 1]。
    
    默认键位映射 (请根据你的环境实际动作定义进行修改):
    - 维度 0 (通常是 Pitch/俯仰): 上箭头(Up) / 下箭头(Down)
    - 维度 1 (通常是 Roll/滚转): 左箭头(Left) / 右箭头(Right)
    - 维度 2 (通常是 Yaw/偏航): A / D
    - 维度 3 (通常是 Throttle/油门): W (加速) / S (减速)
    """
    def __init__(self, action_dim=4):
        self.action_dim = action_dim
        self.current_action = np.zeros(action_dim, dtype=np.float32)
        self.pressed_keys = set()
        
        # 启动监听线程
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.add(key.char)
        else:
            self.pressed_keys.add(key)
        self._update_action()

    def _on_release(self, key):
        if hasattr(key, 'char'):
            if key.char in self.pressed_keys:
                self.pressed_keys.remove(key.char)
        else:
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)
        self._update_action()

    def _update_action(self):
        # 初始化动作向量
        # 注意：这里我们使用瞬时电平控制（按住为1，松开为0），
        # 如果需要累积量（按住持续增加），逻辑需要稍微修改。
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        throttle = 0.0 # 假设初始油门为0，或者中间值

        # --- 键位映射逻辑 ---
        
        # 维度 0: Pitch (俯仰) - 上/下箭头
        if keyboard.Key.up in self.pressed_keys:
            pitch = 1.0 # 或是 1.0，取决于你的环境定义
        elif keyboard.Key.down in self.pressed_keys:
            pitch = -1.0

        # 维度 1: Roll (滚转) - 左/右箭头
        if keyboard.Key.left in self.pressed_keys:
            roll = -1.0
        elif keyboard.Key.right in self.pressed_keys:
            roll = 1.0

        # 维度 2: Yaw (偏航) - A / D
        if 'a' in self.pressed_keys:
            yaw = -1.0
        elif 'd' in self.pressed_keys:
            yaw = 1.0

        # 维度 3: Throttle (油门) - W / S
        # 很多环境中油门是 [0, 1] 或 [-1, 1]。这里假设 W 是最大油门。
        if 'w' in self.pressed_keys:
            throttle = 1.0
        elif 's' in self.pressed_keys:
            throttle = -1.0 # 或 0.0
        
        self.current_action = np.array([roll, pitch, yaw, throttle], dtype=np.float32)

    def predict(self, observation, deterministic=True):
        """
        模仿 Stable Baselines3 模型的 predict 接口
        """
        # 返回形状 (1, 4) 的动作，与模型输出保持一致
        return self.current_action.reshape(1, -1), None

    def close(self):
        self.listener.stop()



import numpy as np
import pygame
import logging

class GamepadController:
    """
    通过 Pygame 读取手柄/摇杆输入控制飞机。
    动作空间: (1, 4) -> [Pitch, Roll, Yaw, Throttle]
    范围: [-1, 1]
    """
    def __init__(self, action_dim=4, deadzone=0.1):
        self.action_dim = action_dim
        self.deadzone = deadzone
        self.joystick = None
        
        # 初始化 Pygame 的 Joystick 模块
        pygame.init()
        pygame.joystick.init()
        
        self._connect_controller()

    def _connect_controller(self):
        count = pygame.joystick.get_count()
        if count > 0:
            # 默认连接第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info(f"Gamepad connected: {self.joystick.get_name()}")
            logging.info(f"Axes: {self.joystick.get_numaxes()}, Buttons: {self.joystick.get_numbuttons()}")
        else:
            logging.warning("No gamepad found! Please connect a controller.")
            raise IOError("No gamepad connected")

    def _apply_deadzone(self, value):
        """应用死区，防止摇杆回中时的抖动"""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def predict(self, observation, deterministic=True):
        """
        获取手柄输入并转换为动作向量。
        """
        # 必须调用 event.pump() 来刷新 Pygame 的内部状态
        pygame.event.pump()
        
        if not self.joystick:
            return np.zeros((1, self.action_dim)), None

        # --- 键位映射 (Xbox/通用手柄 标准映射) ---
        # 注意：不同手柄的 Axis 索引可能不同，以下是 Xbox Controller 的典型布局
        # Axis 0: 左摇杆 左右 (Roll)
        # Axis 1: 左摇杆 上下 (Pitch) -> Pygame中 下是+1, 上是-1
        # Axis 2: 右摇杆 左右 (Yaw) (部分手柄可能是 Axis 3 或 4)
        # Axis 3: 右摇杆 上下 (未使用)
        # Axis 4/5: 左右扳机 (Throttle) (LT/RT)
        
        # 1. Roll (滚转) - 左摇杆横向
        roll = self._apply_deadzone(self.joystick.get_axis(0))
        
        # 2. Pitch (俯仰) - 左摇杆纵向
        # 通常环境定义: 1.0 是拉起机头(Pitch Up)。
        # 手柄物理: 拉杆(Down方向)是 +1。所以通常不需要取反，直接对应。
        pitch = self._apply_deadzone(self.joystick.get_axis(1))

        # 3. Yaw (偏航) - 右摇杆横向 (或者 LB/RB 按钮)
        # 尝试读取 Axis 2 或 3 (取决于手柄驱动，Xbox通常是 Axis 3 或 2)
        # 这里为了兼容性，我们优先尝试用 Shoulder Buttons (LB/RB) 控制 Yaw，
        # 或者如果你喜欢用右摇杆，请取消下面 Axis 代码的注释。
        
        # 方案 A: 使用右摇杆控制 Yaw
        try:
            yaw = self._apply_deadzone(self.joystick.get_axis(3)) # Xbox 右摇杆横向通常是 3 或 2
        except:
            yaw = 0.0
            
        # 方案 B: 使用 L1/R1 (LB/RB) 按钮进行数字量偏航 (覆盖摇杆)
        # Button 4 = LB, Button 5 = RB (索引可能因设备而异)
        if self.joystick.get_numbuttons() > 5:
            if self.joystick.get_button(4): # LB
                yaw = -1.0
            elif self.joystick.get_button(5): # RB
                yaw = 1.0

        # 4. Throttle (油门)
        # 方案 A: 使用按键 (A加速 / B减速) 或 (X/Y)
        # Xbox: A=0, B=1, X=2, Y=3
        throttle = 0.0
        if self.joystick.get_numbuttons() > 1:
            if self.joystick.get_button(0): # A 键 加速
                throttle = 1.0
            elif self.joystick.get_button(1): # B 键 减速/刹车
                throttle = -1.0
        
        # 方案 B: 使用扳机键 (Triggers) 作为油门
        # 许多手柄扳机是 Axis，范围 -1 (松开) 到 1 (按下)
        # axis_rt = self.joystick.get_axis(5) 
        # if axis_rt > -0.9: throttle = (axis_rt + 1) / 2 # 映射到 0~1

        # 组装动作
        action = np.array([roll, -pitch, yaw, throttle], dtype=np.float32)
        
        return action.reshape(1, -1), None

    def close(self):
        pygame.quit()