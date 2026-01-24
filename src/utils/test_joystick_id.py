import pygame
import sys
import time
import os

def debug_gamepad():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("未检测到手柄！请检查连接。")
        return

    # 连接第一个手柄
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    name = joystick.get_name()
    axes = joystick.get_numaxes()
    buttons = joystick.get_numbuttons()
    hats = joystick.get_numhats()

    print(f"检测到手柄: {name}")
    print(f"功能统计 -> 轴(Axes): {axes}, 按钮(Buttons): {buttons}, 苦力帽(Hats): {hats}")
    print("=" * 50)
    print("请移动所有摇杆、按下所有按键，观察数值变化...")
    print("按 Ctrl+C 退出")
    time.sleep(2)

    try:
        while True:
            pygame.event.pump() # 刷新事件
            
            # 清屏 (Linux/Mac使用clear, Windows使用cls)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"手柄: {name}")
            print("-" * 30)

            # --- 显示所有轴 (通常是摇杆和扳机) ---
            print("【轴 (AXIS)】 (范围通常是 -1.0 到 1.0):")
            for i in range(axes):
                val = joystick.get_axis(i)
                # 为了视觉方便，只有当数值明显偏离0时才高亮显示
                mark = " <<< 动作中" if abs(val) > 0.2 else ""
                print(f"Axis {i}: {val:>6.3f}{mark}")
            
            print("-" * 30)

            # --- 显示所有苦力帽 (通常是十字键) ---
            print("【苦力帽 (HAT)】 (通常是十字键):")
            for i in range(hats):
                val = joystick.get_hat(i)
                mark = " <<< 按下" if val != (0,0) else ""
                print(f"Hat {i}: {val}{mark}")

            print("-" * 30)

            # --- 显示所有按钮 ---
            print("【按钮 (BUTTON)】:")
            pressed_buttons = []
            for i in range(buttons):
                if joystick.get_button(i):
                    pressed_buttons.append(str(i))
            print(f"按下的按钮 ID: {', '.join(pressed_buttons)}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n退出调试。")
        pygame.quit()

if __name__ == "__main__":
    debug_gamepad()