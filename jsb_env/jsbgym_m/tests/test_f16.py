import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import jsbgym_m             # type: ignore
import gymnasium as gym
import numpy as np
from time import sleep

# plane = "C172"
plane = "F16"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
# task = "TrajectoryTask"
# task = "TrackingTask"
task = "FlyTask"

shape = "Shaping.STANDARD"
# shape = "Shaping.EXTRA"

render_mode = "flightgear"
# render_mode = None
# render_mode = "graph"
# render_mode = "human"


ENV_ID = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
# ENV_ID = "C172-HeadingControlTask-Shaping.EXTRA-FG-v0"
env = gym.make(ENV_ID, render_mode=render_mode)
# env = gym.make(ENV_ID)
env.reset()
action = env.action_space.sample()
# Function to gradually change the action value
def gradual_change(action, index, target, steps=50):
    step_size = (target - action[index]) / steps
    for _ in range(steps):
        action[index] += step_size
        if render_mode is not None:
            env.render()   
        observation, reward, terminated, truncated, info = env.step(action)
        # sleep(0.05)  # Add a small delay to simulate gradual movement

# Initialize action with all zeros and throttle at 0.4
action = np.zeros(env.action_space.shape)
action[3] = 0.4  # Assuming throttle is at index 3

# Gradually move aileron (index 0) to 0.5
gradual_change(action, 0, 0.5)

# Gradually move aileron to -0.5
gradual_change(action, 0, -0.55)

# Gradually move aileron back to 0
gradual_change(action, 0, 0.0)

gradual_change(action, 1, -0.5)  # Elevator
gradual_change(action, 1, 0.2)  # Elevator
gradual_change(action, 1, 0.0)  # Elevator
# Gradually move rudder (index 2) to 0.5
gradual_change(action, 2, 0.5)
# Gradually move rudder to -0.5
gradual_change(action, 2, -0.5)
# Gradually move rudder back to 0
gradual_change(action, 2, 0.0)  # Rudder
# observation, reward, terminated, truncated, info = env.step(action)
# while not (terminated or truncated):
#     if render_mode is not None:
#         env.render()   
#     observation, reward, terminated, truncated, info = env.step(action)
#     action = env.action_space.sample()
#     # print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()