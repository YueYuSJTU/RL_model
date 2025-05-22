import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import jsbgym_m             # type: ignore
import gymnasium as gym
import numpy as np

# plane = "C172"
plane = "F16"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
# task = "TrajectoryTask"
task = "TrackingTask"

# shape = "Shaping.STANDARD"
# shape = "Shaping.EXTRA"
shape = "stage2"

# render_mode = "flightgear"
# render_mode = None
# render_mode = "graph"
render_mode = "human"


ENV_ID = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
# ENV_ID = "C172-HeadingControlTask-Shaping.EXTRA-FG-v0"
env = gym.make(ENV_ID, render_mode=render_mode)
# env = gym.make(ENV_ID)
env.reset()
action = np.concatenate([env.action_space.sample()[:4],np.array([1,1,1,1])])
observation, reward, terminated, truncated, info = env.step(action)
while not (terminated or truncated):
    if render_mode is not None:
        env.render()   
    observation, reward, terminated, truncated, info = env.step(action)
    # action = env.action_space.sample()
    action = np.concatenate([env.action_space.sample()[:4],np.array([1,1,1,1])])
    # print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()