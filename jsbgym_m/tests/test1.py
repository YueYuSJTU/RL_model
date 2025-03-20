import sys
sys.path.insert(0, "D:\Work_File\RL\jsbgym")

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

shape = "Shaping.STANDARD"
# shape = "Shaping.EXTRA"

# render_mode = "flightgear"
# render_mode = None
# render_mode = "graph"
render_mode = "human"


ENV_ID = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
# ENV_ID = "C172-HeadingControlTask-Shaping.EXTRA-FG-v0"
env = gym.make(ENV_ID, render_mode=render_mode)
# env = gym.make(ENV_ID)
env.reset()
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
while not (terminated or truncated):
    if render_mode is not None:
        env.render()   
    observation, reward, terminated, truncated, info = env.step(action)
    action = env.action_space.sample()
    # print(f"Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()