import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import sys
sys.path.insert(0, "D:\Work_File\RL\jsbgym")
import jsbgym_m             # type: ignore

# 模型存储位置
log_path="D:\Work_File\RL\RL_model\logs\\num12\\"
# 读取模型选择
model_name = "best_model"
# model_name = "best_model_1805.4166278576747"
# model_name = "best_model_1736.615260656178"

skipStep = 5
skipTimes = 3

# # 设置随机种子
# seed = 42
# np.random.seed(seed)
# gym.utils.seeding.np_random(seed)

# Create environment
# plane = "C172"
plane = "F16"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
task = "TrajectoryTask"

# shape = "Shaping.STANDARD"
shape = "Shaping.EXTRA"

# render_mode = "flightgear"
# render_mode = None
# render_mode = "graph"
render_mode = "human"

# ======================================================================================================================

env_id = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode)])
# vec_env = DummyVecEnv([lambda: SkipObsWrapper(gym.make(env_id), skip_step=skipStep, skip_times=skipTimes)])
# vec_env.seed(seed)
vec_env = VecNormalize.load(log_path + "final_train_env.pkl", vec_env)
vec_env.training = False
vec_env.norm_reward = False

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load(log_path + model_name, env=vec_env, device='cpu')

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# vec_env = model.get_env()
obs = vec_env.reset()

position = []
Euler = []
rewardDepart = []
# waypoints = []
actions = []
print("Start to simulate")
for i in range(15000):
    vec_env.render()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, _ = vec_env.step(action)
    if terminated:
        print(f"Terminated at step {i}")
        break

vec_env.close()