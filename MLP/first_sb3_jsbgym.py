import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import sys
sys.path.insert(0, "D:\Work_File\RL\jsbgym")

import jsbgym_m             # type: ignore


# # env_id = "C172-TurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0"
# env_id = "C172-TrajectoryTask-Shaping.STANDARD-NoFG-v0"

# Create environment
plane = "C172"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
task = "TrajectoryTask"

# shape = "Shaping.STANDARD"
shape = "Shaping.EXTRA"

# ======================================================================================================================

env_id = f"{plane}-{task}-{shape}-NoFG-v0"


# 设置随机种子
seed = 42
np.random.seed(seed)
gym.utils.seeding.np_random(seed)

def make_env(env_id: str, rank: int, seed: int = 0, render_mode= None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # 创建训练环境
    train_env = DummyVecEnv([lambda: gym.make(env_id)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False,
                    clip_obs=10.)
    # 创建评估环境
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                    clip_obs=10.)
    eval_env.training = False
    eval_env.norm_reward = False

    # 模型存储位置
    log_path="./logs/"

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                log_path=log_path, eval_freq=10000,
                                deterministic=True, render=False)

    model = PPO("MlpPolicy", train_env, learning_rate=1.5e-4, verbose=1, device='cpu', tensorboard_log="./logs/tensorboard/")
    model.learn(total_timesteps=10000_000, progress_bar=True, callback=[eval_callback])

    # 保存训练结束的模型
    # model.save(log_path + "final_model")
    train_env.save(log_path + "final_train_env.pkl")
