from typing import Optional
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, "D:\Work_File\RL\jsbgym")
import jsbgym_m             # type: ignore

def create_env(env_config: dict, num_cpu: int = 1, training: bool = True) -> DummyVecEnv:
    """创建标准化环境"""
    # 构建环境ID
    plane = env_config["plane"]
    task = env_config["task"]
    shape = env_config["shape"]
    render_mode = env_config.get("render_mode")
    
    env_id = f"{plane}-{task}-{shape}-NoFG-v0"
    if render_mode == "flightgear":
        env_id = f"{plane}-{task}-{shape}-FG-v0"
    
    if training:
        # 创建训练环境
        vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    else:
        # 创建评估环境
        vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode)])
    
    # 标准化处理
    if env_config.get("use_vec_normalize", False):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=env_config["norm_obs"],
            norm_reward=env_config["norm_reward"],
            clip_obs=env_config["clip_obs"],
            training=training
        )
    
    return vec_env


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