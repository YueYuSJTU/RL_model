from typing import Optional
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, "D:\Work_File\RL\jsbgym")
import jsbgym_m             # type: ignore

def create_env(env_config: dict, training: bool = True, load: bool = False) -> DummyVecEnv:
    """创建标准化环境"""
    # 构建环境ID
    plane = env_config["plane"]
    task = env_config["task"]
    shape = env_config["shape"]
    render_mode = env_config.get("render_mode")
    
    env_id = f"{plane}-{task}-{shape}-NoFG-v0"
    if render_mode == "flightgear":
        env_id = f"{plane}-{task}-{shape}-FG-v0"

    # 创建原始环境
    env = gym.make(env_id, render_mode=render_mode)
    
    # 向量化环境
    vec_env = DummyVecEnv([lambda: env])

    # 如果是加载模式，则避免通过VecNormalize初始化
    if load:
        return vec_env
    
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