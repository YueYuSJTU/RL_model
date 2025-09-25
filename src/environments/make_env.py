from typing import Optional, List, Tuple, Callable, Any
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from src.utils.yaml_import import import_class
from src.environments.NN_vec_env import NNVecEnv
from src.environments.wrap_env import create_wrapper_from_config
import os
import sys
# # 将项目根目录添加到sys.path（使用相对路径）
# current_file_path = os.path.abspath(__file__)
# RL_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))  # RL目录
# env_dir = os.path.join(RL_dir, "jsbgym")
# # print(env_dir)
# sys.path.append(env_dir)
import jsb_env.jsbgym_m             # type: ignore

def create_env(
        env_config: dict, 
        num_cpu: int = 1, 
        training: bool = True,
        vec_env_cls: Callable = NNVecEnv,
        vec_env_kwargs: Optional[dict[str, Any]] = None,
    ) -> DummyVecEnv:
    """创建标准化环境"""
    # 构建环境ID
    plane = env_config["plane"]
    task = env_config["task"]
    shape = env_config["shape"]
    render_mode = env_config.get("render_mode")
    wrapper_configs = env_config.get("wrappers") if "wrappers" in env_config else None
    combined_wrapper_class = create_wrapper_from_config(wrapper_configs)
    
    env_id = f"{plane}-{task}-{shape}-NoFG-v0"
    if render_mode == "flightgear":
        env_id = f"{plane}-{task}-{shape}-FG-v0"
    
    if training:
        # 创建训练环境
        # vec_env = SubprocVecEnv([make_Env(env_id, i, wrappers=wrappers) for i in range(num_cpu)])
        vec_env = make_vec_env(
            env_id, 
            n_envs=num_cpu, 
            wrapper_class=combined_wrapper_class,
            vec_env_cls=vec_env_cls, 
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs={"render_mode": render_mode}
        )
    else:
        # 创建评估环境
        # vec_env = DummyVecEnv([make_Env(env_id, 0, render_mode=render_mode, wrappers=wrappers)])
        vec_env = make_vec_env(
            env_id, 
            n_envs=1, 
            wrapper_class=combined_wrapper_class,
            vec_env_cls=vec_env_cls,
            vec_env_kwargs=vec_env_kwargs,
            env_kwargs={"render_mode": render_mode}
        )

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


def make_Env(env_id: str, rank: int, seed: int = 0, render_mode= None, wrappers: Optional[List[Tuple[Callable, dict]]] = None):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if wrappers is not None:
            for item in wrappers:
                wrapper = import_class(item["name"])
                kwargs = item["kwargs"]
                env = wrapper(env, **kwargs)
        env.reset(seed=seed + rank)
        return env
    # set_random_seed(seed)
    return _init