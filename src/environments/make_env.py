from typing import Optional, List, Tuple, Callable, Any
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from src.utils.yaml_import import import_class
from src.environments.self_play_wrapper import SelfPlayWrapper
from src.environments.wrap_env import create_wrapper_from_config
import os
import sys
import jsb_env.jsbgym_m             # type: ignore

def create_env(
        env_config: dict,
        num_cpu: int = 1,
        training: bool = True,
        vec_env_cls: Callable = SubprocVecEnv,
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
    
    # Create a wrapper class that includes SelfPlayWrapper
    def make_wrapper(env):
        if combined_wrapper_class is not None:
            env = combined_wrapper_class(env)

        pool_roots = vec_env_kwargs.get("pool_roots") if vec_env_kwargs else None
        model_num = vec_env_kwargs.get("model_num", 0) if vec_env_kwargs else 0

        return SelfPlayWrapper(env, pool_roots=pool_roots, model_num=model_num)

    if training:
        vec_env = make_vec_env(
            env_id,
            n_envs=num_cpu,
            wrapper_class=make_wrapper,
            vec_env_cls=vec_env_cls,
            env_kwargs={"render_mode": render_mode}
        )
    else:
        vec_env = make_vec_env(
            env_id,
            n_envs=1,
            wrapper_class=make_wrapper,
            vec_env_cls=vec_env_cls,
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
        # Add SelfPlayWrapper if needed
        # env = SelfPlayWrapper(env, pool_roots=pool_roots, model_num=model_num)
        env.reset(seed=seed + rank)
        return env
    # set_random_seed(seed)
    return _init