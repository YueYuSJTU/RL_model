import os
import torch
import warnings
import numpy as np
import multiprocessing as mp
from gymnasium.spaces import Box
from typing import Callable, List, Optional, Tuple, Sequence, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import _worker
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from ..agents.model_wrapper import ObsAdaptingModel

class NNVecEnv(SubprocVecEnv):
    """
    创建一个需要使用神经网络的向量化环境，支持在每次环境重置时随机选择敌机策略

    使用此包装器前，请确保环境的observation和action均为初始环境的两倍
    智能体只需要提供初始的action即可，环境会自动调用成品神经网络进行
    action的补全

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    :param pool_roots: 对手池的路径
    """

    def __init__(
            self, 
            env_fns, 
            start_method=None, 
            pool_roots: Union[str, List[str]] = """/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool4""",
            model_num: int = 0
        ):
        # 为了减半observation space和action space，必须复制父类init代码
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        half_observation_space, half_action_space = self._get_space(observation_space, action_space)
        super(SubprocVecEnv, self).__init__(n_envs, half_observation_space, half_action_space)

        self.model_num = model_num
        self._choose_opponent_models(pool_roots, n_envs)
        self.opponent_observation = None

    def _choose_opponent_models(self, pool_roots: Union[str, List[str]], n_envs) -> None:
        # 选择敌机策略
        self.opponent_model_roots = self._find_strategy_dirs(pool_roots)
        
        # 加载选择的敌机策略
        self.opponent_models = []
        # self.opponent_envs = []
        for root in self.opponent_model_roots:
            model = self._load_opponent_model(root)
            env_config = self._load_opponent_env_config(root)
            # 使用ObsAdaptingModel包装模型
            wrapped_model = ObsAdaptingModel(model, env_config)
            self.opponent_models.append(wrapped_model)
            # self.opponent_envs.append(env)
        
        # 为每个子环境分配一个随机的初始策略
        if len(self.opponent_models) > 0:
            self.env_strategy_indices = np.random.randint(len(self.opponent_models), size=n_envs)
        

    def _get_space(self, observation_space: Box, action_space: Box) -> Tuple[Box, Box]:
        """
        设置环境的观察空间和动作空间。
        通过操作最后一个维度来分割空间，以支持多维观察空间。
        """
        # --- 观察空间减半 ---
        obs_shape = observation_space.shape
        # 总是分割最后一个维度
        obs_split_dim = obs_shape[-1] // 2
        # 构建新的shape，保留前面所有维度，只修改最后一个维度
        new_obs_shape = obs_shape[:-1] + (obs_split_dim,)

        # 使用 Ellipsis (...) 进行切片，以兼容任意维度
        # 适用于 low/high 是一个标量或者一个与原空间同shape的数组的情况
        obs_low = observation_space.low[..., :obs_split_dim]
        obs_high = observation_space.high[..., :obs_split_dim]
        
        half_observation_space = Box(low=obs_low, high=obs_high, shape=new_obs_shape, dtype=observation_space.dtype)
        
        # --- 动作空间减半 ---
        act_shape = action_space.shape
        # 总是分割最后一个维度
        act_split_dim = act_shape[-1] // 2
        # 构建新的shape
        new_act_shape = act_shape[:-1] + (act_split_dim,)
        
        # 使用 Ellipsis (...) 进行切片
        act_low = action_space.low[..., :act_split_dim]
        act_high = action_space.high[..., :act_split_dim]
        
        half_action_space = Box(low=act_low, high=act_high, shape=new_act_shape, dtype=action_space.dtype)
        
        return half_observation_space, half_action_space
    

    def _find_strategy_dirs(self, root_dir: str) -> List[str]:
        """
        查找指定根目录下的所有策略目录
        如果提供了加载模型的编号，则只加载指定编号的模型
        如果编号为-1,则不加载模型
        """
        strategy_dirs = []
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory {root_dir} does not exist")
        
        # 如果提供了加载模型的编号，则只加载指定编号的模型
        if self.model_num > 0:
            model_path = os.path.join(root_dir, f"{self.model_num}")
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "best_model.zip")):
                strategy_dirs.append(model_path)
                return strategy_dirs
            else:
                raise ValueError(f"Model path {model_path} does not exist or is invalid")
        # 如果编号为-1,则不加载模型
        elif self.model_num == -1:
            return strategy_dirs
            
        # 遍历根目录下所有子目录
        for item in os.listdir(root_dir):
            full_path = os.path.join(root_dir, item)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "best_model.zip")):
                strategy_dirs.append(full_path)
        
        if not strategy_dirs:
            raise ValueError(f"No valid strategy directories found in {root_dir}")
        if len(strategy_dirs) > 10:
            warnings.warn(
                f"Found too many strategy directories ({len(strategy_dirs)}).\n"
                f"Since each subprocess can only run one model, the number of models should be less than "
                f"the number of CPU cores.\n"
                f"For better training performance, it's recommended to keep the number of models "
                f"below one-third of available CPU cores."
            )
            
        # print(f"Found {len(strategy_dirs)} strategy directories: {strategy_dirs}")
        return strategy_dirs

    def _load_opponent_model(self, model_path: str):
        """加载单个敌机策略模型"""
        import pickle
        from src.environments.make_env import create_env

        model_file = os.path.join(model_path, "best_model.zip")
        env_config = self._load_opponent_env_config(model_path)
        env_file = os.path.join(model_path, "final_train_env.pkl")

        if not os.path.exists(model_file):
            raise ValueError(f"Model file {model_file} does not exist.")
        if not os.path.exists(env_file):
            raise ValueError(f"VecNormalize file {env_file} does not exist.")
    
        # # 直接从pkl文件加载标准化参数
        # with open(env_file, "rb") as f:
        #     vec_env = pickle.load(f)

        # 这里设置model_num为-1，避免循环依赖
        vec_env = create_env(env_config, training=False, vec_env_kwargs={"model_num": -1})
        vec_env = VecNormalize.load(
            env_file, 
            vec_env
        )
        vec_env.training = False
        vec_env.norm_reward = False

        # 加载模型和标准化参数
        model = PPO.load(model_file, vec_env, device="cuda")
    
        return model
    
    def _load_opponent_env_config(self, model_path: str) -> dict:
        """加载单个敌机策略的环境配置"""
        import yaml

        config_file = os.path.join(model_path, "env_config.yaml")

        if not os.path.exists(config_file):
            raise ValueError(f"Config file {config_file} does not exist.")
        
        # 加载配置
        with open(config_file, encoding="utf-8") as f:
            env_config = yaml.safe_load(f)
        
        return env_config
    
    def _get_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        """批量获取所有子环境的对手动作，每个子环境使用其对应的策略"""
        if self.model_num == -1:
            # 随机输入
            actions = np.random.uniform(-1, 1, size=(obs.shape[0], 4))
            actions[:, -1] = np.abs(actions[:, -1])
            return actions
        
        batch_size = obs.shape[0]
        actions = np.zeros((batch_size, self.opponent_models[0].policy.action_space.shape[0]))
        
        # 按策略分组处理，提高批处理效率
        for strategy_idx in range(len(self.opponent_models)):
            # 找出使用当前策略的环境索引
            env_indices = np.where(self.env_strategy_indices == strategy_idx)[0]
            if len(env_indices) == 0:
                continue
                
            # 获取当前策略对应的包装后模型
            wrapped_model = self.opponent_models[strategy_idx]
            
            # 提取这些环境的观察值
            strategy_obs = obs[env_indices]

            # 使用包装后的模型进行预测，包装器内部会处理观察适配
            strategy_actions, _ = wrapped_model.predict(strategy_obs)
            
            # 将动作放回对应位置
            actions[env_indices] = strategy_actions
        
        return actions
    
    def step_async(self, actions: np.ndarray) -> None:
        """并行执行环境步进"""
        # 生成批量对手动作
        opponent_actions = self._get_opponent_action(self.opponent_observation)
        
        # 合并主控和对手动作
        combined_actions = [
            np.concatenate((main_action, opponent_action))
            for main_action, opponent_action in zip(actions, opponent_actions)
        ]
        
        # 批量发送到子进程
        for remote, action in zip(self.remotes, combined_actions):
            remote.send(("step", action))
        self.waiting = True
    
    def step_wait(self) -> VecEnvStepReturn:
        """收集并行执行结果"""
        full_obs, reward, done, info = super().step_wait()
        
        # Get observations for agent and opponent
        self.opponent_observation = self._get_observation(full_obs, "opponent").copy()
        agent_obs = self._get_observation(full_obs, "agent")
        
        # Update terminal observations only if needed (faster check)
        if np.any(done):
            done_indices = np.where(done)[0]
            for idx in done_indices:
                info[idx]["terminal_observation"] = agent_obs[idx]
        
        return agent_obs, reward, done, info

    def reset(self) -> np.ndarray:
        """重置所有环境，并为每个环境随机选择新的敌机策略"""
        # 为每个环境随机选择新的策略
        if self.model_num >= 0:
            self.env_strategy_indices = np.random.randint(len(self.opponent_models), size=self.num_envs)
        
        raw_observation = super().reset()
        self.opponent_observation = self._get_observation(raw_observation, "opponent").copy()
        agent_obs = self._get_observation(raw_observation, "agent")
        return agent_obs

    def _get_observation(self, obs: np.ndarray, object: str = "agent") -> np.ndarray:
        """
        获取前半段或后半段观察值。
        通过操作最后一个维度来分割，以支持多维观察值。
        """
        # 总是沿着最后一个维度进行分割
        split_point = obs.shape[-1] // 2
        
        if object == "agent":
            return obs[..., :split_point]
        elif object == "opponent":
            return obs[..., split_point:]
        else:
            raise ValueError(f"Invalid object type: {object}. Use 'agent' or 'opponent'.")