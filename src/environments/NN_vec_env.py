import os
import torch
import numpy as np
import multiprocessing as mp
from gymnasium.spaces import Box
from typing import Callable, List, Optional, Tuple, Sequence
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

class NNVecEnv(SubprocVecEnv):
    """
    创建一个需要使用神经网络的向量化环境

    使用此包装器前，请确保环境的observation和action均为初始环境的两倍
    智能体只需要提供初始的action即可，环境会自动调用成品神经网络进行
    action的补全

    :param NN_root: the root path of the neural network
    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
            self, 
            env_fns, 
            start_method=None, 
            NN_root: str = """/home/ubuntu/Workfile/RL/RL_model/experiments/20250526_093055/stage1/20250526_093055_TrackingTask_ppo_1layer1"""):
        # 为了减半observation space和action space，必须复制父类init代码
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
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
        # 使用减半后的空间初始化父类
        super(SubprocVecEnv, self).__init__(n_envs, half_observation_space, half_action_space)


        self.opponent_model_root = NN_root
        self.opponent_model, self.opponent_env = self._load_opponent_model()
        self.opponent_observation = None
        self.obs_normalization_cache = None  # 观察值标准化缓存

    def _get_space(self, observation_space: Box, action_space: Box) -> Tuple[Box, Box]:
        """设置环境的观察空间和动作空间"""
        # 创建减半的观察空间和动作空间
        
        # 观察空间减半
        obs_dim = observation_space.shape[0] // 2
        obs_low = observation_space.low[:obs_dim]
        obs_high = observation_space.high[:obs_dim]
        half_observation_space = Box(low=obs_low, high=obs_high, dtype=observation_space.dtype)
        
        # 动作空间减半
        act_dim = action_space.shape[0] // 2
        act_low = action_space.low[:act_dim]
        act_high = action_space.high[:act_dim]
        half_action_space = Box(low=act_low, high=act_high, dtype=action_space.dtype)
        
        return half_observation_space, half_action_space
    
    def _load_opponent_model(self):
        """加载支持批量推理的神经网络模型"""
        import pickle

        if not os.path.exists(os.path.join(self.opponent_model_root, "best_model.zip")):
            raise ValueError(f"Model file {self.opponent_model_root}/best_model.zip dose not exist.")
        if not os.path.exists(os.path.join(self.opponent_model_root, "final_train_env.pkl")):
            raise ValueError(f"VecNormalize file {self.opponent_model_root}/final_train_env.pkl dose not exist.")
        
        # 加载模型和标准化参数
        model = PPO.load(
            f"{self.opponent_model_root}/best_model.zip",
            device="cuda",
        )
        
        # 直接从pkl文件加载标准化参数
        with open(f"{self.opponent_model_root}/final_train_env.pkl", "rb") as f:
            vec_env = pickle.load(f)
    
        return model, vec_env
    
    def _get_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        """批量获取所有子环境的对手动作"""
        # # 使用缓存机制优化标准化性能
        # if self.obs_normalization_cache is None or not np.array_equal(obs, self.obs_normalization_cache[0]):
        #     half_size = obs.shape[1] // 2
        #     # obs_part1 = obs[:, :half_size]
        #     obs_part2 = obs[:, half_size:]
        #     # normalized_obs_part1 = self.opponent_env.normalize_obs(obs_part1)
        #     normalized_obs_part2 = self.opponent_env.normalize_obs(obs_part2)
        #     # normalized_obs = np.concatenate((normalized_obs_part1, normalized_obs_part2), axis=1)
        #     normalized_obs = normalized_obs_part2
        #     self.obs_normalization_cache = (obs.copy(), normalized_obs)
        # else:
        #     normalized_obs = self.obs_normalization_cache[1]
        normalized_obs = self.opponent_env.normalize_obs(obs)
        
        with torch.no_grad():
            # predict自己会把数据转换到正确的设备上
            actions, _ = self.opponent_model.predict(normalized_obs)
        
        # 随机输入
        # actions = np.random.uniform(-1, 1, size=(obs.shape[0], 4))
        # actions[:, -1] = np.abs(actions[:, -1])
        # 固定对偶输入
        # actions = np.zeros((obs.shape[0], 4))
        # for i in range(obs.shape[0]):
        #     actions[i] = np.array([0.5, 0.6, 0.0, 0.4])
        
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
        """重置所有环境"""
        raw_observation = super().reset()
        self.opponent_observation = self._get_observation(raw_observation, "opponent").copy()
        agent_obs = self._get_observation(raw_observation, "agent")
        self.obs_normalization_cache = None  # 清除缓存
        return agent_obs

    def _get_observation(self, obs: np.ndarray, object: str = "agent") -> np.ndarray:
        """获取前半段或后半段观察值"""
        if object == "agent":
            return obs[:, :obs.shape[1] // 2]
        elif object == "opponent":
            return obs[:, obs.shape[1] // 2:]
        else:
            raise ValueError(f"Invalid object type: {object}. Use 'agent' or 'opponent'.")