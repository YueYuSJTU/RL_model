from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
import gymnasium as gym
import logging

class ComponentEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
        **kwargs
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            **kwargs
        )
        # 根据模型保存路径，自动生成统计文件的保存路径
        if self.best_model_save_path is not None:
            self.stats_path = os.path.join(self.best_model_save_path, "best_env.pkl")
        else:
            self.stats_path = None
            if self.verbose > 0:
                print("Warning: best_model_save_path is not specified. VecNormalize stats will not be saved.")
        
        # 用于追踪上一次的最佳奖励，以检测是否更新
        self._previous_best_reward = self.best_mean_reward
        self.reward_component_values = defaultdict(list)
        
    def _on_step(self) -> bool:
        # 在评估期间收集数据
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 运行评估并收集分项奖励
            self.evaluate_reward_components()
            
            # 记录到TensorBoard
            self.log_component_values()
            
            # 重置收集器
            self.reward_component_values = defaultdict(list)
        
        # 调用父类的 _on_step()。这将执行标准评估、更新 self.best_mean_reward 并保存最佳模型
        continue_training = super()._on_step()

        # 如果父类决定停止训练，我们也应停止
        if not continue_training:
            return False

        # 检查 best_mean_reward 是否被父类更新了
        if self.best_mean_reward > self._previous_best_reward:
            if self.verbose > 0:
                print(f"New best reward: {self.best_mean_reward:.2f} > {self._previous_best_reward:.2f}. ")
            
            # 确认 stats_path 已设置，并且评估环境是 VecNormalize
            if self.stats_path is not None and isinstance(self.eval_env, VecNormalize):
                if self.verbose > 0:
                    print(f"Saving VecNormalize statistics to {self.stats_path}")
                # 保存 VecNormalize 的统计数据
                self.eval_env.save(self.stats_path)

            # 更新我们追踪的最佳奖励值
            self._previous_best_reward = self.best_mean_reward

        return True
    
    def evaluate_reward_components(self):
        """运行评估并收集分项奖励值"""
        # # 确保环境已重置
        # self._reset_eval_env()
        
        # 运行指定次数的评估episode
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_components = defaultdict(list)
            
            while not done:
                action = self.model.predict(obs, deterministic=self.deterministic)[0]
                obs, _, done, infos = self.eval_env.step(action)
                
                # 收集分项奖励值
                for info in infos:  # 处理向量化环境，实际上只会运行一次
                    if 'reward' in info:
                        components = info['reward']
                        for k, v in components.items():
                            episode_components[k].append(v)
            
            # 计算并存储本episode的平均分项值
            for component in episode_components:
                values = episode_components[component]
                self.reward_component_values[component].append(np.mean(values) if values else 0)
    
    def log_component_values(self):
        """将分项奖励值记录到TensorBoard"""
        for component, values in self.reward_component_values.items():
            if values:  # 确保有数据可记录
                mean_value = np.mean(values)
                self.logger.record(f"eval/reward/{component}", mean_value)


class EpisodeCurriculumCallback(BaseCallback):
    """
    一个基于Episode、支持“休眠”的通用课程学习Callback。
    作用是自动调节环境对手难度

    在达到阈值时间步后，此Callback将自动停用以减少计算开销。
    它通过一个通用的接口 `update_task_parameters` 来更新环境参数。
    """
    def __init__(self, threshold_timesteps: int, update_freq_episodes: int, verbose: int = 1):
        """
        初始化Callback。

        :param threshold_timesteps: (int) 达到该时间步时，"goal_point" 概率变为0。
        :param update_freq_episodes: (int) 每隔多少个 episode 结束就更新一次概率。
        :param verbose: (int) 日志级别。
        """
        super().__init__(verbose)
        self.threshold_timesteps = threshold_timesteps
        self.update_freq_episodes = update_freq_episodes
        
        # 内部状态
        self.episode_count = 0
        self.last_update_prob = 1.0
        self.is_active = True  # 关键：Callback的活动状态标志

        if self.verbose > 0:
            logging.info("通用课程学习Callback已初始化。")
            logging.info(f"阈值: {self.threshold_timesteps}步, 更新频率: 每 {self.update_freq_episodes} 个 episodes")

    def _on_step(self) -> bool:
        """
        在每个环境步骤后被调用。
        """
        # === 关键：性能优化 ===
        # 如果Callback已停用，则立即返回，不做任何事情
        if not self.is_active:
            return True

        dones = self.locals.get("dones", np.array([False] * self.training_env.num_envs))

        if np.any(dones):
            self.episode_count += np.sum(dones)

            if self.episode_count >= self.update_freq_episodes:
                progress = self.num_timesteps / self.threshold_timesteps
                current_prob = np.clip(1.0 - progress, 0.0, 1.0)
                
                if not np.isclose(current_prob, self.last_update_prob):
                    if self.verbose > 0:
                        logging.info(
                            f"Timestep {self.num_timesteps}: 课程更新 -> goal_point_prob = {current_prob:.4f}"
                        )
                    
                    # === 关键：调用新的通用接口 ===
                    # 使用关键字参数来指定要更新的参数名和值
                    self.training_env.env_method(
                        "update_task_parameters",
                        goal_point_prob=current_prob
                    )
                    
                    self.last_update_prob = current_prob
                
                self.episode_count = 0

                # === 关键：检查是否应停用Callback ===
                if current_prob == 0.0:
                    self.is_active = False
                    if self.verbose > 0:
                        logging.info(
                            f"课程学习完成于 Timestep {self.num_timesteps}。"
                            "Callback 将进入休眠模式以优化性能。"
                        )
        
        return True