import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

class ObsAdaptingModel:
    """
    一个策略包装器，用于根据环境配置调整观察值，然后调用原始模型进行预测。
    主要用于处理不同任务（如TrackingTask和GoalPointTask）之间的观察值差异。
    一般用在opponent模型上
    """
    def __init__(self, model: BaseAlgorithm, env_config: dict):
        """
        构造函数。

        :param model: 要包装的stable-baselines3模型。
        :param env_config: 与模型关联的环境配置。
        """
        self.model = model
        self.env_config = env_config
        # 暴露内部模型的策略，以便外部代码可以访问action_space等属性
        self.policy = model.policy

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic: bool = False):
        """
        调整观察，然后使用原始模型进行预测。
        """
        # 检查是否需要对时间序列数据进行特殊处理
        observation = self._handle_time_series(observation)
        
        # 根据任务类型调整观察
        adapted_obs = self._adapt_observation_for_task(observation)
        
        with torch.no_grad():
            return self.model.predict(adapted_obs, state=state, episode_start=episode_start, deterministic=deterministic)

    def _handle_time_series(self, obs: np.ndarray) -> np.ndarray:
        """
        如果模型环境不使用ContinueObservation，但输入是时间序列，则只取最后一个时间步。
        """
        has_continue_wrapper = False
        if "wrappers" in self.env_config:
            for wrapper in self.env_config["wrappers"]:
                if isinstance(wrapper, dict) and wrapper.get("name") == "src.environments.ContinueWrapper:ContinueObservation":
                    has_continue_wrapper = True
                    break
        
        if not has_continue_wrapper and len(obs.shape) == 3:
            return obs[:, -1, :]  # 只取最新的观察值
        return obs

    def _adapt_observation_for_task(self, obs: np.ndarray) -> np.ndarray:
        """
        根据环境配置中的任务名称调整观察。
        这个逻辑是从NNVecEnv迁移过来的。
        """
        task_name = self.env_config.get("task", "")

        if "GoalPointTask" in task_name:
            # GoalPointTask的state_variables比TrackingTask少oppo_state_variables
            # 并且extra_state_variables也少一个变量。
            # 这个函数假定输入obs是基于TrackingTask的完整观察结构。
            
            num_base = 9
            num_tracking = 19
            num_extra_tracking = 8
            num_oppo = 18
            
            # 检查输入维度是否符合TrackingTask的结构
            expected_dim = num_base + num_tracking + num_extra_tracking + num_oppo + 4
            if obs.shape[-1] != expected_dim:
                raise ValueError(f"Expected observation dimension {expected_dim}, but got {obs.shape[-1]}")

            base_obs = obs[..., :num_base]
            tracking_obs = obs[..., num_base : num_base + num_tracking]
            extra_obs = obs[..., num_base + num_tracking : num_base + num_tracking + num_extra_tracking]
            action_obs = obs[..., num_base + num_tracking + num_extra_tracking + num_oppo:]

            # 从extra_obs中移除adverse_angle_rad (第7个, index 6)
            extra_obs_for_goal = np.concatenate([extra_obs[..., :6], extra_obs[..., 7:]], axis=-1)
            
            # 重新组合成GoalPointTask的观察
            adapted_obs = np.concatenate([base_obs, tracking_obs, extra_obs_for_goal, action_obs], axis=-1)
            
            return adapted_obs
            
        elif "TrackingTask" in task_name:
            # 如果对手也是TrackingTask，则观察值结构匹配，无需更改
            return obs
        else:
            # 对于未知任务类型，默认不进行任何修改
            return obs
