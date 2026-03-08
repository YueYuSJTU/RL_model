import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
import torch

class ObsAdaptingModel:
    """
    一个策略包装器，用于根据环境配置调整观察值，然后调用原始模型进行预测。
    主要用于处理不同任务（如TrackingTask和GoalPointTask）之间的观察值差异。
    一般用在opponent模型上
    """
    def __init__(self, model: BaseAlgorithm, env_config: dict, current_obs_names: list = None):
        """
        构造函数。

        :param model: 要包装的stable-baselines3模型。
        :param env_config: 与模型关联的环境配置。
        :param current_obs_names: 当前环境的观察特征名称列表。
        """
        self.model = model
        self.env_config = env_config
        self.current_obs_names = current_obs_names or []
        self._model_obs_dim = None
        if hasattr(model, "observation_space") and getattr(model.observation_space, "shape", None) is not None:
            # SB3 models store the observation_space used during training
            self._model_obs_dim = int(model.observation_space.shape[-1])
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
        根据环境配置中的任务名称和观察配置调整观察。
        """
        # 如果没有提供当前环境的特征名称，回退到旧的基于任务名称的硬编码逻辑
        if not self.current_obs_names:
            task_name = self.env_config.get("task", "")

            if "GoalPointTask" in task_name:
                num_base = 9
                num_tracking = 19
                num_extra_tracking = 8
                num_oppo = 18

                expected_dim = num_base + num_tracking + num_extra_tracking + num_oppo + 4
                if obs.shape[-1] != expected_dim:
                    raise ValueError(f"Expected observation dimension {expected_dim}, but got {obs.shape[-1]}")

                base_obs = obs[..., :num_base]
                tracking_obs = obs[..., num_base : num_base + num_tracking]
                extra_obs = obs[..., num_base + num_tracking : num_base + num_tracking + num_extra_tracking]
                action_obs = obs[..., num_base + num_tracking + num_extra_tracking + num_oppo:]

                extra_obs_for_goal = np.concatenate([extra_obs[..., :6], extra_obs[..., 7:]], axis=-1)
                adapted_obs = np.concatenate([base_obs, tracking_obs, extra_obs_for_goal, action_obs], axis=-1)
                return adapted_obs
            else:
                return obs

        # 动态映射逻辑
        # 1. 获取对手模型期望的特征名称
        opponent_obs_names = self._get_opponent_obs_names()

        # 如果无法获取对手的特征名称，或者特征名称完全一致，则直接返回
        if not opponent_obs_names or opponent_obs_names == self.current_obs_names:
            return obs

        # 2. 构建映射
        adapted_obs_shape = list(obs.shape)
        adapted_obs_shape[-1] = len(opponent_obs_names)
        adapted_obs = np.zeros(adapted_obs_shape, dtype=obs.dtype)

        for i, name in enumerate(opponent_obs_names):
            if name in self.current_obs_names:
                # 如果当前环境有这个特征，复制过来
                idx = self.current_obs_names.index(name)
                adapted_obs[..., i] = obs[..., idx]
            else:
                # 如果当前环境没有这个特征，用0填充 (已经在np.zeros中初始化为0)
                pass

        return adapted_obs

    def _get_opponent_obs_names(self) -> list:
        """
        根据对手的环境配置推断其期望的观察特征名称。
        """
        from jsbgym_m.task_tracking import TrackingTask
        from jsbgym_m.task_goal_point import GoalPointTask
        from jsbgym_m.tasks import FlightTask

        task_name = self.env_config.get("task", "")
        obs_config = self.env_config.get("obs_config", None)

        # 临时实例化一个Task来获取state_variables
        # 这里我们只需要获取类属性，不需要真正的环境
        state_variables = []

        if "TrackingTask" in task_name:
            # New semantics: obs_config masks (zeros) dimensions but does not change shape.
            full_state_variables = (
                FlightTask.base_state_variables
                + TrackingTask.tracking_state_variables
                + TrackingTask.extra_state_variables
                + TrackingTask.oppo_state_variables
                + TrackingTask.action_variables
            )
            full_names = [prop.name for prop in full_state_variables]

            # If we can infer the model's expected obs dim and it matches, return full names.
            if self._model_obs_dim is None or self._model_obs_dim == len(full_names):
                return full_names

            # Otherwise, fall back to legacy behavior (older models trained with reduced obs dim).
            state_variables = []
            if obs_config is not None:
                if obs_config.get("base", True):
                    state_variables.extend(FlightTask.base_state_variables)
                if obs_config.get("tracking", True):
                    state_variables.extend(TrackingTask.tracking_state_variables)
                if obs_config.get("extra", True):
                    state_variables.extend(TrackingTask.extra_state_variables)
                if obs_config.get("oppo", True):
                    state_variables.extend(TrackingTask.oppo_state_variables)
                if obs_config.get("action", True):
                    state_variables.extend(TrackingTask.action_variables)

                exclude_list = obs_config.get("exclude", [])
                if exclude_list:
                    state_variables = [prop for prop in state_variables if prop.name not in exclude_list]
            else:
                state_variables = list(full_state_variables)

            names = [prop.name for prop in state_variables]
            if len(names) > self._model_obs_dim:
                return names[: self._model_obs_dim]
            if len(names) < self._model_obs_dim:
                pad = [f"__pad_{i}__" for i in range(self._model_obs_dim - len(names))]
                return names + pad
            return names
        elif "GoalPointTask" in task_name:
            # GoalPointTask 的 extra_state_variables 是在 __init__ 中动态创建的
            # 我们需要手动构建它
            extra_state_variables = (
                TrackingTask.distance_oppo_ft,
                TrackingTask.track_angle_rad,
                TrackingTask.bearing_accountingRollPitch_rad,
                TrackingTask.elevation_accountingRollPitch_rad,
                TrackingTask.bearing_pointMass_rad,
                TrackingTask.elevation_pointMass_rad,
                TrackingTask.closure_rate,
            )

            if obs_config is not None:
                if obs_config.get("base", True):
                    state_variables.extend(FlightTask.base_state_variables)
                if obs_config.get("tracking", True):
                    state_variables.extend(TrackingTask.tracking_state_variables)
                if obs_config.get("extra", True):
                    state_variables.extend(extra_state_variables)
                if obs_config.get("oppo", False):
                    state_variables.extend(TrackingTask.oppo_state_variables)
                if obs_config.get("action", True):
                    state_variables.extend(TrackingTask.action_variables)

                exclude_list = obs_config.get("exclude", [])
                if exclude_list:
                    state_variables = [prop for prop in state_variables if prop.name not in exclude_list]
            else:
                state_variables = (
                    FlightTask.base_state_variables
                    + TrackingTask.tracking_state_variables
                    + extra_state_variables
                    + TrackingTask.action_variables
                )
        else:
            return []

        return [prop.name for prop in state_variables]
