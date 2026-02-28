import os
import numpy as np
import gymnasium as gym
from typing import Union, List, Optional
from stable_baselines3 import PPO
from src.agents.model_wrapper import ObsAdaptingModel

class SelfPlayWrapper(gym.Wrapper):
    """
    A Gym Wrapper that handles opponent policy inference.
    It splits the observation, queries the opponent model for an action,
    and combines it with the agent's action before stepping the base environment.
    """
    def __init__(
        self,
        env: gym.Env,
        pool_roots: Optional[Union[str, List[str]]] = None,
        model_num: int = 0
    ):
        super().__init__(env)
        self.pool_roots = pool_roots
        self.model_num = model_num

        # Split observation and action spaces
        self.observation_space, self.action_space = self._get_space(env.observation_space, env.action_space)

        self.opponent_models = []
        self.opponent_stats = {}
        self.opponent_weights = []
        self.current_strategy_idx = 0
        self.opponent_observation = None

        self.update_opponent_models()

    def _get_space(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        # Observation space
        obs_shape = observation_space.shape
        obs_split_dim = obs_shape[-1] // 2
        new_obs_shape = obs_shape[:-1] + (obs_split_dim,)
        obs_low = observation_space.low[..., :obs_split_dim]
        obs_high = observation_space.high[..., :obs_split_dim]
        half_observation_space = gym.spaces.Box(low=obs_low, high=obs_high, shape=new_obs_shape, dtype=observation_space.dtype)

        # Action space
        act_shape = action_space.shape
        act_split_dim = act_shape[-1] // 2
        new_act_shape = act_shape[:-1] + (act_split_dim,)
        act_low = action_space.low[..., :act_split_dim]
        act_high = action_space.high[..., :act_split_dim]
        half_action_space = gym.spaces.Box(low=act_low, high=act_high, shape=new_act_shape, dtype=action_space.dtype)

        return half_observation_space, half_action_space

    def update_opponent_models(self):
        if self.model_num == -1 or not self.pool_roots:
            return

        strategy_dirs = self._find_strategy_dirs(self.pool_roots)
        self.opponent_models = []

        for root in strategy_dirs:
            model = self._load_opponent_model(root)
            env_config = self._load_opponent_env_config(root)
            wrapped_model = ObsAdaptingModel(model, env_config)
            self.opponent_models.append(wrapped_model)

        self.opponent_stats = {i: {'wins': 0, 'games': 0} for i in range(len(self.opponent_models))}
        self.opponent_weights = np.ones(len(self.opponent_models)) * 9.0

    def _find_strategy_dirs(self, root_dir: str) -> List[str]:
        strategy_dirs = []
        if not os.path.exists(root_dir):
            return strategy_dirs

        if self.model_num > 0:
            model_path = os.path.join(root_dir, f"{self.model_num}")
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "best_model.zip")):
                strategy_dirs.append(model_path)
            return strategy_dirs

        for item in os.listdir(root_dir):
            full_path = os.path.join(root_dir, item)
            if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, "best_model.zip")):
                strategy_dirs.append(full_path)

        return strategy_dirs

    def _load_opponent_model(self, model_path: str):
        from src.environments.make_env import create_env
        model_file = os.path.join(model_path, "best_model.zip")
        env_config = self._load_opponent_env_config(model_path)
        vec_env = create_env(env_config, training=False, vec_env_kwargs={"model_num": -1})
        vec_env.training = False
        vec_env.norm_reward = False
        model = PPO.load(model_file, vec_env, device="cuda")
        return model

    def _load_opponent_env_config(self, model_path: str) -> dict:
        import yaml
        config_file = os.path.join(model_path, "env_config.yaml")
        with open(config_file, encoding="utf-8") as f:
            env_config = yaml.safe_load(f)
        return env_config

    def _sample_strategy(self):
        if not self.opponent_models:
            return
        total_weight = np.sum(self.opponent_weights)
        probs = self.opponent_weights / total_weight if total_weight > 0 else np.ones(len(self.opponent_models)) / len(self.opponent_models)
        self.current_strategy_idx = np.random.choice(len(self.opponent_models), p=probs)

    def _get_opponent_action(self, obs: np.ndarray) -> np.ndarray:
        if not self.opponent_models:
            action = np.random.uniform(-1, 1, size=(4,))
            action[-1] = np.abs(action[-1])
            return action

        wrapped_model = self.opponent_models[self.current_strategy_idx]
        # Add batch dimension for prediction
        action, _ = wrapped_model.predict(np.expand_dims(obs, axis=0))
        return action[0]

    def reset(self, **kwargs):
        self._sample_strategy()
        obs, info = self.env.reset(**kwargs)
        self.opponent_observation = self._get_observation(obs, "opponent")
        agent_obs = self._get_observation(obs, "agent")
        return agent_obs, info

    def step(self, action):
        opponent_action = self._get_opponent_action(self.opponent_observation)
        combined_action = np.concatenate((action, opponent_action))

        obs, reward, terminated, truncated, info = self.env.step(combined_action)

        self.opponent_observation = self._get_observation(obs, "opponent")
        agent_obs = self._get_observation(obs, "agent")

        if terminated or truncated:
            if self.opponent_models:
                is_win = info.get("env_info", {}).get("win", 0) == 1
                self._update_strategy_weight(self.current_strategy_idx, is_win)

        return agent_obs, reward, terminated, truncated, info

    def _update_strategy_weight(self, strategy_idx: int, won: bool):
        stats = self.opponent_stats[strategy_idx]
        stats['games'] += 1
        if won:
            stats['wins'] += 1

        current_weight = self.opponent_weights[strategy_idx]
        new_weight = current_weight - 0.1 if won else current_weight + 0.1
        self.opponent_weights[strategy_idx] = np.clip(new_weight, 1.0, 9.0)

    def _get_observation(self, obs: np.ndarray, object: str = "agent") -> np.ndarray:
        split_point = obs.shape[-1] // 2
        if object == "agent":
            return obs[..., :split_point]
        elif object == "opponent":
            return obs[..., split_point:]
        else:
            raise ValueError(f"Invalid object type: {object}")
