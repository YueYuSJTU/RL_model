import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class ContinueObservation(gym.ObservationWrapper):
    def __init__(
            self, 
            env, 
            skip_frame: int = 4,
            total_frame: int = 5,
        ):
        super().__init__(env)
        self.observation_space = Box(
            low=np.array([self.observation_space.low]*total_frame),
            high=np.array([self.observation_space.high]*total_frame),
            dtype=np.float64
        )
        self.skip_frame = skip_frame
        self.total_frame = total_frame

    def _get_obs(self):
        return np.array([self._state_list[i] for i in range(0, len(self._state_list), self.skip_frame)])

    def reset(self, seed=None):
        self._state_list = []
        obs, info = self.env.reset(seed=seed)
        self._state_list.append(obs)
        # for _ in range(self.total_frame + self.skip_frame*(self.total_frame-1) - 1):
        for _ in range(self.skip_frame*(self.total_frame) - 2):
            self._state_list.append(obs)
        self._state = self._get_obs()
        return self._state, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # self._state_list.pop()
        # self._state_list.insert(0, obs)
        self._state_list.pop(0)
        self._state_list.append(obs)
        self._state = self._get_obs()
        return self._state, reward, terminated, truncated, info


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    print(env.observation_space)
    print(env.reset())
    print("ContinueObservation")
    env = ContinueObservation(env)
    print(env.observation_space.shape)
    print(env.reset())