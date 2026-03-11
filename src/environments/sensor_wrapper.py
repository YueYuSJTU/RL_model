import numpy as np
import gymnasium as gym
from typing import Optional, Sequence


class SensorPerturbationWrapper(gym.ObservationWrapper):
    """Apply missing/noise to enemy-related features in the *agent half* of observation.

    Intended for envs where observation is either N (agent-only) or 2N (agent+opponent).
    We only perturb the agent half, and only dimensions whose Property.name matches
    configured prefixes (default: target/, oppo/).
    """

    def __init__(
        self,
        env: gym.Env,
        missing_prob: float = 0.0,
        rel_error_std: float = 0.0,
        prefixes: Optional[Sequence[str]] = ("target/", "oppo/"),
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        self.missing_prob = float(missing_prob)
        self.rel_error_std = float(rel_error_std)
        self.prefixes = list(prefixes) if prefixes is not None else ["target/", "oppo/"]
        self._mask_agent_half: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(seed)

    def _build_mask_if_needed(self) -> None:
        if self._mask_agent_half is not None:
            return

        task = getattr(self.env.unwrapped, "task", None)
        state_vars = getattr(task, "state_variables", None)
        if not state_vars:
            self._mask_agent_half = np.zeros((0,), dtype=bool)
            return

        names = []
        for prop in state_vars:
            name = getattr(prop, "name", None)
            names.append(name if isinstance(name, str) else "")

        self._mask_agent_half = np.array(
            [any(n.startswith(p) for p in self.prefixes) for n in names],
            dtype=bool,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._build_mask_if_needed()
        return self.observation(obs), info

    def observation(self, obs):
        self._build_mask_if_needed()

        if self.missing_prob <= 0.0 and self.rel_error_std <= 0.0:
            return obs

        arr = np.asarray(obs)
        if arr.ndim not in (1, 2):
            return obs

        mask = self._mask_agent_half
        if mask is None or mask.size == 0:
            return obs

        out = arr.copy()
        N = int(mask.shape[0])
        total_dim = int(out.shape[-1])

        # Decide which slice corresponds to the agent half
        if total_dim == 2 * N:
            sl = slice(0, N)
            mask_eff = mask
        elif total_dim == N:
            sl = slice(0, N)
            mask_eff = mask
        else:
            # If obs_config/exclude changed dims, align conservatively
            N_eff = min(N, total_dim // 2 if total_dim % 2 == 0 else total_dim)
            sl = slice(0, N_eff)
            mask_eff = mask[:N_eff]

        if out.ndim == 1:
            agent = out[sl]

            sel = mask_eff
            if self.missing_prob > 0.0:
                miss = (self._rng.random(agent.shape[0]) < self.missing_prob) & sel
                agent[miss] = 0
                sel_non_missing = sel & (~miss)
            else:
                sel_non_missing = sel

            if self.rel_error_std > 0.0 and np.any(sel_non_missing):
                noise = self._rng.normal(0.0, self.rel_error_std, size=agent.shape[0])
                agent[sel_non_missing] = agent[sel_non_missing] * (1.0 + noise[sel_non_missing])

            out[sl] = agent

        else:
            agent = out[:, sl]
            sel2 = np.broadcast_to(mask_eff, agent.shape)

            if self.missing_prob > 0.0:
                miss = (self._rng.random(agent.shape) < self.missing_prob) & sel2
                agent[miss] = 0
                sel_non_missing2 = sel2 & (~miss)
            else:
                sel_non_missing2 = sel2

            if self.rel_error_std > 0.0 and np.any(sel_non_missing2):
                noise = self._rng.normal(0.0, self.rel_error_std, size=agent.shape)
                agent[sel_non_missing2] = agent[sel_non_missing2] * (1.0 + noise[sel_non_missing2])

            out[:, sl] = agent

        # print(f"SensorPerturbationWrapper: missing_prob={self.missing_prob}, rel_error_std={self.rel_error_std}, "
        #       f"num_perturbed={np.sum(sel)}, num_missing={np.sum(miss) if self.missing_prob > 0.0 else 0}")
        if isinstance(obs, np.ndarray):
            return out.astype(obs.dtype, copy=False)
        return out
