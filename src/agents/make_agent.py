from __future__ import annotations

from typing import Any, Dict, Optional

from stable_baselines3 import PPO

from src.utils.yaml_import import import_class, str2class


def _resolve_algorithm(agent_class: str, agent_cfg: Dict[str, Any]):
    """Resolve SB3 algorithm class.

    Priority:
    1) agent_cfg['algorithm'] if present
    2) backward-compatible: infer from agent_class string ("ppo" -> PPO)

    Supported algorithms:
    - PPO (stable_baselines3.PPO)
    - RecurrentPPO (sb3_contrib.RecurrentPPO)
    """
    algo = agent_cfg.get("algorithm") if isinstance(agent_cfg, dict) else None

    if algo is None:
        # Strict behavior: algorithm MUST be specified in config.
        # Do not silently fallback to PPO.
        raise ValueError("Missing required agent_cfg['algorithm']")

    algo_l = str(algo).lower()
    if algo_l == "ppo":
        return PPO
    if algo_l in {"recurrentppo", "recurrent_ppo", "rppo"}:
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO

    raise ValueError(f"Unknown algorithm: {algo}")


def creat_agent(env, agent_class: str, tensorboard_log, agent_cfg):
    """创建代理（PPO / RecurrentPPO）"""
    model_class = _resolve_algorithm(agent_class, agent_cfg)
    # Print resolved algorithm for verification
    print(f"[algorithm] Resolved class: {getattr(model_class, '__name__', str(model_class))}")

    agent_cfg = str2class(agent_cfg, import_class)

    # Do not forward non-SB3 constructor keys
    agent_cfg.pop("algorithm", None)

    model = model_class(
        env=env,
        tensorboard_log=tensorboard_log,
        **agent_cfg,
    )

    return model

def load_agent(
    env,
    agent_class: str,
    path: str,
    device: str,
    agent_cfg: Optional[Dict[str, Any]] = None,
):
    """加载代理（PPO / RecurrentPPO）"""
    cfg = agent_cfg or {}
    model_class = _resolve_algorithm(agent_class, cfg)
    # Print resolved algorithm for verification
    print(f"[algorithm] Resolved class: {getattr(model_class, '__name__', str(model_class))}")

    model = model_class.load(
        path,
        env=env,
        device=device,
    )

    return model