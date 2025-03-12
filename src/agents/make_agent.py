from stable_baselines3 import PPO
import yaml
from src.utils.yaml_import import import_class, str2class

def creat_agent(env, agent_class: str, tensorboard_log: str, agent_cfg):
    """创建PPO代理"""
    if "ppo" in agent_class.lower():
        model_class = PPO
    else:
        raise ValueError(f"Unknown agent class: {agent_class}")
    
    agent_cfg = str2class(agent_cfg, import_class)
    # print(f"Debug: agent_cfg = {agent_cfg}")
    
    model = model_class(
        env=env,
        tensorboard_log=tensorboard_log,
        **agent_cfg
    )

    return model