from stable_baselines3 import PPO
from agents.gru.gru_nn import RNNEncoder
import yaml

# def custom_loader(loader, node):
#     value = loader.construct_scalar(node)
#     if value == "RNNEncoder":
#         return RNNEncoder
#     return value

# yaml.add_constructor('!custom', custom_loader)


def creat_agent(env, agent_class: str, tensorboard_log: str, agent_cfg):
    """创建PPO代理"""
    if "ppo" in agent_class.lower():
        model_class = PPO
    else:
        raise ValueError(f"Unknown agent class: {agent_class}")
    
    model = model_class(
        env=env,
        tensorboard_log=tensorboard_log,
        **agent_cfg
    )

    return model