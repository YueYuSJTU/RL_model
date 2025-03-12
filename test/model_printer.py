import yaml
from stable_baselines3 import PPO
import sys
import os
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments.make_env import create_env
from src.agents.make_agent import creat_agent

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_model():
    # 加载配置
    with open("configs/train_config.yaml", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    env_path = os.path.join("configs", "env", f"{train_cfg['env']}.yaml")
    agent_path = os.path.join("configs", "agent", f"{train_cfg['agent']}.yaml")
    with open(agent_path, encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)
    with open(env_path, encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    
    train_env = create_env(env_cfg, training=True, num_cpu=1)
    
    model = creat_agent(
        env=train_env,
        agent_class=train_cfg["agent"],
        tensorboard_log=os.path.join("./", "tensorboard"),
        agent_cfg=agent_cfg
    )
    return model

def print_model_structure(model):
    print(model.policy)

if __name__ == "__main__":
    model = load_model()
    print_model_structure(model)