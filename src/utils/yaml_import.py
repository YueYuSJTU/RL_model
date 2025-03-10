
import yaml
import os
import sys
import gymnasium as gym

def add_path(debug = False):
    # 将项目根目录添加到sys.path（使用相对路径）
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file_path))  # src目录
    project_root = os.path.dirname(src_dir)  # 项目根目录
    if debug:
        print(f"Debug: project root is {project_root}")
    sys.path.insert(0, project_root)

def import_class(class_path):
    module_path, class_name = class_path.split(":")
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


if __name__ == "__main__":
    add_path()
    with open("configs/train_config.yaml", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    env_path = os.path.join("configs", "env", f"{train_cfg['env']}.yaml")
    agent_path = os.path.join("configs", "agent", f"{train_cfg['agent']}.yaml")
    with open(agent_path, encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)
    with open(env_path, encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)

    # 动态加载Agent类
    AgentClass = import_class(env_cfg["wrappers"][0]["name"])
    print(AgentClass)
    # agent = AgentClass(**config["agent"]["params"])
    env = AgentClass(gym.make('CartPole-v1'))