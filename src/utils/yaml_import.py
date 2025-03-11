
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

def str2class(config_dict, process_function, flag=":"):
    """
    递归处理字典，将含有":"的字符串值通过指定函数处理
    
    Args:
        config_dict: 要处理的字典
        process_function: 处理函数，接收字符串，返回处理后的值
    
    Returns:
        处理后的字典
    """
    if not isinstance(config_dict, dict):
        return config_dict
    
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # 递归处理嵌套字典
            result[key] = str2class(value, process_function)
        elif isinstance(value, list):
            # 处理列表
            result[key] = [str2class(item, process_function) 
                          if isinstance(item, dict) else item 
                          for item in value]
        elif isinstance(value, str) and flag in value:
            # 处理包含":"的字符串
            result[key] = process_function(value)
        else:
            # 其他值保持不变
            result[key] = value
    
    return result


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