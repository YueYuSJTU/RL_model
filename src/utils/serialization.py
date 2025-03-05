import yaml
import os
from pathlib import Path

def save_config(config: dict, log_path: str, filename: str):
    """保存配置到实验目录"""
    Path(log_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_path, filename), 'w') as f:
        yaml.safe_dump(config, f)

def load_config(config_path: str) -> dict:
    """从文件加载配置"""
    with open(config_path) as f:
        return yaml.safe_load(f)