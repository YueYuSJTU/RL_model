import logging
from typing import Optional
from stable_baselines3.common.logger import configure
import os

def setup_logger(log_path: str) -> logging.Logger:
    """配置统一日志系统"""
    logger = configure(
        folder=os.path.join(log_path, "metrics"),
        format_strings=["csv", "tensorboard"]
    )
    return logger