import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm  # 导入tqdm库
import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from typing import Dict, Tuple
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from src.evaluate_pool import evaluate_versus, save_results
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(exp_path: str, pool_path: str, n_episodes: int = 500) -> None:
    """
    对于exp_path中的模型，遍历pool_name中的所有对手，进行蒙特卡洛仿真评估。
    与每个对手进行对战的结果将保存在exp_path中。
    
    Args:
        exp_path: 实验路径，包含模型和环境配置
        pool_path: 对手池路径
        n_episodes: 蒙特卡洛模拟次数（默认：500）
    """
    # 扫描模型池中的所有模型
    print(f"正在评估模型池 {pool_path} 中的模型...")
    model_dirs = [d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))]
    # model_dirs = sorted(model_dirs, key=int)
    model_dirs = sorted(map(int, model_dirs))  # 将目录名转换为整数
    results = {}

    for model_dir in model_dirs:        
        # 与其他模型进行对战
        print(f"模型 {model_dir} vs 主控 对战中...")
        
        # 进行对战评估
        win_rate, draw_rate, loss_rate, opponent_fall_rate, avg_win_time, avg_reward = evaluate_versus(
            exp_path, 
            pool_path,
            model_dir, 
            n_episodes=n_episodes
        )
        
        # 记录结果
        results[model_dir] = {
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "opponent_fall_rate": opponent_fall_rate,
            "avg_win_time": avg_win_time,
            "avg_reward": avg_reward
        }
    
    # 保存这个模型的所有对战结果
    save_results(
        results=results,
        log_dir=exp_path,
        model_name="ppo_model"
    )


if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='评估强化学习模型性能')
    parser.add_argument('--exp_path', type=str, required=True, 
                        help='实验路径，包含模型和环境配置')
    parser.add_argument('--pool_path', type=str, required=True, 
                        help='对手池路径')
    parser.add_argument('--n_episodes', type=int, default=500, 
                        help='蒙特卡洛模拟次数（默认：500）')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用评估函数
    evaluate(args.exp_path, args.pool_path, args.n_episodes)
