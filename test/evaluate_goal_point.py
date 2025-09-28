import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
import sys
from collections import defaultdict

sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from typing import Dict, Tuple
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate_goal_point(exp_path: str, n_episodes: int = 500, use_tqdm: bool = True) -> None:
    """
    评估模型在目标点任务中的表现。

    Args:
        exp_path: 实验路径，包含模型和环境配置。
        n_episodes: 蒙特卡洛模拟次数。
        use_tqdm: 是否使用tqdm显示评估进度。
    """
    # 加载配置
    env_cfg = load_config(os.path.join(exp_path, "env_config.yaml"))
    agent_cfg = load_config(os.path.join(exp_path, "agent_config.yaml"))

    # 配置评估环境
    env_cfg["render_mode"] = None
    env_cfg["use_vec_normalize"] = False
    
    # 创建评估环境，设置 model_num 为 -1 表示没有对手
    vec_env_kwargs = {"model_num": -1}
    vec_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
    
    # 加载环境标准化器
    env_pkl = "best_env.pkl" if os.path.exists(os.path.join(exp_path, "best_env.pkl")) else "final_train_env.pkl"
    vec_env = VecNormalize.load(
        os.path.join(exp_path, env_pkl),
        vec_env
    )
    vec_env.training = False
    vec_env.norm_reward = False

    # 加载模型
    model = PPO.load(
        os.path.join(exp_path, "best_model"),
        env=vec_env,
        device=agent_cfg["device"]
    )

    # 存储每个模式的结果
    results = defaultdict(lambda: {"success": 0, "crash": 0, "draw": 0, "total": 0})

    print(f"正在评估模型 {exp_path}...")
    
    episodes = range(n_episodes)
    if use_tqdm:
        episodes = tqdm(episodes, desc="评估进度", ncols=80)

    for _ in episodes:
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, info = vec_env.step(action)
            
            if terminated:
                done = True
                env_info = info[0].get("env_info", {})
                win_status = env_info.get("win", 0)
                mode = env_info.get("mode", "unknown")

                results[mode]["total"] += 1
                if win_status == 1:  # 成功
                    results[mode]["success"] += 1
                elif win_status == -1:  # 坠机
                    results[mode]["crash"] += 1
                else:  # 平局
                    results[mode]["draw"] += 1
    
    # 保存结果
    save_goal_point_results(results, exp_path)

def save_goal_point_results(results: Dict, log_dir: str) -> None:
    """
    保存目标点任务的评估结果。

    Args:
        results: 评估结果字典。
        log_dir: 保存路径。
    """
    output_file = os.path.join(log_dir, 'goal_point_evaluation_summary.txt')
    with open(output_file, 'w') as f:
        f.write("目标点任务评估结果\n")
        f.write("=" * 40 + "\n")

        # 计算总体统计数据
        total_success = sum(stats['success'] for stats in results.values())
        total_crash = sum(stats['crash'] for stats in results.values())
        total_draw = sum(stats['draw'] for stats in results.values())
        grand_total = sum(stats['total'] for stats in results.values())

        if grand_total > 0:
            total_success_rate = total_success / grand_total
            total_crash_rate = total_crash / grand_total
            total_draw_rate = total_draw / grand_total
            
            f.write("总体统计:\n")
            f.write(f"  总次数: {grand_total}\n")
            f.write(f"  总成功率: {total_success_rate:.2%}\n")
            f.write(f"  总坠机率: {total_crash_rate:.2%}\n")
            f.write(f"  总平局率: {total_draw_rate:.2%}\n")
            f.write("=" * 40 + "\n")

        f.write("各模式详细统计:\n")
        for mode, stats in sorted(results.items()):
            total = stats['total']
            if total == 0:
                continue
            
            success_rate = stats['success'] / total
            crash_rate = stats['crash'] / total
            draw_rate = stats['draw'] / total

            f.write(f"模式: {mode}\n")
            f.write(f"  总次数: {total}\n")
            f.write(f"  成功率: {success_rate:.2%}\n")
            f.write(f"  坠机率: {crash_rate:.2%}\n")
            f.write(f"  平局率: {draw_rate:.2%}\n")
            f.write("-" * 40 + "\n")

    print(f"评估结果已保存至 {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='评估模型在目标点任务中的性能')
    parser.add_argument('--exp_path', type=str, required=True,
                        help='实验路径，包含模型和环境配置')
    parser.add_argument('--n_episodes', type=int, default=500,
                        help='蒙特卡洛模拟次数 (默认: 500)')

    args = parser.parse_args()
    evaluate_goal_point(args.exp_path, args.n_episodes)
