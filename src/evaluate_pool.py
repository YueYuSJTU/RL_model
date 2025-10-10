import os
import gc
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from typing import Dict, List, Tuple
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate_versus(model_path: str, pool_path: str, opponent_num: int, n_episodes: int = 100, use_tqdm=True) -> Tuple[float, float, float, float, float, float]:
    """
    评估模型与指定编号的对手模型之间的对战
    
    Args:
        model_path: 主模型路径
        pool_path: 对手模型池路径
        opponent_num: 对手模型编号
        n_episodes: 对战次数
        
    Returns:
        win_rate: 主模型胜率
        draw_rate: 平局率
        loss_rate: 主模型失败率
        opponent_fall_rate: 对手自主坠机率
        avg_win_time: 获胜平均时间
        avg_reward: 平均奖励
    """
    # 加载主模型配置
    env_cfg = load_config(os.path.join(model_path, "env_config.yaml"))
    agent_cfg = load_config(os.path.join(model_path, "agent_config.yaml"))
    
    # 配置评估环境（无渲染）
    env_cfg["render_mode"] = None
    env_cfg["use_vec_normalize"] = False
    
    # 创建评估环境，设置对手模型
    vec_env_kwargs = {
        "pool_roots": pool_path,  # 对手模型池路径
        "model_num": opponent_num  # 对手模型编号
    }
    vec_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
    env_pkl = "best_env.pkl" if os.path.exists(os.path.join(model_path, "best_env.pkl")) else "final_train_env.pkl"
    vec_env = VecNormalize.load(
        os.path.join(model_path, env_pkl), 
        vec_env
    )
    vec_env.training = False
    vec_env.norm_reward = False

    # 加载主模型
    model = PPO.load(
        os.path.join(model_path, "best_model"),
        env=vec_env,
        device=agent_cfg["device"]
    )

    # 记录评估结果
    wins = 0
    losses = 0
    draws = 0
    opponent_falls = 0
    win_steps = []
    total_rewards = []  # 用于存储每场对战的总奖励
    
    try:
        # 运行对战，使用tqdm显示进度条
        episodes = range(n_episodes)
        if use_tqdm:
            episodes = tqdm(range(n_episodes), desc=f"对战对手{opponent_num}", ncols=80)

        for episode in episodes:
            obs = vec_env.reset()
            episode_done = False
            episode_reward = 0  # 初始化本场对战的奖励
            
            while not episode_done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, info = vec_env.step(action)
                
                episode_reward += reward[0]  # 累加奖励
                
                if terminated:
                    episode_done = True
                    total_rewards.append(episode_reward)  # 记录本场对战的总奖励
                    env_info = info[0].get("env_info", {})
                    win_status = env_info.get("win", 0)
                    
                    if win_status == 1:  # 主控胜利
                        wins += 1
                        win_steps.append(env_info.get("steps_used", 0))
                    elif win_status == 0:  # 平局
                        draws += 1
                    elif win_status == -1:  # 敌机胜利
                        losses += 1
                    elif win_status == 0.5: # 敌机自主坠机
                        opponent_falls += 1
        
        # 计算统计数据
        win_rate = wins / n_episodes
        draw_rate = draws / n_episodes
        loss_rate = losses / n_episodes
        opponent_fall_rate = opponent_falls / n_episodes
        avg_win_time = np.mean(win_steps) if win_steps else 0
        avg_reward = np.mean(total_rewards)  # 计算平均奖励
    
    finally:
        # === [关键] 显式清理代码 ===

        # 1. 关闭环境，这将终止所有相关的子进程
        if 'vec_env' in locals() and vec_env is not None:
            vec_env.close()
        
        # 2. 从内存中删除模型和环境对象
        if 'model' in locals():
            del model
        if 'vec_env' in locals():
            del vec_env
        
        # 3. 强制运行Python的垃圾回收器
        gc.collect()
        
        # 4. 强制清空PyTorch未被使用的CUDA缓存
        # 这是最重要的一步，它会把显存还给操作系统
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
  
    return win_rate, draw_rate, loss_rate, opponent_fall_rate, avg_win_time, avg_reward

def save_results(results: Dict, log_dir: str, model_name: str) -> None:
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        log_dir: 保存路径
        model_name: 模型名称
    """
    # 保存综合结果
    with open(os.path.join(log_dir, 'versus_summary.txt'), 'w') as f:
        f.write(f"模型: {model_name}\n\n")
        
        # 计算平均胜率
        avg_win_rate = np.mean([r["win_rate"] for r in results.values()])
        avg_draw_rate = np.mean([r["draw_rate"] for r in results.values()])
        avg_loss_rate = np.mean([r["loss_rate"] for r in results.values()])
        avg_opponent_fall_rate = np.mean([r["opponent_fall_rate"] for r in results.values()])
        avg_win_time = np.mean([r["avg_win_time"] for r in results.values() if r["avg_win_time"] > 0])
        avg_reward = np.mean([r["avg_reward"] for r in results.values()])
        
        f.write(f"平均胜率: {avg_win_rate:.2%}\n")
        f.write(f"平均平局率: {avg_draw_rate:.2%}\n")
        f.write(f"平均失败率: {avg_loss_rate:.2%}\n")
        f.write(f"平均对手自主坠机率: {avg_opponent_fall_rate:.2%}\n")
        f.write(f"平均获胜时间: {avg_win_time:.2f} 步\n")
        f.write(f"平均奖励: {avg_reward:.4f}\n\n")
        
        f.write("对战详情:\n")
        f.write("=" * 60 + "\n")
        for opponent, stats in results.items():
            f.write(f"vs {opponent}:\n")
            f.write(f"  胜率: {stats['win_rate']:.2%}\n")
            f.write(f"  平局率: {stats['draw_rate']:.2%}\n")
            f.write(f"  失败率: {stats['loss_rate']:.2%}\n")
            f.write(f"  对手自主坠机率: {stats['opponent_fall_rate']:.2%}\n")
            f.write(f"  平均获胜时间: {stats['avg_win_time']:.2f} 步\n")
            f.write(f"  平均奖励: {stats['avg_reward']:.4f}\n")
            f.write("-" * 40 + "\n")
    
    # 保存详细数据
    np.save(os.path.join(log_dir, 'versus_results.npy'), results)
    
    print(f"模型 {model_name} 的评估结果已保存至 {log_dir}")

def evaluate_pool(pool_path: str, n_episodes: int = 100) -> None:
    """
    评估模型池中所有模型之间的对战表现
    
    Args:
        pool_path: 模型池路径，包含多个模型
        log_dir: 结果保存路径
        n_episodes: 每对模型之间的对战次数
    """
    # # 创建保存目录
    # os.makedirs(log_dir, exist_ok=True)
    
    # 扫描模型池中的所有模型
    model_dirs = [d for d in os.listdir(pool_path) if os.path.isdir(os.path.join(pool_path, d))]
    
    # 对每个模型进行评估
    for model_dir in model_dirs:
        model_path = os.path.join(pool_path, model_dir)
        # model_log_dir = os.path.join(log_dir, model_dir)
        # os.makedirs(model_log_dir, exist_ok=True)
        
        # # 获取当前模型编号
        # try:
        #     model_num = int(model_dir)
        # except ValueError:
        #     print(f"警告: 模型目录 {model_dir} 不是数字，使用索引作为编号")
        #     model_num = model_dirs.index(model_dir) + 1
        
        # 与其他模型进行对战
        results = {}
        for opponent_dir in model_dirs:
            if model_dir == opponent_dir:  # 跳过与自己的对战
                continue
                
            try:
                opponent_num = int(opponent_dir)
            except ValueError:
                print(f"警告: 对手目录 {opponent_dir} 不是数字，使用索引作为编号")
                opponent_num = model_dirs.index(opponent_dir) + 1
                
            print(f"模型 {model_dir} vs {opponent_dir} 对战中...")
            
            # 进行对战评估
            win_rate, draw_rate, loss_rate, opponent_fall_rate, avg_win_time, avg_reward = evaluate_versus(
                model_path, 
                pool_path,
                opponent_num, 
                n_episodes=n_episodes
            )
            
            # 记录结果
            results[opponent_dir] = {
                "win_rate": win_rate,
                "draw_rate": draw_rate,
                "loss_rate": loss_rate,
                "opponent_fall_rate": opponent_fall_rate,
                "avg_win_time": avg_win_time,
                "avg_reward": avg_reward
            }
        
        # 保存这个模型的所有对战结果
        save_results(results, model_path, model_dir)

def show_evaluate_results(log_dir: str) -> None:
    """
    显示评估结果
    
    Args:
        log_dir: 结果保存路径
    """
    import matplotlib.pyplot as plt
    # Get all model directories
    model_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    
    if not model_dirs:
        print(f"Warning: No model directories found in {log_dir}")
        return
    
    # Store win rates for each model
    model_names = []
    win_rates = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(log_dir, model_dir)
        results_file = os.path.join(model_path, 'versus_results.npy')
        
        # Check if results file exists
        if not os.path.exists(results_file):
            print(f"Warning: Evaluation results not found for model {model_dir}")
            continue
        
        try:
            # Load results
            results = np.load(results_file, allow_pickle=True).item()
            
            # Calculate average win rate
            avg_win_rate = np.mean([r["win_rate"] for r in results.values()])
            
            # Store results
            model_names.append(model_dir)
            win_rates.append(avg_win_rate)
        except Exception as e:
            print(f"Error processing model {model_dir}: {e}")
    
    if not model_names:
        print("No valid evaluation results found")
        return
    
    # Sort model names numerically
    sorted_indices = sorted(range(len(model_names)), key=lambda i: int(model_names[i]))
    model_names = [model_names[i] for i in sorted_indices]
    win_rates = [win_rates[i] for i in sorted_indices]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, win_rates)
    
    # Add title and labels
    plt.title('Average Win Rate Comparison Between Models')
    plt.xlabel('Model')
    plt.ylabel('Average Win Rate')
    plt.xticks(rotation=45, ha='right')
    
    # Display values on each bar
    for bar, rate in zip(bars, win_rates):
        plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.01, 
                    f'{rate:.2%}', 
                    ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    output_file = os.path.join(log_dir, 'win_rates.png')
    plt.savefig(output_file)
    plt.close()
    
    print(f"Win rate bar chart saved to {output_file}")


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='评估模型池中所有模型之间的对战表现')
    parser.add_argument('--pool_path', type=str, required=True, 
                        help='模型池路径，包含多个模型')
    # parser.add_argument('--log_dir', type=str, required=True, 
    #                     help='结果保存路径')
    parser.add_argument('--n_episodes', type=int, default=100, 
                        help='每对模型之间的对战次数（默认：100）')
    parser.add_argument('--show', type=bool, default=False,
                        help='是否为显示模式，显示评估结果图表')
    
    # 解析命令行参数
    args = parser.parse_args()

    if args.show:
        # 显示评估结果图表
        show_evaluate_results(args.pool_path)
    else:
        # 调用评估函数
        evaluate_pool(args.pool_path, args.n_episodes)
