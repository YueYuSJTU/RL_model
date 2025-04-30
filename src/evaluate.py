import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm  # 导入tqdm库
if __name__ == "__main__":
    from utils.yaml_import import add_path
    add_path()
from typing import Dict, Tuple
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(exp_path: str, log_dir: str, n_episodes: int = 500) -> Tuple[float, float]:
    """
    对环境进行蒙特卡洛仿真评估
    
    Args:
        exp_path: 实验路径，包含模型和环境配置
        log_dir: 结果保存路径
        n_episodes: 蒙特卡洛模拟次数
        
    Returns:
        win_rate: 胜率
        avg_win_time: 获胜平均时间
    """
    # 创建保存目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 加载实验配置
    env_cfg = load_config(os.path.join(exp_path, "env_config.yaml"))
    agent_cfg = load_config(os.path.join(exp_path, "agent_config.yaml"))
    
    # 配置评估环境（无渲染）
    env_cfg["render_mode"] = None
    env_cfg["use_vec_normalize"] = False
    
    # 创建评估环境
    vec_env = create_env(env_cfg, training=False)
    vec_env = VecNormalize.load(
        os.path.join(exp_path, "final_train_env.pkl"), 
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

    # 记录评估结果
    wins = 0
    win_steps = []
    
    # 运行蒙特卡洛模拟，使用tqdm显示进度条
    for episode in tqdm(range(n_episodes), desc="Evaluating", ncols=80):
        obs = vec_env.reset()
        episode_done = False
        
        while not episode_done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = vec_env.step(action)
            
            if terminated:
                episode_done = True
                env_info = info[0].get("env_info", {})
                if env_info.get("win", False):
                    wins += 1
                    win_steps.append(env_info.get("steps_used", 0))
    
    # 计算统计数据
    win_rate = wins / n_episodes
    avg_win_time = np.mean(win_steps) if win_steps else 0
    
    print(f"\n评估结果:")
    print(f"胜率: {win_rate:.2%}")
    print(f"平均获胜时间: {avg_win_time:.2f} 步")
    
    # 创建可视化图表
    plt.figure(figsize=(12, 6))
    
    # 饼图：胜率
    plt.subplot(1, 2, 1)
    labels = ['Win', 'Fail']  # 使用英文标签避免字体问题
    sizes = [wins, n_episodes - wins]
    colors = ['#66b3ff', '#ff9999']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Win Rate')
    
    # 条形图：获胜时间分布
    plt.subplot(1, 2, 2)
    if win_steps:
        bins = min(30, len(set(win_steps)))
        plt.hist(win_steps, bins=bins, color='#66b3ff', alpha=0.7)
        plt.axvline(avg_win_time, color='r', linestyle='dashed', linewidth=1, 
                   label=f'Average: {avg_win_time:.2f} steps')
        plt.legend()
    plt.title('Distribution of Steps to Win')
    plt.xlabel('Steps')
    plt.ylabel('Frequency')
    
    # 保存结果
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'evaluation_results.png'))
    
    # 保存数值结果
    with open(os.path.join(log_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(f"总模拟次数: {n_episodes}\n")
        f.write(f"获胜次数: {wins}\n")
        f.write(f"胜率: {win_rate:.2%}\n")
        f.write(f"平均获胜时间: {avg_win_time:.2f} 步\n")
    
    # 保存详细数据
    np.save(os.path.join(log_dir, 'win_steps.npy'), np.array(win_steps))
    
    print(f"结果已保存至 {log_dir}")
    
    return win_rate, avg_win_time

if __name__ == "__main__":
    # 示例使用
    evaluate("./experiments/20250428_183617/stage2/20250428_184017_TrackingTask_ppo_1layer1",
              "./experiments/20250428_183617", 500)
