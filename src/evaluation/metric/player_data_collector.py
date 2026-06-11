import os
import sys
import time
import csv
import yaml

sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.evaluation.evaluator import Evaluator

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 硬编码对手模型路径
    # model2_path = "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30"
    model2_path = "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35"
    model1_path = ""  # 留空以标识 human 控制
    player_name = "yuhang"
    
    # 尝试加载环境配置
    env_cfg_path = os.path.join(model2_path, "env_config.yaml")
    env_cfg = None
    if os.path.exists(env_cfg_path):
        env_cfg = load_config(env_cfg_path)
    
    print(f"=== 人机对战开始 ===")
    print(f"对手模型: {model2_path}")
    
    # 执行对战
    results = Evaluator.run_match(
        model1_path=model1_path,
        model2_path=model2_path,
        n_episodes=1,
        render_mode="anim3d",
        manual_control=True,
        env_cfg=env_cfg
    )
    
    # 提取所需统计数据 (以 model1 / human 为主控方视角)
    human_win = results.get("win_rate", 0)
    agent_win = results.get("lose_rate", 0)
    human_hp = results.get("avg_hp", 0)
    agent_hp = results.get("avg_hp_oppo", 0)
    human_energy = results.get("specific_energy_mean", 0)
    agent_energy = results.get("specific_energy_mean_oppo", 0)
    human_tracking_angle = results.get("track_angle_mean", 0)
    agent_tracking_angle = results.get("track_angle_mean_oppo", 0)
    
    print("\n--- 本局结果 ---")
    print(f"人类胜率: {human_win}")
    print(f"人类 HP: {human_hp:.2f} | 能量(mean): {human_energy:.2f} | 跟踪角度(mean): {human_tracking_angle:.2f}")
    print(f"Agent HP: {agent_hp:.2f} | 能量(mean): {agent_energy:.2f} | 跟踪角度(mean): {agent_tracking_angle:.2f}")
    
    # 保存数据到 CSV
    output_dir = "/home/ubuntu/Workfile/RL/RL_model/src/evaluation/player_data"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"{timestamp}_{player_name}.csv")
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Role", "HP", "Energy_Mean", "Win_Flag", "Tracking_Angle_Mean"])
        writer.writerow(["Human", human_hp, human_energy, human_win, human_tracking_angle])
        writer.writerow(["Agent", agent_hp, agent_energy, agent_win, agent_tracking_angle])
        
    print(f"\n战报已保存至: {csv_path}")

if __name__ == "__main__":
    main()
