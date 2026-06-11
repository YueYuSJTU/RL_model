import os
import sys

# 确保能正确导入 src 模块
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.evaluation.evaluator import Evaluator

def main():
    # 在这里硬编码你需要评测的模型路径集合
    # 格式为: {"模型名称": "模型路径"}
    # models = {
    #     "Model_1": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
    #     "Model_2": "experiments/20260429_150222_best_train_mlp2_no_curri_no_pool_pool2/stage1/20260429_150222_stage1_agent",
    #     "Model_3": "experiments/20260502_221525_best_train_mlp2_no_curri_have_pool/stage1/20260503_184759_cycle_30",
    # }
    models = {
        # "20260105-1month": "experiments/20260105_231131/stage2/20260228_032950_cycle_1277",
        # "14h": "experiments/20260301_203610/stage2/20260302_143419_cycle_26",
        # "no_curri_no_pool(123)": "experiments/20260425_204717_best_train_mlp_no_curri_no_pool/stage1/20260425_204717_stage1_agent",
        # "no_curri_no_pool(pool1)": "experiments/20260505_212540_best_train_mlp2_no_curri_no_pool_pool1/stage1/20260505_212540_stage1_agent",
        # "no_curri_no_pool(pool2)": "experiments/20260429_150222_best_train_mlp2_no_curri_no_pool_pool2/stage1/20260429_150222_stage1_agent",
        # "lstm": "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35",
        "now_best": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
        # "no_curri_have_pool": "experiments/20260502_221525_best_train_mlp2_no_curri_have_pool/stage1/20260503_184759_cycle_30",
        "no_curri_have_pool_fix": "experiments/20260506_105525_best_train_mlp2_no_curri_have_pool/stage1/20260506_163252_cycle_9",
        "have_curri_no_pool": "experiments/20260506_211933_best_train_mlp2_have_curri_no_pool/stage2/20260507_155441_cycle_30",
        # "have_curri_no_pool_only1": "experiments/20260507_214055_best_train_mlp2_have_curri_no_pool_only1/stage2/20260508_031313_cycle_11"
        "20250616(no-pool-have-curri)": "experiments/20250616_221656/stage2/20250616_221824_TrackingTask_ppo_1layer1",
    }
    
    # 每个对阵组合测试的回合数
    n_episodes_per_match = 1000 
    
    # 初始化统计数据字典
    stats = {
        name: {
            "total_matches": 0
        } for name in models.keys()
    }
    
    # 建立主控方与对手方数据的映射词典，记录 name2 时自动反转属性
    swap_map = {
        "win_rate": "loss_rate", "loss_rate": "win_rate",
        "avg_hp": "avg_hp_oppo", "avg_hp_oppo": "avg_hp",
        "gun_opportunity_time_ratio": "gun_opportunity_time_ratio_oppo", "gun_opportunity_time_ratio_oppo": "gun_opportunity_time_ratio",
        "damage_to_oppo_total": "damage_to_self_total", "damage_to_self_total": "damage_to_oppo_total",
        "damage_rate": "damage_rate_oppo", "damage_rate_oppo": "damage_rate",
        "track_angle_mean": "track_angle_mean_oppo", "track_angle_mean_oppo": "track_angle_mean",
        "adverse_angle_mean": "adverse_angle_mean_oppo", "adverse_angle_mean_oppo": "adverse_angle_mean",
        "specific_energy_mean": "specific_energy_mean_oppo", "specific_energy_mean_oppo": "specific_energy_mean",
        "specific_energy_final": "specific_energy_final_oppo", "specific_energy_final_oppo": "specific_energy_final",
    }
    
    print("=== 开始两两对战评测 ===")
    
    for name1, path1 in models.items():
        for name2, path2 in models.items():
            if name1 == name2:
                continue
            
            print(f"\n[{name1}] VS [{name2}] (进行 {n_episodes_per_match} 回合)...")
            try:
                # 调用评测器
                results = Evaluator.run_match(
                    model1_path=path1,
                    model2_path=path2,
                    n_episodes=n_episodes_per_match,
                    render_mode=None, # 为了加速，默认不渲染
                    use_tqdm=True
                )
                
                # 累加统计数据 (name1 作为主控)
                stats[name1]["total_matches"] += n_episodes_per_match
                for k, v in results.items():
                    stats[name1][k] = stats[name1].get(k, 0.0) + v * n_episodes_per_match
                
                # 累加统计数据 (name2 作为对手，数据反向映射)
                stats[name2]["total_matches"] += n_episodes_per_match
                for k, v in results.items():
                    mapped_k = swap_map.get(k, k)
                    stats[name2][mapped_k] = stats[name2].get(mapped_k, 0.0) + v * n_episodes_per_match
                
                print(f"当前对战结果 -> {name1} 胜率: {results.get('win_rate', 0):.2%} | {name2} 胜率: {results.get('loss_rate', 0):.2%}")
                
            except Exception as e:
                print(f"对战 {name1} vs {name2} 失败: {e}")

    print("\n" + "="*40)
    print("=== 最终统计结果 ===")
    print("="*40)
    
    for name, st in stats.items():
        if st["total_matches"] > 0:
            total_m = st["total_matches"]
            overall_win_rate = st.get("win_rate", 0) / total_m
            overall_avg_hp = st.get("avg_hp", 0) / total_m
            overall_avg_oppo_hp = st.get("avg_hp_oppo", 0) / total_m
            
            print(f"模型: {name}")
            print(f"  总对战场次 : {total_m}")
            print("  +" + "-"*40 + "+")
            print(f"  | 整体胜率   : {overall_win_rate:^25.2%} |")
            print(f"  | 平均自身HP : {overall_avg_hp:^25.2f} |")
            print(f"  | 平均对手HP : {overall_avg_oppo_hp:^25.2f} |")
            print("  +" + "-"*40 + "+")
            
            # Print other detailed metrics
            print("  --- 详细指标 ---")
            for k, v in st.items():
                if k in ["total_matches", "win_rate", "avg_hp", "avg_hp_oppo"]:
                    continue
                overall_avg = v / total_m
                print(f"  {k:30s} : {overall_avg:.4f}")
            print("-" * 40)
        else:
            print(f"模型: {name} 没有成功完成任何对战。")

if __name__ == "__main__":
    main()
