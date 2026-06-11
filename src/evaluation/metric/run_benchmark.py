import os
import sys

# 确保能正确导入 src 模块
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.evaluation.evaluator import Evaluator

def main():
    # 需要进行 Benchmark 评测的目标模型集合
    models = {
        # "20260105-1month": "experiments/20260105_231131/stage2/20260228_032950_cycle_1277",
        # "14h": "experiments/20260301_203610/stage2/20260302_143419_cycle_26",
        # "no_curri_no_pool(123)": "experiments/20260425_204717_best_train_mlp_no_curri_no_pool/stage1/20260425_204717_stage1_agent",
        # "no_curri_no_pool(pool1)": "experiments/20260505_212540_best_train_mlp2_no_curri_no_pool_pool1/stage1/20260505_212540_stage1_agent",
        # "no_curri_no_pool(pool2)": "experiments/20260429_150222_best_train_mlp2_no_curri_no_pool_pool2/stage1/20260429_150222_stage1_agent",
        "lstm": "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35",
        "now_best": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
        # "no_curri_have_pool": "experiments/20260502_221525_best_train_mlp2_no_curri_have_pool/stage1/20260503_184759_cycle_30",
        "no_curri_have_pool_fix": "experiments/20260506_105525_best_train_mlp2_no_curri_have_pool/stage1/20260506_163252_cycle_9",
        "have_curri_no_pool": "experiments/20260506_211933_best_train_mlp2_have_curri_no_pool/stage2/20260507_155441_cycle_30",
        # "have_curri_no_pool_only1": "experiments/20260507_214055_best_train_mlp2_have_curri_no_pool_only1/stage2/20260508_031313_cycle_11"
        "20250616(no-pool-have-curri)": "experiments/20250616_221656/stage2/20250616_221824_TrackingTask_ppo_1layer1",
    }
    
    # 测试池路径，该目录下的每个文件夹代表一个对手模型
    test_pool_dir = "opponent_pool/test_pool"
    
    # 动态构建测试池字典
    test_pool_models = {}
    if os.path.isdir(test_pool_dir):
        for entry in os.listdir(test_pool_dir):
            full_path = os.path.join(test_pool_dir, entry)
            if os.path.isdir(full_path):
                test_pool_models[entry] = full_path
    else:
        print(f"测试池路径不存在或不是目录: {test_pool_dir}")
        return
    
    if not test_pool_models:
        print(f"在测试池路径 {test_pool_dir} 下未找到任何对手模型文件夹。")
        return

    # 每个对阵组合测试的回合数
    n_episodes_per_match = 1000
    
    # 初始化统计数据字典
    stats = {
        name: {
            "total_matches": 0
        } for name in models.keys()
    }
    
    print("=== 开始 Benchmark 评测 ===")
    
    for agent_name, agent_path in models.items():
        print(f"\n>>> 正在评测目标模型: [{agent_name}] <<<")
        for oppo_name, oppo_path in test_pool_models.items():
            print(f"  VS 测试池对手 [{oppo_name}] (进行 {n_episodes_per_match} 回合)...")
            try:
                # 调用评测器，agent 为模型 1，oppo 为模型 2
                results = Evaluator.run_match(
                    model1_path=agent_path,
                    model2_path=oppo_path,
                    n_episodes=n_episodes_per_match,
                    render_mode=None,
                    use_tqdm=True
                )
                
                # 累加 agent 的统计数据
                stats[agent_name]["total_matches"] += n_episodes_per_match
                for k, v in results.items():
                    stats[agent_name][k] = stats[agent_name].get(k, 0.0) + v * n_episodes_per_match
                
                print(f"  -> 当前对战结果: 胜率 {results.get('win_rate', 0):.2%} | 自身HP {results.get('avg_hp', 0):.2f} | 对手HP {results.get('avg_hp_oppo', 0):.2f}")
                
            except Exception as e:
                print(f"  -> 对战失败 {agent_name} vs {oppo_name} : {e}")

    print("\n" + "="*40)
    print("=== Benchmark 最终统计结果 ===")
    print("="*40)
    
    for name, st in stats.items():
        if st["total_matches"] > 0:
            total_m = st["total_matches"]
            overall_win_rate = st.get("win_rate", 0) / total_m
            overall_avg_hp = st.get("avg_hp", 0) / total_m
            overall_avg_oppo_hp = st.get("avg_hp_oppo", 0) / total_m
            
            print(f"目标模型: {name}")
            print(f"  总对战场次 : {total_m}")
            print("  +" + "-"*40 + "+")
            print(f"  | 综合胜率   : {overall_win_rate:^25.2%} |")
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
            print(f"目标模型: {name} 没有成功完成任何对战。")

if __name__ == "__main__":
    main()
