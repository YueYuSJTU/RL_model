import os
import sys
import yaml

# 确保能正确导入 src 模块
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.evaluation.evaluator import Evaluator

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
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
        "lstm": "experiments/20260316_093625_lstm_train/stage2/20260320_135309_cycle_21",
        "now_best": "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30",
        # "no_curri_have_pool": "experiments/20260502_221525_best_train_mlp2_no_curri_have_pool/stage1/20260503_184759_cycle_30",
        "no_curri_have_pool_fix": "experiments/20260506_105525_best_train_mlp2_no_curri_have_pool/stage1/20260506_163252_cycle_9",
        "have_curri_no_pool": "experiments/20260506_211933_best_train_mlp2_have_curri_no_pool/stage2/20260507_155441_cycle_30",
        # "have_curri_no_pool_only1": "experiments/20260507_214055_best_train_mlp2_have_curri_no_pool_only1/stage2/20260508_031313_cycle_11"
        "20250616(no-pool-have-curri)": "experiments/20250616_221656/stage2/20250616_221824_TrackingTask_ppo_1layer1",
    }
    
    # 每个对阵组合测试的回合数
    n_episodes_per_match = 1000 
    
    # 初始化两两对战统计字典
    pairwise_stats = {
        name1: {
            name2: {} for name2 in models.keys()
        } for name1 in models.keys()
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
                    env_cfg=load_config("/home/ubuntu/Workfile/RL/RL_model/experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30/env_config.yaml"),
                    use_tqdm=True
                )
                
                win_rate = results.get('win_rate', 0.0)
                loss_rate = results.get('loss_rate', 0.0)
                draw_rate = results.get('draw_rate', max(0.0, 1.0 - win_rate - loss_rate))
                
                # 记录两两对战数据
                pairwise_stats[name1][name2] = {
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "draw_rate": draw_rate,
                    "avg_hp": results.get('avg_hp', 0.0),
                    "avg_hp_oppo": results.get('avg_hp_oppo', 0.0)
                }
                
                print(f"当前对战结果 -> {name1} 胜率: {win_rate:.2%} | {name2} 胜率: {loss_rate:.2%} | 平局出场率: {draw_rate:.2%}")
                
            except Exception as e:
                print(f"对战 {name1} vs {name2} 失败: {e}")

    print("\n" + "="*60)
    print("=== 两两对战统计矩阵信息 (用于热力图) ===")
    print("="*60)
    
    for name1 in models.keys():
        for name2 in models.keys():
            if name1 == name2:
                continue
            
            stats = pairwise_stats[name1].get(name2, {})
            if stats:
                print(f"[{name1}] VS [{name2}]")
                print(f"  胜率 (Win) : {stats.get('win_rate', 0):.2%}")
                print(f"  败率 (Loss): {stats.get('loss_rate', 0):.2%}")
                print(f"  平局 (Draw): {stats.get('draw_rate', 0):.2%}")
                print(f"  平均自身HP : {stats.get('avg_hp', 0):.2f}")
                print(f"  平均对手HP : {stats.get('avg_hp_oppo', 0):.2f}")
                print("-" * 40)

if __name__ == "__main__":
    main()
