import os
import yaml
import shutil
import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from src.environments.make_env import create_env
from src.agents.make_agent import creat_agent
from src.utils.serialization import save_config


def create_init_pool(pool_name: str = "pool1", opponent_num: int = 3) -> None:
    """
    创建初始的对手模型池。

    使用TrackingInitTask环境，stage的含义变为不同对手的奖励函数
    不同于stage_train，每个stage均会重新创建模型，从头训练
    训练不会保存在experiments目录下，只会在pool中保存最终结果
    所有的对手模型共用训练参数
    Args:
        pool_name: 对手模型池的名称
        opponent_num: 对手数量
    """
    with open("configs/create_init_pool_config.yaml", encoding="utf-8") as f:
        init_pool_cfg = yaml.safe_load(f)
    stage_path = os.path.join(init_pool_cfg["log_root"], pool_name)

    for i in range(opponent_num):
        env_path = os.path.join("configs", "env", f"{init_pool_cfg['env']}.yaml")
        agent_path = os.path.join("configs", "agent", f"{init_pool_cfg['agent']}.yaml")
        with open(agent_path, encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)
        with open(env_path, encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f)
        env_cfg["shape"] = f"stage{i+1}" 

        # 设置实验路径
        # 获取当前目录下所有子目录
        if os.path.exists(stage_path):
            existing_dirs = [d for d in os.listdir(stage_path) if os.path.isdir(os.path.join(stage_path, d))]
            # 过滤出数字命名的目录并找出最大值
            numeric_dirs = [int(d) for d in existing_dirs if d.isdigit()]
            next_num = max(numeric_dirs, default=0) + 1
        else:
            # 如果目录不存在，创建它并从1开始
            os.makedirs(stage_path, exist_ok=True)
            next_num = 1
        log_path = os.path.join(stage_path, str(next_num))
        os.makedirs(log_path, exist_ok=True)
        
        # 保存配置副本
        save_config(init_pool_cfg, log_path, "train_config.yaml")
        save_config(agent_cfg, log_path, "agent_config.yaml")
        save_config(env_cfg, log_path, "env_config.yaml")

        # 创建环境
        train_env = create_env(env_cfg, training=True, num_cpu=init_pool_cfg["num_cpu"])
        eval_env = create_env(env_cfg, training=False)
        eval_env.training = False
        eval_env.norm_reward = False

        # 初始化回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=init_pool_cfg["eval_freq"],
            deterministic=True,
            render=False
        )
        progress_callback = ProgressBarCallback()

        # 创建模型
        model = creat_agent(
            env=train_env,
            agent_class=init_pool_cfg["agent"],
            tensorboard_log=None,  # 不使用TensorBoard
            agent_cfg=agent_cfg
        )

            
        # 开始训练
        model.learn(
            total_timesteps=init_pool_cfg["total_timesteps"],
            callback=[eval_callback, progress_callback]
        )

        # 保存最终模型和环境
        # model.save(os.path.join(log_path, "final_model"))
        train_env.save(os.path.join(log_path, "final_train_env.pkl"))


if __name__ == "__main__":
    create_init_pool(pool_name="pool1", opponent_num=3)