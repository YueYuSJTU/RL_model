import os
import yaml
import shutil
from utils.yaml_import import add_path
add_path()
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from environments.make_env import create_env
from utils.logger import setup_logger
from utils.serialization import save_config
from agents.make_agent import creat_agent, load_agent
from evaluate import evaluate

def find_latest_training_result(stage_path):
    """
    查找stage_path下最新的训练结果路径
    
    Args:
        stage_path: 包含所有stage的根目录
        
    Returns:
        latest_result_path: 最新训练结果的路径
    """
    # 查找所有stage文件夹
    stage_dirs = [d for d in os.listdir(stage_path) 
                 if os.path.isdir(os.path.join(stage_path, d)) and "stage" in d]
    
    if not stage_dirs:
        raise ValueError(f"在 {stage_path} 中没有找到stage文件夹")
    
    # 找到最新的stage（按修改时间排序）
    latest_stage = max(
        [os.path.join(stage_path, d) for d in stage_dirs],
        key=os.path.getmtime
    )
    
    # 在最新stage中查找训练结果
    result_dirs = [d for d in os.listdir(latest_stage) 
                  if os.path.isdir(os.path.join(latest_stage, d))]
    
    if not result_dirs:
        # 如果没有子文件夹，使用stage目录本身
        return latest_stage
    
    # 找到最新的训练结果（按修改时间排序）
    latest_result = max(
        [os.path.join(latest_stage, d) for d in result_dirs],
        key=os.path.getmtime
    )
    
    return latest_result

def train(stage_path: str = ""):
    # 加载配置
    with open("configs/stage_train_config.yaml", encoding="utf-8") as f:
        stage_train_cfg = yaml.safe_load(f)
    if not stage_path:
        if 'stage1' not in stage_train_cfg.keys():
            raise ValueError("You don't give stage path, so you must start a new train with stage1.")
        home_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_path = os.path.join(next(iter(stage_train_cfg.values()))["log_root"], home_timestamp)
    for train_cfg in stage_train_cfg.values():
        env_path = os.path.join("configs", "env", f"{train_cfg['env']}.yaml")
        agent_path = os.path.join("configs", "agent", f"{train_cfg['agent']}.yaml")
        with open(agent_path, encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)
        with open(env_path, encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f)

        # 设置实验路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{env_cfg['task']}_{train_cfg['agent']}"
        log_path = os.path.join(stage_path, f"stage{train_cfg['stage_num']}", exp_name)
        
        # 初始化日志系统
        logger = setup_logger(log_path)
        
        # 保存配置副本
        save_config(train_cfg, log_path, "train_config.yaml")
        save_config(agent_cfg, log_path, "agent_config.yaml")
        save_config(env_cfg, log_path, "env_config.yaml")

        # 创建环境
        train_env = create_env(env_cfg, training=True, num_cpu=train_cfg["num_cpu"])
        eval_env = create_env(env_cfg, training=False)
        eval_env.training = False
        eval_env.norm_reward = False

        # 初始化回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=train_cfg["eval_freq"],
            deterministic=True,
            render=False
        )
        progress_callback = ProgressBarCallback()

        # 创建或加载模型
        if train_cfg['stage_num'] == 1:
            model = creat_agent(
                env=train_env,
                agent_class=train_cfg["agent"],
                tensorboard_log=os.path.join(train_cfg["log_root"], home_timestamp, "tensorboard", timestamp),
                agent_cfg=agent_cfg
            )
        else:
            load_path = os.path.join(stage_path, f"stage{train_cfg['stage_num'] - 1}")
            model = load_agent(
                env=train_env,
                agent_class=train_cfg["agent"],
                path=os.path.join(load_path, "best_model"),
                device=agent_cfg["device"]
            )
            
        # 开始训练
        model.learn(
            total_timesteps=train_cfg["total_timesteps"],
            callback=[eval_callback, progress_callback]
        )

        # 保存最终模型和环境
        model.save(os.path.join(log_path, "final_model"))
        train_env.save(os.path.join(log_path, "final_train_env.pkl"))

        # 复制 best_model.zip 到父目录
        best_model_path = os.path.join(log_path, "best_model.zip")
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, os.path.join(log_path, ".."))

    # 在训练结束后查找最新的训练结果并进行评估
    latest_result = find_latest_training_result(stage_path)
    evaluate(latest_result, stage_path, n_episodes=1000)

if __name__ == "__main__":
    train()