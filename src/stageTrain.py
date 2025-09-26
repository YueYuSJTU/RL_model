import os
import yaml
import shutil
import argparse
from src.utils.yaml_import import add_path
add_path()
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from src.environments.make_env import create_env
from src.utils.logger import setup_logger
from src.utils.serialization import save_config
from src.agents.make_agent import creat_agent, load_agent
from src.evaluate import evaluate
from src.utils.custom_callback import ComponentEvalCallback

def find_latest_training_result(pretrained_path):
    """
    查找pretrained_path下最新的训练结果路径
    
    Args:
        pretrained_path: 包含所有stage的根目录
        
    Returns:
        latest_result_path: 最新训练结果的路径
    """
    # 查找所有stage文件夹
    stage_dirs = [d for d in os.listdir(pretrained_path) 
                 if os.path.isdir(os.path.join(pretrained_path, d)) and "stage" in d]
    
    if not stage_dirs:
        raise ValueError(f"在 {pretrained_path} 中没有找到stage文件夹")
    
    # 找到最新的stage（按修改时间排序）
    latest_stage = max(
        [os.path.join(pretrained_path, d) for d in stage_dirs],
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

def train(pretrained_path: str = "", config_path: str = "", eval_pool_path: str = ""):
    # 加载配置
    with open(config_path, encoding="utf-8") as f:
        stage_train_cfg = yaml.safe_load(f)
    if not pretrained_path:
        # 从头开始训练必须提供stage1
        if 'stage1' not in stage_train_cfg.keys():
            raise ValueError("You don't give stage path, so you must start a new train with stage1.")
        home_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_path = os.path.join(next(iter(stage_train_cfg.values()))["log_root"], home_timestamp)
    else:
        # 微调模式，目前只支持单阶段微调
        assert len(stage_train_cfg) == 1, "Only one stage is supported in fine-tuning mode."
    for stage_key, train_cfg in stage_train_cfg.items():
        stage_num = int(stage_key[5:])
        env_path = os.path.join("configs", "env", f"{train_cfg['env']}.yaml")
        agent_path = os.path.join("configs", "agent", f"{train_cfg['agent']}.yaml")
        with open(agent_path, encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)
        with open(env_path, encoding="utf-8") as f:
            env_cfg = yaml.safe_load(f)

        # 设置实验路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{timestamp}_{env_cfg['task']}_{train_cfg['agent']}"
        if not pretrained_path:
            log_path = os.path.join(train_path, f"stage{stage_num}", exp_name)
        else:
            stage_dirs = [d for d in os.listdir(pretrained_path) if os.path.isdir(os.path.join(pretrained_path, d)) and "stage" in d]
            if not stage_dirs:
                raise ValueError(f"在 {pretrained_path} 中没有找到stage文件夹")
            latest_stage_dir = max([os.path.join(pretrained_path, d) for d in stage_dirs], key=os.path.getmtime)
            log_path = os.path.join(latest_stage_dir, exp_name)
            print(f"Fine-tuning from {latest_stage_dir}, results will be saved to {log_path}")
        
        # 初始化日志系统
        logger = setup_logger(log_path)
        
        # 保存配置副本
        save_config(train_cfg, log_path, "train_config.yaml")
        save_config(agent_cfg, log_path, "agent_config.yaml")
        save_config(env_cfg, log_path, "env_config.yaml")

        # 创建环境
        if train_cfg.get("model_num") is not None:
            vec_env_kwargs={"model_num": train_cfg["model_num"]}
        else:
            vec_env_kwargs=None
        train_env = create_env(env_cfg, training=True, num_cpu=train_cfg["num_cpu"], vec_env_kwargs=vec_env_kwargs)
        eval_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
        eval_env.training = False
        eval_env.norm_reward = False

        # 初始化回调
        eval_callback = ComponentEvalCallback(
            eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=train_cfg["eval_freq"],
            deterministic=True,
            render=False
        )
        progress_callback = ProgressBarCallback()

        # 创建或加载模型
        if stage_num == 1:
            if not pretrained_path:
                # 预训练模式第一阶段，创建模型
                model = creat_agent(
                    env=train_env,
                    agent_class=train_cfg["agent"],
                    tensorboard_log=os.path.join(train_cfg["log_root"], home_timestamp, "tensorboard", timestamp),
                    agent_cfg=agent_cfg
                )
            else:
                # 微调模式，加载最新stage的模型
                model = load_agent(
                    env=train_env,
                    agent_class=train_cfg["agent"],
                    path=os.path.join(latest_stage_dir, "best_model"),
                    device=agent_cfg["device"]
                )
        else:
            # 预训练模式的非第一阶段，加载上一个阶段的最佳模型
            load_path = os.path.join(train_path, f"stage{stage_num - 1}")
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
    eval_path = pretrained_path if pretrained_path else train_path
    latest_result = find_latest_training_result(eval_path)
    evaluate(latest_result, eval_pool_path, n_episodes=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reinforcement learning model.")
    parser.add_argument("--config", type=str, required=True, default="configs/goal_point_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--eval_pool", type=str, required=True, default="/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool3", help="Path to the evaluation pool.")
    parser.add_argument("--pretrained_path", type=str, default="", help="Path to the stage directory (optional).")
    args = parser.parse_args()

    train(pretrained_path=args.pretrained_path, config_path=args.config, eval_pool_path=args.eval_pool)