import os
import yaml
import shutil
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from environments.make_env import create_env
from utils.logger import setup_logger
from utils.serialization import save_config
from agents.make_agent import creat_agent

def train():
    # 加载配置
    with open("configs/train_config.yaml", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    env_path = os.path.join("configs", "env", f"{train_cfg['env']}.yaml")
    agent_path = os.path.join("configs", "agent", f"{train_cfg['agent']}.yaml")
    with open(agent_path, encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)
    with open(env_path, encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)

    # 设置实验路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_ppo_{env_cfg['task']}"
    log_path = os.path.join(train_cfg["log_root"], exp_name)
    
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

    # 创建模型
    model = creat_agent(
        env=train_env,
        agent_class=train_cfg["agent"],
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        agent_cfg=agent_cfg
    )

    # 开始训练
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=[eval_callback, progress_callback]
    )

    # 保存最终模型和环境
    model.save(os.path.join(log_path, "final_model"))
    train_env.save(os.path.join(log_path, "final_train_env.pkl"))

    # 复制内容到 experiments/last_train 文件夹
    last_train_path = os.path.join("experiments", "last_train")
    os.makedirs(last_train_path, exist_ok=True)
    if os.path.exists(last_train_path):
        shutil.rmtree(last_train_path)
    shutil.copytree(log_path, last_train_path)

if __name__ == "__main__":
    train()