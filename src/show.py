import os
import yaml
import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
import numpy as np
from typing import Dict
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def show(exp_path: str, render_mode: str = "human", random_input: bool = False, model_num: int = 0) -> None:
    # 加载实验配置
    env_cfg = load_config(os.path.join(exp_path, "env_config.yaml"))
    agent_cfg = load_config(os.path.join(exp_path, "agent_config.yaml"))
    
    # 修改渲染模式
    env_cfg["render_mode"] = render_mode
    env_cfg["use_vec_normalize"] = False
    
    # 创建评估环境
    vec_env = create_env(env_cfg, training=False, vec_env_kwargs={"model_num": model_num})
    # vec_env = VecNormalize.load(
    #     os.path.join(exp_path, "final_train_env.pkl"), 
    #     vec_env
    # )
    vec_env.training = False
    vec_env.norm_reward = False

    # 加载模型
    model = PPO.load(
        os.path.join(exp_path, "best_model"),
        env=vec_env,
        device=agent_cfg["device"]
    )

    # 运行演示
    total_reward = 0
    obs = vec_env.reset()
    for i in range(15000):
        if render_mode is not None:
            vec_env.render()
        action, _ = model.predict(obs, deterministic=True)
        if random_input:
            action = np.random.uniform(-1, 1, size=(1,4))
            action[:, -1] = np.abs(action[:, -1])
        obs, reward, terminated, _ = vec_env.step(action)
        total_reward += reward
        if terminated:
            print(f"Episode terminated at step {i}, total reward: {total_reward}")
            break
    vec_env.close()

if __name__ == "__main__":
    # # 示例使用：show("./experiments/20240320_ppo_baseline", "human")
    # show("""/home/ubuntu/Workfile/RL/RL_model/experiments/20250514_164357/stage1/20250514_164357_TrackingTask_ppo_1layer1""", "human")
    # # show("./experiments/last_train", "flightgear")
    # # show("./experiments/20250416_192206_ppo_1layer_TrackingTask", "human")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--random_input", type=bool, default=False)
    parser.add_argument("--model_num", type=int, default=0, help="Model number for multi-agent environments")
    args = parser.parse_args()
    show(args.exp_path, args.render_mode, args.random_input)
