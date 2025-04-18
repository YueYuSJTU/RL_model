import os
import yaml
from utils.yaml_import import add_path
add_path()
from typing import Dict
from stable_baselines3 import PPO
from environments.make_env import create_env
from utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def evaluate(exp_path: str, render_mode: str = "human"):
    # 加载实验配置
    env_cfg = load_config(os.path.join(exp_path, "env_config.yaml"))
    agent_cfg = load_config(os.path.join(exp_path, "agent_config.yaml"))
    
    # 修改渲染模式
    env_cfg["render_mode"] = render_mode
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

    # 运行演示
    total_reward = 0
    obs = vec_env.reset()
    for i in range(15000):
        if render_mode is not None:
            vec_env.render()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _ = vec_env.step(action)
        total_reward += reward
        # print(f"step{i}, obs: {obs}, action: {action}")
        if terminated:
            print(f"Episode terminated at step {i}, total reward: {total_reward}")
            break
    vec_env.close()

if __name__ == "__main__":
    # 示例使用：evaluate("./experiments/20240320_ppo_baseline", "human")
    evaluate("./experiments/last_train", "human")
    # evaluate("./experiments/last_train", "flightgear")
    # evaluate("./experiments/20250416_192206_ppo_1layer_TrackingTask", "human")
