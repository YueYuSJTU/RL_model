import os
import yaml
import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
import numpy as np
import logging
from typing import Dict, Tuple, List
from stable_baselines3 import PPO
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import gc
from tqdm import tqdm
from .agents.model_wrapper import ObsAdaptingModel

# def show(exp_path: str, render_mode: str = "human", random_input: bool = False, model_num: int = 0, pool_path: str = None) -> None:
#     # 加载实验配置
#     env_cfg = load_config(os.path.join(exp_path, "env_config.yaml"))
#     agent_cfg = load_config(os.path.join(exp_path, "agent_config.yaml"))
    
#     # 修改渲染模式
#     env_cfg["render_mode"] = render_mode
#     env_cfg["use_vec_normalize"] = False
    
#     # 创建评估环境
#     vec_env = create_env(env_cfg, training=False, vec_env_kwargs={"model_num": model_num, "pool_roots": pool_path})
#     env_pkl = "best_env.pkl" if os.path.exists(os.path.join(exp_path, "best_env.pkl")) else "final_train_env.pkl"
#     vec_env = VecNormalize.load(
#         os.path.join(exp_path, env_pkl), 
#         vec_env
#     )
#     vec_env.training = False
#     vec_env.norm_reward = False
#     vec_env.env_method("update_task_parameters", goal_point_prob=0.0)

#     # 加载模型
#     model = PPO.load(
#         os.path.join(exp_path, "best_model"),
#         env=vec_env,
#         device=agent_cfg["device"]
#     )

#     # 运行演示
#     total_reward = 0
#     obs = vec_env.reset()
#     for i in range(15000):
#         if render_mode is not None:
#             vec_env.render()
#         action, _ = model.predict(obs, deterministic=True)
#         if random_input:
#             action = np.random.uniform(-1, 1, size=(1,4))
#             action[:, -1] = np.abs(action[:, -1])
#         obs, reward, terminated, _ = vec_env.step(action)
#         total_reward += reward
#         if terminated:
#             print(f"Episode terminated at step {i}, total reward: {total_reward}")
#             break
#     vec_env.close()

def show(exp_path: str, render_mode: str = "human", model_num: int = 0, pool_path: str = None) -> None:
    model1_path = exp_path
    model2_path = os.path.join(pool_path, str(model_num))
    results = evaluate_without_NN(
        model1_path=model1_path,
        model2_path=model2_path,
        n_episodes=1,
        render_mode=render_mode,
    )
    win_rate = results["win_rate"]
    draw_rate = results["draw_rate"]
    loss_rate = results["loss_rate"]
    opponent_fall_rate = results["opponent_fall_rate"]
    avg_win_time = results["avg_win_time"]
    avg_reward = results["avg_reward"]
    avg_hp = results["avg_hp"]
    avg_hp_oppo = results["avg_hp_oppo"]
    print(f"Win Rate: {win_rate:.2%}, Draw Rate: {draw_rate:.2%}, Loss Rate: {loss_rate:.2%}")
    print(f"Opponent Fall Rate: {opponent_fall_rate:.2%}, Avg Win Time: {avg_win_time:.2f}, Avg Reward: {avg_reward:.2f}")
    print(f"Avg HP: {avg_hp:.2f}, Avg Opponent HP: {avg_hp_oppo:.2f}")


def evaluate_without_NN(
        model1_path: str, 
        model2_path: str, 
        n_episodes: int = 1, 
        render_mode: str = None, 
        env_cfg: Dict = None, 
        use_tqdm: bool = True
    ) -> Tuple[float, float, float, float, float, float]:
    """
    评估两个模型在对战环境中的表现，不经过NN包装器，直接调用原始环境。
    对于评估任务，建议设置render_mode为None以提高速度；对于演示任务，建议设置n_episodes为1。
    Arguments:
        model1_path: (str) 模型1路径。
        model2_path: (str) 模型2路径。
        n_episodes: (int) 评估的对战回合数。
        render_mode: (str or None) 渲染模式，如 "human"。若为 None 则不渲染。
        env_cfg: (Dict or None) 环境配置字典，若为 None 则从 model1_path 目录加载。
        use_tqdm: (bool) 是否使用 tqdm 显示进度条。
    Returns:
        win_rate: (float) 主控胜率。
        draw_rate: (float) 平局率。
        loss_rate: (float) 主控失败率。
        opponent_fall_rate: (float) 敌机自主坠机率。
        avg_win_time: (float) 主控获胜的平均时间步数。
        avg_reward: (float) 主控的平均总奖励。
        avg_hp: (float) 主控的平均剩余生命值。
        avg_hp_oppo: (float) 敌机的平均剩余生命值。
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1.")
    if n_episodes > 1 and render_mode is not None:
        logging.warning("Rendering multiple evaluation episodes may slow down the process.")
    
    # 创建评估环境
    if env_cfg is None:
        logging.warning("No environment configuration provided, using model1 settings.")
        env_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))
    env_cfg["render_mode"] = render_mode
    env_cfg["use_vec_normalize"] = False
    vec_env = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv, vec_env_kwargs=None)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.env_method("update_task_parameters", goal_point_prob=0.0)
    # 由于模型obs维度不匹配，必须创建两个虚假的使用NN包装器的环境用以加载模型

    # 加载模型
    agent1_cfg = load_config(os.path.join(model1_path, "agent_config.yaml"))
    env1_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))        # 这里需要加载环境配置以适配ObsAdaptingModel，它包含了帧堆叠包装器和环境obs维度等信息
    fake_env1 = create_env(env1_cfg, training=False, vec_env_kwargs=None)
    model1 = PPO.load(
        os.path.join(model1_path, "best_model"),
        env=fake_env1,
        device=agent1_cfg["device"]
    )
    model1 = ObsAdaptingModel(model1, env1_cfg)
    agent2_cfg = load_config(os.path.join(model2_path, "agent_config.yaml"))
    env2_cfg = load_config(os.path.join(model2_path, "env_config.yaml"))        # 这里需要加载环境配置以适配ObsAdaptingModel，它包含了帧堆叠包装器和环境obs维度等信息
    fake_env2 = create_env(env2_cfg, training=False, vec_env_kwargs=None)
    model2 = PPO.load(
        os.path.join(model2_path, "best_model"),
        env=fake_env2,
        device=agent2_cfg["device"]
    )
    model2 = ObsAdaptingModel(model2, env2_cfg)

    # 记录评估结果
    wins = 0
    losses = 0
    draws = 0
    opponent_falls = 0
    avg_hp = 0
    avg_hp_oppo = 0
    win_steps = []
    total_rewards = []  # 用于存储每场对战的总奖励

    try:
        # 运行对战，使用tqdm显示进度条
        episodes = range(n_episodes)
        if use_tqdm:
            episodes = tqdm(episodes, desc=f"Evaluating models", ncols=80)

        for episode in episodes:
            obs = vec_env.reset()
            obs_length = obs.shape[1]
            episode_done = False
            episode_reward = 0  # 初始化本场对战的奖励

            while not episode_done:
                if render_mode is not None:
                    vec_env.render()
                action1, _ = model1.predict(obs[:, :obs_length//2], deterministic=True)
                action2, _ = model2.predict(obs[:, obs_length//2:], deterministic=True)
                # action1和action2需要合并成一个动作输入
                combined_action = np.concatenate([action1, action2], axis=-1)
                obs, reward, terminated, info = vec_env.step(combined_action)
                
                episode_reward += reward[0]  # 累加奖励

                if terminated:
                    episode_done = True
                    total_rewards.append(episode_reward)  # 记录本场对战的总奖励
                    env_info = info[0].get("env_info", {})
                    win_status = env_info["win"]
                    
                    if win_status == 1:  # 主控胜利
                        wins += 1
                        win_steps.append(env_info.get("steps_used", 0))
                    elif win_status == 0:  # 平局
                        draws += 1
                    elif win_status == -1:  # 敌机胜利
                        losses += 1
                    elif win_status == 0.5: # 敌机自主坠机
                        opponent_falls += 1
                    avg_hp += env_info["HP_self"]
                    avg_hp_oppo += env_info["HP_oppo"]
        
        # 计算统计数据
        win_rate = wins / n_episodes
        draw_rate = draws / n_episodes
        loss_rate = losses / n_episodes
        opponent_fall_rate = opponent_falls / n_episodes
        avg_win_time = np.mean(win_steps) if win_steps else 0
        avg_reward = np.mean(total_rewards) if total_rewards else 0
        avg_hp = avg_hp / n_episodes
        avg_hp_oppo = avg_hp_oppo / n_episodes

    finally:
        # === [关键] 显式清理代码 ===
        if 'vec_env' in locals() and vec_env is not None:
            vec_env.close()
        if 'model1' in locals():
            del model1
        if 'model2' in locals():
            del model2
        if 'vec_env' in locals():
            del vec_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "opponent_fall_rate": opponent_fall_rate,
        "avg_win_time": avg_win_time,
        "avg_reward": avg_reward,
        "avg_hp": avg_hp,
        "avg_hp_oppo": avg_hp_oppo,
    }


def _compute_overall_average(aggregated: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], int]:
    total_episodes = 0
    metric_totals: Dict[str, float] = {}
    for metrics in aggregated.values():
        episodes = int(metrics.get("episodes", 0)) or 0
        total_episodes += episodes
        for key, value in metrics.items():
            if key == "episodes" or not isinstance(value, (int, float)):
                continue
            metric_totals[key] = metric_totals.get(key, 0.0) + float(value) * episodes
    overall_average = {key: (metric_totals[key] / total_episodes) if total_episodes else 0.0 for key in metric_totals}
    return overall_average, total_episodes


def save_evaluation_results(
    base_path: str,
    filename: str,
    opponents: Dict[str, Dict[str, float]],
    overall_average: Dict[str, float],
    total_episodes: int,
) -> str:
    result_path = os.path.join(base_path, filename)
    payload = {
        "opponents": opponents,
        "total_episodes": total_episodes,
        "overall_average": overall_average,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    logging.info("Evaluation results saved to %s", result_path)
    return result_path


def evaluate(
    model1_path: str,
    target_path: str,
    n_episodes: int = 1,
    render_mode: str = None,
    use_tqdm: bool = True,
    result_filename: str = "evaluation_results.yaml",
) -> Dict[str, Dict[str, float]]:
    def _is_valid_model_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        required = ("agent_config.yaml", "env_config.yaml")
        if not all(os.path.exists(os.path.join(path, name)) for name in required):
            return False
        stem = os.path.join(path, "best_model")
        return os.path.exists(stem) or os.path.exists(f"{stem}.zip")

    model1_path = os.path.abspath(model1_path)
    target_path = os.path.abspath(target_path)
    if not os.path.isdir(model1_path):
        raise FileNotFoundError(f"model1_path not found: {model1_path}")

    opponents: List[Tuple[str, str]] = []
    if os.path.isdir(target_path):
        for entry in sorted(os.listdir(target_path)):
            entry_path = os.path.join(target_path, entry)
            if _is_valid_model_dir(entry_path):
                opponents.append((entry, entry_path))
        if not opponents and _is_valid_model_dir(target_path):
            opponents.append((os.path.basename(target_path.rstrip(os.sep)), target_path))
    if not opponents:
        raise FileNotFoundError(f"No valid opponent models found under: {target_path}")

    aggregated: Dict[str, Dict[str, float]] = {}
    for opponent_name, opponent_path in opponents:
        logging.info("Evaluating %s vs %s", os.path.basename(model1_path), opponent_name)
        raw_metrics = evaluate_without_NN(
            model1_path=model1_path,
            model2_path=opponent_path,
            n_episodes=n_episodes,
            render_mode=render_mode,
            use_tqdm=use_tqdm,
        )
        sanitized: Dict[str, float] = {}
        for key, value in raw_metrics.items():
            if isinstance(value, np.ndarray):
                sanitized[key] = float(value.item())
            elif isinstance(value, (np.floating, np.integer)):
                sanitized[key] = float(value)
            else:
                sanitized[key] = value
        sanitized["episodes"] = int(n_episodes)
        aggregated[opponent_name] = sanitized

    overall_average, total_episodes = _compute_overall_average(aggregated)
    result_path = save_evaluation_results(
        model1_path,
        result_filename,
        aggregated,
        overall_average,
        total_episodes,
    )
    return {
        "opponents": aggregated,
        "overall_average": overall_average,
        "total_episodes": total_episodes,
        "result_path": result_path,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--random_input", type=bool, default=False)
    parser.add_argument("--pool_path", type=str, default=None, help="Path to the opponent pool directory.")
    parser.add_argument("--model_num", type=int, default=0, help="Model number for multi-agent environments")
    parser.add_argument("--n_episode", type=int, default=1, help="Number of episodes for evaluation.")
    args = parser.parse_args()
    if args.render_mode == "none" or args.render_mode == "None":
        args.render_mode = None
    if args.n_episode <= 1:
        show(args.exp_path, args.render_mode, model_num=args.model_num, pool_path=args.pool_path)
    else:
        if args.pool_path is None:
            raise ValueError("pool_path is required when n_episode > 1.")
        result = evaluate(
            model1_path=args.exp_path,
            target_path=args.pool_path,
            n_episodes=args.n_episode,
            render_mode=args.render_mode,
        )
        print(f"Evaluation results saved to {result['result_path']}")