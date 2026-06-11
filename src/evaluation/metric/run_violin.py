import os
import json
import logging
import numpy as np
import torch
import gc
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

import sys
sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.agents.make_agent import load_agent
from src.environments.make_env import create_env
from src.utils.serialization import load_config
from src.agents.model_wrapper import ObsAdaptingModel

def run_granular_evaluation(nameA: str, pathA: str, nameB: str, pathB: str, n_episodes: int, output_file: str):
    logging.basicConfig(level=logging.INFO)
    
    # Load env config from Model A
    env_cfg = load_config(os.path.join(pathA, "env_config.yaml"))
    env_cfg["render_mode"] = None
    env_cfg["use_vec_normalize"] = False
    
    vec_env = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.env_method("update_task_parameters", goal_point_prob=0.0)

    # Load Model A
    logging.info(f"Loading {nameA}...")
    agentA_cfg = load_config(os.path.join(pathA, "agent_config.yaml"))
    envA_cfg = load_config(os.path.join(pathA, "env_config.yaml"))
    fake_envA = create_env(envA_cfg, training=False, vec_env_kwargs=None)
    modelA_raw = load_agent(
        env=fake_envA, agent_class=agentA_cfg.get("algorithm", "PPO"),
        path=os.path.join(pathA, "best_model"), device=agentA_cfg["device"], agent_cfg=agentA_cfg,
    )
    modelA = ObsAdaptingModel(modelA_raw, envA_cfg)

    # Load Model B
    logging.info(f"Loading {nameB}...")
    agentB_cfg = load_config(os.path.join(pathB, "agent_config.yaml"))
    envB_cfg = load_config(os.path.join(pathB, "env_config.yaml"))
    fake_envB = create_env(envB_cfg, training=False, vec_env_kwargs=None)
    modelB_raw = load_agent(
        env=fake_envB, agent_class=agentB_cfg.get("algorithm", "PPO"),
        path=os.path.join(pathB, "best_model"), device=agentB_cfg["device"], agent_cfg=agentB_cfg,
    )
    modelB = ObsAdaptingModel(modelB_raw, envB_cfg)

    episode_results = []
    
    phases = [
        (nameA, modelA, nameB, modelB),
        (nameB, modelB, nameA, modelA)
    ]

    try:
        episode_counter = 0
        for phase_idx, (name1, m1, name2, m2) in enumerate(phases):
            logging.info(f"Phase {phase_idx+1}: [Agent 1: {name1}] vs [Agent 2: {name2}]")
            for episode in tqdm(range(n_episodes), desc=f"Evaluating Phase {phase_idx+1}", ncols=80):
                episode_counter += 1
                obs = vec_env.reset()
                obs_length = obs.shape[1]
                episode_done = False
                
                state1 = None
                episode_start1 = np.ones((1,), dtype=bool)
                step_series = []

                while not episode_done:
                    try:
                        action1, state1 = m1.predict(
                            obs[:, :obs_length//2], state=state1, episode_start=episode_start1, deterministic=True,
                        )
                    except TypeError:
                        action1, _ = m1.predict(obs[:, :obs_length//2], deterministic=True)

                    action2, _ = m2.predict(obs[:, obs_length//2:], deterministic=True)
                    combined_action = np.concatenate([action1, action2], axis=-1)
                    obs, reward, dones, info = vec_env.step(combined_action)
                    episode_start1 = dones

                    metrics_step = info[0].get("metrics_step")
                    if metrics_step is not None:
                        step_series.append(dict(metrics_step))

                    if bool(dones[0]):
                        episode_done = True
                        env_info = info[0].get("env_info", {})
                        
                        g_ft = 32.174
                        es_1 = [s.get("altitude_sl_ft", 0.0) + (s.get("u_fps", 0.0)**2)/(2.0*g_ft) for s in step_series]
                        es_2 = [s.get("oppo_altitude_sl_ft", 0.0) + (s.get("oppo_u_fps", 0.0)**2)/(2.0*g_ft) for s in step_series]

                        se_mean_1 = sum(es_1) / len(es_1) if es_1 else 0.0
                        se_mean_2 = sum(es_2) / len(es_2) if es_2 else 0.0

                        win_1 = env_info.get("win", 0) # 1: win, 0: draw, -1: loss, 0.5: opponent falls
                        if win_1 == 1: win_2 = -1
                        elif win_1 == -1: win_2 = 1
                        elif win_1 == 0: win_2 = 0
                        else: win_2 = -0.5

                        result = {
                            "episode_global": episode_counter,
                            "phase_matchup": f"{name1}_vs_{name2}",
                            name1: {
                                "win_status": win_1,
                                "hp": env_info.get("HP_self", 0.0),
                                "specific_energy_mean": se_mean_1,
                            },
                            name2: {
                                "win_status": win_2,
                                "hp": env_info.get("HP_oppo", 0.0),
                                "specific_energy_mean": se_mean_2,
                            }
                        }
                        episode_results.append(result)

    finally:
        vec_env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(episode_results, f, indent=4)
    logging.info(f"Granular evaluation results saved to {output_file}")


if __name__ == "__main__":
    # 硬编码参数，不再需要从命令行传入
    model_A_name = "Ours-LSTM"
    model_A_path = "experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35"
    
    model_B_name = "Ours-MLP"
    model_B_path = "experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30"

    n_episodes_per_phase = 1000
    output_filename = "src/evaluation/plot/results_violin.json"

    run_granular_evaluation(
        nameA=model_A_name, 
        pathA=model_A_path, 
        nameB=model_B_name, 
        pathB=model_B_path, 
        n_episodes=n_episodes_per_phase, 
        output_file=output_filename
    )
