import os
import yaml
import logging
import numpy as np
import torch
import gc
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environments.make_env import create_env
from src.utils.serialization import load_config
from src.agents.model_wrapper import ObsAdaptingModel
from src.utils.manual_control import KeyboardController, GamepadController

class Evaluator:
    """
    Unified Evaluator class for running matches between models or manual control.
    Handles environment stepping, reward accumulation, and statistics.
    """

    @staticmethod
    def run_match(
        model1_path: str,
        model2_path: str,
        n_episodes: int = 1,
        render_mode: Optional[str] = None,
        env_cfg: Optional[Dict] = None,
        use_tqdm: bool = True,
        manual_control: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate two models in a combat environment without NN wrapper.
        """
        if n_episodes < 1:
            raise ValueError("n_episodes must be at least 1.")
        if n_episodes > 1 and render_mode is not None:
            logging.warning("Rendering multiple evaluation episodes may slow down the process.")

        # Create evaluation environment
        if env_cfg is None:
            logging.warning("No environment configuration provided, using model1 settings.")
            env_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))

        env_cfg["render_mode"] = render_mode
        env_cfg["use_vec_normalize"] = False
        vec_env = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv, vec_env_kwargs=None)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.env_method("update_task_parameters", goal_point_prob=0.0)

        # Initialize Agent 1 (Manual or Model)
        if manual_control:
            logging.info("Initializing Manual Control for Agent 1...")
            try:
                manual_agent = GamepadController(action_dim=4)
                logging.info(">>> Using GAMEPAD Control <<<")
                logging.info("Controls: Left Stick=Pitch/Roll, RB/LB=Yaw, A/B=Throttle")
                model1 = None
            except Exception as e:
                logging.warning(f"Gamepad initialization failed: {e}")
                logging.info(">>> Falling back to KEYBOARD Control <<<")
                manual_agent = KeyboardController(action_dim=4)
                model1 = None
        else:
            agent1_cfg = load_config(os.path.join(model1_path, "agent_config.yaml"))
            env1_cfg = load_config(os.path.join(model1_path, "env_config.yaml"))
            fake_env1 = create_env(env1_cfg, training=False, vec_env_kwargs=None)
            model1 = PPO.load(
                os.path.join(model1_path, "best_model"),
                env=fake_env1,
                device=agent1_cfg["device"]
            )
            model1 = ObsAdaptingModel(model1, env1_cfg)

        # Initialize Agent 2 (Model)
        agent2_cfg = load_config(os.path.join(model2_path, "agent_config.yaml"))
        env2_cfg = load_config(os.path.join(model2_path, "env_config.yaml"))
        fake_env2 = create_env(env2_cfg, training=False, vec_env_kwargs=None)
        model2 = PPO.load(
            os.path.join(model2_path, "best_model"),
            env=fake_env2,
            device=agent2_cfg["device"]
        )
        model2 = ObsAdaptingModel(model2, env2_cfg)

        # Metrics
        wins = 0
        losses = 0
        draws = 0
        opponent_falls = 0
        avg_hp = 0
        avg_hp_oppo = 0
        win_steps = []
        total_rewards = []

        try:
            episodes = range(n_episodes)
            if use_tqdm:
                episodes = tqdm(episodes, desc=f"Evaluating models", ncols=80)

            for episode in episodes:
                obs = vec_env.reset()
                obs_length = obs.shape[1]
                episode_done = False
                episode_reward = 0

                while not episode_done:
                    if render_mode is not None:
                        vec_env.render()

                    if manual_control:
                        action1, _ = manual_agent.predict(None)
                        action1[:, -1] = np.abs(action1[:, -1])
                    else:
                        action1, _ = model1.predict(obs[:, :obs_length//2], deterministic=True)

                    action2, _ = model2.predict(obs[:, obs_length//2:], deterministic=True)
                    combined_action = np.concatenate([action1, action2], axis=-1)
                    obs, reward, terminated, info = vec_env.step(combined_action)

                    episode_reward += reward[0]

                    if terminated:
                        episode_done = True
                        total_rewards.append(episode_reward)
                        env_info = info[0].get("env_info", {})
                        win_status = env_info.get("win", 0)

                        if win_status == 1:
                            wins += 1
                            win_steps.append(env_info.get("steps_used", 0))
                        elif win_status == 0:
                            draws += 1
                        elif win_status == -1:
                            losses += 1
                        elif win_status == 0.5:
                            opponent_falls += 1

                        avg_hp += env_info.get("HP_self", 0)
                        avg_hp_oppo += env_info.get("HP_oppo", 0)

            # Calculate statistics
            win_rate = wins / n_episodes
            draw_rate = draws / n_episodes
            loss_rate = losses / n_episodes
            opponent_fall_rate = opponent_falls / n_episodes
            avg_win_time = np.mean(win_steps) if win_steps else 0
            avg_reward = np.mean(total_rewards) if total_rewards else 0
            avg_hp = avg_hp / n_episodes
            avg_hp_oppo = avg_hp_oppo / n_episodes

        finally:
            # Cleanup
            if 'vec_env' in locals() and vec_env is not None:
                vec_env.close()
            if 'manual_agent' in locals() and manual_agent is not None:
                manual_agent.close()
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

    @staticmethod
    def compute_overall_average(aggregated: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], int]:
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

    @staticmethod
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

    @classmethod
    def evaluate_pool(
        cls,
        model1_path: str,
        target_path: str,
        n_episodes: int = 1,
        render_mode: Optional[str] = None,
        use_tqdm: bool = True,
        result_filename: str = "evaluation_results.yaml",
    ) -> Dict:
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
            raw_metrics = cls.run_match(
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

        overall_average, total_episodes = cls.compute_overall_average(aggregated)
        result_path = cls.save_evaluation_results(
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
