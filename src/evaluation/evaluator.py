import os
import yaml
import logging
import numpy as np
import torch
import gc
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.make_agent import load_agent

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
        obs_wrappers: Optional[List[Dict]] = None,
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
        if obs_wrappers:
            env_cfg["wrappers"] = list(obs_wrappers)
        vec_env = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv)
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
            model1 = load_agent(
                env=fake_env1,
                agent_class=agent1_cfg.get("algorithm", "PPO"),
                path=os.path.join(model1_path, "best_model"),
                device=agent1_cfg["device"],
                agent_cfg=agent1_cfg,
            )
            print(f"debug: model1 class is {model1.__class__}")
            model1 = ObsAdaptingModel(model1, env1_cfg)

        # Initialize Agent 2 (Model)
        agent2_cfg = load_config(os.path.join(model2_path, "agent_config.yaml"))
        env2_cfg = load_config(os.path.join(model2_path, "env_config.yaml"))
        fake_env2 = create_env(env2_cfg, training=False, vec_env_kwargs=None)
        model2 = load_agent(
            env=fake_env2,
            agent_class=agent2_cfg.get("algorithm", "PPO"),
            path=os.path.join(model2_path, "best_model"),
            device=agent2_cfg["device"],
            agent_cfg=agent2_cfg,
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

        # Extended metrics (episode-level aggregates) - model1(self)
        gun_time_ratios: List[float] = []
        damage_to_oppo_totals: List[float] = []
        damage_rates: List[float] = []
        track_angle_means: List[float] = []
        adverse_angle_means: List[float] = []
        overshoot_time_ratios: List[float] = []
        delta_specific_energy_means: List[float] = []
        delta_specific_energy_finals: List[float] = []

        # Extended metrics (episode-level aggregates) - model2(opponent)
        gun_time_ratios_oppo: List[float] = []
        damage_to_self_totals: List[float] = []
        damage_rates_oppo: List[float] = []
        track_angle_means_oppo: List[float] = []
        adverse_angle_means_oppo: List[float] = []
        delta_specific_energy_means_oppo: List[float] = []
        delta_specific_energy_finals_oppo: List[float] = []

        # No timeseries files are saved during evaluation.

        try:
            episodes = range(n_episodes)
            if use_tqdm:
                episodes = tqdm(episodes, desc=f"Evaluating models", ncols=80)

            for episode in episodes:
                obs = vec_env.reset()
                obs_length = obs.shape[1]
                episode_done = False
                episode_reward = 0

                # recurrent state for both agents
                state1 = None
                state2 = None
                episode_start1 = np.ones((vec_env.num_envs,), dtype=bool)
                episode_start2 = np.ones((vec_env.num_envs,), dtype=bool)

                # per-episode step metrics collection
                step_series: List[Dict[str, float]] = []
                prev_hp_self: Optional[float] = None
                prev_hp_oppo: Optional[float] = None

                while not episode_done:
                    if render_mode is not None:
                        vec_env.render()

                    if manual_control:
                        action1, _ = manual_agent.predict(None)
                        action1[:, -1] = np.abs(action1[:, -1])
                    else:
                        try:
                            action1, state1 = model1.predict(
                                obs[:, :obs_length//2],
                                state=state1,
                                episode_start=episode_start1,
                                deterministic=True,
                            )
                        except TypeError:
                            action1, _ = model1.predict(obs[:, :obs_length//2], deterministic=True)

                    # Update model2 to also handle recurrent states
                    try:
                        action2, state2 = model2.predict(
                            obs[:, obs_length//2:],
                            state=state2,
                            episode_start=episode_start2,
                            deterministic=True,
                        )
                    except TypeError:
                        action2, _ = model2.predict(obs[:, obs_length//2:], deterministic=True)

                    combined_action = np.concatenate([action1, action2], axis=-1)
                    obs, reward, dones, info = vec_env.step(combined_action)
                    
                    episode_start1 = dones
                    episode_start2 = dones

                    episode_reward += reward[0]

                    # collect step-level metrics if provided by env/task
                    metrics_step = info[0].get("metrics_step") if isinstance(info, (list, tuple)) else None
                    if metrics_step is not None:
                        # add derived delta HP (from adjacent samples)
                        hp_self = float(metrics_step.get("hp_self", 0.0))
                        hp_oppo = float(metrics_step.get("hp_oppo", 0.0))
                        if prev_hp_self is None:
                            delta_hp_self = 0.0
                            delta_hp_oppo = 0.0
                        else:
                            delta_hp_self = hp_self - float(prev_hp_self)
                            delta_hp_oppo = hp_oppo - float(prev_hp_oppo)
                        prev_hp_self, prev_hp_oppo = hp_self, hp_oppo

                        row = dict(metrics_step)
                        row["delta_hp_self"] = float(delta_hp_self)
                        row["delta_hp_oppo"] = float(delta_hp_oppo)
                        row["step"] = float(len(step_series))
                        step_series.append(row)

                    if bool(dones[0]):
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

                        # Aggregate extended episode metrics (no timeseries saved)
                        if step_series:
                            gun_series = np.array([r.get("gun_opportunity", 0.0) for r in step_series], dtype=np.float32)
                            gun_series_oppo = np.array([r.get("gun_opportunity_oppo", 0.0) for r in step_series], dtype=np.float32)
                            overshoot_series = np.array([r.get("overshoot_flag", 0.0) for r in step_series], dtype=np.float32)

                            track_angle_series = np.array([r.get("track_angle_rad", 0.0) for r in step_series], dtype=np.float32)
                            track_angle_series_oppo = np.array([r.get("oppo_track_angle_rad", 0.0) for r in step_series], dtype=np.float32)

                            hp_self_series = np.array([r.get("hp_self", 0.0) for r in step_series], dtype=np.float32)
                            hp_oppo_series = np.array([r.get("hp_oppo", 0.0) for r in step_series], dtype=np.float32)

                            dhp_oppo = np.diff(hp_oppo_series, prepend=hp_oppo_series[0])
                            damage_to_oppo_total = float(np.sum(np.maximum(0.0, -dhp_oppo)))
                            dhp_self = np.diff(hp_self_series, prepend=hp_self_series[0])
                            damage_to_self_total = float(np.sum(np.maximum(0.0, -dhp_self)))

                            episode_steps = len(step_series)
                            step_hz = None
                            if env_cfg is not None:
                                step_hz = env_cfg.get("agent_interaction_freq") or env_cfg.get("step_frequency_hz")
                            episode_time = float(episode_steps / float(step_hz)) if step_hz else float(episode_steps)

                            gun_time_ratio = float(np.mean(gun_series)) if gun_series.size else 0.0
                            gun_time_ratio_oppo = float(np.mean(gun_series_oppo)) if gun_series_oppo.size else 0.0
                            overshoot_time_ratio = float(np.mean(overshoot_series)) if overshoot_series.size else 0.0

                            track_angle_mean = float(np.mean(track_angle_series)) if track_angle_series.size else 0.0
                            track_angle_mean_oppo = float(np.mean(track_angle_series_oppo)) if track_angle_series_oppo.size else 0.0

                            g_ft = 32.174
                            u = np.array([r.get("u_fps", 0.0) for r in step_series], dtype=np.float32)
                            h = np.array([r.get("altitude_sl_ft", 0.0) for r in step_series], dtype=np.float32)
                            ou = np.array([r.get("oppo_u_fps", 0.0) for r in step_series], dtype=np.float32)
                            oh = np.array([r.get("oppo_altitude_sl_ft", 0.0) for r in step_series], dtype=np.float32)
                            es = h + (u * u) / (2.0 * g_ft)
                            oes = oh + (ou * ou) / (2.0 * g_ft)
                            # specific energy (not difference): Es = h + u^2/(2g)
                            g_ft = 32.174
                            u = np.array([r.get("u_fps", 0.0) for r in step_series], dtype=np.float32)
                            h = np.array([r.get("altitude_sl_ft", 0.0) for r in step_series], dtype=np.float32)
                            ou = np.array([r.get("oppo_u_fps", 0.0) for r in step_series], dtype=np.float32)
                            oh = np.array([r.get("oppo_altitude_sl_ft", 0.0) for r in step_series], dtype=np.float32)
                            es = h + (u * u) / (2.0 * g_ft)
                            oes = oh + (ou * ou) / (2.0 * g_ft)

                            specific_energy_mean = float(np.mean(es)) if es.size else 0.0
                            specific_energy_final = float(es[-1]) if es.size else 0.0
                            specific_energy_mean_oppo = float(np.mean(oes)) if oes.size else 0.0
                            specific_energy_final_oppo = float(oes[-1]) if oes.size else 0.0

                            gun_time_ratios.append(gun_time_ratio)
                            overshoot_time_ratios.append(overshoot_time_ratio)
                            track_angle_means.append(track_angle_mean)
                            damage_to_oppo_totals.append(damage_to_oppo_total)
                            delta_specific_energy_means.append(specific_energy_mean)
                            delta_specific_energy_finals.append(specific_energy_final)

                            gun_time_ratios_oppo.append(gun_time_ratio_oppo)
                            track_angle_means_oppo.append(track_angle_mean_oppo)
                            damage_to_self_totals.append(damage_to_self_total)
                            delta_specific_energy_means_oppo.append(specific_energy_mean_oppo)
                            delta_specific_energy_finals_oppo.append(specific_energy_final_oppo)

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

            # model1 (self) metrics
            "gun_opportunity_time_ratio": float(np.mean(gun_time_ratios)) if gun_time_ratios else 0.0,
            "damage_to_oppo_total": float(np.mean(damage_to_oppo_totals)) if damage_to_oppo_totals else 0.0,
            "damage_rate": float(np.mean(damage_rates)) if damage_rates else 0.0,
            "track_angle_mean": float(np.mean(track_angle_means)) if track_angle_means else 0.0,
            "adverse_angle_mean": float(np.mean(adverse_angle_means)) if adverse_angle_means else 0.0,
            "overshoot_time_ratio": float(np.mean(overshoot_time_ratios)) if overshoot_time_ratios else 0.0,
            "specific_energy_mean": float(np.mean(delta_specific_energy_means)) if delta_specific_energy_means else 0.0,
            "specific_energy_final": float(np.mean(delta_specific_energy_finals)) if delta_specific_energy_finals else 0.0,

            # model2 (opponent) metrics
            "gun_opportunity_time_ratio_oppo": float(np.mean(gun_time_ratios_oppo)) if gun_time_ratios_oppo else 0.0,
            "damage_to_self_total": float(np.mean(damage_to_self_totals)) if damage_to_self_totals else 0.0,
            "damage_rate_oppo": float(np.mean(damage_rates_oppo)) if damage_rates_oppo else 0.0,
            "track_angle_mean_oppo": float(np.mean(track_angle_means_oppo)) if track_angle_means_oppo else 0.0,
            "adverse_angle_mean_oppo": float(np.mean(adverse_angle_means_oppo)) if adverse_angle_means_oppo else 0.0,
            "specific_energy_mean_oppo": float(np.mean(delta_specific_energy_means_oppo)) if delta_specific_energy_means_oppo else 0.0,
            "specific_energy_final_oppo": float(np.mean(delta_specific_energy_finals_oppo)) if delta_specific_energy_finals_oppo else 0.0,
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

        # Also generate a summary figure for quick comparison across opponents
        try:
            from src.evaluation.plot_eval_summary import plot_opponent_metrics

            fig_path = os.path.join(base_path, "evaluation_summary.png")
            plot_opponent_metrics(opponents, fig_path, overall_average=overall_average)
            logging.info("Evaluation summary figure saved to %s", fig_path)
        except Exception as e:
            logging.warning("Failed to generate evaluation summary figure: %s", e)

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
        obs_wrappers: Optional[List[Dict]] = None,
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
                obs_wrappers=obs_wrappers,
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
