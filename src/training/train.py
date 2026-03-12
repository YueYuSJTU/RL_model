import os
import gc
import torch
import yaml
import shutil
import argparse
import logging
from datetime import datetime
from typing import Dict, Any


def _sanitize_exp_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import VecNormalize
from src.environments.make_env import create_env
from src.agents.make_agent import creat_agent, load_agent
from src.utils.logger import setup_logger
from src.utils.serialization import save_config, load_config
from src.utils.custom_callback import ComponentEvalCallback, EpisodeCurriculumCallback
from src.training.pool_manager import PoolManager
from src.evaluation.evaluator import Evaluator

# Setup root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UnifiedTrainer:
    """
    Orchestrates a multi-stage training process, including standard stages and a final "battle training"
    stage that interacts with an opponent pool.
    """
    def __init__(self, config_path: str, pool_path: str, pretrained_path: str = "", debug_mode: bool = False, exp_name: str = ""):
        """
        Initializes the UnifiedTrainer.

        Args:
            config_path (str): Path to the unified training configuration file.
            pool_path (str): Path to the opponent pool directory.
            pretrained_path (str, optional):
                Path to a root training directory (e.g., experiments/20250928_...) to resume a run.
                If empty, a new training run will be started.
            debug_mode (bool): If True, limits training to 2 battle cycles for testing.
        """
        with open(config_path, encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)

        self.pool_path = pool_path
        self.pretrained_path = pretrained_path
        self.debug_mode = debug_mode
        self.exp_name = exp_name

        # Optional: per-experiment opponent pool config (kept at top-level in YAML)
        self.opponent_pool_cfg = self.full_config.get("opponent_pool")

        # Only treat stage* keys as training stages (top-level metadata like opponent_pool
        # should not be interpreted as a stage)
        self.stage_keys = sorted([k for k in self.full_config.keys() if str(k).startswith("stage")])

        self.model = None
        self.last_best_model_path = ""

        self._setup_paths()

        # If configured, materialize a per-experiment opponent pool by copying from base
        if self.opponent_pool_cfg is not None:
            self.pool_path = self._setup_experiment_opponent_pool(self.opponent_pool_cfg)

        whitelist = []
        max_pool_size = 8
        if isinstance(self.opponent_pool_cfg, dict):
            whitelist = self.opponent_pool_cfg.get("whitelist") or []
            max_pool_size = int(self.opponent_pool_cfg.get("max_pool_size", 8))

        self.pool_manager = PoolManager(
            pool_path=self.pool_path,
            max_pool_size=max_pool_size,
            whitelist=whitelist,
        )
        logging.info("Unified Trainer initialized.")

    def _setup_paths(self):
        """Sets up the main training directory."""
        if self.pretrained_path:
            self.train_path = self.pretrained_path
            logging.info(f"Resuming training in existing directory: '{self.train_path}'")
        else:
            home_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Determine log_root from first stage config (stage keys only)
            first_stage_key = self.stage_keys[0] if self.stage_keys else None
            if not first_stage_key:
                raise ValueError("No stage* entries found in config")
            log_root = self.full_config[first_stage_key]["log_root"]
            exp = _sanitize_exp_name(self.exp_name)
            folder = home_timestamp if exp == "" else f"{home_timestamp}_{exp}"
            self.train_path = os.path.join(log_root, folder)
            os.makedirs(self.train_path, exist_ok=True)
            logging.info(f"Starting new training run in: '{self.train_path}'")
            save_config(self.full_config, self.train_path, "full_config.yaml")

    def _setup_experiment_opponent_pool(self, cfg: Dict[str, Any]) -> str:
        if not isinstance(cfg, dict):
            raise ValueError("opponent_pool must be a mapping/dict")

        base_dir = cfg.get("base_dir", "./base_opponent_model")
        opponents = cfg.get("opponents")
        pool_subdir = cfg.get("pool_subdir", "opponent_pool")

        if not opponents or not isinstance(opponents, (list, tuple)):
            raise ValueError("opponent_pool.opponents must be a non-empty list")

        base_dir_abs = os.path.abspath(base_dir)
        if not os.path.isdir(base_dir_abs):
            raise FileNotFoundError(f"base opponent dir not found: {base_dir_abs}")

        exp_pool_root = os.path.join(self.train_path, pool_subdir)
        os.makedirs(exp_pool_root, exist_ok=True)

        for opp_id in opponents:
            opp_id = str(opp_id)
            src = os.path.join(base_dir_abs, opp_id)
            dst = os.path.join(exp_pool_root, opp_id)
            if not os.path.isdir(src):
                raise FileNotFoundError(f"base opponent id dir not found: {src}")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        logging.info(f"[opponent_pool] Using per-experiment pool at: {exp_pool_root}")
        logging.info(f"[opponent_pool] Seeded from base: {base_dir_abs} (ids={list(map(str, opponents))})")
        return exp_pool_root

    def run(self):
        """Executes the entire training pipeline, stage by stage."""
        for i, stage_key in enumerate(self.stage_keys):
            stage_cfg = self.full_config[stage_key]
            is_last_stage = (i == len(self.stage_keys) - 1)
            is_battle_stage = is_last_stage and 'battle_step' in stage_cfg

            stage_path = os.path.join(self.train_path, stage_key)
            os.makedirs(stage_path, exist_ok=True)

            logging.info(f"\n{'='*25} Starting {stage_key} {'='*25}")

            if is_battle_stage:
                self._run_battle_stage(stage_cfg, stage_path)
            else:
                self._run_normal_stage(stage_cfg, stage_path, stage_key)

        logging.info("All training stages completed.")

        # Evaluate the final model
        latest_result = self._find_latest_training_result(self.train_path)
        if latest_result and self.full_config.get("opponent_pool", None):
            logging.info(f"Evaluating final model from {latest_result}")
            Evaluator.evaluate_pool(latest_result, self.pool_path, n_episodes=100)

    def _find_latest_training_result(self, base_path):
        """Finds the latest training result directory."""
        stage_dirs = [d for d in os.listdir(base_path)
                     if os.path.isdir(os.path.join(base_path, d)) and "stage" in d]

        if not stage_dirs:
            return None

        latest_stage = max(
            [os.path.join(base_path, d) for d in stage_dirs],
            key=os.path.getmtime
        )

        result_dirs = [d for d in os.listdir(latest_stage)
                      if os.path.isdir(os.path.join(latest_stage, d))]

        if not result_dirs:
            return latest_stage

        latest_result = max(
            [os.path.join(latest_stage, d) for d in result_dirs],
            key=os.path.getmtime
        )

        return latest_result

    def _run_normal_stage(self, stage_cfg: Dict[str, Any], stage_path: str, stage_key: str):
        """Executes a standard training stage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # For naming, allow agent to be dict; fall back to stage_key.
        env_name = stage_cfg.get('env')
        agent_name = stage_cfg.get('agent')
        env_tag = env_name if isinstance(env_name, str) else (env_name.get('shape') if isinstance(env_name, dict) else stage_key)
        agent_tag = agent_name if isinstance(agent_name, str) else (agent_name.get('name') if isinstance(agent_name, dict) else "agent")
        exp_path = os.path.join(stage_path, f"{timestamp}_{env_tag}_{agent_tag}")

        logger = setup_logger(exp_path)

        # --- Load configs (unified with shared sections) ---
        # Supported structure:
        # - top-level: agent / env_base
        # - per-stage: env (merged over env_base) and optional agent override
        base_env = self.full_config.get("env_base") or {}
        stage_env = stage_cfg.get("env") or {}
        if not isinstance(stage_env, dict):
            raise ValueError("stage.env must be dict in unified config")
        if isinstance(base_env, dict):
            env_cfg = dict(base_env)
            env_cfg.update(stage_env)
        else:
            env_cfg = dict(stage_env)

        base_agent = self.full_config.get("agent") or {}
        stage_agent = stage_cfg.get("agent") or {}
        if not isinstance(stage_agent, dict):
            # backward-compatible: allow string agent id
            stage_agent = {"name": stage_agent}
        if isinstance(base_agent, dict):
            agent_cfg = dict(base_agent)
            agent_cfg.update(stage_agent)
        else:
            agent_cfg = dict(stage_agent)

        save_config(stage_cfg, exp_path, "stage_config.yaml")
        save_config(agent_cfg, exp_path, "agent_config.yaml")
        save_config(env_cfg, exp_path, "env_config.yaml")

        # Remove non-SB3 keys before passing into PPO(...)
        agent_cfg_for_sb3 = dict(agent_cfg)
        agent_cfg_for_sb3.pop("name", None)

        # --- Create Environments ---
        vec_env_kwargs = {"pool_roots": self.pool_path}
        if stage_cfg.get("model_num") is not None:
            vec_env_kwargs["model_num"] = stage_cfg["model_num"]

        train_env = create_env(env_cfg, training=True, num_cpu=stage_cfg["num_cpu"], vec_env_kwargs=vec_env_kwargs)
        eval_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
        eval_env.training = False
        eval_env.norm_reward = False

        # --- Create or Load Model ---
        stage_num = int(stage_key.replace('stage', ''))
        if stage_num > 1 and self.last_best_model_path:
            logger.info(f"Loading model from previous stage: {self.last_best_model_path}")
            self.model = load_agent(
                env=train_env, agent_class=agent_cfg.get("name", agent_tag),
                path=self.last_best_model_path, device=agent_cfg["device"]
            )
        elif self.pretrained_path and stage_num == 1:
            # Fine-tuning mode
            latest_stage_dir = self._find_latest_training_result(self.pretrained_path)
            logger.info(f"Fine-tuning from {latest_stage_dir}")
            self.model = load_agent(
                env=train_env, agent_class=agent_cfg.get("name", agent_tag),
                path=os.path.join(latest_stage_dir, "best_model"),
                device=agent_cfg["device"]
            )
        else:
            logger.info("Creating new model for the first stage.")
            self.model = creat_agent(
                env=train_env, agent_class=agent_cfg.get("name", agent_tag),
                tensorboard_log=os.path.join(self.train_path, "tensorboard", timestamp),
                agent_cfg=agent_cfg_for_sb3
            )

        # --- Callbacks and Training ---
        eval_callback = ComponentEvalCallback(
            eval_env, best_model_save_path=exp_path, log_path=exp_path,
            eval_freq=stage_cfg["eval_freq"], deterministic=True
        )
        progress_callback = ProgressBarCallback()

        total_timesteps = stage_cfg.get("total_timesteps", int(1e12)) # Default to huge number for "infinite"
        if total_timesteps == int(1e12):
            logger.info("'total_timesteps' not set. Training indefinitely until Ctrl+C.")

        try:
            self.model.learn(total_timesteps=total_timesteps, callback=[eval_callback, progress_callback])
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (Ctrl+C).")
        finally:
            logger.info("Saving final model and environment state...")
            self.model.save(os.path.join(exp_path, "final_model"))
            self.last_best_model_path = os.path.join(exp_path, "best_model.zip")
            if not os.path.exists(self.last_best_model_path):
                 self.last_best_model_path = os.path.join(exp_path, "final_model.zip")

            # Copy best_model.zip to parent directory for easier access
            if os.path.exists(self.last_best_model_path):
                shutil.copy(self.last_best_model_path, os.path.join(exp_path, ".."))

            logger.info(f"Normal stage '{stage_key}' finished. Best model is at: {self.last_best_model_path}")
            # === Explicit resource cleanup ===
            eval_env.close()
            del eval_env
            del train_env
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_battle_stage(self, stage_cfg: Dict[str, Any], stage_path: str):
        """Executes the cyclical battle training stage."""
        logger = setup_logger(stage_path)
        logger.info(f"Entering BATTLE-TRAINING mode for stage '{self.stage_keys[-1]}'.")

        # --- Update PoolManager with config ---
        if isinstance(self.opponent_pool_cfg, dict) and self.opponent_pool_cfg.get("max_pool_size") is not None:
            self.pool_manager.max_pool_size = int(self.opponent_pool_cfg["max_pool_size"])

        # --- Load configs (unified with shared sections) ---
        base_env = self.full_config.get("env_base") or {}
        stage_env = stage_cfg.get("env") or {}
        if not isinstance(stage_env, dict):
            raise ValueError("stage.env must be dict in unified config")
        if isinstance(base_env, dict):
            env_cfg = dict(base_env)
            env_cfg.update(stage_env)
        else:
            env_cfg = dict(stage_env)

        base_agent = self.full_config.get("agent") or {}
        stage_agent = stage_cfg.get("agent") or {}
        if not isinstance(stage_agent, dict):
            # backward-compatible: allow string agent id
            stage_agent = {"name": stage_agent}
        if isinstance(base_agent, dict):
            agent_cfg = dict(base_agent)
            agent_cfg.update(stage_agent)
        else:
            agent_cfg = dict(stage_agent)

        # --- Create Environment with Opponent Pool ---
        logger.info(f"Creating training environment with opponent pool: {self.pool_path}")
        vec_env_kwargs = {"pool_roots": self.pool_path}
        train_env = create_env(
            env_cfg, training=True, num_cpu=stage_cfg["num_cpu"], vec_env_kwargs=vec_env_kwargs
        )

        # --- Load Model from Previous Stage ---
        if not self.last_best_model_path:
             raise RuntimeError("Cannot start battle stage without a model from a previous stage.")

        logger.info(f"Loading model for battle training: {self.last_best_model_path}")
        self.model = load_agent(
            env=train_env, agent_class=agent_cfg.get("name", "ppo"),
            path=self.last_best_model_path, device=agent_cfg["device"]
        )

        # --- Initialize persistent callbacks ---
        logger.info("initializing curriculum learning callback for battle stage.")
        curriculum_callback = EpisodeCurriculumCallback(
            threshold_timesteps=stage_cfg['threshold_timesteps'],
            update_freq_episodes=stage_cfg['update_freq_episodes'],
            verbose=1
        )

        # --- Main Battle Cycle ---
        total_timesteps = stage_cfg.get("total_timesteps", int(1e12))
        battle_step = stage_cfg['battle_step']
        n_episodes_eval = stage_cfg.get('n_episodes_eval', 500)
        battle_cycles = int(stage_cfg.get('battle_cycles', 0) or 0)
        num_cycles = total_timesteps // battle_step

        try:
            completed_cycles = 0
            while self.model.num_timesteps < total_timesteps:
                completed_cycles += 1
                cycle_info = f"Cycle {completed_cycles}"
                if total_timesteps != int(1e12):
                    cycle_info += f"/{num_cycles}"

                logger.info(f"\n{'-'*20} BATTLE-TRAINING {cycle_info} {'-'*20}")

                # 1. Create cycle-specific path
                cycle_dir_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cycle_{completed_cycles}"
                cycle_path = os.path.join(stage_path, cycle_dir_name)
                os.makedirs(cycle_path)

                save_config(stage_cfg, cycle_path, "train_config.yaml")
                save_config(agent_cfg, cycle_path, "agent_config.yaml")

                agent_cfg_for_sb3 = dict(agent_cfg)
                agent_cfg_for_sb3.pop("name", None)
                save_config(env_cfg, cycle_path, "env_config.yaml")

                # === Create cycle-specific EvalCallback ===
                eval_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
                eval_env.training = False
                eval_env.norm_reward = False
                eval_env.env_method("update_task_parameters", goal_point_prob=curriculum_callback.last_update_prob)

                eval_callback_for_cycle = ComponentEvalCallback(
                    eval_env,
                    best_model_save_path=cycle_path,
                    log_path=cycle_path,
                    eval_freq=stage_cfg["eval_freq"],
                    deterministic=True
                )

                callbacks_for_this_cycle = [eval_callback_for_cycle, curriculum_callback]

                # 2. Train for this cycle
                self.model.learn(
                    total_timesteps=battle_step,
                    reset_num_timesteps=False,
                    callback=callbacks_for_this_cycle
                )

                # 3. Save cycle model
                self.model.save(os.path.join(cycle_path, "final_model"))

                logger.info(f"Cycle {completed_cycles} completed. Model saved to '{cycle_path}'")

                # === Explicit resource cleanup ===
                eval_env.close()
                del eval_env
                del eval_callback_for_cycle
                del callbacks_for_this_cycle
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 4. Challenge the opponent pool
                self.pool_manager.update_pool(new_model_path=cycle_path, n_episodes=n_episodes_eval)
                train_env.update_opponent_models()

                if battle_cycles > 0 and completed_cycles >= battle_cycles:
                    logger.info(f"Reached configured battle_cycles={battle_cycles}. Terminating battle training.")
                    break

                if self.debug_mode and completed_cycles >= 2:
                    logger.info("Debug mode: Reached 2 battle cycles. Terminating training.")
                    break
        except KeyboardInterrupt:
            logger.warning("Battle training interrupted by user (Ctrl+C).")
        finally:
            if 'train_env' in locals():
                train_env.close()
            logger.info("Battle stage finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified trainer for multi-stage and battle training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the unified training configuration file.")
    parser.add_argument("--pool_path", type=str, default="", help="Path to the opponent pool directory (legacy mode).")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="Path to a root training directory (e.g., experiments/20250928_...) to resume a run.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (terminates after 2 battle cycles).")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name appended to timestamp folder")

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    if (not isinstance(_cfg, dict)) or ("opponent_pool" not in _cfg):
        if not args.pool_path:
            raise SystemExit("--pool_path is required when config has no top-level opponent_pool")

    trainer = UnifiedTrainer(
        config_path=args.config,
        pool_path=args.pool_path,
        pretrained_path=args.pretrained_path,
        debug_mode=args.debug,
        exp_name=args.exp_name
    )
    trainer.run()
