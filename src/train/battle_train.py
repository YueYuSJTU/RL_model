import os
import gc
import torch
import yaml
import shutil
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# # --- Import your project's modules ---
# from src.utils.yaml_import import add_path
# add_path()

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from src.environments.make_env import create_env
from src.agents.make_agent import creat_agent, load_agent
from src.utils.logger import setup_logger
from src.utils.serialization import save_config, load_config
from src.utils.custom_callback import ComponentEvalCallback, EpisodeCurriculumCallback
from src.train.pool_manager import PoolManager # Our PoolManager class

# Setup root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class UnifiedTrainer:
    """
    Orchestrates a multi-stage training process, including a final "battle training"
    stage that interacts with an opponent pool.
    """
    def __init__(self, config_path: str, pool_path: str, pretrained_path: str = ""):
        """
        Initializes the UnifiedTrainer.
        
        Args:
            config_path (str): Path to the unified training configuration file.
            pool_path (str): Path to the opponent pool directory.
            pretrained_path (str, optional):
                Path to a root training directory (e.g., experiments/20250928_...) to resume a run.
                If empty, a new training run will be started."""
        with open(config_path, encoding="utf-8") as f:
            self.full_config = yaml.safe_load(f)
        
        self.pool_path = pool_path
        self.pretrained_path = pretrained_path
        self.stage_keys = sorted(self.full_config.keys())
        
        self.model = None
        self.last_best_model_path = ""
        
        self._setup_paths()
        self.pool_manager = PoolManager(
            pool_path=self.pool_path,
            max_pool_size=8 # Default, will be overridden by battle config
        )
        logging.info("Unified Trainer initialized.")

    def _setup_paths(self):
        """Sets up the main training directory."""
        if self.pretrained_path:
            self.train_path = self.pretrained_path
            logging.info(f"Resuming training in existing directory: '{self.train_path}'")
        else:
            home_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_root = next(iter(self.full_config.values()))['log_root']
            self.train_path = os.path.join(log_root, home_timestamp)
            os.makedirs(self.train_path, exist_ok=True)
            logging.info(f"Starting new training run in: '{self.train_path}'")

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

    def _run_normal_stage(self, stage_cfg: Dict[str, Any], stage_path: str, stage_key: str):
        """Executes a standard training stage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        env_name = stage_cfg['env']
        agent_name = stage_cfg['agent']
        exp_path = os.path.join(stage_path, f"{timestamp}_{env_name}_{agent_name}")
        
        logger = setup_logger(exp_path)
        
        # --- Load configs ---
        env_cfg = load_config(os.path.join("configs", "env", f"{env_name}.yaml"))
        agent_cfg = load_config(os.path.join("configs", "agent", f"{agent_name}.yaml"))
        
        save_config(stage_cfg, exp_path, "stage_config.yaml")
        save_config(agent_cfg, exp_path, "agent_config.yaml")
        save_config(env_cfg, exp_path, "env_config.yaml")
        
        # --- Create Environments ---
        vec_env_kwargs = {"pool_roots": self.pool_path}
        train_env = create_env(env_cfg, training=True, num_cpu=stage_cfg["num_cpu"], vec_env_kwargs=vec_env_kwargs)
        eval_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
        eval_env.training = False
        eval_env.norm_reward = False

        # --- Create or Load Model ---
        stage_num = int(stage_key.replace('stage', ''))
        if stage_num > 1 and self.last_best_model_path:
            logger.info(f"Loading model from previous stage: {self.last_best_model_path}")
            self.model = load_agent(
                env=train_env, agent_class=agent_name,
                path=self.last_best_model_path, device=agent_cfg["device"]
            )
        else:
            logger.info("Creating new model for the first stage.")
            self.model = creat_agent(
                env=train_env, agent_class=agent_name,
                tensorboard_log=os.path.join(self.train_path, "tensorboard", timestamp),
                agent_cfg=agent_cfg
            )
        
        # --- Callbacks and Training ---
        eval_callback = ComponentEvalCallback(
            eval_env, best_model_save_path=exp_path, log_path=exp_path,
            eval_freq=stage_cfg["eval_freq"], deterministic=True
        )
        
        total_timesteps = stage_cfg.get("total_timesteps", int(1e12)) # Default to huge number for "infinite"
        if total_timesteps == int(1e12):
            logger.info("'total_timesteps' not set. Training indefinitely until Ctrl+C.")

        try:
            self.model.learn(total_timesteps=total_timesteps, callback=[eval_callback])
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user (Ctrl+C).")
        finally:
            logger.info("Saving final model and environment state...")
            self.model.save(os.path.join(exp_path, "final_model"))
            train_env.save(os.path.join(exp_path, "final_train_env.pkl"))
            self.last_best_model_path = os.path.join(exp_path, "best_model.zip")
            if not os.path.exists(self.last_best_model_path):
                 self.last_best_model_path = os.path.join(exp_path, "final_model.zip")
            logger.info(f"Normal stage '{stage_key}' finished. Best model is at: {self.last_best_model_path}")
            # === 显式资源清理，否则爆显存 ===
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

        # --- Update PoolManager with battle-specific config ---
        self.pool_manager.max_pool_size = stage_cfg.get("max_pool_size", 8)

        # --- Load configs ---
        env_cfg = load_config(os.path.join("configs", "env", f"{stage_cfg['env']}.yaml"))
        agent_cfg = load_config(os.path.join("configs", "agent", f"{stage_cfg['agent']}.yaml"))

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
            env=train_env, agent_class=stage_cfg['agent'],
            path=self.last_best_model_path, device=agent_cfg["device"]
        )

        # --- 初始化在循环外持续存在的Callback ---
        # 课程学习Callback的状态(is_active, num_timesteps)需要在整个战斗阶段中保持
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
        num_cycles = total_timesteps // battle_step

        try:
            completed_cycles = 0
            while self.model.num_timesteps < total_timesteps:
                completed_cycles += 1
                cycle_info = f"Cycle {completed_cycles}"
                if total_timesteps != int(1e12):
                    cycle_info += f"/{num_cycles}"

                logger.info(f"\n{'-'*20} BATTLE-TRAINING {cycle_info} {'-'*20}")

                # 1. 为当前周期创建专属路径
                cycle_dir_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cycle_{completed_cycles}"
                cycle_path = os.path.join(stage_path, cycle_dir_name)
                os.makedirs(cycle_path)

                save_config(stage_cfg, cycle_path, "train_config.yaml")
                save_config(agent_cfg, cycle_path, "agent_config.yaml")
                save_config(env_cfg, cycle_path, "env_config.yaml")

                # === 为每个周期创建专属的EvalCallback ===                
                # 创建一个独立的评估环境，确保评估过程纯净
                eval_env = create_env(env_cfg, training=False, vec_env_kwargs=vec_env_kwargs)
                eval_env.training = False
                eval_env.norm_reward = False

                eval_callback_for_cycle = ComponentEvalCallback(
                    eval_env,
                    best_model_save_path=cycle_path, # <--- 使用当前周期的路径
                    log_path=cycle_path,             # <--- 日志也保存在周期路径下
                    eval_freq=stage_cfg["eval_freq"],
                    deterministic=True
                )
                
                # 组合当前周期需要的所有Callback
                callbacks_for_this_cycle = [eval_callback_for_cycle, curriculum_callback]

                # 2. 使用为本周期定制的Callback列表进行训练
                self.model.learn(
                    total_timesteps=battle_step,
                    reset_num_timesteps=False,
                    callback=callbacks_for_this_cycle
                )

                # 3. 保存周期结束时的最终模型和环境状态
                self.model.save(os.path.join(cycle_path, "final_model"))
                train_env.save(os.path.join(cycle_path, "final_train_env.pkl"))
                # # 复制配置文件
                # shutil.copy(os.path.join(cycle_path, "agent_config.yaml"))
                # shutil.copy(os.path.join(cycle_path, "env_config.yaml"))

                logger.info(f"周期 {completed_cycles} 训练完成。模型已保存至 '{cycle_path}'")

                # === 显式资源清理，否则爆显存 ===
                eval_env.close()
                del eval_env
                del eval_callback_for_cycle
                del callbacks_for_this_cycle
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 3. Challenge the opponent pool
                self.pool_manager.update_pool(new_model_path=cycle_path, n_episodes=n_episodes_eval)
        
        except KeyboardInterrupt:
            logger.warning("Battle training interrupted by user (Ctrl+C).")
        finally:
            if 'train_env' in locals():
                train_env.close()
            logger.info("Battle stage finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified trainer for multi-stage and battle training.")
    parser.add_argument("--config", type=str, required=True, help="Path to the unified training configuration file.")
    parser.add_argument("--pool_path", type=str, required=True, help="Path to the opponent pool directory.")
    parser.add_argument("--pretrained_path", type=str, default="", 
                        help="Path to a root training directory (e.g., experiments/20250928_...) to resume a run.")
    
    args = parser.parse_args()
    
    trainer = UnifiedTrainer(
        config_path=args.config,
        pool_path=args.pool_path,
        pretrained_path=args.pretrained_path
    )
    trainer.run()