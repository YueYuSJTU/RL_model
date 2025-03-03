import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

from GRU_NN import RNNEncoder
from ContinueWrapper import ContinueObservation
import sys
sys.path.insert(0, "D:\yy_workspace\Reinforcement_learning\jsbgym")

import jsbgym_m             # type: ignore


# # env_id = "C172-TurnHeadingControlTask-Shaping.EXTRA_SEQUENTIAL-NoFG-v0"
# env_id = "C172-TrajectoryTask-Shaping.STANDARD-NoFG-v0"

# Create environment
plane = "C172"

# task = "HeadingControlTask"
# task = "SmoothHeadingTask"
# task = "TurnHeadingControlTask"
task = "TrajectoryTask"

# shape = "Shaping.STANDARD"
shape = "Shaping.EXTRA"

# ======================================================================================================================

env_id = f"{plane}-{task}-{shape}-NoFG-v0"

timesteps = 5000_000
num_cpu = 4  # Number of processes to use
render_mode = None
is_gru = True
# is_gru = False
skip_frame = 3
total_frame = 3
log_path = "./gru_train/logs/"
tensorboard_path = "./gru_train/logs/tensorboard_log/"

features_extractor_class = RNNEncoder
features_extractor_kwargs = dict(
    features_dim=128,
    embed_dim=128,
    rnn_hidden_size=128,
    num_layers=1,
    dropout=None,
    encoder="GRU",
    obs_length=15,
)

def net_arch(on_policy: bool):
    return (
        dict(pi=[256, 256], vf=[256, 256])
        if on_policy
        else dict(pi=[256, 256], qf=[256, 256])
    )

def get_model(model_name: str, env: gym.Env, is_gru: bool = False, seed: int = 0):
    """
    Get the model class from the model name.
    """
    if model_name == "PPO":
        model_class = PPO
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=1e-5,
        verbose=1,
        tensorboard_log=tensorboard_path,
        ent_coef=0.01,
        batch_size=1024,
        policy_kwargs=dict(
            share_features_extractor=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            net_arch=net_arch(on_policy=True),
        ) if is_gru else None,
    )
    return model

def get_env(
        env_id: str, 
        num_cpu: int = 1, 
        render_mode=None,
        is_gru: bool = False,
        skip_frame: int = 4,
        total_frame: int = 5,
        load_mode: bool = False,
    ):
    """
    Create the vectorized environment
    """
    env = SubprocVecEnv([make_env(
        env_id, 
        i, 
        render_mode=render_mode,
        is_gru=is_gru, 
        skip_frame=skip_frame, 
        total_frame=total_frame)
          for i in range(num_cpu)])
    if load_mode:
        env = VecNormalize.load(log_path + "final_train_env", env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env

def make_env(env_id: str, rank: int, seed: int = 0, render_mode= None, is_gru: bool = False, skip_frame: int = 4, total_frame: int = 5):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if is_gru:
            env = ContinueObservation(env, skip_frame=skip_frame, total_frame=total_frame)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    # Create the vectorized environment
    vec_env = get_env(env_id, num_cpu, render_mode, is_gru=is_gru, skip_frame=skip_frame, total_frame=total_frame)

    # 创建评估环境
    eval_env = get_env(env_id, 1, render_mode, is_gru=is_gru, skip_frame=skip_frame, total_frame=total_frame)
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
    #                 clip_obs=10.)
    eval_env.training = False
    eval_env.norm_reward = False

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                                log_path=log_path, eval_freq=50000,
                                deterministic=True, render=False)

    model = get_model("PPO", vec_env, is_gru=is_gru, seed=0)
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=[eval_callback])
    # model.save(log_path + "/model")
    vec_env.save(log_path + "final_train_env")
    vec_env.close()
