import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo
from ContinueWrapper import ContinueObservation
from main import log_path, plane, task, shape, get_env, is_gru, skip_frame, total_frame

render_mode = "human"
# render_mode = "flightgear"

model_name = "best_model"
# model_name = "model"

# ======================================================================================================================
if __name__ == "__main__":
    env_id = f"{plane}-{task}-{shape}-FG-v0" if render_mode == "flightgear" else f"{plane}-{task}-{shape}-NoFG-v0"
    vec_env = get_env(env_id, num_cpu=1, render_mode=render_mode, is_gru=is_gru, load_mode=True, skip_frame=skip_frame, total_frame=total_frame)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(log_path + model_name, env=vec_env, device='cpu')

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    obs = vec_env.reset()
    print("Start to simulate")
    for i in range(15000):
        vec_env.render()
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, _ = vec_env.step(action)
        if terminated:
            print(f"Terminated at step {i}")
            break

    vec_env.close()