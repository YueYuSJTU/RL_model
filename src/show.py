import os
import sys
import argparse

sys.path.insert(0, "/home/ubuntu/Workfile/RL/RL_model")
from src.evaluation.evaluator import Evaluator

def show(exp_path: str, render_mode: str = "human", model_num: int = 0, pool_path: str = None, manual_control: bool = False,
         sensor_missing_prob: float = 0.0, sensor_rel_error_std: float = 0.0) -> None:
    model1_path = exp_path
    model2_path = os.path.join(pool_path, str(model_num))
    obs_wrappers = None
    if sensor_missing_prob > 0.0 or sensor_rel_error_std > 0.0:
        obs_wrappers = [
            {
                "name": "src.environments.sensor_wrapper:SensorPerturbationWrapper",
                "kwargs": {
                    "missing_prob": sensor_missing_prob,
                    "rel_error_std": sensor_rel_error_std,
                    "prefixes": ["target/", "oppo/"]
                },
            }
        ]

    results = Evaluator.run_match(
        model1_path=model1_path,
        model2_path=model2_path,
        n_episodes=1,
        render_mode=render_mode,
        manual_control=manual_control,
        obs_wrappers=obs_wrappers,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--random_input", type=bool, default=False)
    parser.add_argument("--pool_path", type=str, default=None, help="Path to the opponent pool directory.")
    parser.add_argument("--model_num", type=int, default=0, help="Model number for multi-agent environments")
    parser.add_argument("--n_episode", type=int, default=1, help="Number of episodes for evaluation.")
    parser.add_argument("--manual", action="store_true", help="Enable keyboard manual control for agent 1.")
    parser.add_argument("--sensor_missing_prob", type=float, default=0.0, help="Missing probability for SensorPerturbationWrapper.")
    parser.add_argument("--sensor_rel_error_std", type=float, default=0.0, help="Relative error std for SensorPerturbationWrapper.")

    args = parser.parse_args()
    if args.render_mode == "none" or args.render_mode == "None":
        args.render_mode = None
    if args.n_episode <= 1:
        show(
            args.exp_path,
            args.render_mode,
            model_num=args.model_num,
            pool_path=args.pool_path,
            manual_control=args.manual,
            sensor_missing_prob=args.sensor_missing_prob,
            sensor_rel_error_std=args.sensor_rel_error_std,
        )
    else:
        if args.pool_path is None:
            raise ValueError("pool_path is required when n_episode > 1.")
        obs_wrappers = None
        if args.sensor_missing_prob > 0.0 or args.sensor_rel_error_std > 0.0:
            obs_wrappers = [
                {
                    "name": "src.environments.sensor_wrapper:SensorPerturbationWrapper",
                    "kwargs": {
                        "missing_prob": args.sensor_missing_prob,
                        "rel_error_std": args.sensor_rel_error_std,
                        "prefixes": ["target/", "oppo/"]
                    },
                }
            ]

        result = Evaluator.evaluate_pool(
            model1_path=args.exp_path,
            target_path=args.pool_path,
            n_episodes=args.n_episode,
            render_mode=args.render_mode,
            obs_wrappers=obs_wrappers,
        )
        print(f"Evaluation results saved to {result['result_path']}")
