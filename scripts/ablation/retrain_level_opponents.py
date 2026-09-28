"""按 pool2 平飞型对手（pool2/11=stage3, pool2/16=stage4, pool2/17=stage5）的原始配方重训对手。

与 src/multiAgent/creat_init_pool.py 完全同构：
  - env: configs/env/f16_tracking_init.yaml + shape=stage{3,4,5}（TrackingInitTask 的三档奖励组合）
  - agent: pool2/11 保存的 agent_config.yaml（ppo_1layer1 的超参；原 configs/agent/ppo_1layer1.yaml 已不在）
  - 训练 600k 步，EvalCallback(eval_freq=1000, deterministic) 存 best_model
  - 每个 stage 都从零创建模型（与原脚本一致），结果存 experiments/level_opponents_retrain/<shape>/
"""
import argparse
import os
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback  # noqa: E402

from src.agents.make_agent import creat_agent  # noqa: E402
from src.environments.make_env import create_env  # noqa: E402
from src.utils.serialization import save_config  # noqa: E402

DEFAULT_AGENT_CFG = 'opponent_pool/pool2/11/agent_config.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', required=True, choices=['stage1', 'stage2', 'stage3', 'stage4', 'stage5'])
    ap.add_argument('--out', default=None, help='输出目录（默认 experiments/level_opponents_retrain/<shape>）')
    ap.add_argument('--steps', type=int, default=None, help='覆盖总步数（默认取 create_init_pool_config）')
    ap.add_argument('--num-cpu', type=int, default=None)
    ap.add_argument('--agent-cfg', default=DEFAULT_AGENT_CFG)
    args = ap.parse_args()

    with open('configs/create_init_pool_config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    steps = args.steps or cfg['total_timesteps']
    num_cpu = args.num_cpu or cfg['num_cpu']

    out = args.out or os.path.join('experiments', 'level_opponents_retrain', args.shape)
    os.makedirs(out, exist_ok=True)

    env_cfg = yaml.safe_load(open(os.path.join('configs', 'env', cfg['env'] + '.yaml'), encoding='utf-8'))
    env_cfg['shape'] = args.shape
    agent_cfg = yaml.safe_load(open(args.agent_cfg, encoding='utf-8'))

    save_config(cfg, out, 'train_config.yaml')
    save_config(agent_cfg, out, 'agent_config.yaml')
    save_config(env_cfg, out, 'env_config.yaml')

    train_env = create_env(env_cfg, training=True, num_cpu=num_cpu,
                           vec_env_kwargs={'model_num': -1})
    eval_env = create_env(env_cfg, training=False, vec_env_kwargs={'model_num': -1})
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(eval_env, best_model_save_path=out, log_path=out,
                                 eval_freq=cfg['eval_freq'], deterministic=True, render=False)
    progress_callback = ProgressBarCallback()

    model = creat_agent(env=train_env, agent_class=cfg['agent'],
                        tensorboard_log=None, agent_cfg=agent_cfg)
    model.learn(total_timesteps=steps, callback=[eval_callback, progress_callback])

    train_env.save(os.path.join(out, 'final_train_env.pkl'))
    print('[done] %s -> %s (best_model.zip %s)' % (
        args.shape, out, os.path.exists(os.path.join(out, 'best_model.zip'))), flush=True)


if __name__ == '__main__':
    main()
