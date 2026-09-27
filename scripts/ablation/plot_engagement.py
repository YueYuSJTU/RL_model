"""L3 定性展示：同一初始态势下，不同臂与同一对手的交战轨迹（俯视 N-E 平面）。

完全复刻 Evaluator.run_match 的交互协议：
  obs = env.reset()；action1 = 臂在 obs[..., :D/2] 上预测；action2 = 对手在 obs[..., D/2:] 上预测；
  合并成 8 维动作 step。额外采集 info['trajectory']（self/oppo 的 NED 位置）与
  metrics_step 里的 gun_opportunity 标志。

用法：
  python scripts/ablation/plot_engagement.py \
      --opponent experiments/20250715_105409/stage2/20250715_105533_TrackingTask_ppo_1layer1 \
      --model "Ctrl=experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30" \
      --model "A=experiments/20260924_225859_selfplay_init/stage2/20260926_124349_cycle_30" \
      --model "B=experiments/20260924_225859_weakinit_nogoalpoint/stage2/20260926_110948_cycle_30" \
      [--np-seed 12345] [--max-steps 1000] --out experiments/ablation_L3_trajectory.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.agents.make_agent import load_agent                # noqa: E402
from src.agents.model_wrapper import ObsAdaptingModel       # noqa: E402
from src.environments.make_env import create_env            # noqa: E402
from src.utils.serialization import load_config             # noqa: E402


def build(env_cfg, model_dir):
    env = create_env(env_cfg, training=False, vec_env_cls=DummyVecEnv)
    env.training = False
    env.norm_reward = False
    env.env_method('update_task_parameters', goal_point_prob=0.0)
    ag = load_config(os.path.join(model_dir, 'agent_config.yaml'))
    ag.setdefault('algorithm', 'PPO')   # 老格式配置缺少该键（load_agent 故意严格）
    m = load_agent(env=create_env(load_config(os.path.join(model_dir, 'env_config.yaml')),
                                  training=False, vec_env_kwargs=None),
                   agent_class=ag.get('algorithm', 'PPO'), path=os.path.join(model_dir, 'best_model'),
                   device=ag.get('device', 'cuda'), agent_cfg=ag)
    return env, ObsAdaptingModel(m, load_config(os.path.join(model_dir, 'env_config.yaml')))


def rollout(model_dir, opponent_dir, np_seed=None, max_steps=1000):
    env_cfg = load_config(os.path.join(model_dir, 'env_config.yaml'))
    env_cfg['render_mode'] = None
    env_cfg['use_vec_normalize'] = False
    vec_env, m1 = build(env_cfg, model_dir)

    ag2 = load_config(os.path.join(opponent_dir, 'agent_config.yaml'))
    ag2.setdefault('algorithm', 'PPO')
    m2 = load_agent(env=create_env(load_config(os.path.join(opponent_dir, 'env_config.yaml')),
                                   training=False, vec_env_kwargs=None),
                    agent_class=ag2.get('algorithm', 'PPO'),
                    path=os.path.join(opponent_dir, 'best_model'),
                    device=ag2.get('device', 'cuda'), agent_cfg=ag2)
    m2 = ObsAdaptingModel(m2, load_config(os.path.join(opponent_dir, 'env_config.yaml')))

    if np_seed is not None:
        np.random.seed(np_seed)
    obs = vec_env.reset()
    D = obs.shape[-1]
    self_xy, oppo_xy, gun, hp = [], [], [], []
    st1 = st2 = None
    done = False
    steps = 0
    win = None
    while not done and steps < max_steps:
        a1, st1 = m1.predict(obs[:, :D // 2], state=st1, episode_start=np.zeros(1, dtype=bool), deterministic=True)
        a2, st2 = m2.predict(obs[:, D // 2:], state=st2, episode_start=np.zeros(1, dtype=bool), deterministic=True)
        obs, _r, dones, info = vec_env.step(np.concatenate([a1, a2], axis=-1))
        i0 = info[0] if isinstance(info, (list, tuple)) else info
        tr = i0.get('trajectory')
        if tr:
            self_xy.append((tr['self']['x_ft'], tr['self']['y_ft']))
            oppo_xy.append((tr['oppo']['x_ft'], tr['oppo']['y_ft']))
        ms = i0.get('metrics_step') or {}
        gun.append(float(ms.get('gun_opportunity', 0.0)))
        hp.append((float(ms.get('hp_self', 0.0)), float(ms.get('hp_oppo', 0.0))))
        if bool(dones[0]):
            done = True
            win = (i0.get('env_info') or {}).get('win', 0)
        steps += 1
    vec_env.close()
    return dict(self_xy=np.array(self_xy), oppo_xy=np.array(oppo_xy), gun=np.array(gun),
                hp=np.array(hp), win=win, steps=steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='append', required=True, help='label=model_dir（可多次）')
    ap.add_argument('--opponent', required=True, help='对手模型目录')
    ap.add_argument('--np-seed', type=int, default=12345)
    ap.add_argument('--episodes', type=int, default=1, help='每臂跑多少局（同一组初始态势种子），用于选"典型局"')
    ap.add_argument('--select', default='median', choices=['median', 'best', 'worst', 'first'],
                    help='按枪口机会比选典型局：median=中位数局（默认，避免抽到好局）')
    ap.add_argument('--max-steps', type=int, default=1000)
    ap.add_argument('--out', default='experiments/ablation_L3_trajectory.png')
    args = ap.parse_args()

    models = [m.split('=', 1) for m in args.model]
    res, stats = {}, {}
    for label, path in models:
        runs = [rollout(path, args.opponent, np_seed=args.np_seed + k, max_steps=args.max_steps)
                for k in range(args.episodes)]
        gun = np.array([float(r['gun'].mean()) if len(r['gun']) else 0.0 for r in runs])
        wins = sum(1 for r in runs if r['win'] == 1)
        if args.select == 'median':
            idx = int(np.argsort(gun)[len(gun) // 2])
        elif args.select == 'best':
            idx = int(np.argmax(gun))
        elif args.select == 'worst':
            idx = int(np.argmin(gun))
        else:
            idx = 0
        res[label] = runs[idx]
        stats[label] = dict(n=len(runs), win=100.0 * wins / len(runs), median_gun=float(np.median(gun)),
                            idx=idx, idx_gun=float(gun[idx]))
        print('%-8s K=%d 局: 胜率 %.0f%% | 枪口机会比 中位数 %.3f (典型局=第 %d 局, %.3f) | 典型局结果=%s steps=%d HP %.2f/%.2f'
              % (label, len(runs), stats[label]['win'], stats[label]['median_gun'], idx,
                 stats[label]['idx_gun'],
                 {1: 'WIN', 0: 'DRAW', -1: 'LOSS', 0.5: 'OPP-FALL'}.get(runs[idx]['win'], '?'),
                 runs[idx]['steps'], runs[idx]['hp'][-1][0], runs[idx]['hp'][-1][1]), flush=True)

    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), squeeze=False)
    for ax, (label, r) in zip(axes[0], res.items()):
        s, o = r['self_xy'], r['oppo_xy']
        ax.plot(s[:, 0], s[:, 1], color='#307bb5', lw=1.6, label='ego')
        ax.plot(o[:, 0], o[:, 1], color='#c43f3c', lw=1.6, label='opponent')
        ax.scatter(*s[0], color='#307bb5', marker='o', s=45, zorder=5)
        ax.scatter(*o[0], color='#c43f3c', marker='o', s=45, zorder=5)
        # 枪口机会（对手在本机武器包线内）的时段用灰带标出
        gun_idx = np.where(r['gun'] > 0)[0]
        for i in gun_idx[::max(1, len(gun_idx) // 40)]:
            if i < len(s):
                ax.scatter(s[i, 0], s[i, 1], color='#f0a000', s=6, alpha=.5, zorder=4)
        verdict = {1: 'WIN', 0: 'DRAW', -1: 'LOSS', 0.5: 'OPP-FALL'}.get(r['win'], '?')
        st = stats[label]
        ax.set_title('%s: typical episode (%s, %d steps, gun-opp %.0f%%)\n'
                     'K=%d eps: win %.0f%%, median gun-opp %.0f%%'
                     % (label, verdict, r['steps'], 100 * r['gun'].mean(), st['n'], st['win'], 100 * st['median_gun']),
                     fontsize=9)
        ax.set_xlabel('North (ft)'); ax.set_ylabel('East (ft)')
        ax.grid(alpha=.3); ax.axis('equal')
    axes[0][0].legend(fontsize=8, loc='best')
    plt.suptitle('Same initial condition, same opponent: engagement geometry per arm'
                 '  (yellow dots = opponent inside ego weapon envelope)', fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
