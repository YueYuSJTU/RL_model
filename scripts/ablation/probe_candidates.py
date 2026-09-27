"""临时探针：测量若干候选模型的行为（用于把新 benchmark 与旧槽位的行为特征对齐）。

用法：PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/probe_candidates.py name=path [name=path ...]
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'opponent_behavior'))

from common import load_policy, make_eval_env                       # noqa: E402
from run_behaviour import run_free, behaviour_metrics               # noqa: E402


def shim_if_needed(path):
    """老格式目录缺 agent_config.yaml 的 algorithm 键时，生成 shim（软链 + 补键），不改原目录。"""
    ag_path = os.path.join(path, 'agent_config.yaml')
    ag = yaml.safe_load(open(ag_path, encoding='utf-8')) or {}
    if 'algorithm' in ag:
        return path
    dst = os.path.join('experiments', 'ablation_eval', 'shims',
                       os.path.basename(os.path.dirname(path.rstrip('/'))) + '_' + os.path.basename(path.rstrip('/')))
    os.makedirs(dst, exist_ok=True)
    for f in ('best_model.zip', 'env_config.yaml'):
        tgt = os.path.join(dst, f)
        if os.path.islink(tgt) or os.path.exists(tgt):
            os.remove(tgt)
        os.symlink(os.path.abspath(os.path.join(path, f)), tgt)
    ag.setdefault('algorithm', 'PPO')
    ag.setdefault('device', 'cuda')
    yaml.safe_dump(ag, open(os.path.join(dst, 'agent_config.yaml'), 'w', encoding='utf-8'), allow_unicode=True)
    return dst


def main():
    for spec in sys.argv[1:]:
        name, path = spec.split('=', 1)
        if not os.path.exists(os.path.join(path, 'best_model.zip')):
            print('[cand] %-16s 缺模型' % name)
            continue
        path = shim_if_needed(path)
        vec = make_eval_env(seed=0)
        sub = load_policy(path)
        eps = run_free(vec, sub, repeats=1)
        vec.close()
        per = [p for p in (behaviour_metrics(e['series']) for e in eps) if p]
        agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
        print('[cand] %-16s |bank|=%5.1f climb=%+7.1f ele=%+6.3f speed=%5.1f closure=%+6.0f'
              % (name, agg['bank_mean_abs'], agg['climb_rate_mean'], agg['ctrl_elevator_mean'],
                 agg['speed_mean'], agg['los_closure_mean']), flush=True)


if __name__ == '__main__':
    main()
