"""检查对手模型能否用当前代码加载（用于筛掉 obs 空间不兼容的老模型）。

对每个候选目录执行与 Evaluator 相同的加载流程：
  agent_config.yaml + env_config.yaml -> create_env(NNVecEnv) -> load_agent -> ObsAdaptingModel
并打印保存的 obs 形状 vs 当前 env 的 obs 形状。
"""
import base64
import json
import os
import pickle
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.environments.make_env import create_env            # noqa: E402
from src.agents.make_agent import load_agent                # noqa: E402
from src.agents.model_wrapper import ObsAdaptingModel       # noqa: E402
from src.utils.serialization import load_config             # noqa: E402


def saved_spaces(model_dir):
    z = zipfile.ZipFile(os.path.join(model_dir, 'best_model.zip'))
    d = json.loads(z.read('data'))
    out = {}
    for k in ('observation_space', 'action_space'):
        v = d.get(k)
        if isinstance(v, dict) and ':serialized:' in v:
            out[k] = pickle.loads(base64.b64decode(v[':serialized:']))
    return out


def check(model_dir):
    rec = {'dir': model_dir}
    try:
        sp = saved_spaces(model_dir)
        rec['saved_obs'] = tuple(sp['observation_space'].shape) if 'observation_space' in sp else None
        rec['saved_act'] = tuple(sp['action_space'].shape) if 'action_space' in sp else None
    except Exception as exc:  # noqa: BLE001
        rec['saved_obs'] = rec['saved_act'] = None
        rec['saved_err'] = repr(exc)[:200]
    try:
        cfg = load_config(os.path.join(model_dir, 'env_config.yaml'))
        env = create_env(cfg, training=False, vec_env_kwargs=None)
        rec['env_obs'] = tuple(env.observation_space.shape)
        rec['env_act'] = tuple(env.action_space.shape)
        ag = load_config(os.path.join(model_dir, 'agent_config.yaml'))
        model = load_agent(env=env, agent_class=ag.get('algorithm', 'PPO'),
                           path=os.path.join(model_dir, 'best_model'),
                           device=ag['device'], agent_cfg=ag)
        model = ObsAdaptingModel(model, cfg)
        rec['ok'] = True
        del model
        env.close()
    except Exception as exc:  # noqa: BLE001
        rec['ok'] = False
        rec['err'] = repr(exc)[:300]
    return rec


def main():
    cand = json.load(open(os.path.join(ROOT, 'experiments', 'ablation_heldout_candidates.json')))
    dirs = sorted(cand.keys())
    # 附上三臂自身与四个"已见过"的模型作为对照
    extra = ['experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30',
             'experiments/20260924_225859_selfplay_init/stage2/20260926_124349_cycle_30',
             'experiments/20260924_225859_weakinit_nogoalpoint/stage2/20260926_110948_cycle_30',
             'base_opponent_model/1', 'base_opponent_model/2', 'base_opponent_model/3',
             'base_opponent_model_weakinit/1']
    out = []
    for d in dirs + extra:
        if not os.path.isdir(d):
            continue
        r = check(d)
        out.append(r)
        flag = 'OK  ' if r.get('ok') else 'FAIL'
        print('%s %-58s saved=%s/%s env=%s/%s %s' % (
            flag, d, r.get('saved_obs'), r.get('saved_act'), r.get('env_obs'), r.get('env_act'),
            '' if r.get('ok') else r.get('err', r.get('saved_err', ''))[:120]), flush=True)
    json.dump(out, open(os.path.join(ROOT, 'experiments', 'ablation_opponent_loadcheck.json'), 'w'),
              indent=1, ensure_ascii=False)
    ok = [r['dir'] for r in out if r.get('ok')]
    print('\n可加载: %d / %d' % (len(ok), len(out)))


if __name__ == '__main__':
    main()
