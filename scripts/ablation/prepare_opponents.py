"""准备中立 held-out 对手集：可加载性检查 + 老格式 shim + 生成最终 HO 集合。

背景（已在 2026-09-26 实测）：
  * 部分 2025 年老模型目录的 agent_config.yaml 缺少 `algorithm` 键，而
    src/agents/make_agent.py::_resolve_algorithm 是**故意严格**的（缺键即报错）。
    这里不修改原目录，而是生成 shim 目录（符号链接 best_model.zip / env_config.yaml，
    再写一份补上 algorithm: PPO 的 agent_config.yaml）。
  * opponent_pool/pool1/* 的 obs 空间 low/high 与当前 spec 不一致（SB3 空间校验拒绝加载），
    本集合将其排除，并在报告里注明（其"平飞型"战术类别由 pool2/7/11/12/16/17 代表）。

输出：
  experiments/ablation_ho_set.json   {显示名: 目录路径}
  experiments/ablation_eval/shims/<name>/...
"""
import base64
import json
import os
import pickle
import shutil
import sys
import zipfile

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.environments.make_env import create_env            # noqa: E402
from src.agents.make_agent import load_agent                # noqa: E402
from src.utils.serialization import load_config             # noqa: E402

SHIM_ROOT = os.path.join(ROOT, 'experiments', 'ablation_eval', 'shims')

# 最终 HO 集合（按 md5 审计确认：均未出现在三臂训练池中）
HO = [
    ('HO01-level-flight(pool2/11)',      'opponent_pool/pool2/11'),
    ('HO02-level-flight(pool2/12)',      'opponent_pool/pool2/12'),
    ('HO03-level-flight(pool2/16)',      'opponent_pool/pool2/16'),
    ('HO04-level-flight(pool2/17)',      'opponent_pool/pool2/17'),
    ('HO05-level-flight(pool2/7)',       'opponent_pool/pool2/7'),
    ('HO06-combat(best_models/2)',       'opponent_pool/best_models/2'),
    ('HO07-combat(best_models/3)',       'opponent_pool/best_models/3'),
    ('HO08-combat(pool3/4=0715_105409)', 'opponent_pool/pool3/4'),
    ('HO09-weakref(pool3/1=Baseline)',   'opponent_pool/pool3/1'),
    ('HO10-strong(LSTM-cycle21)',        'opponent_pool/LSTM_20260320_135309_cycle_21'),
    ('HO11-strong(LSTM-c35)',            'opponent_pool/LSTM_eval_version_from_cycle35'),
    ('HO12-strong(1month-1277)',         'experiments/20260105_231131/stage2/20260228_032950_cycle_1277'),
    ('HO13-strong(14h-0326)',            'experiments/20260301_203610/stage2/20260302_143419_cycle_26'),
]


def saved_obs(model_dir):
    z = zipfile.ZipFile(os.path.join(model_dir, 'best_model.zip'))
    data = json.loads(z.read('data'))
    return pickle.loads(base64.b64decode(data['observation_space'][':serialized:']))


def make_shim(src):
    """为缺少 algorithm 键的老目录建 shim（不修改原目录）。"""
    name = os.path.basename(src.rstrip('/'))
    dst = os.path.join(SHIM_ROOT, name)
    os.makedirs(dst, exist_ok=True)
    for f in ('best_model.zip', 'env_config.yaml'):
        tgt = os.path.join(dst, f)
        if os.path.islink(tgt) or os.path.exists(tgt):
            os.remove(tgt)
        os.symlink(os.path.abspath(os.path.join(src, f)), tgt)
    ag = yaml.safe_load(open(os.path.join(src, 'agent_config.yaml'), encoding='utf-8')) or {}
    ag.setdefault('algorithm', 'PPO')
    ag.setdefault('device', 'cuda')
    yaml.safe_dump(ag, open(os.path.join(dst, 'agent_config.yaml'), 'w', encoding='utf-8'), allow_unicode=True)
    return dst


def try_load(path):
    cfg = load_config(os.path.join(path, 'env_config.yaml'))
    env = create_env(cfg, training=False, vec_env_kwargs=None)
    try:
        ag = load_config(os.path.join(path, 'agent_config.yaml'))
        load_agent(env=env, agent_class=ag.get('algorithm', 'PPO'),
                   path=os.path.join(path, 'best_model'), device=ag.get('device', 'cuda'), agent_cfg=ag)
        return True, ''
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:200]
    finally:
        env.close()


def main():
    os.makedirs(SHIM_ROOT, exist_ok=True)
    final = {}
    report = []
    for name, path in HO:
        ok, err = try_load(path)
        used = path
        note = ''
        if not ok and "agent_cfg['algorithm']" in err:
            used = make_shim(path)
            ok2, err2 = try_load(used)
            note = 'shim(algorithm=PPO)' if ok2 else 'shim 后仍失败: ' + err2
            ok = ok2
        if not ok and 'Observation spaces do not match' in err:
            so, cfg = saved_obs(path), load_config(os.path.join(path, 'env_config.yaml'))
            env = create_env(cfg, training=False, vec_env_kwargs=None)
            eo = env.observation_space
            env.close()
            note = ('obs bounds 不一致: low 差异维度 %d/58, high 差异维度 %d/58'
                    % (int((~np.isclose(so.low, eo.low)).sum()), int((~np.isclose(so.high, eo.high)).sum())))
        report.append((name, path, ok, note or err))
        if ok:
            final[name] = used
        print('%-36s %-52s %s %s' % (name, path, 'OK ' if ok else 'FAIL', note or err), flush=True)

    out = os.path.join(ROOT, 'experiments', 'ablation_ho_set.json')
    json.dump(final, open(out, 'w'), indent=1, ensure_ascii=False)
    print('\n最终 HO 集合: %d 个 -> %s' % (len(final), out))
    json.dump(report, open(os.path.join(ROOT, 'experiments', 'ablation_ho_report.json'), 'w'),
              indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
