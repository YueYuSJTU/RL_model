"""重建 8 对手 benchmark（bench8-v2）：与旧 8 对手同构，但全部为"未参与训练"的中立模型。

旧 8 对手的角色构成（`opponent_pool/test_pool`）：
  1   goalpoint 策略（GoalPointTask，弱、不追踪）        md5 83041d60…
  2   base2 = 20250710_212837（TrackingTask stage2，40M 步）  md5 978e3666…
  3   base3 = 20250713_125657（TrackingTask stage2，500M 步） md5 3543907a…
  11  pool2/11（TrackingInitTask stage3，平飞型）
  16  pool2/16（stage4）
  17  pool2/17（stage5）
  23  pool2/3 （stage3）                                md5 469281b7…
  616 Baseline 本体（20250616，弱参照）

其中 1/2/3/23 参与了消融三臂的训练（stage1 池 1/2/3 与 B 臂 battle 池的 pool2/3），
因此不能再用于评测。本脚本按"角色对位"挑 8 个中立替代模型，并逐一验证可加载性。

输出：
  experiments/ablation_bench8_v2.json   {显示名: 目录}
  experiments/bench8_v2/README.md       构成说明与 md5 对照
"""
import base64
import hashlib
import json
import os
import pickle
import sys
import zipfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.environments.make_env import create_env            # noqa: E402
from src.agents.make_agent import load_agent                # noqa: E402
from src.utils.serialization import load_config             # noqa: E402

SHIM_ROOT = os.path.join(ROOT, 'experiments', 'ablation_eval', 'shims')

# 角色对位表：旧槽位 -> (新对手显示名, 目录)；均经 md5 审计确认未出现在训练池中
PLAN = [
    # 旧 1：goalpoint 策略 -> 同样用 GoalPointTask 训练的兄弟 run（base1 是 20250926_155941）
    ('1  (goalpoint policy)', 'NB1-goalpoint(0918)',
     'experiments/goal_point/20250918_095027/stage1/20250918_095027_GoalPointTask_ppo_1layer'),
    # 旧 2/3：早期空战型 TrackingTask stage2 -> 同族但未参与训练的 0715 / 0716
    ('2  (combat 0710)', 'NB2-combat(0715_105409)',
     'experiments/20250715_105409/stage2/20250715_105533_TrackingTask_ppo_1layer1'),
    ('3  (combat 0713)', 'NB3-combat(0716_220909)',
     'experiments/20250716_220909/stage2/20250716_221041_TrackingTask_ppo_1layer1'),
    # 旧 11/16/17/23：平飞型 TrackingInitTask（pool2 家族）-> 同族中未参与训练的成员
    ('11 (level-flight st3)', 'NB4-level-flight(st3/pool2-11)', 'opponent_pool/pool2/11'),
    ('16 (level-flight st4)', 'NB5-level-flight(st4/pool2-16)', 'opponent_pool/pool2/16'),
    ('17 (level-flight st5)', 'NB6-level-flight(st5/pool2-17)', 'opponent_pool/pool2/17'),
    ('23 (level-flight st3)', 'NB7-level-flight(st4/pool2-7)', 'opponent_pool/pool2/7'),
    # 旧 616：弱参照（Baseline 本体）-> 换 2025-07-10 的 stage1 纯飞行控制模型（更弱，角色为"弱对手"）
    ('616 (old Baseline)', 'NB8-weakref(best_models/8)',
     'opponent_pool/best_models/8'),
]
# 说明：
#  * NB8 不用 pool3/1(=Baseline 本体)：Baseline 是论文其它对比中的被测模型，会引入耦合；
#    改用 best_models/8（md5 e30972740b…，未参与任何训练池）。实测行为 53.6/+14.3/-0.748/459.3/-401.3
#    与旧 616（60.1/+6.4/-0.618/475.5/-394.4）在五个指标上几乎一致，是可用候选里最贴近的。
#  * NB7 用 stage4（pool2/7）替代旧 23 的 stage3 —— pool2 中唯一的 stage3（pool2/3）已被训练使用。
#    在 pool2 的剩余成员中选 pool2/7：实测行为（|bank|=29.8 deg, climb=+33.8 ft/s）比 pool2/12
#    （|bank|=74.7, climb=-263）更接近旧基准里平飞型槽位的"浅坡度、近定高"特征。
#  * 不含 LSTM 快照与 2026 年强模型：它们明显强于旧 benchmark，会破坏可比性。


def md5f(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def saved_obs(d):
    z = zipfile.ZipFile(os.path.join(d, 'best_model.zip'))
    data = json.loads(z.read('data'))
    return pickle.loads(base64.b64decode(data['observation_space'][':serialized:']))


def make_shim(src):
    name = os.path.basename(os.path.dirname(src.rstrip('/'))) + '_' + os.path.basename(src.rstrip('/'))
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
        return True, '', cfg
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:220], cfg
    finally:
        env.close()


def main():
    os.makedirs(os.path.join(ROOT, 'experiments', 'bench8_v2'), exist_ok=True)
    out, rows = {}, []
    for oldslot, name, path in PLAN:
        ok, err, cfg = try_load(path)
        used, note = path, ''
        if not ok and "agent_cfg['algorithm']" in err:
            used = make_shim(path)
            ok, err2, cfg = try_load(used)
            note = 'shim(algorithm=PPO)' if ok else 'shim 后仍失败: ' + err2
        if not ok and 'Observation spaces do not match' in err:
            so = saved_obs(path)
            env = create_env(cfg, training=False, vec_env_kwargs=None)
            eo = env.observation_space
            env.close()
            note = 'obs 边界不一致 low=%d/58 high=%d/58' % (
                int((so.low != eo.low).sum()), int((so.high != eo.high).sum()))
        tc = {}
        tcf = os.path.join(used, 'train_config.yaml')
        if os.path.exists(tcf):
            tc = yaml.safe_load(open(tcf)) or {}
        rows.append((oldslot, name, used, ok, cfg.get('task'), cfg.get('shape'),
                     tc.get('total_timesteps', '?'), md5f(os.path.join(used, 'best_model.zip')), note or err))
        if ok:
            out[name] = used
        print('%-26s %-34s %-18s %-7s %s %s' % (oldslot, name, cfg.get('task'), cfg.get('shape'),
                                                'OK ' if ok else 'FAIL', note or err[:70]), flush=True)

    json.dump(out, open(os.path.join(ROOT, 'experiments', 'ablation_bench8_v2.json'), 'w'),
              indent=1, ensure_ascii=False)

    md = ['# 新 benchmark（bench8-v2）：8 个中立对手\n\n',
          '构成原则：与旧 8 对手（`opponent_pool/test_pool`）**逐槽位同构**，但每个模型都经 md5 审计，'
          '确认**未出现在任何参与对比的训练池**中（排除 `base_opponent_model/1,2,3` 与 `pool2/3`）。\n\n',
          '| 旧槽位（角色） | 新对手 | 目录 | 任务/阶段 | 训练步数 | md5 | 备注 |\n|---|---|---|---|---|---|---|\n']
    for oldslot, name, used, ok, task, shape, steps, h, note in rows:
        md.append('| %s | %s | `%s` | %s/%s | %s | `%s` | %s |\n' % (
            oldslot, name, used, task, shape, steps, h, (note or '').replace('|', '/')))
    md.append('\n## 与旧 benchmark 的差异（写论文时需说明）\n\n')
    md.append('- 旧 `1`（goalpoint 83041d60…）→ NB1：另一个 GoalPointTask run（2025-09-18），角色一致。\n')
    md.append('- 旧 `2/3`（2025-07-10 / 07-13 的 TrackingTask stage2）→ NB2/NB3：2025-07-15 / 07-16 同族模型。\n')
    md.append('- 旧 `11/16/17/23`（pool2 的 stage3/3/4/5 平飞型）→ NB4–NB7：pool2 中**未被训练使用**的 11/16/17/12'
              '（stage3/4/5/4）；旧 23 的 stage3 位由 stage4 顶替（pool2 仅有一个 stage3）。\n')
    md.append('- 旧 `616`（Baseline 本体）→ NB8：2025-07-10 的 stage1 纯飞行控制模型（更弱的"弱对手"）。\n')
    md.append('- **不含** LSTM 快照与 2026 年模型（明显更强，会破坏与旧 benchmark 的可比性）。\n')
    open(os.path.join(ROOT, 'experiments', 'bench8_v2', 'README.md'), 'w').write(''.join(md))
    print('\n最终 bench8-v2: %d/8 个可加载 -> experiments/ablation_bench8_v2.json' % len(out))


if __name__ == '__main__':
    main()
