"""用 scripts/opponent_behavior 的同一套协议，测量 bench8-v2 的 8 个新对手的飞行行为。

目的：证明新基准在"行为覆盖"上与旧 benchmark 相近（论文 tab:bench_behaviour 的口径）。
复用既有工具，不修改 jsb_env/**、src/** 的任何文件：
  - common.py     : load_policy / make_eval_env / write_json
  - run_behaviour : run_free / behaviour_metrics

输出：experiments/opponent_behavior/raw/bh_bench8_<name>.json
     （另外复测旧对手 23 作为协议一致性校验，输出 bh_recheck_23.json）

用法：PYTHONDONTWRITEBYTECODE=1 python scripts/ablation/run_bench8_behaviour.py [--only NAME]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'opponent_behavior'))

from common import load_policy, make_eval_env, write_json          # noqa: E402
from run_behaviour import run_free, behaviour_metrics              # noqa: E402

RAW = Path('experiments/opponent_behavior/raw')
BENCH8 = json.load(open('experiments/ablation_bench8_v2.json'))

# 旧对手（用于协议一致性校验）：与 experiments/opponent_behavior/raw/bh_23.json 比较
RECHECK = {'23': 'opponent_pool/test_pool/23'}


def measure(name, path, out, repeats=1, seed=0):
    t0 = time.time()
    vec = make_eval_env(seed=seed)
    subject = load_policy(path)
    eps = run_free(vec, subject, repeats=repeats)
    vec.close()
    per_ep = [p for p in (behaviour_metrics(e['series']) for e in eps) if p]
    agg = {k: float(np.mean([p[k] for p in per_ep])) for k in per_ep[0]}
    write_json(out, dict(subject=name, subject_path=path, repeats=repeats, seed=seed,
                         wall_s=time.time() - t0, metrics=agg, per_episode=per_ep))
    print('[bench8-behaviour] %-32s |bank|=%5.1f climb=%+7.1f ele=%+6.3f speed=%5.1f closure=%+6.0f  (%.0fs)'
          % (name, agg['bank_mean_abs'], agg['climb_rate_mean'], agg['ctrl_elevator_mean'],
             agg['speed_mean'], agg['los_closure_mean'], time.time() - t0), flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)

    items = [(k, v) for k, v in BENCH8.items()]
    items += [('recheck-%s' % k, v) for k, v in RECHECK.items()]
    if args.only:
        items = [it for it in items if args.only in it[0]]
    for name, path in items:
        out = RAW / ('bh_bench8_%s.json' % name.replace('/', '_') if not name.startswith('recheck')
                     else 'bh_%s.json' % name)
        if out.exists():
            print('skip (exists):', out)
            continue
        measure(name, path, out)


if __name__ == '__main__':
    main()
