"""构建初始化消融实验的中立 held-out 对手集。

原则：排除任何在三臂（BDC-MLP 对照 / A 纯自博弈 / B 弱能力）训练池中出现过的模型，
包括三臂共用的 stage1 对手池 base_opponent_model 1/2/3 与 B 臂 battle 池的 pool2/3。
输出：experiments/ablation_heldout_candidates.json
"""
import os
import hashlib
import json

# 本文件位于 <repo>/scripts/ablation/，上三级即仓库根目录
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def md5f(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    used_paths = [
        'base_opponent_model/1', 'base_opponent_model/2', 'base_opponent_model/3',
        'base_opponent_model_weakinit/1', 'base_opponent_model_weakinit/2', 'base_opponent_model_weakinit/3',
    ]
    used = {}
    for p in used_paths:
        fp = os.path.join(ROOT, p, 'best_model.zip')
        if os.path.exists(fp):
            used[p] = md5f(fp)
    used_md5 = set(used.values())
    print('[1] 三臂训练中见过的外部对手:')
    for k, v in used.items():
        print('    %s  %s' % (v, k))

    cands = []
    roots = ['opponent_pool/pool1', 'opponent_pool/pool2', 'opponent_pool/pool3',
             'opponent_pool/pool5', 'opponent_pool/pool5_2', 'opponent_pool/best_models',
             'opponent_pool/test_pool', 'opponent_pool/LSTM_20260320_135309_cycle_21',
             'opponent_pool/LSTM_eval_version_from_cycle35']
    for root in roots:
        rp = os.path.join(ROOT, root)
        if not os.path.isdir(rp):
            continue
        if os.path.exists(os.path.join(rp, 'best_model.zip')):
            subs = [rp]
        else:
            subs = [os.path.join(rp, d) for d in sorted(os.listdir(rp))
                    if os.path.isdir(os.path.join(rp, d))]
        for d in subs:
            bm = os.path.join(d, 'best_model.zip')
            if os.path.exists(bm) and os.path.exists(os.path.join(d, 'env_config.yaml')):
                cands.append((os.path.relpath(d, ROOT), md5f(bm)))

    extra = [
        'experiments/20260105_231131/stage2/20260228_032950_cycle_1277',
        'experiments/20260301_203610/stage2/20260302_143419_cycle_26',
        'experiments/20260316_093625_lstm_train/stage2/20260323_133647_cycle_35',
        'experiments/20250715_105409/stage2/20250715_105533_TrackingTask_ppo_1layer1',
        'experiments/20250716_220909/stage1/20250716_220909_TrackingTask_ppo_1layer1',
    ]
    for d in extra:
        dp = os.path.join(ROOT, d)
        if os.path.exists(os.path.join(dp, 'best_model.zip')) and os.path.exists(os.path.join(dp, 'env_config.yaml')):
            cands.append((d, md5f(os.path.join(dp, 'best_model.zip'))))

    heldout, seen = [], set()
    for d, h in cands:
        if h in used_md5 or h in seen:
            continue
        seen.add(h)
        heldout.append((d, h))

    print('\n[2] 候选 held-out（去重、且未出现在三臂训练池中）: %d 个' % len(heldout))
    for d, h in heldout:
        print('    %s  %s' % (h, d))

    out = os.path.join(ROOT, 'experiments', 'ablation_heldout_candidates.json')
    json.dump({d: h for d, h in heldout}, open(out, 'w'), indent=1, ensure_ascii=False)
    print('\n[3] 已写出 %s' % out)


if __name__ == '__main__':
    main()
