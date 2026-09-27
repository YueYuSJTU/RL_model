"""初始化消融：统一评测脚本（三臂 × 对手集，可分片并行）。

用法：
  # 分片跑（建议 --shards 8，配合 0..7 各起一个进程）
  python scripts/ablation/eval_ablation.py --set ho10 --episodes 300 --shards 8 --shard 0
  # 合并 + 生成 markdown/CSV 摘要
  python scripts/ablation/eval_ablation.py --set ho10 --merge

输出：
  experiments/ablation_eval/<set>/shard<i>.jsonl   每行一个对阵的完整指标
  experiments/ablation_eval/<set>/merged.json      合并后的结果
  experiments/ablation_eval/<set>/summary.md       汇总表

协议与论文一致：Evaluator.run_match（对手以 jsbsim 模式追逐 ego，goal_point_prob=0），
每对阵 n 局，随机化初始态势；模型 1 = 被测臂，模型 2 = 对手。
"""
import argparse
import json
import os
import sys
import time
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.evaluation.evaluator import Evaluator  # noqa: E402

OUT_ROOT = os.path.join(ROOT, 'experiments', 'ablation_eval')

# ---- 被测臂（论文对照 + 两个新臂）----
ARMS = {
    'Ctrl-full-init': 'experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30',
    'A-selfplay-init': 'experiments/20260924_225859_selfplay_init/stage2/20260926_124349_cycle_30',
    'B-weak-init': 'experiments/20260924_225859_weakinit_nogoalpoint/stage2/20260926_110948_cycle_30',
}

# ---- 中立 held-out 对手（md5 审计确认未出现在三臂训练池中）----
HO10 = {
    'HO1-level-flight(pool1/1)': 'opponent_pool/pool1/1',
    'HO2-level-flight(pool1/3)': 'opponent_pool/pool1/3',
    'HO3-level-flight(pool2/7)': 'opponent_pool/pool2/7',
    'HO4-level-flight(pool2/12)': 'opponent_pool/pool2/12',
    'HO5-level-flight(pool2/16)': 'opponent_pool/pool2/16',
    'HO6-combat(pool3/4=0715)': 'opponent_pool/pool3/4',
    'HO7-combat(best_models/2)': 'opponent_pool/best_models/2',
    'HO8-combat(best_models/3)': 'opponent_pool/best_models/3',
    'HO9-strong(LSTM-c35)': 'opponent_pool/LSTM_eval_version_from_cycle35',
    'HO10-strong(14h-0326)': 'experiments/20260301_203610/stage2/20260302_143419_cycle_26',
}

HO22_FILE = os.path.join(ROOT, 'experiments', 'ablation_heldout_candidates.json')
HO_FILE = os.path.join(ROOT, 'experiments', 'ablation_ho_set.json')
BENCH8_FILE = os.path.join(ROOT, 'experiments', 'ablation_bench8_v2.json')

# 交叉对战：三臂互为对手
CROSS = dict(ARMS)

SETS = {'ho10': HO10, 'cross': CROSS}


def load_ho():
    if not os.path.exists(HO_FILE):
        raise SystemExit('缺少 %s，请先运行 scripts/ablation/prepare_opponents.py' % HO_FILE)
    return json.load(open(HO_FILE))


def load_bench8():
    """新的 8 对手 benchmark（bench8-v2）：与旧 benchmark 同构、但全部为未参与训练的中立模型。"""
    if not os.path.exists(BENCH8_FILE):
        raise SystemExit('缺少 %s，请先运行 scripts/ablation/build_bench8_v2.py' % BENCH8_FILE)
    return json.load(open(BENCH8_FILE))


def load_ho22():
    if not os.path.exists(HO22_FILE):
        raise SystemExit('缺少 %s，请先运行 scripts/ablation/build_heldout.py' % HO22_FILE)
    d = json.load(open(HO22_FILE))
    out = {}
    for i, (path, md5) in enumerate(sorted(d.items()), 1):
        out['HO%02d-%s' % (i, os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path))] = path
    return out


def build_pairs(set_name):
    if set_name == 'ho22':
        opps = load_ho22()
    elif set_name == 'ho':
        opps = load_ho()
    elif set_name == 'bench8':
        opps = load_bench8()
    else:
        opps = SETS[set_name]
    pairs = []
    for arm, arm_path in ARMS.items():
        for oname, opath in opps.items():
            if set_name == 'cross' and os.path.abspath(opath) == os.path.abspath(arm_path):
                continue
            pairs.append((arm, arm_path, oname, opath))
    return pairs


def run_shard(set_name, n_episodes, shards, shard, only_opponent=None, out_name=None):
    out_dir = os.path.join(OUT_ROOT, set_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, (out_name or ('shard%d' % shard)) + '.jsonl')
    done = set()
    if os.path.exists(out_file):
        for line in open(out_file):
            try:
                r = json.loads(line)
                done.add((r['arm'], r['opponent']))
            except Exception:
                pass
    pairs = build_pairs(set_name)
    if only_opponent:
        pairs = [p for p in pairs if only_opponent in p[2]]
    mine = [p for i, p in enumerate(pairs) if i % shards == shard]
    print('[shard %d/%d] 共 %d 个对阵，已完成 %d' % (shard, shards, len(mine), len(done)), flush=True)
    with open(out_file, 'a') as fh:
        for arm, arm_path, oname, opath in mine:
            if (arm, oname) in done:
                print('  skip %s vs %s' % (arm, oname), flush=True)
                continue
            t0 = time.time()
            rec = {'arm': arm, 'arm_path': arm_path, 'opponent': oname, 'opponent_path': opath,
                   'n_episodes': n_episodes}
            try:
                res = Evaluator.run_match(model1_path=arm_path, model2_path=opath,
                                          n_episodes=n_episodes, render_mode=None, use_tqdm=False)
                # numpy 标量 -> python float，保证 JSON 可序列化
                rec['metrics'] = {k: (float(v) if isinstance(v, (int, float)) or hasattr(v, 'item') else v)
                                  for k, v in dict(res).items()}
                rec['ok'] = True
            except Exception as exc:  # noqa: BLE001
                rec['ok'] = False
                rec['error'] = repr(exc)[:500]
            rec['seconds'] = round(time.time() - t0, 1)
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            m = rec.get('metrics', {})
            print('  %-16s vs %-30s win=%.1f%% draw=%.1f%% loss=%.1f%% hp=%.2f/%.2f (%.0fs)' % (
                arm, oname, 100 * m.get('win_rate', 0), 100 * m.get('draw_rate', 0),
                100 * m.get('loss_rate', 0), m.get('avg_hp', 0), m.get('avg_hp_oppo', 0), rec['seconds']),
                flush=True)


def merge(set_name, n_episodes):
    out_dir = os.path.join(OUT_ROOT, set_name)
    recs = []
    for fp in sorted(glob.glob(os.path.join(out_dir, 'shard*.jsonl'))):
        for line in open(fp):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    if not recs:
        raise SystemExit('没有可合并的结果')
    json.dump(recs, open(os.path.join(out_dir, 'merged.json'), 'w'), indent=1, ensure_ascii=False)

    arms = sorted({r['arm'] for r in recs})
    opps = sorted({r['opponent'] for r in recs})
    keys = ['win_rate', 'draw_rate', 'loss_rate', 'opponent_fall_rate', 'avg_hp', 'avg_hp_oppo',
            'gun_opportunity_time_ratio', 'gun_opportunity_time_ratio_oppo',
            'specific_energy_mean', 'track_angle_mean', 'damage_to_oppo_total', 'avg_win_time']
    lines = ['# 初始化消融评测汇总（set=%s, episodes/对阵=%d）\n' % (set_name, n_episodes),
             '模型 1 = 被测臂，模型 2 = 对手；每格为 %d 局的统计。\n' % n_episodes]
    # 总体表
    lines.append('\n## 总体（对全部对手加权平均，权重=局数）\n\n| 臂 | 胜率 | 平局 | 负率 | 对手坠机 | 自身HP | 对手HP | 枪口机会 | 比能量 | 跟踪角 | 总伤害 |\n|---|---|---|---|---|---|---|---|---|---|---|\n')
    for arm in arms:
        rs = [r for r in recs if r['arm'] == arm and r.get('ok')]
        tot = sum(r['n_episodes'] for r in rs) or 1
        def w(k):
            return sum(r['metrics'].get(k, 0) * r['n_episodes'] for r in rs) / tot
        lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f | %.2f | %.3f | %.0f | %.3f | %.2f |\n' % (
            arm, 100 * w('win_rate'), 100 * w('draw_rate'), 100 * w('loss_rate'), 100 * w('opponent_fall_rate'),
            w('avg_hp'), w('avg_hp_oppo'), w('gun_opportunity_time_ratio'), w('specific_energy_mean'),
            w('track_angle_mean'), w('damage_to_oppo_total')))
    # 逐对手胜率表
    lines.append('\n## 逐对手胜率\n\n| 对手 | ' + ' | '.join(arms) + ' |\n|---|' + '---|' * len(arms) + '\n')
    for opp in opps:
        cells = []
        for arm in arms:
            rr = [r for r in recs if r['arm'] == arm and r['opponent'] == opp and r.get('ok')]
            cells.append('%.1f%%' % (100 * rr[0]['metrics'].get('win_rate', 0)) if rr else '—')
        lines.append('| %s | %s |\n' % (opp, ' | '.join(cells)))
    # 失败记录
    bad = [r for r in recs if not r.get('ok')]
    if bad:
        lines.append('\n## 失败对阵（%d）\n\n' % len(bad))
        for r in bad:
            lines.append('- %s vs %s: %s\n' % (r['arm'], r['opponent'], r.get('error', '')))
    md = os.path.join(out_dir, 'summary.md')
    open(md, 'w').write(''.join(lines))
    print('已写出 %s 与 %s（%d 条记录）' % (os.path.join(out_dir, 'merged.json'), md, len(recs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=['bench8', 'ho', 'ho10', 'ho22', 'cross'])
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--only-opponent', default=None, help='只评测名字包含该子串的对手（局部重跑用）')
    ap.add_argument('--out-name', default=None, help='自定义输出文件名（不含 .jsonl）')
    args = ap.parse_args()
    if args.merge:
        merge(args.set, args.episodes)
    else:
        run_shard(args.set, args.episodes, args.shards, args.shard,
                  only_opponent=args.only_opponent, out_name=args.out_name)


if __name__ == '__main__':
    main()
