"""L2/L3 结果分析与作图（消费 eval_ablation.py 的 merged.json）。

用法：python scripts/ablation/analyze_l2.py [--episodes 300]

产出：
  experiments/ablation_L2_wdl.png            胜负平堆叠柱（全部对手 / 仅强对手）
  experiments/ablation_L2_heatmap.png        逐对手胜率热图（臂 × 对手）
  experiments/ablation_L2_capability.png     2x2 能力指标（枪口机会/比能量/跟踪角/伤害）
  experiments/ablation_L3_crossplay.png      交叉对战胜率矩阵
  experiments/ablation_L2_results.md         汇总表（含 Wilson 95% 置信区间）
"""
import argparse
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
EVAL = os.path.join(ROOT, 'experiments', 'ablation_eval')


def heatmap(ax, M, row_labels, col_labels, cbar_label):
    """纯 matplotlib 热图（避免依赖 seaborn）。"""
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, '%.0f' % M[i, j], ha='center', va='center', fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(set_name):
    """直接读取分片 jsonl 并汇总（不产生任何中间"合并文件"，也不涉及 git）。"""
    d = os.path.join(EVAL, set_name)
    PAPER_ARMS = {'Ctrl-full-init', 'A-selfplay-init'}      # 论文只比较"加不加初始化"
    recs = []
    for fp in sorted(glob.glob(os.path.join(d, '*.jsonl'))):
        for line in open(fp):
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    # 同一 (arm, opponent) 若被重复写入，保留最后一条
    uniq = {}
    for r in recs:
        uniq[(r['arm'], r['opponent'])] = r
    return [r for r in uniq.values() if r.get('ok') and r['arm'] in PAPER_ARMS]


def agg(records, keys=('win_rate', 'draw_rate', 'loss_rate', 'opponent_fall_rate', 'avg_hp',
                       'avg_hp_oppo', 'gun_opportunity_time_ratio', 'specific_energy_mean',
                       'track_angle_mean', 'damage_to_oppo_total')):
    tot = sum(r['n_episodes'] for r in records) or 1
    return {k: sum(r['metrics'].get(k, 0) * r['n_episodes'] for r in records) / tot for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='bench8', help='主评测集（默认 bench8 = 新的中立 8 对手基准）')
    ap.add_argument('--episodes', type=int, default=1000)
    args = ap.parse_args()

    MAIN = args.set
    ho = load(MAIN)
    cross = load('cross')
    if not ho and not cross:
        raise SystemExit('主集 %s 与 cross 都没有数据' % MAIN)

    arms = sorted({r['arm'] for r in ho}) or sorted({r['arm'] for r in cross})
    opps = sorted({r['opponent'] for r in ho})
    # 对手分组（按名字前缀）
    def classify(o):
        if 'level-flight' in o: return 'level-flight'
        if 'weakref' in o or 'weak-ref' in o: return 'weak-ref'
        if 'goalpoint' in o: return 'goalpoint'
        if 'combat' in o: return 'combat'
        return 'other'
    grp = {o: classify(o) for o in opps}
    GROUPS = ['level-flight', 'weak-ref', 'combat', 'goalpoint', 'other']
    gcount = {g: sum(1 for o in opps if grp[o] == g) for g in GROUPS}

    lines = [f'# 初始化消融评测结果（主集 = {MAIN}）\n',
             f'每对阵 {args.episodes} 局；对手集为 md5 审计后的中立基准（未出现在任何参与对比的训练池中）。\n',
             '模型 1 = 被测臂，模型 2 = 对手（jsbsim 追逐模式，`goal_point_prob=0`）。\n']
    if not ho:
        lines.append(f'\n> ⚠️ 主集 {MAIN} 尚无数据（评测进行中），本文件当前仅含交叉对战部分。\n')

    # ---------- 总体表 ----------
    if ho:
        lines.append('\n## 1. 总体（对 %d 个中立对手加权平均）\n\n' % len(opps)
                     + '| 臂 | 胜率 [95% CI] | 平局 | 负率 | 对手坠机 | 自身HP | 对手HP | 枪口机会 | 比能量(ft) | 跟踪角(rad) | 总伤害 |\n'
                     + '|---|---|---|---|---|---|---|---|---|---|---|\n')
    for a in arms:
        rs = [r for r in ho if r['arm'] == a]
        n = sum(r['n_episodes'] for r in rs)
        k = sum(r['metrics']['win_rate'] * r['n_episodes'] for r in rs)
        lo, hi = wilson(k, n)
        g = agg(rs)
        lines.append('| %s | **%.2f%%** [%.2f, %.2f] | %.2f%% | %.2f%% | %.2f%% | %.2f | %.2f | %.3f | %.0f | %.3f | %.2f |\n' % (
            a, 100 * g['win_rate'], 100 * lo, 100 * hi, 100 * g['draw_rate'], 100 * g['loss_rate'],
            100 * g['opponent_fall_rate'], g['avg_hp'], g['avg_hp_oppo'], g['gun_opportunity_time_ratio'],
            g['specific_energy_mean'], g['track_angle_mean'], g['damage_to_oppo_total']))

    # ---------- 分组表 ----------
    hdr = ' | '.join('%s(%d)' % (g, gcount[g]) for g in GROUPS if gcount[g])
    lines.append('\n## 2. 按对手类型拆分\n\n| 臂 | ' + hdr + ' |\n|---|' + '---|' * sum(1 for g in GROUPS if gcount[g]) + '\n')
    for a in arms:
        cells = []
        for gname in [g for g in GROUPS if gcount[g]]:
            rs = [r for r in ho if r['arm'] == a and grp[r['opponent']] == gname]
            if not rs:
                cells.append('—')
                continue
            g = agg(rs)
            n = sum(r['n_episodes'] for r in rs)
            k = sum(r['metrics']['win_rate'] * r['n_episodes'] for r in rs)
            lo, hi = wilson(k, n)
            cells.append('%.1f%% [%.1f, %.1f]' % (100 * g['win_rate'], 100 * lo, 100 * hi))
        lines.append('| %s | %s |\n' % (a, ' | '.join(cells)))

    # ---------- 逐对手 ----------
    lines.append('\n## 3. 逐对手胜率\n\n| 对手 | 类型 | ' + ' | '.join(arms) + ' |\n|---|---|' + '---|' * len(arms) + '\n')
    for o in opps:
        cells = []
        for a in arms:
            rs = [r for r in ho if r['arm'] == a and r['opponent'] == o]
            cells.append('%.1f%%' % (100 * rs[0]['metrics']['win_rate']) if rs else '—')
        lines.append('| %s | %s | %s |\n' % (o, grp[o], ' | '.join(cells)))

    # ---------- 交叉对战 ----------
    if cross:
        lines.append('\n## 4. 三臂交叉对战（胜率 / 自身-对手 HP 差）\n\n| 行=模型1，列=模型2 | '
                     + ' | '.join(sorted({r['opponent'] for r in cross})) + ' |\n|---|'
                     + '---|' * len({r['opponent'] for r in cross}) + '\n')
        for a in sorted({r['arm'] for r in cross}):
            cells = []
            for o in sorted({r['opponent'] for r in cross}):
                rs = [r for r in cross if r['arm'] == a and r['opponent'] == o]
                cells.append('%.1f%% / %+.2f' % (100 * rs[0]['metrics']['win_rate'],
                                                 rs[0]['metrics']['avg_hp'] - rs[0]['metrics']['avg_hp_oppo'])
                             if rs else '—')
            lines.append('| %s | %s |\n' % (a, ' | '.join(cells)))

    open('experiments/ablation_L2_results.md', 'w').write(''.join(lines))

    # ================= 图 =================
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10})

    # 图1：胜负平堆叠柱
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (title, filt) in zip(axes, [('All 13 held-out opponents', lambda o: True),
                                        ('Strong opponents only (4)', lambda o: grp[o] == 'strong')]):
        ys = np.arange(len(arms))
        w = [100 * agg([r for r in ho if r['arm'] == a and filt(r['opponent'])])['win_rate'] for a in arms]
        d = [100 * agg([r for r in ho if r['arm'] == a and filt(r['opponent'])])['draw_rate'] for a in arms]
        l = [100 * agg([r for r in ho if r['arm'] == a and filt(r['opponent'])])['loss_rate'] for a in arms]
        ax.barh(ys, w, color='#307bb5', edgecolor='black', lw=.8, label='Win')
        ax.barh(ys, d, left=w, color='#EAEAEA', edgecolor='black', lw=.8, label='Draw')
        ax.barh(ys, l, left=np.add(w, d), color='#c43f3c', edgecolor='black', lw=.8, label='Loss')
        for i, (wi, di, li) in enumerate(zip(w, d, l)):
            ax.text(min(wi / 2, 12), i, '%.1f%%' % wi, va='center', ha='center',
                    color='white' if wi > 6 else 'black', fontsize=8, fontweight='bold')
            ax.text(wi + di / 2, i, '%.0f%%' % di, va='center', ha='center', fontsize=8)
        ax.set_yticks(ys); ax.set_yticklabels([a.split(' ')[0] for a in arms], fontweight='bold')
        ax.set_xlim(0, 100); ax.set_xlabel('percentage of %d engagements' % args.episodes)
        ax.set_title(title); ax.grid(axis='x', alpha=.3); ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc='lower right')
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig('experiments/ablation_L2_wdl.png', dpi=150); plt.close()

    # 图2：逐对手胜率热图
    M = np.full((len(arms), len(opps)), np.nan)
    for i, a in enumerate(arms):
        for j, o in enumerate(opps):
            rs = [r for r in ho if r['arm'] == a and r['opponent'] == o]
            if rs:
                M[i, j] = 100 * rs[0]['metrics']['win_rate']
    fig, ax = plt.subplots(figsize=(11, 3.4))
    heatmap(ax, M, [a.split(' ')[0] for a in arms],
            [o.split('(', 1)[0] for o in opps], 'win rate (%)')
    ax.set_title('Win rate (%) per held-out benchmark opponent')
    plt.tight_layout(); plt.savefig('experiments/ablation_L2_heatmap.png', dpi=150); plt.close()

    # 图3：能力指标 2x2
    metrics = [('gun_opportunity_time_ratio', 'Gun opportunity ratio (↑)', 100),
               ('specific_energy_mean', 'Mean specific energy (ft) (↑)', 1),
               ('track_angle_mean', 'Mean track angle (rad) (↓)', 1),
               ('damage_to_oppo_total', 'Total damage to opponent (↑)', 1)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    for ax, (k, title, sc) in zip(axes.ravel(), metrics):
        vals = [sc * agg([r for r in ho if r['arm'] == a])[k] for a in arms]
        bars = ax.bar(np.arange(len(arms)), vals, color='#307bb5', edgecolor='black', lw=.6, width=.55)
        for b, v in zip(bars, vals):
            ax.annotate('%.3f' % v if v < 10 else '%.1f' % v, (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([a.split(' ')[0] for a in arms], fontsize=8, fontweight='bold')
        ax.set_title(title); ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig('experiments/ablation_L2_capability.png', dpi=150); plt.close()

    # 图4：交叉对战
    if cross:
        cols = sorted({r['opponent'] for r in cross})
        rows = sorted({r['arm'] for r in cross})
        W = np.full((len(rows), len(cols)), np.nan)
        for i, a in enumerate(rows):
            for j, o in enumerate(cols):
                rs = [r for r in cross if r['arm'] == a and r['opponent'] == o]
                if rs:
                    W[i, j] = 100 * rs[0]['metrics']['win_rate']
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        heatmap(ax, W, [r.split(' ')[0] for r in rows], [c.split(' ')[0] for c in cols], 'win rate (%)')
        ax.set_title('Cross-play win rate (%)  (row = model 1, column = model 2)')
        plt.tight_layout(); plt.savefig('experiments/ablation_L3_crossplay.png', dpi=150); plt.close()

    print('wrote experiments/ablation_L2_results.md + 4 figures')
    for a in arms:
        g = agg([r for r in ho if r['arm'] == a])
        print('%-16s held-out win=%.1f%% draw=%.1f%% loss=%.1f%% gunOpp=%.3f' % (
            a, 100 * g['win_rate'], 100 * g['draw_rate'], 100 * g['loss_rate'], g['gun_opportunity_time_ratio']))


if __name__ == '__main__':
    main()
