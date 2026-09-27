"""从评测分片直接生成论文用的 LaTeX 表格（不手抄数字）。

用法：python scripts/ablation/make_paper_tables.py
输出：configs/ablation/tables_init_ablation.tex
  - tab:init_ablation_results   总体（胜率+Wilson CI / 平局 / 负率 / 枪口机会 / 跟踪角 / ΔHP）
  - tab:init_ablation_bytype    按对手类型拆分
  - tab:init_crossplay          三臂交叉对战（胜率 / ΔHP）
"""
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

ARM_LABEL = {
    'Ctrl-full-init': 'With initialization',
    'A-selfplay-init': 'Without initialization',
}
ARM_ORDER = ['Ctrl-full-init', 'A-selfplay-init']          # 论文只比较"加不加初始化"
KEYS = ['win_rate', 'draw_rate', 'loss_rate', 'gun_opportunity_time_ratio', 'track_angle_mean',
        'avg_hp', 'avg_hp_oppo', 'specific_energy_mean', 'damage_to_oppo_total']


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(set_name):
    recs = {}
    for fp in sorted(glob.glob(os.path.join('experiments', 'ablation_eval', set_name, '*.jsonl'))):
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('ok'):
                recs[(r['arm'], r['opponent'])] = r      # 同名后写覆盖前写
    return list(recs.values())


def agg(recs, keys=KEYS):
    tot = sum(r['n_episodes'] for r in recs) or 1
    return {k: sum(r['metrics'][k] * r['n_episodes'] for r in recs) / tot for k in keys}


GROUP_TITLE = {'level-flight': 'Level flight', 'combat': 'Early combat',
               'goal-point': 'Goal point', 'weak-ref': 'Weak reference', 'other': 'Other'}


def classify(o):
    if 'level-flight' in o:
        return 'level-flight'
    if 'weakref' in o or 'weak-ref' in o:
        return 'weak-ref'
    if 'goalpoint' in o:
        return 'goal-point'
    if 'combat' in o:
        return 'combat'
    return 'other'


def main():
    bench = load('bench8')
    cross = load('cross')
    if not bench:
        raise SystemExit('没有 bench8 数据')
    opps = sorted({r['opponent'] for r in bench})
    groups = ['level-flight', 'combat', 'goal-point', 'weak-ref']
    groups = [g for g in groups if any(classify(o) == g for o in opps)]

    L = ['% 由 scripts/ablation/make_paper_tables.py 自动生成，请勿手改\n']

    # ---- 总体表 ----
    L.append('\\begin{table}[h]')
    L.append('\\caption{Effect of opponent pool initialization on the benchmark of '
             'Section~\\ref{sec:eval_benchmark} ($8\\times1000$ engagements per arm). $\\Delta$HP is the '
             'mean difference in remaining health points between the ego and the opponent at the end of '
             'an engagement.}')
    L.append('\\label{tab:init_ablation_results}')
    L.append('\\small')
    L.append('\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lccccc}')
    L.append('\\toprule')
    L.append('Arm & Win rate & Draw & Loss & Gun opportunity & $\\Delta$HP\\\\')
    L.append('\\midrule')
    for arm in ARM_ORDER:
        rs = [r for r in bench if r['arm'] == arm]
        if not rs:
            continue
        n = sum(r['n_episodes'] for r in rs)
        k = sum(r['metrics']['win_rate'] * r['n_episodes'] for r in rs)
        lo, hi = wilson(k, n)
        g = agg(rs)
        L.append('%s & $%.2f\\%%$ & $%.1f\\%%$ & $%.2f\\%%$ & $%.3f$ & $%+.2f$\\\\'
                 % (ARM_LABEL[arm], 100 * g['win_rate'] + 1e-9, 100 * g['draw_rate'] + 1e-9,
                    100 * g['loss_rate'] + 1e-9, g['gun_opportunity_time_ratio'],
                    g['avg_hp'] - g['avg_hp_oppo']))
    L.append('\\botrule')
    L.append('\\end{tabular*}')
    L.append('\\end{table}')
    L.append('')

    # ---- 按类型 ----
    L.append('\\begin{table}[h]')
    L.append('\\caption{Win rate of each arm against the four opponent types of the benchmark pool, '
             'which contains four level-flight, two early-combat, one goal-point and one weak-reference '
             'opponent.}')
    L.append('\\label{tab:init_ablation_bytype}')
    L.append('\\small')
    L.append('\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l' + 'c' * len(groups) + '}')
    L.append('\\toprule')
    L.append('Arm & ' + ' & '.join(GROUP_TITLE[g] for g in groups) + '\\\\')
    L.append('\\midrule')
    for arm in ARM_ORDER:
        cells = []
        for g in groups:
            rs = [r for r in bench if r['arm'] == arm and classify(r['opponent']) == g]
            cells.append('$%.1f\\%%$' % (100 * agg(rs)['win_rate']) if rs else '--')
        L.append('%s & %s\\\\' % (ARM_LABEL[arm], ' & '.join(cells)))
    L.append('\\botrule')
    L.append('\\end{tabular*}')
    L.append('\\end{table}')
    L.append('')

    # ---- 交叉对战 ----
    if cross and False:      # 交叉对战改为正文行内叙述，不再单独出表
        opp_arms = [a for a in ARM_ORDER if any(r['opponent'] == a for r in cross)]
        L.append('\\begin{table}[h]')
        L.append('\\caption{Cross-play between the two arms (300 engagements per ordered pair). '
                 'Each cell gives the win rate of the row model and, in parentheses, the mean '
                 'difference in remaining health points between the row and the column model.}')
        L.append('\\label{tab:init_crossplay}')
        L.append('\\small')
        L.append('\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}l' + 'c' * len(opp_arms) + '}')
        L.append('\\toprule')
        L.append(' & ' + ' & '.join(ARM_LABEL[a].split(' (')[0] for a in opp_arms) + '\\\\')
        L.append('\\midrule')
        for a in ARM_ORDER:
            cells = []
            for o in opp_arms:
                rs = [r for r in cross if r['arm'] == a and r['opponent'] == o]
                if not rs:
                    cells.append('--')
                elif a == o:
                    cells.append('--')
                else:
                    m = rs[0]['metrics']
                    cells.append('$%.1f\\%%$ $(%+.2f)$' % (100 * m['win_rate'], m['avg_hp'] - m['avg_hp_oppo']))
            L.append('%s & %s\\\\' % (ARM_LABEL[a].split(' (')[0], ' & '.join(cells)))
        L.append('\\botrule')
        L.append('\\end{tabular*}')
        L.append('\\end{table}')
        L.append('')

    out = 'configs/ablation/tables_init_ablation.tex'
    open(out, 'w').write('\n'.join(L))
    print('wrote', out)
    # 控制台回显关键数字，便于写正文
    for arm in ARM_ORDER:
        rs = [r for r in bench if r['arm'] == arm]
        if rs:
            g = agg(rs)
            print('%-16s win=%.2f%% draw=%.2f%% loss=%.2f%% dHP=%+.2f gunOpp=%.3f track=%.3f dmg=%.2f energy=%.0f'
                  % (arm, 100 * g['win_rate'], 100 * g['draw_rate'], 100 * g['loss_rate'],
                     g['avg_hp'] - g['avg_hp_oppo'], g['gun_opportunity_time_ratio'],
                     g['track_angle_mean'], g['damage_to_oppo_total'], g['specific_energy_mean']))
    print('对手数:', len(opps), '| 记录数:', len(bench))


if __name__ == '__main__':
    main()
