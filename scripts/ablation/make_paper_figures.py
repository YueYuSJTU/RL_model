"""初始化消融：论文用图（风格与原有绘图脚本一致）。

风格来源（严格对齐）：
  src/evaluation/metric/plot_reward_curve.py   折线：serif/Times New Roman，title 17 / label 16 /
      tick 15 / legend 10，figsize (8,5)，配色 #307bb5 / #c43f3c / #4f9b6b，EMA 平滑 w=0.95，
      linewidth 2，whitegrid + despine，图例 lower right（黑边方框），dpi=300 + bbox_inches='tight'
  src/evaluation/metric/plot_eval_results.py   堆叠条形：win #307bb5 / draw #EAEAEA / loss #c43f3c，
      黑色描边，条内白色加粗数值，Loss 标在条形右侧，xlim 0-100 + PercentFormatter，
      xlabel "Match Percentage Indicator (%)"，图例置于图下方（ncol=3, frameon=False）
  src/evaluation/metric/plot_crossplay_heatmap.py  热图：annot 加粗 fontsize 13，坐标刻度加粗，
      标题加粗 pad=15，colorbar，dpi=300

臂命名（与正文一致）：With initialization / Without initialization
输出：configs/ablation/figures/init_ablation_{stability,wdl,heatmap}.png
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
OUT = os.path.join('configs', 'ablation', 'figures')
CYCLE = 10_000_000

ARMS = {
    'With initialization': 'experiments/20260428_141850_best_train_mlp2',
    'Without initialization': 'experiments/20260924_225859_selfplay_init',
}
ARM_KEY = {'With initialization': 'Ctrl-full-init', 'Without initialization': 'A-selfplay-init'}
COLORS = ['#307bb5', '#c43f3c']          # 与 plot_reward_curve.py 相同的蓝/红
WDL_COLORS = ['#307bb5', '#EAEAEA', '#c43f3c']   # 与 plot_eval_results.py 相同


def rcparams():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'axes.titlesize': 17,
        'axes.labelsize': 16,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 10,
        'figure.figsize': (8, 5),
    })


def whitegrid(ax):
    """seaborn whitegrid + despine 的等效实现（环境中没有 seaborn）。"""
    ax.set_axisbelow(True)
    ax.grid(True, color='#DCDCDC', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def smooth(vals, weight=0.95):
    last, out = vals[0], []
    for v in vals:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return out


def load_curve(run, tag='eval/mean_reward'):
    files = sorted(glob.glob(os.path.join(run, 'tensorboard', '*', 'PPO_*', 'events.out.tfevents*')),
                   key=os.path.getmtime)
    d = {}
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception:
            continue
        if tag in ea.Tags().get('scalars', []):
            for e in ea.Scalars(tag):
                d[e.step] = e.value
    steps = sorted(d)
    return np.array(steps), np.array([d[s] for s in steps])


START_STEP = 1_000_000        # stage-1（0–300k 步）的评估点与消融变量无关，且首段剧烈跳变，故不绘制


def fig_stability():
    rcparams()
    fig, ax = plt.subplots(figsize=(8, 5))
    kept = []
    for (label, run), color in zip(ARMS.items(), COLORS):
        steps, vals = load_curve(run)
        m = steps >= START_STEP                     # 从 battle 阶段开始
        steps, vals = steps[m], vals[m]
        ys = np.array(smooth(list(vals), 0.95))     # EMA 从第一个保留点重新起算
        kept.append(ys)
        ax.plot(steps, ys, color=color, label=label, linewidth=2)
    lo = min(float(y.min()) for y in kept)
    hi = max(float(y.max()) for y in kept)
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)                 # 露出曲线最低点
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Evaluation Return')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    whitegrid(ax)
    ax.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'init_ablation_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()


def bench_records():
    rec = {}
    for fp in sorted(glob.glob(os.path.join('experiments', 'ablation_eval', 'bench8', '*.jsonl'))):
        for line in open(fp):
            if line.strip():
                r = json.loads(line)
                if r.get('ok'):
                    rec[(r['arm'], r['opponent'])] = r
    return rec


def fig_wdl():
    """胜负平堆叠条形（扁平、细条、单面板，风格同 plot_eval_results.py）。"""
    rcparams()
    rec = bench_records()
    opps = sorted({k[1] for k in rec})
    labels = list(ARMS)
    win, draw, loss = [], [], []
    for lab in labels:
        rs = [rec[(ARM_KEY[lab], o)] for o in opps if (ARM_KEY[lab], o) in rec]
        n = sum(r['n_episodes'] for r in rs)
        win.append(100 * sum(r['metrics']['win_rate'] * r['n_episodes'] for r in rs) / n + 1e-9)
        draw.append(100 * sum(r['metrics']['draw_rate'] * r['n_episodes'] for r in rs) / n + 1e-9)
        loss.append(100 * sum(r['metrics']['loss_rate'] * r['n_episodes'] for r in rs) / n + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 2.6))
    order = ['With initialization', 'Without initialization']
    ys = np.array([0.55, 0.0])          # 上=With，下=Without（与下方热图行序一致）
    h = 0.34
    win = [win[labels.index(o)] for o in order]
    draw = [draw[labels.index(o)] for o in order]
    loss = [loss[labels.index(o)] for o in order]

    p1 = ax.barh(ys, win, h, color=WDL_COLORS[0], edgecolor='black', linewidth=1.2, label='Win')
    p2 = ax.barh(ys, draw, h, left=win, color=WDL_COLORS[1], edgecolor='black', linewidth=1.2, label='Draw')
    p3 = ax.barh(ys, loss, h, left=np.add(win, draw), color=WDL_COLORS[2], edgecolor='black',
                 linewidth=1.2, label='Loss')
    FS = 13                              # 与下方热图的注释字号一致
    for yv, w, d, l in zip(ys, win, draw, loss):
        ax.text(w / 2, yv, '%.1f%%' % w, va='center', ha='center', color='white',
                fontweight='bold', fontsize=FS)
        ax.text(w + d / 2, yv, '%.1f%%' % d, va='center', ha='center', fontsize=FS)
        ax.text(w + d + l + 1.0, yv, 'Loss: %.2f%%' % l, va='center', ha='left',
                fontweight='bold', fontsize=FS)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontweight='bold', fontsize=FS)
    ax.set_ylim(-0.45, 0.95)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='x', labelsize=FS)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
    ax.set_axisbelow(True)
    ax.set_xlabel('Match Percentage Indicator (%)', fontweight='bold', fontsize=FS + 1)
    ax.legend([p1, p2, p3], ['Win', 'Draw', 'Loss'], loc='upper center', bbox_to_anchor=(0.5, -0.50),
              ncol=3, frameon=False, handlelength=1.4, handleheight=1.4, fontsize=FS - 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'init_ablation_wdl.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_heatmap():
    """逐对手胜率热图（风格同 plot_crossplay_heatmap.py：加粗刻度与注释、colorbar、标题 pad 15）。"""
    rcparams()
    rec = bench_records()
    opps = sorted({k[1] for k in rec})
    labels = list(ARMS)
    M = np.full((len(labels), len(opps)), np.nan)
    for i, lab in enumerate(labels):
        for j, o in enumerate(opps):
            r = rec.get((ARM_KEY[lab], o))
            if r:
                M[i, j] = 100 * r['metrics']['win_rate']

    fig, ax = plt.subplots(figsize=(9, 2.6))
    # 只展示胜率（非发散量），采用白→蓝的顺序色图（高胜率=深蓝，与论文主色 #307bb5 同一色系）
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                v = M[i, j]
                ax.text(j, i, '%.1f' % v, ha='center', va='center', fontsize=13,
                        fontweight='bold', color='white' if v >= 60 else 'black')
    ax.set_xticks(range(len(opps)))
    ax.set_xticklabels([str(i + 1) for i in range(len(opps))], fontweight='bold', fontsize=13)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontweight='bold', fontsize=13)
    ax.set_xlabel('Benchmark opponent', fontweight='bold', fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('Win rate (%)', fontsize=12)
    cb.ax.tick_params(labelsize=12)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'init_ablation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_stability()
    fig_wdl()
    fig_heatmap()
    print('wrote', OUT, '->', sorted(os.listdir(OUT)))
