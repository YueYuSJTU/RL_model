"""L1 训练稳定性分析：三臂 per-cycle 曲线 + 稳定性指标。

输入：各 run 的 tensorboard/<ts>/PPO_*/events.out.tfevents*（每 cycle 一个文件）
输出：
  experiments/ablation_L1_stability.md   稳定性指标表 + 逐 cycle 均值
  experiments/ablation_L1_stability.png  三臂曲线（细线=原始 eval 点，粗线=per-cycle 均值）
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

# 论文只比较"加不加初始化"：对照（完整初始化） vs 无初始化（纯自博弈）
ARMS = {
    'Full initialization (control)': 'experiments/20260428_141850_best_train_mlp2',
    'No initialization (self-play)': 'experiments/20260924_225859_selfplay_init',
}
CYCLE = 10_000_000


def load_series(run):
    files = sorted(glob.glob(os.path.join(run, 'tensorboard', '*', 'PPO_*', 'events.out.tfevents*')),
                   key=os.path.getmtime)
    out = {}
    for f in files:
        try:
            ea = EventAccumulator(f, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ('eval/mean_reward', 'eval/mean_ep_length', 'train/entropy_loss', 'train/std',
                    'train/approx_kl', 'train/explained_variance', 'train/value_loss'):
            if tag in ea.Tags().get('scalars', []):
                for e in ea.Scalars(tag):
                    out.setdefault(tag, {})[e.step] = e.value
    return out


def per_cycle(series, tag):
    d = series.get(tag, {})
    buckets = {}
    for step, v in d.items():
        c = int(step // CYCLE)      # 0 = stage1 段
        buckets.setdefault(c, []).append(v)
    return {c: st.mean(vs) for c, vs in sorted(buckets.items()) if c >= 1}


def main():
    data = {name: load_series(run) for name, run in ARMS.items()}
    lines = ['# L1 训练稳定性（三臂对照）\n',
             '数据源：各 run 的 `tensorboard/<ts>/PPO_*/events.out.tfevents*`；每 cycle = 10M env steps。\n',
             '> 注意：对照（Ctrl）是在旧代码版本（HP=5、`betaR=-2.5`）下训练的，绝对值仅供趋势参考；'
             'A/B 与它同配置、同 seed 无关随机性，参数差异对早期训练影响很小（用户已确认可视为相同）。\n',
             '\n## 1. 稳定性指标（基于 per-cycle eval return）\n',
             '| 臂 | 早段均值(1-10) | 中段(11-20) | 晚段(21-30) | 趋势斜率/cycle | 晚/早 | cycle间波动(std) | 最差单次回撤 | 崩盘次数(<50%历史峰值) | 峰值 |\n',
             '|---|---|---|---|---|---|---|---|---|---|\n']
    table = {}
    for name, series in data.items():
        pc = per_cycle(series, 'eval/mean_reward')
        cyc = [c for c in sorted(pc) if c <= 30]
        vals = [pc[c] for c in cyc]
        if len(vals) < 5:
            continue
        e = st.mean(vals[:10]); m = st.mean(vals[10:20]) if len(vals) >= 15 else float('nan')
        l = st.mean(vals[20:30]) if len(vals) >= 25 else float('nan')
        slope = (vals[-1] - vals[0]) / max(1, (cyc[-1] - cyc[0]))
        runmax, drops, collapses = vals[0], [], 0
        for i, v in enumerate(vals):
            if i > 0:
                drops.append(100 * (vals[i - 1] - v) / max(1e-6, abs(vals[i - 1])))
            runmax = max(runmax, v)
            if v < 0.5 * runmax:
                collapses += 1
        table[name] = dict(early=e, mid=m, late=l, slope=slope, ratio=(l / e if e else float('nan')),
                           vol=st.pstdev(vals), worst=max(drops) if drops else 0, collapses=collapses,
                           peak=max(vals), vals=vals, cycles=cyc)
        lines.append('| %s | %.0f | %.0f | %.0f | %+.1f | %.2f | %.0f | %.0f%% | %d | %.0f |\n' % (
            name, e, m, l, slope, (l / e if e else float('nan')), st.pstdev(vals),
            max(drops) if drops else 0, collapses, max(vals)))

    lines.append('\n## 2. 其他稳定性相关标量（全期均值 ± 标准差）\n')
    lines.append('| 臂 | eval episode 长度 | entropy_loss | policy std | approx_kl | explained_variance |\n|---|---|---|---|---|---|\n')
    for name, series in data.items():
        cells = []
        for tag in ('eval/mean_ep_length', 'train/entropy_loss', 'train/std', 'train/approx_kl',
                    'train/explained_variance'):
            vs = list(series.get(tag, {}).values())
            cells.append('%.3f ± %.3f' % (st.mean(vs), st.pstdev(vs)) if vs else '—')
        lines.append('| %s | %s |\n' % (name, ' | '.join(cells)))

    lines.append('\n## 3. 逐 cycle eval return\n\n| cycle | ' + ' | '.join(data.keys()) + ' |\n|---|'
                 + '---|' * len(data) + '\n')
    for c in range(1, 31):
        row = []
        for name in data:
            pc = per_cycle(data[name], 'eval/mean_reward')
            row.append('%.0f' % pc[c] if c in pc else '—')
        lines.append('| %d | %s |\n' % (c, ' | '.join(row)))

    open('experiments/ablation_L1_stability.md', 'w').write(''.join(lines))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    colors = {'Full initialization (control)': '#307bb5',
              'No initialization (self-play)': '#c43f3c'}
    for name, series in data.items():
        d = series.get('eval/mean_reward', {})
        xs = sorted(d)
        axes[0].plot([x / 1e6 for x in xs], [d[x] for x in xs], lw=0.6, alpha=0.35, color=colors[name])
        pc = per_cycle(series, 'eval/mean_reward')
        cyc = [c for c in sorted(pc) if c <= 30]
        axes[0].plot([c * 10 for c in cyc], [pc[c] for c in cyc], lw=2.0, marker='o', ms=3,
                     color=colors[name], label=name)
        pc2 = per_cycle(series, 'eval/mean_ep_length')
        cyc2 = [c for c in sorted(pc2) if c <= 30]
        axes[1].plot([c * 10 for c in cyc2], [pc2[c] for c in cyc2], lw=1.6, marker='o', ms=3,
                     color=colors[name], label=name)
    axes[0].set_xlabel('env steps (millions)'); axes[0].set_ylabel('eval mean reward')
    axes[0].set_title('Training stability: per-cycle eval return'); axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel('env steps (millions)'); axes[1].set_ylabel('eval mean episode length (agent steps)')
    axes[1].set_title('Engagement termination behaviour'); axes[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig('experiments/ablation_L1_stability.png', dpi=150)
    print('wrote experiments/ablation_L1_stability.md / .png')
    for name, t in table.items():
        print('%-42s early=%.0f late=%.0f slope=%+.1f vol=%.0f worst_drop=%.0f%% collapses=%d peak=%.0f'
              % (name, t['early'], t['late'], t['slope'], t['vol'], t['worst'], t['collapses'], t['peak']))


if __name__ == '__main__':
    main()
