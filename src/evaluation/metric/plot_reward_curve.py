import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def smooth_curve(scalars, weight=0.85):
    """用于平滑曲线的指数移动平均算法"""
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    # ================= 1. 硬编码配置区域 =================
    # 定义需要读取的 Tensorboard 日志路径及其对应的图例标签
    # 路径请指向具体的 events.out.tfevents... 文件或其所在目录
    log_paths = {
        "BDC": "experiments/20260428_141850_best_train_mlp2/tensorboard/20260428_141850/PPO_1",
        "BD": "experiments/20260501_210431_best_train_mlp2_no_curri_have_pool/tensorboard/battle/PPO_0",
        "Baseline": "experiments/20250616_221656/tensorboard/20250616_221656/example_tensorboard",
    }
    
    # 需要提取的指标名称（根据你的 SB3 或自定义 Tensorboard 日志确定，如 rollout/ep_rew_mean）
    tag_name = "rollout/ep_rew_mean"
    
    # 图表保存路径
    save_path = "src/evaluation/plot/reward_curve_plot.png" # 也可以改为 .pdf 或 .svg 以获无损矢量图
    
    # 平滑系数 (0 到 1 之间，越大越平滑)
    smooth_weight = 0.95
    
    # 最大显示的 timestep 限制（例如限制在 3 亿步）
    max_steps = 3e8
    # =====================================================

    # 配置科研图表样式
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5)
    })

    fig, ax = plt.subplots()

    # 将原有 husl 颜色空间替换为需要的蓝、红、绿配色组合
    # 蓝色: #307bb5, 红色: #c43f3c, 绿色: #4f9b6b
    colors = ['#307bb5', '#c43f3c', '#4f9b6b']
    
    for idx, (label, log_path) in enumerate(log_paths.items()):
        # 加载 Tensorboard 数据
        print(f"Loading {label} from {log_path}...")
        event_acc = EventAccumulator(log_path, size_guidance={'scalars': 0})
        event_acc.Reload()

        try:
            scalars = event_acc.Scalars(tag_name)
        except KeyError:
            print(f"Warning: Tag '{tag_name}' not found in {log_path}. Skipping.")
            continue

        # 过滤超出 max_steps 阈值的数据
        filtered_scalars = [s for s in scalars if s.step <= max_steps]
        
        steps = [s.step for s in filtered_scalars]
        vals = [s.value for s in filtered_scalars]
        
        if not steps:
            print(f"Warning: No data found within max_steps limits for {label}.")
            continue
            
        smoothed_vals = smooth_curve(vals, weight=smooth_weight)

        # 循环使用我们给定的颜色列表，以防越界
        color = colors[idx % len(colors)]
        
        # 绘制半透明的原始波动曲线
        # ax.plot(steps, vals, color=color, alpha=0.3, linewidth=1)
        # 绘制平滑后的主曲线
        ax.plot(steps, smoothed_vals, color=color, label=label, linewidth=2)

    # 修饰图表
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title("Training Curve Comparison")
    ax.set_ylim(-300, None)
    
    # 科学计数法显示X轴
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # 去除顶部和右侧多余的边框线
    sns.despine()

    # 图例设置
    ax.legend(loc="lower right", frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    main()
