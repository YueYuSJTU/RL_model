import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

eval_result = """
============================================================
=== 两两对战统计矩阵信息 (用于热力图) ===
============================================================
[lstm] VS [now_best]
  胜率 (Win) : 21.50%
  败率 (Loss): 1.80%
  平局 (Draw): 76.50%
  平均自身HP : 2.46
  平均对手HP : 1.34
----------------------------------------
[lstm] VS [no_curri_have_pool_fix]
  胜率 (Win) : 27.20%
  败率 (Loss): 2.30%
  平局 (Draw): 70.40%
  平均自身HP : 2.42
  平均对手HP : 1.17
----------------------------------------
[lstm] VS [have_curri_no_pool]
  胜率 (Win) : 13.10%
  败率 (Loss): 2.70%
  平局 (Draw): 84.20%
  平均自身HP : 2.58
  平均对手HP : 1.56
----------------------------------------
[lstm] VS [20250616(no-pool-have-curri)]
  胜率 (Win) : 22.30%
  败率 (Loss): 0.40%
  平局 (Draw): 76.90%
  平均自身HP : 2.84
  平均对手HP : 1.21
----------------------------------------
[now_best] VS [lstm]
  胜率 (Win) : 2.70%
  败率 (Loss): 25.40%
  平局 (Draw): 71.90%
  平均自身HP : 1.27
  平均对手HP : 2.36
----------------------------------------
[now_best] VS [no_curri_have_pool_fix]
  胜率 (Win) : 18.00%
  败率 (Loss): 6.70%
  平局 (Draw): 75.30%
  平均自身HP : 2.07
  平均对手HP : 1.77
----------------------------------------
[now_best] VS [have_curri_no_pool]
  胜率 (Win) : 13.90%
  败率 (Loss): 11.40%
  平局 (Draw): 74.70%
  平均自身HP : 2.01
  平均对手HP : 1.80
----------------------------------------
[now_best] VS [20250616(no-pool-have-curri)]
  胜率 (Win) : 12.20%
  败率 (Loss): 0.60%
  平局 (Draw): 87.20%
  平均自身HP : 2.74
  平均对手HP : 1.69
----------------------------------------
[no_curri_have_pool_fix] VS [lstm]
  胜率 (Win) : 2.40%
  败率 (Loss): 27.60%
  平局 (Draw): 70.00%
  平均自身HP : 1.20
  平均对手HP : 2.40
----------------------------------------
[no_curri_have_pool_fix] VS [now_best]
  胜率 (Win) : 8.00%
  败率 (Loss): 18.10%
  平局 (Draw): 73.90%
  平均自身HP : 1.78
  平均对手HP : 2.09
----------------------------------------
[no_curri_have_pool_fix] VS [have_curri_no_pool]
  胜率 (Win) : 6.30%
  败率 (Loss): 13.60%
  平局 (Draw): 80.10%
  平均自身HP : 1.83
  平均对手HP : 2.14
----------------------------------------
[no_curri_have_pool_fix] VS [20250616(no-pool-have-curri)]
  胜率 (Win) : 9.70%
  败率 (Loss): 1.40%
  平局 (Draw): 88.80%
  平均自身HP : 2.74
  平均对手HP : 1.77
----------------------------------------
[have_curri_no_pool] VS [lstm]
  胜率 (Win) : 2.50%
  败率 (Loss): 12.90%
  平局 (Draw): 84.60%
  平均自身HP : 1.56
  平均对手HP : 2.53
----------------------------------------
[have_curri_no_pool] VS [now_best]
  胜率 (Win) : 11.60%
  败率 (Loss): 10.90%
  平局 (Draw): 77.50%
  平均自身HP : 1.93
  平均对手HP : 1.92
----------------------------------------
[have_curri_no_pool] VS [no_curri_have_pool_fix]
  胜率 (Win) : 12.50%
  败率 (Loss): 7.40%
  平局 (Draw): 80.10%
  平均自身HP : 2.07
  平均对手HP : 1.81
----------------------------------------
[have_curri_no_pool] VS [20250616(no-pool-have-curri)]
  胜率 (Win) : 15.30%
  败率 (Loss): 0.40%
  平局 (Draw): 84.30%
  平均自身HP : 2.78
  平均对手HP : 1.55
----------------------------------------
[20250616(no-pool-have-curri)] VS [lstm]
  胜率 (Win) : 0.10%
  败率 (Loss): 23.40%
  平局 (Draw): 76.50%
  平均自身HP : 1.15
  平均对手HP : 2.89
----------------------------------------
[20250616(no-pool-have-curri)] VS [now_best]
  胜率 (Win) : 1.20%
  败率 (Loss): 15.90%
  平局 (Draw): 82.90%
  平均自身HP : 1.63
  平均对手HP : 2.77
----------------------------------------
[20250616(no-pool-have-curri)] VS [no_curri_have_pool_fix]
  胜率 (Win) : 3.50%
  败率 (Loss): 10.00%
  平局 (Draw): 86.50%
  平均自身HP : 1.78
  平均对手HP : 2.61
----------------------------------------
[20250616(no-pool-have-curri)] VS [have_curri_no_pool]
  胜率 (Win) : 0.50%
  败率 (Loss): 11.20%
  平局 (Draw): 88.30%
  平均自身HP : 1.78
  平均对手HP : 2.73
----------------------------------------
"""

def parse_heatmap_data(text):
    """解析文本提取跨局胜率与HP信息"""
    parsed_data = {}
    models_set = set()
    
    pattern = r'\[(.*?)\] VS \[(.*?)\]\s+胜率 \(Win\) : ([\d\.]+)%\s+败率 \(Loss\): ([\d\.]+)%\s+平局 \(Draw\): ([\d\.]+)%\s+平均自身HP : ([\d\.]+)\s+平均对手HP : ([\d\.]+)'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        model_a = match.group(1).strip()
        model_b = match.group(2).strip()
        win_rate = float(match.group(3))
        self_hp = float(match.group(6))
        oppo_hp = float(match.group(7))
        
        hp_diff = self_hp - oppo_hp
        
        if model_a not in parsed_data:
            parsed_data[model_a] = {}
        parsed_data[model_a][model_b] = {
            "Win Rate": win_rate,
            "HP Diff": hp_diff
        }
        models_set.add(model_a)
        models_set.add(model_b)
        
    # 为保证模型顺序固定，进行排序
    models = sorted(list(models_set))
    return parsed_data, models[::-1]

def plot_crossplay_heatmap(data_tuple, save_path="src/evaluation/plot/crossplay_heatmap.png"):
    data_dict, original_models = data_tuple
    
    # --- 自定义模型展示顺序 (需更改顺序请调整此列表) ---
    target_order = [
        "lstm",
        "now_best",
        "have_curri_no_pool",
        "no_curri_have_pool_fix",
        "20250616(no-pool-have-curri)",
    ]
    # 按 target_order 排序，若有未指定的模型则追加到末尾
    models = [m for m in target_order if m in original_models]
    models += [m for m in original_models if m not in target_order]
    
    # --- 自定义模型名称映射 (需更改显示名称请调整此字典) ---
    # name_map = {
    #     "now_best": "Full-MLP",
    #     "have_curri_no_pool": " Baseline + Curriculum",
    #     "no_curri_have_pool_fix": "Baseline + Dynamic Pool",
    #     "20250616(no-pool-have-curri)": "Baseline",
    #     "lstm": "Full-LSTM"
    # }
    name_map = {
        "now_best": "BDC(MLP)",
        "have_curri_no_pool": "BC",
        "no_curri_have_pool_fix": "BD",
        "20250616(no-pool-have-curri)": "Baseline",
        "lstm": "BDC(LSTM)"
    }
    
    # ======= 科研图表样式配置 =======
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })
    
    n = len(models)
    # 构造数据矩阵 (NxN)
    win_rates = np.zeros((n, n))
    hp_diffs = np.zeros((n, n))
    annotations = np.empty((n, n), dtype=object)
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if m1 == m2:
                # 自己对战自己：胜率标记为 -，hp 差值为 0
                win_rates[i, j] = np.nan
                hp_diffs[i, j] = 0.0
                annotations[i, j] = "-\n$\\Delta$HP: 0.00"
            elif m2 not in data_dict.get(m1, {}):
                win_rates[i, j] = np.nan
                hp_diffs[i, j] = np.nan
                annotations[i, j] = "N/A"
            else:
                wr = data_dict[m1][m2]["Win Rate"] / 100.0
                hd = data_dict[m1][m2]["HP Diff"]
                win_rates[i, j] = wr
                hp_diffs[i, j] = hd
                annotations[i, j] = f"{wr*100:.1f}%\n$\\Delta$HP: {hd:+.2f}"
    
    # 绘图基础设置
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 获取最大HP差距的绝对值，使色彩条以0为中心对称
    max_hp_diff = np.nanmax(np.abs(hp_diffs)) if not np.all(np.isnan(hp_diffs)) else 1.0
    
    # 改为红蓝对比色 (蓝色代表正收益，红色代表负收益)
    cmap = "RdBu"
    
    # 以 hp_diffs 为底色进行热力图绘制，调小方格线宽度 (linewidths)
    # 增加 annot_kws 的 fontsize 使文本字体变大，占满整个方格
    sns.heatmap(hp_diffs, annot=annotations, fmt="", cmap=cmap, center=0, cbar=True,
                cbar_kws={'label': r'$\Delta$ HP'}, linewidths=0.5, linecolor='black',
                ax=ax, vmin=-max_hp_diff, vmax=max_hp_diff, annot_kws={"fontsize": 13, "fontweight": "bold"},
                mask=np.isnan(hp_diffs))
    
    # # 遍历所有文本对象，将delta HP为负数的文本标为红色
    # for t in ax.texts:
    #     if "HP: -" in t.get_text():
    #         t.set_color("red")
            
    # 格式化过长的模型名称，以便折行显示 (应用名称映射字典)
    formatted_models = [name_map.get(m, m).replace("(", "\n(") for m in models]

    # 坐标轴修饰
    ax.set_yticklabels(formatted_models, rotation=0, fontweight='bold')
    ax.set_xticklabels(formatted_models, rotation=0, ha='center', fontweight='bold')
    ax.set_title("Cross-play Matrix Evaluation", fontweight='bold', pad=15)
    ax.set_ylabel("Agent", fontweight='bold')
    ax.set_xlabel("Opponent", fontweight='bold')
    
    # 加粗外边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    print(f"Heatmap plotted successfully to {save_path}")

if __name__ == "__main__":
    data_tuple = parse_heatmap_data(eval_result)
    plot_crossplay_heatmap(data_tuple)
