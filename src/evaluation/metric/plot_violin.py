import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

# ==========================================
# 1. 全局字体与样式设置 (完美复刻学术风)
# ==========================================
# 设置为衬线字体 (Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 2. 读取并提取数据
# ==========================================
data_path = 'src/evaluation/plot/results_violin.json'  # 请替换为您的真实JSON文件路径

with open(data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

lstm_hp, mlp_hp = [], []
lstm_energy, mlp_energy = [], []
lstm_wins, mlp_wins = [], []  

for episode in raw_data:
    if "Ours-LSTM" in episode and "Ours-MLP" in episode:
        lstm_hp.append(episode["Ours-LSTM"]["hp"])
        mlp_hp.append(episode["Ours-MLP"]["hp"])
        
        lstm_energy.append(episode["Ours-LSTM"]["specific_energy_mean"])
        mlp_energy.append(episode["Ours-MLP"]["specific_energy_mean"])
        
        # 提取 win_status
        lstm_wins.append(episode["Ours-LSTM"]["win_status"])
        mlp_wins.append(episode["Ours-MLP"]["win_status"])

n_valid = len(lstm_hp)

if n_valid == 0:
    raise ValueError("未找到有效数据，请检查 JSON 格式和模型名称。")

# ==========================================
# 3. 计算各项指标，并分别生成多行标签
# ==========================================
# 计算胜率 (获胜场次 / 总场次)
lstm_win_rate = sum(1 for w in lstm_wins if w == 1) / n_valid * 100
mlp_win_rate = sum(1 for w in mlp_wins if w == 1) / n_valid * 100

# 计算平均剩余血量
lstm_avg_hp = np.mean(lstm_hp)
mlp_avg_hp = np.mean(mlp_hp)

# 计算平均特定能量
lstm_avg_energy = np.mean(lstm_energy)
mlp_avg_energy = np.mean(mlp_energy)

# ==========================================
# 构建统一的汇总标签，包含Win Rate, HP, Energy
# ==========================================
lstm_label = f"BDC(LSTM)\nWin Rate: {lstm_win_rate:.1f}%\nAvg HP: {lstm_avg_hp:.1f}\nAvg Energy: {lstm_avg_energy:.2f}"
mlp_label = f"BDC(MLP)\nWin Rate: {mlp_win_rate:.1f}%\nAvg HP: {mlp_avg_hp:.1f}\nAvg Energy: {mlp_avg_energy:.2f}"

# 将新标签赋给 DataFrame
df_hp = pd.DataFrame({
    'Architecture': [lstm_label] * n_valid + [mlp_label] * n_valid,
    'Value': lstm_hp + mlp_hp,
    'Metric': ['HP'] * (n_valid * 2) 
})

df_energy = pd.DataFrame({
    'Architecture': [lstm_label] * n_valid + [mlp_label] * n_valid,
    'Value': lstm_energy + mlp_energy,
    'Metric': ['Mean Specific Energy'] * (n_valid * 2)
})

# ==========================================
# 4. 绘图函数
# ==========================================
import matplotlib.patches as mpatches

def draw_violin_half(df, ax, order):
    # 提取原图的颜色 (蓝、橙褐)
    colors = ['#307bb5', '#c43f3c']
    palette = sns.color_palette(colors)

    # --- 图层 1: 底层散点 (Strip Plot) ---
    sns.stripplot(
        data=df, x='Metric', y='Value',
        hue='Architecture', palette=palette, legend=False,
        dodge=True, alpha=0.03, jitter=0.1, size=10.5,
        ax=ax, order=order, zorder=1 
    )

    # 手动调整散点，使其向各列柱的中心轴聚拢
    shift_val = 0.13
    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PathCollection):
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                base_x = np.round(offsets[:, 0]) # 获得对应的列基准 x（0 或 1）
                offsets[:, 0] = np.where(offsets[:, 0] < base_x, offsets[:, 0] + shift_val, offsets[:, 0] - shift_val)
                collection.set_offsets(offsets)

    # --- 图层 2: 中层小提琴图 (Violin Plot) ---
    sns.violinplot(
        data=df, x='Metric', y='Value',
        hue='Architecture', split=True, legend=False,
        inner=None, linewidth=1.0, ax=ax, order=order, zorder=2, color='white'
    )
    for collection in ax.collections:
        if isinstance(collection, plt.matplotlib.collections.PolyCollection):
            collection.set_facecolor('none')  # 内部透明
            collection.set_edgecolor('#404040') # 边缘颜色

    # --- 图层 3: 顶层箱线图 (Box Plot) ---
    sns.boxplot(
        data=df, x='Metric', y='Value',
        hue='Architecture', dodge=True, legend=False,
        width=0.25, fliersize=0, showcaps=True,
        boxprops={'facecolor': 'none', 'edgecolor': '#404040', 'linewidth': 1.2, 'zorder': 3},
        whiskerprops={'color': '#404040', 'linewidth': 1.2, 'zorder': 3},
        capprops={'color': '#404040', 'linewidth': 1.2, 'zorder': 3},
        medianprops={'color': '#404040', 'linewidth': 1.2, 'zorder': 3},
        ax=ax, order=order
    )


# 创建单张画布，采用双轴 (twinx)
fig, ax1 = plt.subplots(figsize=(8.5, 7.5), dpi=300)
ax2 = ax1.twinx()

order = ['HP', 'Mean Specific Energy']

# 用 ax1 绘制左轴 (HP)，ax2 绘制右轴 (Energy)
draw_violin_half(df_hp, ax1, order)
draw_violin_half(df_energy, ax2, order)

# ==========================================
# 5. 坐标轴、网格与细节调整
# ==========================================
ax1.set_xlabel('', fontsize=15, fontweight='bold')
ax1.set_ylabel('HP', fontsize=15, fontweight='bold', labelpad=12)
ax2.set_ylabel('Mean Specific Energy', fontsize=15, fontweight='bold', labelpad=12)

ax1.tick_params(axis='both', which='major', labelsize=14, direction='out', length=4)
ax2.tick_params(axis='both', which='major', labelsize=14, direction='out', length=4)

# 仅左侧轴保留网格线，防止两边刻度不同导致网格杂乱
ax1.yaxis.grid(True, linestyle='-', which='major', color='#d3d3d3', alpha=0.8, linewidth=1.2)
ax2.yaxis.grid(False)
ax1.xaxis.grid(False) 
ax1.set_axisbelow(True) 

for ax in [ax1, ax2]:
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)

# 手动添加图例，并放置于图片顶部中央避免遮挡
colors = ['#307bb5', '#c43f3c']
patch1 = mpatches.Patch(color=colors[0], label=lstm_label)
patch2 = mpatches.Patch(color=colors[1], label=mlp_label)
ax1.legend(handles=[patch1, patch2], title='Architecture Status', fontsize=15, 
           title_fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.33), ncol=2)

plt.tight_layout()

# 合成保存
save_path = 'src/evaluation/plot/violin_combined.png'
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.close()

print(f"绘制完成！图片已保存至 {save_path}")