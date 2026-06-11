import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 全局字体与样式设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 

def main():
    # ==========================================
    # 2. 读取并提取数据
    # ==========================================
    data_dir = "/home/ubuntu/Workfile/RL/RL_model/src/evaluation/player_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"在 {data_dir} 中未找到 CSV 文件。")

    all_data = []
    for file in csv_files:
        df_temp = pd.read_csv(file)
        # 提取相关列，包含胜负标志和HP以用于统计
        df_temp = df_temp[['Role', 'Tracking_Angle_Mean', 'Win_Flag', 'HP']]
        all_data.append(df_temp)
        
    df = pd.concat(all_data, ignore_index=True)
    
    if df.empty:
        raise ValueError("提取的数据为空，请检查 CSV 内容。")

    # ==========================================
    # 2.5 计算各项指标，并分别生成多行标签
    # ==========================================
    human_data = df[df['Role'] == 'Human']
    agent_data = df[df['Role'] == 'Agent']
    
    human_win_rate = human_data['Win_Flag'].mean() * 100 if not human_data.empty else 0
    human_avg_hp = human_data['HP'].mean() if not human_data.empty else 0
    
    agent_win_rate = agent_data['Win_Flag'].mean() * 100 if not agent_data.empty else 0
    agent_avg_hp = agent_data['HP'].mean() if not agent_data.empty else 0
    
    human_label = f"Human\nAvg HP: {human_avg_hp:.3f}"
    agent_label = f"Agent\nAvg HP: {agent_avg_hp:.3f}"
    
    # 替换 DataFrame 中的 Role 以用于后续绘图的 X 轴标签
    df['Role'] = df['Role'].replace({'Human': human_label, 'Agent': agent_label})

    # ==========================================
    # 3. 绘图
    # ==========================================
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300) 
    colors = ['#307bb5', '#c43f3c'] # 修改颜色为蓝色和红色
    palette = sns.color_palette(colors)

    # 数据重塑以适应图表中展示两个指标 (HP 和 Tracking_Angle_Mean)
    df_melt = df.melt(id_vars='Role', value_vars=['HP', 'Tracking_Angle_Mean'], 
                      var_name='Metric', value_name='Value')
    df_melt['Metric'] = df_melt['Metric'].replace({'HP': 'Remain HP', 'Tracking_Angle_Mean': 'Track Angle'})

    # 绘制分组箱线图 (仅保留此图层)
    sns.boxplot(
        data=df_melt, x='Metric', y='Value', hue='Role',
        hue_order=[agent_label, human_label],  # 确保 Agent 排在前面
        palette=palette,
        width=0.45, fliersize=4, showcaps=True,
        boxprops={'linewidth': 1.5, 'zorder': 3},
        whiskerprops={'color': '#404040', 'linewidth': 1.5, 'zorder': 3},
        capprops={'color': '#404040', 'linewidth': 1.5, 'zorder': 3},
        medianprops={'color': 'black', 'linewidth': 2.0, 'zorder': 4}, # 改为红色加粗以突出代表中位数的横线
        ax=ax
    )

    # ==========================================
    # 4. 坐标轴与细节调整
    # ==========================================
    ax.set_xlabel('Evaluation Metric', fontsize=14, fontweight='bold', labelpad=12)
    ax.set_ylabel('Value', fontsize=14, fontweight='bold', labelpad=12)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='out', length=4)
    ax.yaxis.grid(True, linestyle='-', which='major', color='#d3d3d3', alpha=0.8, linewidth=1.2)
    ax.xaxis.grid(False) 
    ax.set_axisbelow(True) 

    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1.5)
        
    ax.legend(title='Role', fontsize=11, title_fontsize=12, loc='lower right')

    plt.tight_layout()

    # ==========================================
    # 5. 保存图片
    # ==========================================
    save_dir = "/home/ubuntu/Workfile/RL/RL_model/src/evaluation/plot"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "human_vs_agent.png")
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"绘制完成！图片已保存至 {save_path}")

if __name__ == "__main__":
    main()
