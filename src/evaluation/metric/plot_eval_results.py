import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

eval_result = """
========================================
=== Benchmark 最终统计结果 ===
========================================
目标模型: BDC-(LSTM)
  总对战场次 : 8000
  +----------------------------------------+
  | 综合胜率   :          55.59%           |
  | 平均自身HP :           2.91            |
  | 平均对手HP :           0.75            |
  +----------------------------------------+
  --- 详细指标 ---
  draw_rate                      : 0.4430
  loss_rate                      : 0.0003
  opponent_fall_rate             : 0.0009
  avg_win_time                   : 283.4012
  avg_reward                     : 748.3967
  gun_opportunity_time_ratio     : 0.1908
  damage_to_oppo_total           : 2.2501
  damage_rate                    : 0.0000
  track_angle_mean               : 0.6960
  adverse_angle_mean             : 0.0000
  overshoot_time_ratio           : 0.0072
  specific_energy_mean           : 22401.6497
  specific_energy_final          : 19179.5225
  gun_opportunity_time_ratio_oppo : 0.0033
  damage_to_self_total           : 0.0921
  damage_rate_oppo               : 0.0000
  track_angle_mean_oppo          : 2.1982
  adverse_angle_mean_oppo        : 0.0000
  specific_energy_mean_oppo      : 18629.0129
  specific_energy_final_oppo     : 14305.3351
----------------------------------------
目标模型: BDC-(MLP)
  总对战场次 : 8000
  +----------------------------------------+
  | 综合胜率   :          40.98%           |
  | 平均自身HP :           2.80            |
  | 平均对手HP :           0.97            |
  +----------------------------------------+
  --- 详细指标 ---
  draw_rate                      : 0.5885
  loss_rate                      : 0.0018
  opponent_fall_rate             : 0.0000
  avg_win_time                   : 284.5975
  avg_reward                     : 549.1157
  gun_opportunity_time_ratio     : 0.1598
  damage_to_oppo_total           : 2.0258
  damage_rate                    : 0.0000
  track_angle_mean               : 0.7577
  adverse_angle_mean             : 0.0000
  overshoot_time_ratio           : 0.0093
  specific_energy_mean           : 20177.4209
  specific_energy_final          : 18514.7887
  gun_opportunity_time_ratio_oppo : 0.0063
  damage_to_self_total           : 0.1959
  damage_rate_oppo               : 0.0000
  track_angle_mean_oppo          : 2.0907
  adverse_angle_mean_oppo        : 0.0000
  specific_energy_mean_oppo      : 18178.6938
  specific_energy_final_oppo     : 13858.9636
----------------------------------------
目标模型: BD
  总对战场次 : 8000
  +----------------------------------------+
  | 综合胜率   :          33.29%           |
  | 平均自身HP :           2.89            |
  | 平均对手HP :           1.28            |
  +----------------------------------------+
  --- 详细指标 ---
  draw_rate                      : 0.6647
  loss_rate                      : 0.0019
  opponent_fall_rate             : 0.0005
  avg_win_time                   : 368.7916
  avg_reward                     : -28.9097
  gun_opportunity_time_ratio     : 0.1310
  damage_to_oppo_total           : 1.7196
  damage_rate                    : 0.0000
  track_angle_mean               : 0.7208
  adverse_angle_mean             : 0.0000
  overshoot_time_ratio           : 0.0040
  specific_energy_mean           : 18458.2136
  specific_energy_final          : 16077.2718
  gun_opportunity_time_ratio_oppo : 0.0039
  damage_to_self_total           : 0.1055
  damage_rate_oppo               : 0.0000
  track_angle_mean_oppo          : 2.1714
  adverse_angle_mean_oppo        : 0.0000
  specific_energy_mean_oppo      : 18004.1533
  specific_energy_final_oppo     : 14009.3324
----------------------------------------
目标模型: BC
  总对战场次 : 8000
  +----------------------------------------+
  | 综合胜率   :          22.44%           |
  | 平均自身HP :           2.89            |
  | 平均对手HP :           1.67            |
  +----------------------------------------+
  --- 详细指标 ---
  draw_rate                      : 0.7742
  loss_rate                      : 0.0006
  opponent_fall_rate             : 0.0008
  avg_win_time                   : 374.1632
  avg_reward                     : 361.8250
  gun_opportunity_time_ratio     : 0.0906
  damage_to_oppo_total           : 1.3257
  damage_rate                    : 0.0000
  track_angle_mean               : 0.7390
  adverse_angle_mean             : 0.0000
  overshoot_time_ratio           : 0.0034
  specific_energy_mean           : 18231.7844
  specific_energy_final          : 16222.4672
  gun_opportunity_time_ratio_oppo : 0.0039
  damage_to_self_total           : 0.1072
  damage_rate_oppo               : 0.0000
  track_angle_mean_oppo          : 2.2012
  adverse_angle_mean_oppo        : 0.0000
  specific_energy_mean_oppo      : 17955.5061
  specific_energy_final_oppo     : 13730.5838
----------------------------------------
目标模型: Baseline
  总对战场次 : 8000
  +----------------------------------------+
  | 综合胜率   :           0.39%           |
  | 平均自身HP :           2.94            |
  | 平均对手HP :           2.86            |
  +----------------------------------------+
  --- 详细指标 ---
  draw_rate                      : 0.9954
  loss_rate                      : 0.0008
  opponent_fall_rate             : 0.0000
  avg_win_time                   : 166.6981
  avg_reward                     : 35.3539
  gun_opportunity_time_ratio     : 0.0092
  damage_to_oppo_total           : 0.1449
  damage_rate                    : 0.0000
  track_angle_mean               : 0.8136
  adverse_angle_mean             : 0.0000
  overshoot_time_ratio           : 0.0029
  specific_energy_mean           : 13401.3072
  specific_energy_final          : 9071.4529
  gun_opportunity_time_ratio_oppo : 0.0031
  damage_to_self_total           : 0.0587
  damage_rate_oppo               : 0.0000
  track_angle_mean_oppo          : 2.1289
  adverse_angle_mean_oppo        : 0.0000
  specific_energy_mean_oppo      : 17622.9384
  specific_energy_final_oppo     : 13637.5766
----------------------------------------
"""

def parse_eval_text(text):
    """通过正则表达式将文本解析为字典"""
    parsed_data = {}
    
    # 根据模型分割文本块
    blocks = re.split(r'模型:\s+', text)[1:]
    
    for block in blocks:
        lines = block.strip().split('\n')
        model_name = lines[0].strip()
        
        # 提取胜负平等关键率
        win_match = re.search(r'综合胜率\s*:\s*([\d\.]+)%', block)
        win_rate = float(win_match.group(1)) / 100.0 if win_match else 0.0
        
        draw_match = re.search(r'draw_rate\s*:\s*([\d\.]+)', block)
        draw_rate = float(draw_match.group(1)) if draw_match else 0.0
        
        loss_match = re.search(r'loss_rate\s*:\s*([\d\.]+)', block)
        loss_rate = float(loss_match.group(1)) if loss_match else 0.0
        
        # 提取额外详细指标（注意 \s+: 强制匹配空格，避免匹配到 _oppo 的数据）
        gun_opp_match = re.search(r'gun_opportunity_time_ratio\s+:\s*([\d\.]+)', block)
        spec_energy_match = re.search(r'specific_energy_mean\s+:\s*([\d\.]+)', block)
        track_angle_match = re.search(r'track_angle_mean\s+:\s*([\d\.]+)', block)
        # 修改：匹配 damage_to_oppo_total
        damage_oppo_match = re.search(r'damage_to_oppo_total\s+:\s*([\d\.]+)', block)
        
        parsed_data[model_name] = {
            "Win Rate": win_rate,
            "Draw Rate": draw_rate,
            "Loss Rate": loss_rate,
            "Gun Opportunity": float(gun_opp_match.group(1)) if gun_opp_match else 0.0,
            "Specific Energy": float(spec_energy_match.group(1)) if spec_energy_match else 0.0,
            "Track Angle": float(track_angle_match.group(1)) if track_angle_match else 0.0,
            # 修改：将 Overshoot Ratio 改为 Damage to Oppo
            "Damage to Oppo": float(damage_oppo_match.group(1)) if damage_oppo_match else 0.0
        }
        
    return parsed_data

def plot_win_rates(data_dict, save_path="model_win_rates.png"):
    # ======= 科研图表样式配置 =======
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 11,
        "hatch.linewidth": 0.5,
    })
    
    models = list(data_dict.keys())[::-1]
    
    wrapped_models = []
    for m in models:
        # 如果包含 '+'（例如 Baseline + Dynamic Pool），在 '+' 后面换行
        if '-' in m:
            m_wrapped = m.replace('-', ' \n')
        # 如果名字较长且存在空格，则挑一个主要空格进行折行
        elif len(m) > 12 and ' ' in m:
            parts = m.split(' ', 1)
            m_wrapped = f"{parts[0]}\n{parts[1]}"
        else:
            m_wrapped = m
        wrapped_models.append(m_wrapped)
        
    win_rates = [data_dict[m]["Win Rate"] * 100 for m in models]
    draw_rates = [data_dict[m]["Draw Rate"] * 100 for m in models]
    loss_rates = [data_dict[m]["Loss Rate"] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # 学术经典配色: 沉稳蓝(胜), 纯净灰(平), 砖石红(负)
    colors = ['#307bb5', '#EAEAEA', '#c43f3c'] 

    # 获取堆叠的基准位置
    y_pos = np.arange(len(models))
    height = 0.55

    # 为了更好的学术质感，边框线加深并增加细微阴影效果
    p1 = ax.barh(y_pos, win_rates, height, color=colors[0], edgecolor="black", linewidth=1.2)
    p2 = ax.barh(y_pos, draw_rates, height, left=win_rates, color=colors[1], edgecolor="black", linewidth=1.2)
    p3 = ax.barh(y_pos, loss_rates, height, left=np.add(win_rates, draw_rates), color=colors[2], edgecolor="black", linewidth=1.2)

    # 在条形图内直接添加数值标签 (当比例足够大时才显示，避免拥挤)
    for idx, (w, d, l) in enumerate(zip(win_rates, draw_rates, loss_rates)):
        if w > 3.0:
            ax.text(w / 2, idx, f"{w:.1f}%", va='center', ha='center', color='white', fontweight='bold', fontsize=10, fontname='Times New Roman')
        if d > 5.0:
            ax.text(w + d / 2, idx, f"{d:.1f}%", va='center', ha='center', color='black', fontsize=10, fontname='Times New Roman')
        
        # 将 Loss 的比例统一写到条形的右侧外部，不论多大都展示，以便看清楚小的 Loss
        total_len = w + d + l
        ax.text(total_len + 1.5, idx, f"Loss: {l:.2f}%", va='center', ha='left', color='black', fontweight='bold', fontsize=15, fontname='Times New Roman')

    # 图表修饰
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped_models, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter())
    
    # 添加轻量级垂直网格线，并将网格置于底层
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
    ax.set_axisbelow(True)
    
    ax.set_xlabel('Match Percentage Indicator (%)', fontweight='bold')
    ax.set_title('Evaluation Outcomes per Model in Benchmark Environment', fontweight='bold', pad=15)
    
    # 将图例居中放置在图表上方或下方，更符合双栏论文排版
    ax.legend([p1, p2, p3], ['Win', 'Draw', 'Loss'], loc='upper center', 
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, 
              handlelength=1.5, handleheight=1.5)
    
    # 仅保留左和下边框线，加粗坐标轴边框
    sns.despine(top=True, right=True)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    print(f"Plot saved successfully to {save_path}")

def plot_capability_metrics(data_dict, save_path="src/evaluation/plot/model_capabilities.png"):
    """绘制 2x2 网格结构的详细指标对比图"""
    # 统一使用 tick 样式，与图1一致
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "hatch.linewidth": 0.5,
    })
    
    models = list(data_dict.keys())
    # 提取四个绘图用的指标列表
    metrics = {
        "Gun Opp. Ratio (↑ Better) (%)": [data_dict[m]["Gun Opportunity"] * 100 for m in models],
        "Mean Specific Energy (↑ Better) (ft)": [data_dict[m]["Specific Energy"] for m in models],
        "Mean Track Angle (↓ Better) (rad)": [data_dict[m]["Track Angle"] for m in models],
        # 修改：将 Overshoot 改为 Total Damage to Oppo（不乘以 100，因为它是总伤害值）
        "Total Damage to Oppo (↑ Better)": [data_dict[m]["Damage to Oppo"] for m in models]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # 统一将所有条形图设置为蓝色
    bar_color = '#307bb5'
    x_pos = np.arange(len(models))
    width = 0.5
    
    # 获取支持多行展示的模型名称（用换行符替换某些空格或长文本中的截断）
    # 如果字符串较长，根据空格或特征符号将其分成两行
    wrapped_models = []
    for m in models:
        # 如果包含 '+'（例如 Baseline + Dynamic Pool），在 '+' 后面换行
        if '-' in m:
            m_wrapped = m.replace('-', ' \n')
        # 如果名字较长且存在空格，则挑一个主要空格进行折行
        elif len(m) > 12 and ' ' in m:
            parts = m.split(' ', 1)
            m_wrapped = f"{parts[0]}\n{parts[1]}"
        else:
            m_wrapped = m
        wrapped_models.append(m_wrapped)

    for i, (title, vals) in enumerate(metrics.items()):
        ax = axes[i]
        # 使用统一蓝色，并保持细边框
        bars = ax.bar(x_pos, vals, width, color=bar_color, edgecolor="black", linewidth=0.5)
        
        # 略微增加标题和图之间的距离，修改 pad 为 15
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        # 旋转角度调整为 0，因为多行文本不需要太大倾斜，或者保持极小倾斜角
        ax.set_xticklabels(wrapped_models, rotation=0, ha='center', fontweight='bold')
        
        # 添加在柱形上的数值标注
        for bar in bars:
            height = bar.get_height()
            # 动态判断标注格式：针对极大或极小的差别处理
            fmt = f"{height:.1f}" if height > 10 else f"{height:.3f}"
            ax.annotate(fmt,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=13, fontname='Times New Roman', fontweight='bold')
        
        # 统一的网格与坐标轴边框样式（与图1完全对齐）
        ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='grey')
        ax.set_axisbelow(True)
        
        # 去除多余边框且将保留的边框线加粗
        sns.despine(ax=ax, top=True, right=True)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    print(f"Capability plots saved successfully to {save_path}")

def main():
    parsed_data = parse_eval_text(eval_result)
    
    # 打印解析出来的字典，以备验证
    import pprint
    print("Parsed Data:")
    pprint.pprint(parsed_data)
    
    # 调用绘图逻辑
    plot_win_rates(parsed_data)
    plot_capability_metrics(parsed_data)

if __name__ == "__main__":
    main()