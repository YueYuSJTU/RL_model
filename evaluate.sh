#!/bin/bash

# 设置当前工作目录
cd "$(dirname "$0")"
# 使用source方式初始化conda并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate js

# 检查 experiments 目录是否存在
if [ ! -d "./experiments" ]; then
    echo "错误：./experiments 目录不存在"
    exit 1
fi

# 列出 experiments 目录下的所有文件夹
echo "可用的实验："
experiments=($(ls -d ./experiments/*/))
for i in "${!experiments[@]}"; do
    # 去掉路径前缀和尾部斜杠
    folder_name=$(basename "${experiments[$i]}")
    echo "[$i] $folder_name"
done

# 询问用户选择哪个实验
read -p "请选择要评估的实验 [0-$((${#experiments[@]}-1))]: " exp_idx

# 验证输入
if ! [[ "$exp_idx" =~ ^[0-9]+$ ]] || [ "$exp_idx" -ge "${#experiments[@]}" ]; then
    echo "错误：无效的选择"
    exit 1
fi

selected_exp="${experiments[$exp_idx]}"
selected_exp_name=$(basename "$selected_exp")

echo "已选择实验: $selected_exp_name"
echo ""

# 列出选定实验中的所有目录（可能包含stage）
all_dirs=($(ls -d "$selected_exp"/*/))
if [ ${#all_dirs[@]} -eq 0 ]; then
    echo "错误：所选实验中没有任何子文件夹"
    exit 1
fi

# 过滤出可能的stage文件夹（排除tensorboard等特殊文件夹）
stages=()
for dir in "${all_dirs[@]}"; do
    dir_name=$(basename "$dir")
    # 如果文件夹名不是tensorboard，则认为是stage
    if [ "$dir_name" != "tensorboard" ]; then
        stages+=("$dir")
    fi
done

if [ ${#stages[@]} -eq 0 ]; then
    echo "错误：所选实验中没有stage文件夹"
    exit 1
fi

# 找到编号最大的stage（按stage后面的数字排序）
latest_stage=""
largest_num=-1  # 从-1开始，确保至少有一个有效数字会被选择

for stage in "${stages[@]}"; do
    # 获取文件夹名
    stage_name=$(basename "$stage")
    
    # 提取stage后面的数字部分
    if [[ $stage_name =~ ^stage([0-9]+) ]]; then
        stage_num=${BASH_REMATCH[1]}
        
        # 如果这个数字更大，则更新最大值
        if (( stage_num > largest_num )); then
            largest_num=$stage_num
            latest_stage=$stage
        fi
    fi
done

# 确保找到了有效的stage
if [ -z "$latest_stage" ]; then
    echo "错误：无法从stage文件夹名中提取数字，无法确定最新stage"
    exit 1
fi

latest_stage_name=$(basename "$latest_stage")
echo "自动选择了最新的stage: $latest_stage_name"
echo ""

# 在最新的stage中查找训练结果
results=($(ls -d "$latest_stage"/*/))
if [ ${#results[@]} -eq 0 ]; then
    # 如果没有子文件夹，则使用当前stage目录
    echo "在此stage中未找到训练结果子文件夹，将使用stage目录本身"
    latest_result="$latest_stage"
    latest_result_name="."
else
    # 找到最新的训练结果（按文件夹修改时间排序）
    latest_result=""
    latest_result_time=0
    
    for result in "${results[@]}"; do
        # 获取文件夹的最后修改时间
        mod_time=$(stat -c %Y "$result")
        if (( mod_time > latest_result_time )); then
            latest_result_time=$mod_time
            latest_result=$result
        fi
    done
    
    latest_result_name=$(basename "$latest_result")
fi

echo "自动选择了最新的训练结果: $latest_result_name"
echo ""

# 询问用户评估次数
read -p "请输入评估次数 [默认: 500]: " n_episodes
n_episodes=${n_episodes:-500}  # 如果用户未输入，则使用默认值

# 询问用户对手池
read -p "请输入对手池的名称 [Eg: pool1]: " opponent_pool
opponent_pool=${opponent_pool:-pool1}  # 如果用户未输入，则使用默认值
opponent_pool="/home/ubuntu/Workfile/RL/RL_model/opponent_pool/$opponent_pool"

# 评估结果保存在与stage同级的文件夹
eval_log_dir="$selected_exp"

# 显示评估信息
echo "开始评估..."
echo "评估路径: $latest_result"
echo "评估次数: $n_episodes"
echo "结果将保存至: $eval_log_dir"
echo ""

# 调用Python脚本进行评估
python -m src.evaluate --exp_path "$latest_result" --n_episodes $n_episodes --pool_path "$opponent_pool"

echo "评估完成"
