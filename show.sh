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

# 检查是否存在tensorboard文件夹
has_tensorboard=0
if [ -d "${selected_exp}tensorboard" ]; then
    has_tensorboard=1
    echo "[0] tensorboard"
fi

# 列出选定实验中的所有stage
echo "可用的stages："
stages=($(ls -d "$selected_exp"/*/))
if [ ${#stages[@]} -eq 0 ]; then
    echo "错误：所选实验中没有stage文件夹"
    exit 1
fi

stage_count=0
for i in "${!stages[@]}"; do
    # 去掉路径前缀和尾部斜杠
    folder_name=$(basename "${stages[$i]}")
    # 跳过tensorboard文件夹
    if [ "$folder_name" != "tensorboard" ]; then
        stage_count=$((stage_count+1))
        echo "[$stage_count] $folder_name"
    fi
done

# 询问用户选择哪个stage或者tensorboard
max_option=$stage_count
if [ $has_tensorboard -eq 1 ]; then
    read -p "请选择tensorboard或要评估的stage [0-$max_option]: " option_idx
    
    # 处理tensorboard选项
    if [ "$option_idx" -eq 0 ]; then
        echo "已选择tensorboard，启动tensorboard..."
        # 这里可以添加启动tensorboard的代码
        tensorboard --logdir="${selected_exp}tensorboard"
        exit 0
    fi
    
    # 将用户选择映射回实际的stage索引
    stage_idx=0
    current_option=0
    for i in "${!stages[@]}"; do
        folder_name=$(basename "${stages[$i]}")
        if [ "$folder_name" != "tensorboard" ]; then
            current_option=$((current_option+1))
            if [ $current_option -eq $option_idx ]; then
                stage_idx=$i
                break
            fi
        fi
    done
else
    read -p "请选择要评估的stage [1-$max_option]: " option_idx
    
    # 将用户选择映射回实际的stage索引
    stage_idx=0
    current_option=0
    for i in "${!stages[@]}"; do
        folder_name=$(basename "${stages[$i]}")
        if [ "$folder_name" != "tensorboard" ]; then
            current_option=$((current_option+1))
            if [ $current_option -eq $option_idx ]; then
                stage_idx=$i
                break
            fi
        fi
    done
fi

# 验证输入
if ! [[ "$option_idx" =~ ^[0-9]+$ ]] || [ "$option_idx" -lt 0 ] || ([ $has_tensorboard -eq 0 ] && [ "$option_idx" -eq 0 ]) || [ "$option_idx" -gt "$max_option" ]; then
    echo "错误：无效的选择"
    exit 1
fi

selected_stage="${stages[$stage_idx]}"
selected_stage_name=$(basename "$selected_stage")

echo "已选择: $selected_stage_name"
echo ""

# 列出选定stage中的所有训练结果
echo "可用的训练结果："
results=($(ls -d "$selected_stage"/*/))
if [ ${#results[@]} -eq 0 ]; then
    # 如果没有子文件夹，则使用当前目录
    results=("$selected_stage")
    for i in "${!results[@]}"; do
        echo "[$i] ."  # 表示当前目录
    done
else
    for i in "${!results[@]}"; do
        folder_name=$(basename "${results[$i]}")
        echo "[$i] $folder_name"
    done
fi

# 询问用户选择哪个训练结果
read -p "请选择要评估的训练结果 [0-$((${#results[@]}-1))]: " result_idx

# 验证输入
if ! [[ "$result_idx" =~ ^[0-9]+$ ]] || [ "$result_idx" -ge "${#results[@]}" ]; then
    echo "错误：无效的选择"
    exit 1
fi

selected_result="${results[$result_idx]}"
selected_result_name=$(basename "$selected_result")

echo "已选择训练结果: $selected_result_name"
echo ""

# 询问用户使用什么渲染模式
echo "可用的渲染模式:"
echo "[0] human"
echo "[1] flightgear"
echo "[2] none"

read -p "请选择渲染模式 [0-2]: " mode_idx

case $mode_idx in
    0)
        render_mode="human"
        ;;
    1)
        render_mode="flightgear"
        ;;
    2)
        render_mode="none"
        ;;
    *)
        echo "错误：无效的选择"
        exit 1
        ;;
esac

echo "已选择渲染模式: $render_mode"
echo ""

# 调用Python脚本进行评估
echo "开始评估..."
python3 -m src.show --exp_path "$selected_result" --render_mode "$render_mode"