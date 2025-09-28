#!/bin/bash

# 设置当前工作目录
cd "$(dirname "$0")"
# 使用source方式初始化conda并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate js_gpu

# 模型池的根目录
POOL_ROOT="./opponent_pool"

# 检查模型池目录是否存在
if [ ! -d "$POOL_ROOT" ]; then
    echo "错误：$POOL_ROOT 目录不存在"
    echo "创建 $POOL_ROOT 目录..."
    mkdir -p "$POOL_ROOT"
fi

# 列出所有可用的模型池
clear
echo "===== 模型池评估工具 ====="
echo "可用的模型池："

# 检查是否有模型池
if [ -z "$(ls -A $POOL_ROOT 2>/dev/null)" ]; then
    echo "当前没有模型池。请先创建一个模型池目录。"
    exit 1
fi

# 列出所有模型池
pools=($(ls -d $POOL_ROOT/*/))
for i in "${!pools[@]}"; do
    pool_name=$(basename "${pools[$i]}")
    # 显示的序号从1开始
    echo "[$(($i+1))] $pool_name"
done

# 用户选择模型池
read -p "请选择要评估的模型池 [1-${#pools[@]}]: " pool_idx

# 验证输入 - 调整为从1开始的范围
if ! [[ "$pool_idx" =~ ^[0-9]+$ ]] || [ "$pool_idx" -lt 1 ] || [ "$pool_idx" -gt "${#pools[@]}" ]; then
    echo "错误：无效的选择"
    exit 1
fi

# 将用户输入转换为数组索引（减1）
selected_pool="${pools[$(($pool_idx-1))]}"
selected_pool_name=$(basename "$selected_pool")
echo "已选择模型池: $selected_pool_name"

# 选择评估模式
echo ""
echo "评估模式:"
echo "[1] 执行模型评估 (对模型进行对战评估)"
echo "[2] 显示评估结果 (显示已有评估结果的图表)"
read -p "请选择评估模式 [1-2]: " mode_choice

# 根据选择设置show参数
if [[ "$mode_choice" == "2" ]]; then
    show_mode="True"
else
    show_mode="False"
    # 询问对战次数
    read -p "请输入每对模型之间的对战次数 [默认: 100]: " n_episodes
    n_episodes=${n_episodes:-100}  # 如果用户未输入，则使用默认值
fi

# 确认将要执行的操作
echo ""
echo "将执行以下操作:"
if [[ "$show_mode" == "True" ]]; then
    echo "显示模型池 '$selected_pool_name' 的评估结果图表"
else
    echo "对模型池 '$selected_pool_name' 中的模型进行 $n_episodes 次对战评估"
fi
read -p "确认执行? [y/n]: " confirm

if [[ "$confirm" != "y" ]]; then
    echo "操作已取消"
    exit 0
fi

# 调用Python脚本执行评估
echo "开始执行..."
if [[ "$show_mode" == "True" ]]; then
    python -m src.evaluate_pool --pool_path "$selected_pool" --show True
else
    python -m src.evaluate_pool --pool_path "$selected_pool" --n_episodes $n_episodes
fi

echo "评估完成"
