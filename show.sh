# #!/bin/bash

# # 设置当前工作目录
# cd "$(dirname "$0")"
# # 使用source方式初始化conda并激活环境
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate js_gpu

# # 检查 experiments 目录是否存在
# if [ ! -d "./experiments" ]; then
#     echo "错误：./experiments 目录不存在"
#     exit 1
# fi

# # 列出 experiments 目录下的所有文件夹
# echo "可用的实验："
# experiments=($(ls -d ./experiments/*/))
# for i in "${!experiments[@]}"; do
#     # 去掉路径前缀和尾部斜杠
#     folder_name=$(basename "${experiments[$i]}")
#     echo "[$i] $folder_name"
# done

# # 询问用户选择哪个实验
# read -p "请选择要评估的实验 [0-$((${#experiments[@]}-1))]: " exp_idx

# # 验证输入
# if ! [[ "$exp_idx" =~ ^[0-9]+$ ]] || [ "$exp_idx" -ge "${#experiments[@]}" ]; then
#     echo "错误：无效的选择"
#     exit 1
# fi

# selected_exp="${experiments[$exp_idx]}"
# selected_exp_name=$(basename "$selected_exp")

# is_goal_point_mode=0
# if [ "$selected_exp_name" == "goal_point" ]; then
#     is_goal_point_mode=1
#     echo "选择了展示GoalPoint模型"
    
#     # 列出 goal_point 文件夹中的实验
#     echo "可用的GoalPoint实验："
#     experiments=($(ls -d "$selected_exp"/*/))
#     if [ ${#experiments[@]} -eq 0 ]; then
#         echo "错误：goal_point文件夹中没有实验"
#         exit 1
#     fi
    
#     for i in "${!experiments[@]}"; do
#         folder_name=$(basename "${experiments[$i]}")
#         echo "[$i] $folder_name"
#     done

#     # 再次询问用户选择哪个实验
#     read -p "请选择要评估的实验 [0-$((${#experiments[@]}-1))]: " exp_idx

#     # 验证输入
#     if ! [[ "$exp_idx" =~ ^[0-9]+$ ]] || [ "$exp_idx" -ge "${#experiments[@]}" ]; then
#         echo "错误：无效的选择"
#         exit 1
#     fi

#     selected_exp="${experiments[$exp_idx]}"
#     selected_exp_name=$(basename "$selected_exp")
# fi

# echo "已选择实验: $selected_exp_name"
# echo ""

# # 检查是否存在tensorboard文件夹
# has_tensorboard=0
# if [ -d "${selected_exp}tensorboard" ]; then
#     has_tensorboard=1
#     echo "[0] tensorboard"
# fi

# # 列出选定实验中的所有stage
# echo "可用的stages："
# stages=($(ls -d "$selected_exp"/*/))
# if [ ${#stages[@]} -eq 0 ]; then
#     echo "错误：所选实验中没有stage文件夹"
#     exit 1
# fi

# stage_count=0
# for i in "${!stages[@]}"; do
#     # 去掉路径前缀和尾部斜杠
#     folder_name=$(basename "${stages[$i]}")
#     # 跳过tensorboard文件夹
#     if [ "$folder_name" != "tensorboard" ]; then
#         stage_count=$((stage_count+1))
#         echo "[$stage_count] $folder_name"
#     fi
# done

# # 询问用户选择哪个stage或者tensorboard
# max_option=$stage_count
# if [ $has_tensorboard -eq 1 ]; then
#     read -p "请选择tensorboard或要评估的stage [0-$max_option]: " option_idx
    
#     # 处理tensorboard选项
#     if [ "$option_idx" -eq 0 ]; then
#         echo "已选择tensorboard，启动tensorboard..."
#         # 这里可以添加启动tensorboard的代码
#         tensorboard --logdir="${selected_exp}tensorboard"
#         exit 0
#     fi
    
#     # 将用户选择映射回实际的stage索引
#     stage_idx=0
#     current_option=0
#     for i in "${!stages[@]}"; do
#         folder_name=$(basename "${stages[$i]}")
#         if [ "$folder_name" != "tensorboard" ]; then
#             current_option=$((current_option+1))
#             if [ $current_option -eq $option_idx ]; then
#                 stage_idx=$i
#                 break
#             fi
#         fi
#     done
# else
#     read -p "请选择要评估的stage [1-$max_option]: " option_idx
    
#     # 将用户选择映射回实际的stage索引
#     stage_idx=0
#     current_option=0
#     for i in "${!stages[@]}"; do
#         folder_name=$(basename "${stages[$i]}")
#         if [ "$folder_name" != "tensorboard" ]; then
#             current_option=$((current_option+1))
#             if [ $current_option -eq $option_idx ]; then
#                 stage_idx=$i
#                 break
#             fi
#         fi
#     done
# fi

# # 验证输入
# if ! [[ "$option_idx" =~ ^[0-9]+$ ]] || [ "$option_idx" -lt 0 ] || ([ $has_tensorboard -eq 0 ] && [ "$option_idx" -eq 0 ]) || [ "$option_idx" -gt "$max_option" ]; then
#     echo "错误：无效的选择"
#     exit 1
# fi

# selected_stage="${stages[$stage_idx]}"
# selected_stage_name=$(basename "$selected_stage")

# echo "已选择: $selected_stage_name"
# echo ""

# # 列出选定stage中的所有训练结果
# echo "可用的训练结果："
# results=($(ls -d "$selected_stage"/*/))
# if [ ${#results[@]} -eq 0 ]; then
#     # 如果没有子文件夹，则使用当前目录
#     results=("$selected_stage")
#     for i in "${!results[@]}"; do
#         echo "[$i] ."  # 表示当前目录
#     done
# else
#     for i in "${!results[@]}"; do
#         folder_name=$(basename "${results[$i]}")
#         echo "[$i] $folder_name"
#     done
# fi

# # 询问用户选择哪个训练结果
# read -p "请选择要评估的训练结果 [0-$((${#results[@]}-1))]: " result_idx

# # 验证输入
# if ! [[ "$result_idx" =~ ^[0-9]+$ ]] || [ "$result_idx" -ge "${#results[@]}" ]; then
#     echo "错误：无效的选择"
#     exit 1
# fi

# selected_result="${results[$result_idx]}"
# selected_result_name=$(basename "$selected_result")

# echo "已选择训练结果: $selected_result_name"
# echo ""

# # 新增：选择模式
# echo "请选择运行模式:"
# echo "[0] 可视化 (Visualization)"
# echo "[1] 定量评估 (Quantitative Evaluation)"
# read -p "请输入选择 [0-1]: " run_mode_idx

# n_episode=1
# render_mode="human"

# if [ "$run_mode_idx" -eq 0 ]; then
#     # 可视化模式
#     if [ "$is_goal_point_mode" -eq 1 ]; then
#         render_mode="human"
#         echo "GoalPoint模式下，渲染模式固定为human"
#     else
#         # 询问用户使用什么渲染模式
#         echo "可用的渲染模式:"
#         echo "[0] human"
#         echo "[1] anim3d"
#         echo "[2] flightgear"
#         echo "[3] none"

#         read -p "请选择渲染模式 [0-3]: " mode_idx

#         case $mode_idx in
#             0) render_mode="human" ;;
#             1) render_mode="anim3d" ;;
#             2) render_mode="flightgear" ;;
#             3) render_mode="none" ;;
#             *) echo "错误：无效的选择"; exit 1 ;;
#         esac
#     fi
#     echo "已选择渲染模式: $render_mode"
# elif [ "$run_mode_idx" -eq 1 ]; then
#     # 定量评估模式
#     render_mode="none"
#     read -p "请输入评估次数 (n_episode > 1): " n_episode
#     if ! [[ "$n_episode" =~ ^[0-9]+$ ]] || [ "$n_episode" -le 1 ]; then
#         echo "错误：评估次数必须为大于1的整数"
#         exit 1
#     fi
# else
#     echo "错误：无效的选择"
#     exit 1
# fi
# echo ""

# opponent_pool_path_param=""
# selected_pool_path=""

# if [ "$is_goal_point_mode" -eq 0 ]; then
#     # 选择对手池路径
#     echo "选择对手池路径:"
#     default_pool_dir="./opponent_pool"
#     pools=()
#     if [ -d "$default_pool_dir" ]; then
#         pools=($(ls -d "$default_pool_dir"/*/))
#     fi
    
#     for i in "${!pools[@]}"; do
#         pool_name=$(basename "${pools[$i]}")
#         echo "[$i] $pool_name"
#     done
#     echo "[${#pools[@]}] 输入自定义路径"

#     read -p "请选择对手池 [0-${#pools[@]}]: " pool_idx

#     if [ "$pool_idx" -eq "${#pools[@]}" ]; then
#         read -p "请输入对手池路径 (相对或绝对路径): " custom_path
#         # 处理相对路径
#         if [[ "$custom_path" != /* ]]; then
#             custom_path="$(pwd)/$custom_path"
#         fi
        
#         if [ -d "$custom_path" ]; then
#             selected_pool_path="$custom_path"
#         else
#             echo "错误：路径 '$custom_path' 不存在或不是一个目录。"
#             exit 1
#         fi
#     elif [[ "$pool_idx" =~ ^[0-9]+$ ]] && [ "$pool_idx" -lt "${#pools[@]}" ]; then
#         selected_pool_path="${pools[$pool_idx]}"
#     else
#         echo "错误：无效的选择"
#         exit 1
#     fi
    
#     echo "已选择对手池: $(basename "$selected_pool_path")"
#     opponent_pool_path_param="--pool_path $selected_pool_path"
#     echo ""
# fi

# model_num_param=""
# if [ "$is_goal_point_mode" -eq 0 ]; then
#     # 如果是可视化模式，需要选择具体的对手模型
#     if [ "$run_mode_idx" -eq 0 ]; then
#         echo "选择对手模型:"
#         # 列出池中的模型文件夹
#         if [ -d "$selected_pool_path" ]; then
#             echo "对手池中的可用模型:"
#             ls -F "$selected_pool_path" | grep /$ | head -n 10
#             echo "..."
#         fi
#         read -p "请输入对手模型编号 (对应文件夹名): " opponent_model_num
#         model_num_param="--model_num $opponent_model_num"
#         echo "已选择对手模型: $opponent_model_num"
#     fi
# fi

# # 调用Python脚本进行评估
# echo "开始运行..."
# python3 -m src.show --exp_path "$selected_result" --render_mode "$render_mode" --n_episode "$n_episode" $model_num_param $opponent_pool_path_param

#!/bin/bash

# 设置当前工作目录
cd "$(dirname "$0")"
# 使用source方式初始化conda并激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate js_gpu

# === 修改点 1: 检查命令行参数是否包含 --manual ===
manual_mode=0
for arg in "$@"; do
    if [ "$arg" == "--manual" ]; then
        manual_mode=1
        echo ">>> 启动手动控制模式 (Manual Mode) <<<"
        break
    fi
done

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

# === 修改点 2: 自动选择实验 ===
if [ "$manual_mode" -eq 1 ]; then
    # 手动模式下，默认选择第一个实验 (索引0)
    # 如果你想选择最后一个(通常是最新的)，可以将 0 改为 $((${#experiments[@]}-1))
    exp_idx=0
    echo "Manual Mode: 自动选择实验 [0]"
else
    # 询问用户选择哪个实验
    read -p "请选择要评估的实验 [0-$((${#experiments[@]}-1))]: " exp_idx
fi

# 验证输入
if ! [[ "$exp_idx" =~ ^[0-9]+$ ]] || [ "$exp_idx" -ge "${#experiments[@]}" ]; then
    echo "错误：无效的选择"
    exit 1
fi

selected_exp="${experiments[$exp_idx]}"
selected_exp_name=$(basename "$selected_exp")

is_goal_point_mode=0
if [ "$selected_exp_name" == "goal_point" ]; then
    is_goal_point_mode=1
    echo "选择了展示GoalPoint模型"
    
    # 列出 goal_point 文件夹中的实验
    echo "可用的GoalPoint实验："
    experiments=($(ls -d "$selected_exp"/*/))
    if [ ${#experiments[@]} -eq 0 ]; then
        echo "错误：goal_point文件夹中没有实验"
        exit 1
    fi
    
    for i in "${!experiments[@]}"; do
        folder_name=$(basename "${experiments[$i]}")
        echo "[$i] $folder_name"
    done

    # GoalPoint 子实验选择
    if [ "$manual_mode" -eq 1 ]; then
        exp_idx=0
        echo "Manual Mode: 自动选择GoalPoint子实验 [0]"
    else
        read -p "请选择要评估的实验 [0-$((${#experiments[@]}-1))]: " exp_idx
    fi

    # 验证输入
    if ! [[ "$exp_idx" =~ ^[0-9]+$ ]] || [ "$exp_idx" -ge "${#experiments[@]}" ]; then
        echo "错误：无效的选择"
        exit 1
    fi

    selected_exp="${experiments[$exp_idx]}"
    selected_exp_name=$(basename "$selected_exp")
fi

echo "已选择实验: $selected_exp_name"
echo ""

# 检查是否存在tensorboard文件夹
has_tensorboard=0
if [ -d "${selected_exp}tensorboard" ]; then
    has_tensorboard=1
    # echo "[0] tensorboard" # 为了简洁，手动模式下不显示这个，普通模式下保持原样
    if [ "$manual_mode" -eq 0 ]; then
         echo "[0] tensorboard"
    fi
fi

# 列出选定实验中的所有stage
echo "可用的stages："
stages=($(ls -d "$selected_exp"/*/))
if [ ${#stages[@]} -eq 0 ]; then
    echo "错误：所选实验中没有stage文件夹"
    exit 1
fi

stage_count=0
# 创建一个映射数组来存储有效的stage索引
declare -a valid_stage_indices
for i in "${!stages[@]}"; do
    folder_name=$(basename "${stages[$i]}")
    if [ "$folder_name" != "tensorboard" ]; then
        stage_count=$((stage_count+1))
        if [ "$manual_mode" -eq 0 ]; then
            echo "[$stage_count] $folder_name"
        fi
        valid_stage_indices+=($i)
    fi
done

max_option=$stage_count

# === 修改点 3: 自动选择 Stage ===
if [ "$manual_mode" -eq 1 ]; then
    # 手动模式下，跳过 Tensorboard 选择，直接选择第一个有效的 Stage
    # 这里的 option_idx 对应显示的 [1]，即 valid_stage_indices 的第0个元素
    if [ "$has_tensorboard" -eq 1 ]; then
        option_idx=1 
    else
        option_idx=1
    fi
    echo "Manual Mode: 自动选择 Stage [1]"
else
    if [ $has_tensorboard -eq 1 ]; then
        read -p "请选择tensorboard或要评估的stage [0-$max_option]: " option_idx
    else
        read -p "请选择要评估的stage [1-$max_option]: " option_idx
    fi
fi

# 处理tensorboard和索引映射逻辑 (保持不变，只是 option_idx 来源不同了)
if [ $has_tensorboard -eq 1 ] && [ "$option_idx" -eq 0 ]; then
    echo "已选择tensorboard，启动tensorboard..."
    tensorboard --logdir="${selected_exp}tensorboard"
    exit 0
fi

# 验证输入
if ! [[ "$option_idx" =~ ^[0-9]+$ ]] || [ "$option_idx" -lt 0 ] || ([ $has_tensorboard -eq 0 ] && [ "$option_idx" -eq 0 ]) || [ "$option_idx" -gt "$max_option" ]; then
    echo "错误：无效的选择"
    exit 1
fi

# 计算实际的 array index
# option_idx 是从1开始计数的有效stage (在tensorboard之后)
# valid_stage_indices 存储了真实的文件系统索引，下标从0开始 (对应 option_idx-1)
stage_idx=${valid_stage_indices[$((option_idx-1))]}

selected_stage="${stages[$stage_idx]}"
selected_stage_name=$(basename "$selected_stage")

echo "已选择: $selected_stage_name"
echo ""

# 列出选定stage中的所有训练结果
echo "可用的训练结果："
results=($(ls -d "$selected_stage"/*/))
if [ ${#results[@]} -eq 0 ]; then
    results=("$selected_stage")
    if [ "$manual_mode" -eq 0 ]; then
        for i in "${!results[@]}"; do echo "[$i] ."; done
    fi
else
    if [ "$manual_mode" -eq 0 ]; then
        for i in "${!results[@]}"; do
            folder_name=$(basename "${results[$i]}")
            echo "[$i] $folder_name"
        done
    fi
fi

# === 修改点 4: 自动选择训练结果 ===
if [ "$manual_mode" -eq 1 ]; then
    result_idx=0
    echo "Manual Mode: 自动选择训练结果 [0]"
else
    read -p "请选择要评估的训练结果 [0-$((${#results[@]}-1))]: " result_idx
fi

# 验证输入
if ! [[ "$result_idx" =~ ^[0-9]+$ ]] || [ "$result_idx" -ge "${#results[@]}" ]; then
    echo "错误：无效的选择"
    exit 1
fi

selected_result="${results[$result_idx]}"
selected_result_name=$(basename "$selected_result")

echo "已选择训练结果: $selected_result_name"
echo ""

# === 修改点 5: 自动设置运行模式和渲染模式 ===
if [ "$manual_mode" -eq 1 ]; then
    echo "Manual Mode: 自动设置为可视化模式 (anim3d), Episodes=1"
    run_mode_idx=0
    render_mode="anim3d"
    n_episode=1
else
    # 原有逻辑
    echo "请选择运行模式:"
    echo "[0] 可视化 (Visualization)"
    echo "[1] 定量评估 (Quantitative Evaluation)"
    read -p "请输入选择 [0-1]: " run_mode_idx
    # ... (省略了中间大部分并未改动的逻辑，直接跳到下面的赋值) ...
    
    n_episode=1
    render_mode="human"

    if [ "$run_mode_idx" -eq 0 ]; then
        if [ "$is_goal_point_mode" -eq 1 ]; then
            render_mode="human"
            echo "GoalPoint模式下，渲染模式固定为human"
        else
            echo "可用的渲染模式:"
            echo "[0] human"
            echo "[1] anim3d"
            echo "[2] flightgear"
            echo "[3] none"
            read -p "请选择渲染模式 [0-3]: " mode_idx
            case $mode_idx in
                0) render_mode="human" ;;
                1) render_mode="anim3d" ;;
                2) render_mode="flightgear" ;;
                3) render_mode="none" ;;
                *) echo "错误：无效的选择"; exit 1 ;;
            esac
        fi
        echo "已选择渲染模式: $render_mode"
    elif [ "$run_mode_idx" -eq 1 ]; then
        render_mode="none"
        read -p "请输入评估次数 (n_episode > 1): " n_episode
        if ! [[ "$n_episode" =~ ^[0-9]+$ ]] || [ "$n_episode" -le 1 ]; then
            echo "错误：评估次数必须为大于1的整数"
            exit 1
        fi
    else
        echo "错误：无效的选择"
        exit 1
    fi
fi
echo ""

# === 此处开始依然保留交互，因为用户需要选择对手 ===

opponent_pool_path_param=""
selected_pool_path=""

if [ "$is_goal_point_mode" -eq 0 ]; then
    # 选择对手池路径
    echo "选择对手池路径:"
    default_pool_dir="./opponent_pool"
    pools=()
    if [ -d "$default_pool_dir" ]; then
        pools=($(ls -d "$default_pool_dir"/*/))
    fi
    
    for i in "${!pools[@]}"; do
        pool_name=$(basename "${pools[$i]}")
        echo "[$i] $pool_name"
    done
    echo "[${#pools[@]}] 输入自定义路径"

    read -p "请选择对手池 [0-${#pools[@]}]: " pool_idx

    if [ "$pool_idx" -eq "${#pools[@]}" ]; then
        read -p "请输入对手池路径 (相对或绝对路径): " custom_path
        if [[ "$custom_path" != /* ]]; then
            custom_path="$(pwd)/$custom_path"
        fi
        if [ -d "$custom_path" ]; then
            selected_pool_path="$custom_path"
        else
            echo "错误：路径 '$custom_path' 不存在或不是一个目录。"
            exit 1
        fi
    elif [[ "$pool_idx" =~ ^[0-9]+$ ]] && [ "$pool_idx" -lt "${#pools[@]}" ]; then
        selected_pool_path="${pools[$pool_idx]}"
    else
        echo "错误：无效的选择"
        exit 1
    fi
    
    echo "已选择对手池: $(basename "$selected_pool_path")"
    opponent_pool_path_param="--pool_path $selected_pool_path"
    echo ""
fi

model_num_param=""
if [ "$is_goal_point_mode" -eq 0 ]; then
    # 如果是可视化模式(手动模式必然是可视化)，需要选择具体的对手模型
    if [ "$run_mode_idx" -eq 0 ]; then
        echo "选择对手模型:"
        if [ -d "$selected_pool_path" ]; then
            echo "对手池中的可用模型:"
            ls -F "$selected_pool_path" | grep /$ | head -n 10
            echo "..."
        fi
        read -p "请输入对手模型编号 (对应文件夹名): " opponent_model_num
        model_num_param="--model_num $opponent_model_num"
        echo "已选择对手模型: $opponent_model_num"
    fi
fi

# === 修改点 6: 传递 --manual 参数给 Python 脚本 ===
manual_arg=""
if [ "$manual_mode" -eq 1 ]; then
    manual_arg="--manual"
fi

# 构造并打印最终将要执行的命令（但不执行）
cmd=(python3 -m src.show --exp_path "$selected_result" --render_mode "$render_mode" --n_episode "$n_episode")

# 如果 model_num_param/opponent_pool_path_param/manual_arg 非空，追加为单个参数元素
if [ -n "$model_num_param" ]; then cmd+=("$model_num_param"); fi
if [ -n "$opponent_pool_path_param" ]; then cmd+=("$opponent_pool_path_param"); fi
if [ -n "$manual_arg" ]; then cmd+=("$manual_arg"); fi

echo "最终指令:"
# 使用 %q 以便显示带引号/转义的参数，便于复制粘贴执行
printf '%q ' "${cmd[@]}"
echo
# 调用Python脚本进行评估
echo "开始运行..."
python3 -m src.show --exp_path "$selected_result" --render_mode "$render_mode" --n_episode "$n_episode" $model_num_param $opponent_pool_path_param $manual_arg