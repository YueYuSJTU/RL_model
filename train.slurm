#!/bin/bash
set -e
#=======================================================================
# SLURM 资源申请指令
#=======================================================================
# -- Job Name: 任务名，方便您识别
#SBATCH --job-name=hpc_train1

# -- Output/Error files: 指定标准输出和错误日志的路径
#    %j 是一个占位符，Slurm会自动替换为任务ID (Job ID)
#SBATCH --output=logs/hpc_%j.out
#SBATCH --error=logs/hpc_%j.err

# -- Partition: 指定任务提交到哪个计算分区/队列
#SBATCH --partition=compute

# -- GPU Request: 申请GPU资源
#    格式为 --gres=gpu:[类型]:[数量]
#SBATCH --gres=gpu:rtx5090:1

# -- CPU Request: 每个任务申请的CPU核心数
#SBATCH --cpus-per-task=16

# -- Memory Request: 申请的内存大小
#SBATCH --mem=64G

#=======================================================================
# 您的程序运行命令
#=======================================================================
cd $SLURM_SUBMIT_DIR
echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job started on: $(hostname) at $(date)"
echo "========================================================"

# --- 1. 激活您的虚拟环境 ---
# 如果您使用Conda, 命令应为: conda activate <您的环境名>
source $(conda info --base)/etc/profile.d/conda.sh
conda activate js_gpu

echo "Virtual environment activated."

# --- 2. 运行您的Python训练脚本 ---
python -u -m src.train.battle_train --config "configs/battle_train_config.yaml" --pool_path "opponent_pool/pool4"

echo "========================================================"
echo "Job finished at: $(date)"
echo "========================================================"
