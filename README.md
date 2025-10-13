# JSBSim 双飞机空战强化学习库

## 介绍

这是一个基于 [JSBSim](http://jsbsim.sourceforge.net/) 的双飞机空战强化学习库。在 [jsbgym](https://github.com/Gor-Ren/jsbgym) (JSBSim的Gymnasium封装) 的基础上，扩展了新的任务：

-   **`goal_point` 任务**: 目标点跟随任务，主要用于初始化对手。
-   **`tracking` 任务**: 双飞机1v1空战对抗环境。

---

## 0. 环境配置

本项目使用 [Conda](https://docs.conda.io/en/latest/miniconda.html) 进行环境管理，以确保所有依赖（包括Python包和CUDA工具链）的一致性和可复现性。

### 安装步骤

1.  **克隆本仓库**
    ```bash
    git clone [您的仓库地址]
    cd [您的仓库目录]
    ```

2.  **通过 `environment.yml` 文件创建Conda环境**
    本项目所有依赖都已定义在 `environment.yml` 文件中。请运行以下命令来创建名为 `js_gpu` 的虚拟环境并安装所有依赖：
    ```bash
    conda env create -f environment.yml
    ```

3.  **激活新创建的环境**
    创建成功后，您需要激活该环境才能开始工作：
    ```bash
    conda activate js_gpu
    ```
    当您看到终端提示符前出现 `(js_gpu)` 字样时，表示环境已成功激活。

---

## 1. 启动训练

通过 `src.stageTrain` 模块启动训练。所有训练都应在 `js_gpu` 环境下进行。

**本地直接启动 (用于调试或小型实验):**

-   **预训练模式**: 不提供 `--pretrained_path` 参数。脚本将自动创建一个以当前日期和时间命名的文件夹来保存训练记录。
-   **微调模式**: 提供 `--pretrained_path` 参数，指定一个已有的日期文件夹。脚本会自动搜索该文件夹下最新的 `stage` 并进行微调。

    > **注意**: 微调模式的配置文件 (`config`) 只支持一个 `stage`。

**本地启动示例:**
```bash
python -m src.stageTrain \
    --config "configs/stage_train_config.yaml" \
    --eval_pool "/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool3" \
    --pretrained_path "experiments/20250921_162658"
```
---
**battle 训练模式 (本地后台运行)**
```bash
nohup python -m src.train.battle_train --config "configs/battle_train_config.yaml" --pool_path "/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool4" >output.log 2>&1 &
```
---
## 2. 在HPC平台通过Slurm提交训练任务

对于大规模或长时间的训练，建议使用HPC平台的Slurm作业调度系统。

### 2.1 编写Slurm脚本

您可以使用项目根目录下的 `run_training.sh` 脚本模板。该脚本负责申请计算资源并执行您的训练命令。

**`run_training.sh` 示例:**
```bash
#!/bin/bash

#=======================================================================
# SLURM 资源申请指令
#=======================================================================
# -- 任务名，方便您识别
#SBATCH --job-name=exp1_resnet50

# -- 指定标准输出和错误日志的路径 (%j 会被替换为任务ID)
#SBATCH --output=logs/resnet50_%j.out
#SBATCH --error=logs/resnet50_%j.err

# -- 指定任务提交到哪个计算分区
#SBATCH --partition=compute

# -- 申请GPU资源 (请根据集群可用资源修改)
#SBATCH --gres=gpu:rtx5090:1

# -- 申请CPU和内存资源
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#=======================================================================
# 您的程序运行命令
#=======================================================================
echo "========================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job started on: $(hostname) at $(date)"
echo "========================================================"

# --- 1. 激活您的Conda环境 ---
source $(conda info --base)/etc/profile.d/conda.sh
conda activate js_gpu

echo "Conda environment 'js_gpu' activated."

# --- 2. 运行您的Python训练脚本 ---
# !!! 重要: 请在此处修改为您本次实验的真实参数 !!!
python -u -m src.stageTrain \
    --config "configs/stage_train_config.yaml" \
    --eval_pool "/path/to/your/opponent_pool/pool3"

echo "========================================================"
echo "Job finished at: $(date)"
echo "========================================================"
```

### 2.2 提交与监控任务

1.  **提交任务**
    根据您的实验需求修改好 `run_training.sh` 脚本中的资源申请和Python命令参数后，使用 `sbatch` 命令提交任务：
    ```bash
    sbatch run_training.sh
    ```

2.  **监控任务状态**
    您可以使用以下命令来查看您的任务状态：
    ```bash
    # 查看您自己提交的所有任务
    squeue -u <您的用户名>

    # 查看任务的详细信息
    scontrol show job <任务ID>
    ```

3.  **查看日志**
    任务运行产生的输出和错误信息会保存在您脚本中指定的 `logs/` 目录下。例如，查看输出日志：
    ```bash
    cat logs/resnet50_<任务ID>.out
    ```

4.  **取消任务**
    如果需要提前终止任务，可以使用 `scancel` 命令：
    ```bash
    scancel <任务ID>
    ```

---

## 3. 配置文件编写

所有配置文件位于 `configs/` 文件夹下。

-   `train_config` 中的 `agent` 和 `env` 参数分别指向对应Agent和环境的 `.yaml` 配置文件名。
-   **多阶段训练**: 本库支持多阶段训练。您可以在配置文件中定义多个 `stage` (最多10个)，并为每个 `stage` 指定不同的奖励函数。
-   **自定义奖励函数**: 修改奖励函数需要同步修改任务定义文件。具体代码示例请参考 `jsb_env/jsbgym_m/task_tracking.py`。

---

## 4. 评估实验

### 4.1 对战评估

此脚本用于评估某个已训练模型的胜率，结果将以 `.txt` 和 `.npy` 格式存储。

**运行方式:**

```bash
./evalate.sh
```

### 4.2 goal point模型评估
```bash
python ./test/evaluate_goal_point.py --n_episodes 1000 --exp_path "experiments/goal_point/20250922_095754/stage1/20250923_203916_GoalPointTask_ppo_1layer"
```

---

## 5. 展示实验

通过可视化方式回放和展示实验效果。

**渲染支持:**

-   `tracking` 任务: 支持 `human`, `anim3d`, `flightgear` 渲染。
-   `goal_point` 任务: 支持 `human` 渲染。

**运行方式:**

```bash
./show.sh
```