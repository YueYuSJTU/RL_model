# JSBSim 双飞机空战强化学习库

## 介绍

这是一个基于 [JSBSim](http://jsbsim.sourceforge.net/) 的双飞机空战强化学习库。在 [jsbgym](https://github.com/Gor-Ren/jsbgym) (JSBSim的Gymnasium封装) 的基础上，扩展了新的任务，并提供了一套完整的训练、评估和可视化流水线。

**核心任务**:
- **`goal_point` 任务**: 目标点跟随任务，主要用于训练基础的飞行控制能力，其模型可作为对战训练的初始对手。
- **`tracking` 任务**: 双飞机1v1空战对抗环境，支持自博弈（Self-Play）和对手池（Opponent Pool）训练机制。

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

## 1. 项目架构

经过重构，项目结构更加清晰模块化：

- **`src/training/`**: 训练流水线
  - `train.py`: 统一的训练入口，支持多阶段（Multi-stage）和对战（Battle）训练。
  - `pool_manager.py`: 对手池管理器，用于在对战训练中动态更新和选择对手模型。
- **`src/evaluation/`**: 评估流水线
  - `evaluator.py`: 统一的评估器，处理环境步进、奖励累加和胜率统计，支持模型对战和人工控制。
- **`src/environments/`**: 环境与包装器
  - `make_env.py`: 环境创建工厂，负责构建 Gymnasium 环境和向量化环境。
  - `self_play_wrapper.py`: 自博弈包装器，解耦了环境与策略推理，负责在环境中加载和查询对手模型。
- **`src/agents/`**: 智能体与模型
  - `make_agent.py`: 创建和加载 Stable-Baselines3 模型（支持 PPO 及自定义网络架构）。
  - `model_wrapper.py`: 模型包装器，用于适配不同任务（如 GoalPoint 和 Tracking）之间的观察值差异。
- **`src/visualization/`**: 可视化工具
  - 包含用于 3D 渲染和通信的脚本（如 `dogfight_client.py`）。
- **`jsb_env/`**: 底层 JSBSim 环境
  - 包含本地修改的 `jsbgym_m` 包，定义了飞行器动力学、任务逻辑（如 `task_tracking.py`）和奖励函数。
- **`configs/`**: 配置文件
  - 包含环境配置 (`env/`)、智能体配置 (`agent/`) 和训练流程配置 (`*train_config.yaml`)。

---

## 2. 启动训练

通过 `src.training.train` 模块启动训练。所有训练都应在 `js_gpu` 环境下进行。

**统一训练入口 (支持多阶段和对战):**

```bash
python -m src.training.train \
    --config "configs/stage_train_config.yaml" \
    --pool_path "/path/to/opponent_pool/pool3" \
    --pretrained_path "experiments/20250921_162658"
```

- **预训练模式**: 省略 `--pretrained_path` 参数。脚本将自动创建一个以当前日期和时间命名的文件夹（如 `experiments/YYYYMMDD_HHMMSS/`）来保存训练记录。
- **微调模式**: 提供 `--pretrained_path` 参数，指定一个已有的实验目录。脚本会自动加载该目录下的模型进行微调。
- **对战模式**: 如果配置文件中包含 `battle_step` 参数，最后一个阶段将作为对战阶段运行，期间会通过 `PoolManager` 定期更新对手池。

**后台运行示例:**
```bash
nohup python -m src.training.train --config "configs/battle_train_config.yaml" --pool_path "./opponent_pool/pool4" > output.log 2>&1 &
```

---

## 3. 评估与展示

项目提供了一个交互式的 Shell 脚本 `show.sh`，用于方便地选择实验结果进行可视化回放或定量评估。

### 3.1 交互式评估/可视化

运行以下命令启动交互式助手：

```bash
./show.sh
```

根据终端提示：
1. 选择要评估的实验批次和具体阶段。
2. 选择运行模式（可视化演示 或 定量评估）。
3. 选择渲染模式（如 `anim3d`, `human`, `flightgear` 或无渲染）。
4. 选择对手模型（从对手池中选择或指定特定模型）。

### 3.2 人工/手动控制

支持使用键盘或手柄手动控制一架飞机与训练好的模型进行对战。只需在运行 `show.sh` 时添加 `--manual` 参数：

```bash
./show.sh --manual
```

> **提示**:
> - 手柄控制优先，如果未检测到手柄则回退到键盘控制。
> - 如果手柄键位不正确，可以使用 `tests/test_joystick_id.py` 测试键位，并在 `src/utils/manual_control.py` 中进行修改。

---

## 4. 配置文件编写

所有配置文件位于 `configs/` 文件夹下。

- **训练配置 (`*train_config.yaml`)**: 定义训练的阶段（stages）。每个阶段可以指定不同的环境配置（`env`）和智能体配置（`agent`）。
- **多阶段训练**: 本库支持多阶段课程学习。您可以在配置文件中定义多个 `stage`，模型会按顺序训练，并将前一阶段的最佳模型作为下一阶段的初始权重。
- **自定义奖励函数**: 奖励函数的定义位于底层环境代码中。如需修改，请编辑 `jsb_env/jsbgym_m/task_tracking.py` 等任务文件。

---

## 5. 在HPC平台通过Slurm提交训练任务

对于大规模或长时间的训练，建议使用HPC平台的Slurm作业调度系统。

您可以使用项目根目录下的 `train.slurm` 脚本模板。该脚本负责申请计算资源并执行您的训练命令。

**提交任务:**
```bash
sbatch train.slurm
```

**监控任务状态:**
```bash
squeue -u <您的用户名>
```

**查看日志:**
```bash
cat logs/train_<任务ID>.out
```
