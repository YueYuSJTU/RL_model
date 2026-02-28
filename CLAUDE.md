这份文件为 Claude Code (claude.ai/code) 在处理本仓库代码时提供指南。

# 项目概述

本仓库是一个基于 JSBSim 的双机空战强化学习代码库，建立在 Gymnasium 包装器（衍生自 jsbgym）之上。它添加了自定义任务以及使用 Stable-Baselines3 (主要是 PPO) 的训练/评估流水线，外加可视化和可选的人工/手动控制功能。

**关键任务** (参见 `README.md`):
- `tracking`: 1v1 双机空战 / 追踪任务。

**主要工作流**:
1. 配置 conda 环境 (`environment.yml`)
2. 通过 `python -m src.stageTrain ...` (多阶段) 或 `python -m src.train.battle_train ...` (带有对手池的对战/自博弈风格) 进行训练
3. 通过 `./show.sh` 进行评估 / 可视化 (它是 `python -m src.show ...` 的包装器)

# 环境 / 依赖

**Conda 环境** (根目录 `environment.yml`):
- 环境名称: `js_gpu`
- python: 3.9
- 关键 pip 依赖: `stable-baselines3[extra]`, `jsbgym`, `pyyaml`, `pyquaternion`, `opencv-python` 等。

**本地修改的环境包**:
- `jsb_env/` 包含一个本地 `jsbgym_m` 包的 `pyproject.toml` (修改版的 jsbgym/jsbsim Gymnasium 环境 + 新增任务)。

# 常用命令


## 多阶段训练 (推荐入口)

使用 `src/stageTrain.py` (README 中引用的模块入口点):
```bash
python -m src.stageTrain \
--config "configs/stage_train_config.yaml" \
--eval_pool "/path/to/opponent_pool/pool3" \
--pretrained_path "experiments/20250921_162658"
```
- 省略 `--pretrained_path` 以开始新运行 (创建 `experiments/YYYYMMDD_HHMMSS/`)。
- 提供 `--pretrained_path` 以恢复/微调 (该根目录下的阶段目录)。

## 对战训练模式 (后台运行)

包装器入口点: `src/train/battle_train.py` (根据 README)
```bash
nohup python -m src.train.battle_train \
--config "configs/battle_train_config.yaml" \
--pool_path "/path/to/opponent_pool/pool4" \
> output.log 2>&1 &
```

## 展示 / 评估 (交互式助手)

运行交互式评估/可视化选择器:
```bash
./show.sh
```

人工/手动游玩 (键盘/手柄) 通过 `show.sh` 透传参数:
```bash
./show.sh --manual
```


# 代码架构 (宏观视角)

## 训练流水线 (SB3 PPO)

**主要部分**:
- `src/stageTrain.py`: 编排多阶段训练。
    - 读取描述阶段的 YAML 配置 (例如 `configs/stage_train_config.yaml`)。
    - 为每个阶段加载:
        - 环境配置: `configs/env/*.yaml`
        - 智能体配置: `configs/agent/*.yaml`
        - 阶段/训练设置: 阶段 cfg (时间步, 评估频率, CPU 数量等)
    - 创建 `experiments/<timestamp>/stageN/<timestamp>_<env>_<agent>/` 文件夹并快照配置 YAML。
- `src/train/battle_train.py`: 带有最终“对战”阶段的统一阶段训练器。
    - 实现 `UnifiedTrainer`:
        - 普通阶段: 顺序训练，将最佳模型带入下一阶段
        - 对战阶段: 分块循环训练 (`battle_step`)，通过 `PoolManager` 定期更新对手池
- `src/train/pool_manager.py`: 管理磁盘上的对手池并更新选择 (用于对战阶段)。
- `src/environments/make_env.py`: 训练器使用的 `create_env(...)` 工厂，构建 Gymnasium 环境和矢量化环境。
- `src/agents/make_agent.py`: 创建/加载 SB3 智能体和策略 kwargs (支持 `src/agents/` 下的 GRU/transformer 变体)。
- `src/utils/custom_callback.py`: 包含训练期间使用的回调，包括:
    - `ComponentEvalCallback`
    - `EpisodeCurriculumCallback` (用于在对战阶段随时间调整难度 / `goal_point_prob`)

## 评估 / 可视化 / 手动控制

- `src/show.py`: CLI 入口点，用于:
    - 一次性可视化运行 (`show(...)`)
    - 基于模型池的定量评估 (`evaluate(...)` / `evaluate_without_NN(...)`)
    - 支持 `--manual` 标志以手动控制智能体 1 (键盘/手柄)
- `src/utils/manual_control.py`: 键盘和手柄控制器。
- `show.sh`: 交互式 shell 助手；选择实验/阶段/运行、渲染模式、对手池/模型，并转发 `--manual`。

## 环境 (JSBSim Gymnasium 包装器)

**本地环境包**:
- `jsb_env/jsbgym_m/environment.py`:
    - 定义 `JsbSimEnv` 和双机环境 `DoubleJsbSimEnv` (+ NoFG 变体)。
    - `DoubleJsbSimEnv` 添加对手模拟并暴露 `update_task_parameters(**kwargs)` 用于课程学习回调。
    - 渲染模式包括 `human`, `flightgear`, 和 `anim3d` (增强版 3D 可视化器)。

**任务**:
- `jsb_env/jsbgym_m/task_tracking.py`: 追踪/1v1 任务逻辑 (观测, 奖励塑形, 终止条件等)。
- 相关任务: `task_tracking_goal_point.py`, `task_tracking_init.py`, `task_goal_point.py` (目标点和初始化变体)。

## 配置

- `configs/env/*.yaml` 通常定义 `env_id`/任务参数 (环境工厂期望 `env_id` 并将 `config=env_config` 传入 `gym.make`)。
- `configs/agent/*.yaml` 定义 PPO 超参数和模型架构参数。
- `configs/*train*_config.yaml` 定义阶段级和对战级的时间表。

# 修改注意事项

- 当更改奖励塑形或终止条件时，规范位置在 `jsb_env/jsbgym_m/task_tracking.py` 下 (README 明确引用了这一点)。
- 对于对手池 / 自博弈动态，请检查:
    - `src/train/pool_manager.py`
    - `src/utils/custom_callback.py` 中的回调
    - `DoubleJsbSimEnv.update_opponent_models()` (在对战循环中更新池后调用)
- 对于手动控制行为和按键映射:
    - `src/utils/test_joystick_id.py`
    - `src/utils/manual_control.py`

