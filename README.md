# JSBSim 双飞机空战强化学习库

## 介绍

这是一个基于 [JSBSim](http://jsbsim.sourceforge.net/) 的双飞机空战强化学习库。在 [jsbgym](https://github.com/Gor-Ren/jsbgym) (JSBSim的Gymnasium封装) 的基础上，扩展了新的任务：

-   **`goal_point` 任务**: 目标点跟随任务，主要用于初始化对手。
-   **`tracking` 任务**: 双飞机1v1空战对抗环境。

---

## 0. 环境配置

安装本项目所需依赖。核心依赖包括：

-   `jsbgym`
-   `stable-baselines3`

*具体的安装步骤和其余零散库将在项目稳定后补充。*

---

## 1. 启动训练

通过 `src.stageTrain` 模块启动训练。

**启动参数:**

-   `--config`: 训练配置文件路径 (e.g., `"configs/stage_train_config.yaml"`)。
-   `--eval_pool`: 评估时使用的对手池路径 (e.g., `"/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool3"`)。
-   `--pretrained_path`: 预训练模型路径。用于微调或继续训练。

**训练模式:**

-   **预训练模式**: 不提供 `--pretrained_path` 参数。脚本将自动创建一个以当前日期和时间命名的文件夹来保存训练记录。
-   **微调模式**: 提供 `--pretrained_path` 参数，指定一个已有的日期文件夹。脚本会自动搜索该文件夹下最新的 `stage` 并进行微调。
    > **注意**: 微调模式的配置文件 (`config`) 只支持一个 `stage`。

**示例:**

```bash
python -m src.stageTrain \
    --config "configs/stage_train_config.yaml" \
    --eval_pool "/home/ubuntu/Workfile/RL/RL_model/opponent_pool/pool3" \
    --pretrained_path "experiments/20250921_162658"
```

---

## 2. 配置文件编写

所有配置文件位于 `configs/` 文件夹下。

-   `train_config` 中的 `agent` 和 `env` 参数分别指向对应Agent和环境的 `.yaml` 配置文件名。
-   **多阶段训练**: 本库支持多阶段训练。您可以在配置文件中定义多个 `stage` (最多10个)，并为每个 `stage` 指定不同的奖励函数。
-   **自定义奖励函数**: 修改奖励函数需要同步修改任务定义文件。具体代码示例请参考 `jsb_env/jsbgym_m/task_tracking.py`。

---

## 3. 评估实验

### 3.1 对战评估

此脚本用于评估某个已训练模型的胜率，结果将以 `.txt` 和 `.npy` 格式存储。

**运行方式:**

```bash
./evalate.sh
```

### 3.2 goal point模型评估
```bash
python ./test/evaluate_goal_point.py --n_episodes 1000 --exp_path "experiments/goal_point/20250922_095754/stage1/20250923_203916_GoalPointTask_ppo_1layer"
```

---

## 4. 展示实验

通过可视化方式回放和展示实验效果。

**渲染支持:**

-   `tracking` 任务: 支持 `human`, `anim3d`, `flightgear` 渲染。
-   `goal_point` 任务: 支持 `human` 渲染。

**运行方式:**

```bash
./show.sh
```