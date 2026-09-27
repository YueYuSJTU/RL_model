# Benchmark 对手行为量表（固定目标点体系）

本目录用**固定目标点体系**测量 8 个 benchmark 对手的**自身行为**，并据此划分等级。
全程**不使用任何外部模型当评测尺子**：量表上的每个数字都是被测对手自己的量。

> **约束遵守**：本目录及其产物只写入 `scripts/opponent_behavior/` 与
> `experiments/opponent_behavior/`。`jsb_env/**`、`src/environments/**`、
> `src/agents/**`、`src/evaluation/**`、`src/utils/custom_callback.py`
> 只被 import、**从未修改**（已用 `git status` 与 `find -newermt` 核对：零改动、
> 零 `.pyc` 写入）。所有脚本通过 `PYTHONDONTWRITEBYTECODE=1` 运行，防止 Python
> 往上述目录写 `__pycache__`。

---

## 1. 测量协议

### 1.1 接线方式（全部使用仓库既有机制，未改代码）

| 环节 | 做法 | 依据 |
|---|---|---|
| 被测对手所在席位 | **对手席位**（model2） | 只有对手席位的相对几何观测会被目标点替换 |
| 目标点如何被"看见" | `update_task_parameters(goal_point_prob=1.0)` 打开环境**原生 goal_point 模式** | `task_tracking.py:849-850`：该模式下 `_cal_oppo_state` 用 `self.goal_point` 计算**对手**的相对几何 |
| 目标点固定 | 运行时把实例方法 `get_goal_point` 替换为返回固定点 | `_new_episode_init` 每回合调用一次（`task_tracking.py:1043`） |
| 初始几何固定 | 运行时把实例方法 `_maybe_init_pair_geometry` 替换为写入固定值 | 该方法本身只负责赋值 `_init_*` 并置缓存位 |
| ego 席位 | 脚本化零学习 **trim drone**（常值指令） | 只为让仿真成立；放在 20000 ft 之外并爬升，永不进入机炮包线，且对所有对手完全相同 |

目标点坐标约定：**本地 NED 的 (north_ft, east_ft, altitude_sl_ft)**，原点为 ego 起始位置
（`_update_extra_properties` 第 901-916 行把 `set_oppo_position` 与
`(ned_X, ned_Y, altitude_sl_ft)` 直接相减，据此确认）。

固定初值：drone NED(0, 0) / 13000 ft / 航向 0°；被测对手 NED(0, −20000) / 10000 ft / 航向 0°。
目标点以**被测对手起始点**为参考定义，所以 `bearing_deg` 即"相对其初始机头方向"。

### 1.2 目标点体系（16 点）

| 组 | 点数 | 覆盖 |
|---|---|---|
| 方位扫掠 `bearing_*` | 8 | 0/±45/±90/±135/180°，距离 12000 ft、高度 14000 ft |
| 距离扫掠 `range_*` | 3 | 正前方 6000/12000/20000 ft |
| 高度扫掠 `alt_*` | 3 | 正前方 8000/14000/20000 ft |
| 交叉项 `cross_*` | 4 | ±90° 方位 × 8000/20000 ft 高度 × 20000 ft 距离 |

去重后共 **16 点**（`goalpoint.build_battery()`）。

### 1.3 测什么

全部为被测对手自身的量（`goalpoint.goalpoint_descriptors`）：

- `straightness` 净位移/航迹长度（1.0 = 完全不理会目标、直线飞行）
- `yaw_rate_deg` 每步平均航向变化
- `bank_std_rad` / `bank_abs_mean_rad` 滚转活动强度
- `alt_std_ft` / `speed_mean_fps` / `Es_ft = h + V²/2g`
- **`point_frac`** 机头指向目标点 <30° 的时间占比（主量表轴）
- **`gun_frac`** 对目标点构成有效射击条件的时间占比（机头 <15° 且 500 ≤ 距离 ≤ 4500 ft，与 `GUN_ANGLE`/`GUN_RANGE` 一致）
- `closure_mean_fps` 朝目标点的平均闭合率
- `dist_mean_ft` / `dist_min_ft` / `dist_end_ft` 到目标点的距离

### 1.4 确定性

`repeats=1` 与 `repeats=2` 的逐点 `point_frac` **完全一致**（实测 0 处差异），
即该探测是可复现的确定性测量。

---

## 2. 结果

### 2.1 等级划分

分级判据为**实测的"持续指向"目标点数**：`point_frac ≥ 0.30` 的目标点个数。
分档边界取实测计数的**自然间隙**（下降 ≥ 2），不含任何人为性能常数。

| 对手 | 来源家族 | reward set | 持续指向 | 平均指向率 | 枪口率 | 平均闭合 | 直线度 | 等级 |
|---|---|---|---|---|---|---|---|---|
| **616** | legacy-run | Tracking/stage2 | **12/16** | 0.384 | 0.0418 | +170 | 0.659 | **A** |
| **2** | protected-init | Tracking/stage2 | **4/16** | 0.193 | 0.0053 | +154 | 0.478 | **B** |
| 23 | level-flight | TrackingInit/stage3 | 2/16 | 0.064 | 0.0077 | −99 | **1.000** | C |
| 11 | level-flight | TrackingInit/stage3 | 1/16 | 0.045 | 0.0052 | −316 | **0.999** | C |
| 16 | level-flight | TrackingInit/stage4 | 1/16 | 0.040 | 0.0000 | −405 | **0.976** | C |
| 1 | protected-init | GoalPoint/stage1 | 0/16 | 0.080 | 0.0010 | +192 | 0.351 | C |
| 3 | protected-init | Tracking/stage2 | 0/16 | 0.027 | 0.0000 | −412 | **0.986** | C |
| 17 | level-flight | TrackingInit/stage5 | 0/16 | 0.026 | 0.0000 | −292 | **0.955** | C |

**核心结论**：8 个对手里只有 **2 个**能对指令目标点作出实质响应（616 强、2 部分），
其余 6 个**完全不理会**被赋予的目标点——其中 5 个 `straightness ≥ 0.95`（笔直飞过），
且 `closure` 为负（持续远离目标点）。

### 2.2 一个必须标注的退化目标点

`range_20000`（正前方 20000 ft、高度 14000 ft）**不能用于判能力**：直飞的飞机也会
"顺便"掠过它，产生瞬时指向。实测证据——所有 6 个不追踪的对手在该点上
`point_frac` 分别为 0.41/0.45/0.38/0.22/0.09/0.17，而它们在其他点上几乎全为 0。
**6 个对手中每一个的"唯一 1/16"命中点都是 `range_20000`。**
因此判据采用"持续指向 ≥30%"而非"曾进入枪口"，以排除这类偶然掠过。

### 2.3 与 reward 定义的交叉印证

`TrackingInitTask` 的 stage3/4/5（对手 11/16/17/23）奖励项**只有**
`small_roll / small_throttle / deck1 / action_penalty / altitude_contain`
（`task_tracking.py:1190-1301`），**完全没有距离、航迹角、瞄准项**；且其
`_is_hp_zero` 直接 `return False`（`task_tracking.py:1313-1314`）。
它们在本次探测中"直线飞过、不理会目标"的表现，与其训练目标完全一致——
**是设计使然，不是测量失败**。

---

## 3. 必须向审稿人/合作者说明的三条保留意见

1. **对手 1（GoalPointTask 导航策略）出现 0/16 持续指向，与预期不符。**
   按理导航策略应当飞向目标点。可能原因有二：(a) 它在本容器环境（`TrackingTask`
   + goal_point 模式）下确实不追踪；(b) `ObsAdaptingModel._adapt_observation_for_task`
   对 GoalPointTask 走的是**硬编码分支**（按维度切分 + 丢弃 `extra[...,6]`），
   可能与本环境的特征顺序不匹配。
   **在写进论文前必须单独验证**（例如直接对比该模型在 `GoalPointTask` 环境里的
   目标点跟随表现）。本目录的结论**不依赖**对手 1。

2. **容器任务的训练分布差异。** 测量在 `TrackingTask` 容器中以原生 goal_point 模式进行，
   而 11/16/17/23 训练于 `TrackingInitTask`、1 训练于 `GoalPointTask`。
   本测量回答的是"**给定一个指令目标点，该策略是否作出目标导向的行为**"，
   这是一个标准化刺激，但与各自训练分布并不相同。已在协议中显式说明。

3. **脚本化"定直飞 drone"不可行（已实测排除）。** 见
   `calibrate_level_probe.py`：常值舵面**无法配平**该 F-16 模型（42 组
   (elevator, throttle) 扫掠中最好也有 2396 ft/60 s 漂移）；闭环定高控制器在前
   ~35 s 稳定于 9900 ft，随后因长周期模态发散、饱和进深失速而崩塌。
   故 ego 席位改用**常值指令的 trim drone**，其唯一职责是维持仿真成立。

---

## 4. 如何复现

```bash
# 全部测量（8 对手 × 16 目标点 × 2 重复），6 路并行，约 2 分钟
bash scripts/opponent_behavior/run_goalpoints_all.sh 2 6

# 汇总成表
python scripts/opponent_behavior/analyze.py

# 出图
python scripts/opponent_behavior/plot_figures.py
```

产物：

```
experiments/opponent_behavior/
  raw/gp_<对手>.json            逐目标点原始测量
  goalpoint_battery.csv         对手 × 目标点 明细表
  opponent_capability.csv       对手级汇总表（含等级）
  analysis.json                 绘图用汇总
  figures/fig1_behaviour_scale.png    (a) 量表梯 (b) 行为向量 (c) 逐目标点图
  figures/fig2_behaviour_map.png      直线度–指向率行为地图
  altitude_hold.csv             各对手自身高度保持能力（方法学记录）
  out_of_scope/                 已被否决的"用外部模型评测"链路，仅留档
```

| 脚本 | 作用 |
|---|---|
| `goalpoint.py` | 目标点体系、trim drone、环境接线、行为描述量 |
| `run_goalpoints.py` | 单对手测量 CLI |
| `run_goalpoints_all.sh` | 全部对手并行测量 |
| `analyze.py` | 汇总 + 分级 |
| `plot_figures.py` | 两张图 |
| `measure_hold.py` | 各对手自身高度/速度保持（方法学记录） |
| `calibrate_level_probe.py` | 脚本化 drone 不可行的实测证据（方法学记录） |
| `common.py` | 固定环境配置、8 对手溯源登记、加载与 IO |
