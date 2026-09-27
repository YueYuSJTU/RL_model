# 初始化消融：证据与可复现性索引

本文件记录论文小节 `configs/ablation/section_init_ablation.tex` 中**每一个数字的来源**，
以及产生它的脚本与原始产物路径。所有实验均在 2026-09-24 ~ 09-26 完成。

## 1. 三个实验臂（训练）

| 臂 | 配置 | 运行目录 | 时长 | 备注 |
|---|---|---|---|---|
| Full initialization（对照，= 论文 BDC-MLP 设定） | `experiments/20260428_141850_best_train_mlp2/full_config.yaml` | `experiments/20260428_141850_best_train_mlp2` | 20.4 h | 旧代码版本（HP=5、`betaR=-2.5`） |
| Weak initialization（去 goalpoint） | `configs/ablation/weakinit_nogoalpoint.yaml` | `experiments/20260924_225859_weakinit_nogoalpoint` | 37.6 h | 与 A 臂同批、同代码版本 |
| Self-play initialization（纯自博弈） | `configs/ablation/selfplay_init.yaml` | `experiments/20260924_225859_selfplay_init` | 38.6 h | 同上 |

- 三臂共用同一 stage1 配置（对手池 = `base_opponent_model` 1/2/3），battle 阶段池通过 stage 级
  `opponent_pool` 覆盖（`src/training/train.py` 的向后兼容改动）。
- 纯自博弈臂的池种子 = **本 run 自己的 stage1 ckpt**（`base_dir: "@self_stage1"`），无白名单。
- 日志：`logs/ablation/{selfplay_init,weakinit_nogoalpoint}.log`；
  池演化：`experiments/<run>/POOL_HISTORY.md`。

## 2. 训练稳定性（L1）

- 脚本：`scripts/ablation/analyze_stability.py`
- 产物：`experiments/ablation_L1_stability.md`、`experiments/ablation_L1_stability.png`
  （论文图：`configs/ablation/figures/init_ablation_stability.png`）
- 数据源：各 run 的 `tensorboard/<ts>/PPO_*/events.out.tfevents*`（每 cycle 一个文件）
- 结论数字：早段/晚段均值、斜率、波动、最差回撤、崩盘次数、峰值（见 md 表格）。

## 3. 最终泛化（L2）

- 基准：`configs/ablation/bench8_v2.yaml`（8 个中立对手，md5 审计见 §5）
- 脚本：`scripts/ablation/eval_ablation.py --set bench8 --episodes 1000 --shards 12 --shard <i>`
- 原始记录：`experiments/ablation_eval/bench8/shard*.jsonl`（每行一个对阵的完整指标）
- 汇总与作图：`scripts/ablation/analyze_l2.py` → `experiments/ablation_L2_results.md`
  + `experiments/ablation_L2_{wdl,heatmap,capability}.png`
- 协议：`Evaluator.run_match`，1000 局/对阵，初始态势随机化，`goal_point_prob=0`。

## 4. 交叉对战（L3）

- 脚本：同 `eval_ablation.py --set cross --episodes 300`
- 原始记录：`experiments/ablation_eval/cross/shard*.jsonl`；图：`experiments/ablation_L3_crossplay.png`

## 5. bench8-v2 的中立性（md5 审计）

- 构建脚本：`scripts/ablation/build_bench8_v2.py`；定义：`configs/ablation/bench8_v2.yaml`
- 被排除（参与过三臂训练的外部对手）：

| 模型 | md5 | 参与方式 |
|---|---|---|
| `base_opponent_model/1`（goalpoint） | `83041d60afcb0efa6a599d188922939c` | 三臂 stage1 池 + 对照 battle 池 |
| `base_opponent_model/2`（0710） | `978e3666bf41473618f04c764b6d4a8e` | 同上 |
| `base_opponent_model/3`（0713） | `3543907abceaca018f3f1219fdfba141` | 同上 |
| `opponent_pool/pool2/3` | `469281b7897bc9c304153f0e1b0bf8ea` | 弱初始化臂 battle 池 id 1 |

- 旧 8 对手（`opponent_pool/test_pool`）因此**不再用于本消融**；其成员 1/2/3/23 正是上表四个模型。

## 6. 新基准的飞行行为测量（与旧基准可比性）

- 脚本：`scripts/ablation/run_bench8_behaviour.py`（复用 `scripts/opponent_behavior/` 的协议，未改动其代码）
- 原始产物：`experiments/opponent_behavior/raw/bh_bench8_*.json`（新 8 个）；
  旧 8 个为既有的 `raw/bh_{1,2,3,11,16,17,23,616}.json`
- **协议一致性校验**：用本脚本复测旧对手 `test_pool/23`，得到
  `|bank|=3.2, climb=+33.2, ele=-0.165, speed=443.6, closure=-500`，与既有测量/论文表完全一致。

## 7. 池动力学机制指标

- 来源：训练日志里 `PoolManager` 的挑战记录行
  `vs. 对手 'N': Win=..%, Loss=..%, Draw=..%, Opponent fall=..%, HP_avg=.., HP_oppo_avg=.. --> 得分: ...`
- 解析：见本轮分析（脚本片段，结果写入论文小节）：
  - 纯自博弈臂：108 条挑战记录，挑战者平均胜率 **0.26%**、平局率 99.7%、仅 4 条记录出现过胜利，
    池仍被刷新 27 次（平均 HP 差 +0.03）。
  - 弱初始化臂：106 条挑战记录，挑战者平均胜率 **38.9%**，刷新由真实胜负驱动。
- 池名册与替换链：`experiments/<run>/POOL_HISTORY.md`。

## 8. 复现命令（按顺序）

```bash
PY=/home/ubuntu/miniconda3/envs/js_gpu/bin/python

# L1 稳定性
$PY scripts/ablation/analyze_stability.py

# 新基准构建 + 中立性审计
$PY scripts/ablation/build_heldout.py          # 22 个中立候选
$PY scripts/ablation/prepare_opponents.py      # 可加载性检查 + 老格式 shim
$PY scripts/ablation/build_bench8_v2.py        # 最终 8 对手基准

# L2/L3 评测（分片并行）
for i in $(seq 0 11); do $PY scripts/ablation/eval_ablation.py --set bench8 --episodes 1000 \
    --shards 12 --shard $i > logs/ablation/eval_bench8_shard$i.log 2>&1 & done; wait
for i in 0 1 2; do $PY scripts/ablation/eval_ablation.py --set cross --episodes 300 \
    --shards 3 --shard $i > logs/ablation/eval_cross_shard$i.log 2>&1 & done; wait

# 汇总与作图（直接读分片，无中间合并步骤）
$PY scripts/ablation/analyze_l2.py

# 定性轨迹图
$PY scripts/ablation/plot_engagement.py --episodes 12 \
  --model Ctrl=experiments/20260428_141850_best_train_mlp2/stage2/20260429_104611_cycle_30 \
  --model B=experiments/20260924_225859_weakinit_nogoalpoint/stage2/20260926_110948_cycle_30 \
  --model A=experiments/20260924_225859_selfplay_init/stage2/20260926_124349_cycle_30 \
  --opponent experiments/20250715_105409/stage2/20250715_105533_TrackingTask_ppo_1layer1 \
  --out experiments/ablation_L3_trajectory_combat.png

# 新基准行为测量
PYTHONDONTWRITEBYTECODE=1 $PY scripts/ablation/run_bench8_behaviour.py
```

---

## 9. 最终结果快照（2026-09-27 定稿，24 条记录 = 3 臂 × 8 对手 × 1000 局）

| 臂 | 胜率 [95% CI] | 平局 | 负率 | ΔHP | 枪口机会 | 跟踪角 | 总伤害 | 比能量 |
|---|---|---|---|---|---|---|---|---|
| Full init（对照） | 42.49% [41.41,43.57] | 57.24% | 0.26% | +1.80 | 0.156 | 0.735 | 2.02 | 21068 |
| Weak init | **56.84%** [55.75,57.92] | 43.09% | 0.07% | **+2.06** | **0.175** | **0.668** | **2.15** | 21946 |
| Self-play init | 14.44% [13.68,15.22] | 84.09% | 1.47% | +0.99 | 0.078 | 1.004 | 1.32 | 20710 |

按对手类型（胜率）：level-flight(4) 25.1 / 66.6 / 84.8；combat(2) 6.7 / 30.6 / 55.4；
goal-point(1) 1.6 / 5.0 / 0.2；weak-ref(1) 0.1 / 7.3 / 4.7（顺序为 A / Ctrl / B）。

交叉对战（300 局/有序对，胜率 / ΔHP）：
Ctrl vs A 4.0% (+0.81)、A vs Ctrl 0.0% (−0.83)、Ctrl vs B 6.7% (−0.27)、B vs Ctrl 7.7% (+0.26)、
B vs A 1.7% (+0.86)、A vs B 0.0% (−0.85)。

训练稳定性（per-cycle eval return）：A 370→275（−0.5/cycle，峰值 539，崩盘 8）；
B 693→706（+13.8，峰值 1282，崩盘 3）；Ctrl 759→744（+19.9，峰值 1324，崩盘 7）。

池替换机制：A 108 次挑战、挑战者平均胜率 0.26%（99.7% 平局，仅 4 次出现过胜利）、刷新 27 次（HP 差 +0.03）；
B 106 次挑战、挑战者平均胜率 38.9%。

## 10. bench8-v2 的两次角色贴合优化（依据实测行为）

| 槽位 | 初选 | 定稿 | 理由（实测 5 指标） |
|---|---|---|---|
| 旧 23（平飞 st3） | pool2/12 | **pool2/7** | pool2/12 是"大坡度俯冲"（\|φ\|=74.7, climb −263），旧槽位是"浅坡度近定高"；pool2/7（29.8, +33.8）更贴合 |
| 旧 616（弱参照） | 0710-stage1 | **best_models/8** | 旧 616 画像 60.1/+6.4/−0.618/475.5/−394.4；best_models/8 为 53.6/+14.3/−0.748/459.3/−401.0，逐项吻合 |

两处替换后，NB7/NB8 两列各重跑 3 臂 × 1000 局；旧的对应记录已从分片中删除，故当前每臂恰好 8 个对手。

## 11. 交付物清单

| 类型 | 路径 |
|---|---|
| 论文小节（LaTeX，可编译） | `configs/ablation/section_init_ablation.tex` |
| 表格（自动生成，已内联进小节） | `configs/ablation/tables_init_ablation.tex` |
| 基准定义（版本库） | `configs/ablation/bench8_v2.yaml` |
| 图（6 张） | `configs/ablation/figures/init_ablation_{stability,wdl,heatmap,capability,crossplay,trajectory}.png` |
| 机器可读结果 | `experiments/ablation_eval/bench8/*.jsonl`、`experiments/ablation_L2_results.md`、`experiments/ablation_L1_stability.md` |
| 池演化记录 | `experiments/<run>/POOL_HISTORY.md` |
| 行为测量原始数据 | `experiments/opponent_behavior/raw/bh_bench8_*.json`、`experiments/bench8_v2/behaviour_comparison.md` |

图片需复制到论文目录：`cp configs/ablation/figures/init_ablation_*.png 小论文-终稿/graph/`
