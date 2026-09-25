# 对手池初始化消融实验（Opponent Pool Initialization Ablation）

本轮只跑两个 arm（BDC-MLP 对照重训、第三臂「4 空槽」、中立 held-out 评测集见 §7）。
**状态：两个 arm 已于 2026-09-24 22:58:59 启动，正在跑（stage1 已完成，处于 battle cycle 1）。**

| arm | 运行目录 | stdout 日志 |
|---|---|---|
| A. 纯自博弈初始化 | `experiments/20260924_225859_selfplay_init` | `logs/ablation/selfplay_init.log` |
| B. 弱能力初始化（仅去 goalpoint） | `experiments/20260924_225859_weakinit_nogoalpoint` | `logs/ablation/weakinit_nogoalpoint.log` |

监控：`bash scripts/ablation/monitor.sh <run_dir>`

## 1. 两个 arm 的定义（臂内唯一变量 = battle 阶段的对手池初始化）

| | **A. 纯自博弈初始化** | **B. 弱能力初始化（仅去 goalpoint）** |
|---|---|---|
| 配置 | `configs/ablation/selfplay_init.yaml` | `configs/ablation/weakinit_nogoalpoint.yaml` |
| stage1（两臂一致） | 300k 步纯飞行预训练，对手池 = `base_opponent_model` 1/2/3（**与对照 BDC-MLP 的 stage1 完全相同**） | 同左 |
| battle 初始池 | 仅 1 个 = **本 run 自己的 stage1 ckpt**（`base_dir: "@self_stage1"`） | 3 个：id1=`opponent_pool/pool2/3` `469281b7…`（TrackingInitTask stage3，平飞型）、id2=`base_opponent_model/2` `978e3666…`、id3=`base_opponent_model/3` `3543907a…` |
| `whitelist` | `[]`（**无受保护对手**，4 槽全可替换；stage1 种子从第 2 个 cycle 起即可被淘汰） | `['1','2','3']`（3 受保护 + 1 动态槽，与对照同构） |
| `max_pool_size` | 4（与对照同容量） | 4 |
| 与对照的差别 | 初始化内容 + 无冻结槽（不设白名单的必然结果） | **仅**把 id1 的 goalpoint（`83041d60…`）换成平飞型对手 |
| 池演化预期 | cycle1 后池 1/4 → 无条件插入 `2`；cycle2、3 再插入 `3`、`4`；第 4 轮起挑战全池、替换最弱被击败者 | cycle1 后池 3/4 → 无条件插入 `4`；此后只能替换 4 号槽（1/2/3 白名单保护）——与对照一致 |

对照（后补）配置 = 顶层 `opponent_pool` 用 `base_opponent_model` 1/2/3 + `whitelist` 1/2/3 + `max_pool_size 4`，且**不加** stage 级覆盖。

## 2. 冒烟测试结论（`smoke/` 子目录，文件名与实验名 `_smoke` 双重标记）

正式配置与冒烟配置的唯一差别：`num_cpu 30→4`；stage1 `total_timesteps 300000→20000`、`eval_freq→5000`；
`battle_step 1e7→20000`、`battle_cycles 30→1`、`n_episodes_eval 50→2`、`threshold_timesteps 2e7→20000`。

两个冒烟均 **exit 0**，关键校验点全部通过：

| 校验点 | A 臂结果 | B 臂结果 |
|---|---|---|
| stage1 播种 | `Seeded from base: .../base_opponent_model (ids=['1','2','3'])`、3 个对手模型 | 同左 |
| battle 池切换 | `Clearing existing pool at: <run>/opponent_pool` → `Seeded from SELF stage1 ckpt: <run>/stage1/<ts>_stage1_agent (ids=['1'])` | `Seeded from base: .../base_opponent_model_weakinit (ids=['1','2','3'])` |
| battle 初始对手数 | `NNVecEnv initialized with 1 opponent models`，**无**“白名单已激活”行 | `NNVecEnv initialized with 3 opponent models` + `白名单已激活: ['1','2','3']` |
| cycle1 后池变化 | `对手池未满 (当前 1/4)。正在添加新模型为 '2'` | `对手池未满 (当前 3/4)。正在添加新模型为 '4'` |
| 池名册 md5 复核 | `1` = 本 run stage1 ckpt（一致） | `1/2/3` = `469281b7…`/`978e3666…`/`3543907a…`（一致），`4` = cycle_1 |

冒烟产生的 `experiments/*_smoke/`（含 `20260924_225204_*`）**不是实验数据**，可随时删除。

## 3. 过程中发现并已解决的两个阻塞（务必保留此记录）

1. **"空对手池"不可行**：`train.py::_run_normal_stage` 也会传 `vec_env_kwargs={"pool_roots": pool_path}`，
   而 `NN_vec_env._find_strategy_dirs()` 在池目录为空时直接
   `raise ValueError("No valid strategy directories found")`（2026-09-24 22:44 实测复现）。
   → 因此 A 臂改用「stage1 ckpt 作为池种子」，而非空池。
2. **`--pretrained_path` 跑 battle-only 配置在仓库里是坏的**：`_find_latest_training_result()` 会选中
   刚被 logger 创建出来的 `<run>/stage2/metrics` 目录，导致
   `FileNotFoundError: <run>/stage2/metrics/best_model.zip`（2026-09-24 22:50 实测复现）。
   → 因此改为**单进程两阶段**配置 + `train.py` 的小改动（见 §4），两个 arm 各一条命令即可跑完。

## 4. 对 `src/training/train.py` 的改动（未提交，向后兼容）

```python
# 1) _setup_experiment_opponent_pool(cfg, clear=False)
#    - clear=True: 先清空 <run>/opponent_pool，避免 stage1 遗留的 1/2/3 残留污染 battle 池
#    - base_dir 支持特殊值 "@self_stage1": 用本 run 自己的 stage1 ckpt 目录作为池来源
# 2) _run_battle_stage(): 支持 stage 级 `opponent_pool:` 段作为 battle 阶段覆盖池
#    （重新播种 + 重建 PoolManager，使 stage1 与 battle 可用不同初始池）
```

不影响任何未使用 stage 级覆盖的配置（旧配置行为逐字不变）。**但控制组与后续臂必须在同一代码版本上跑**
（当前 `git rev-parse HEAD = a0a8e0240dddcb817e3471fac23b462adc733f92` + 上述未提交改动）。

## 5. 冻结不变量（两臂一致，且与对照一致）

- **MLP only**：`algorithm: PPO`、`name: ppo_1layer`、`policy: MlpPolicy`、`lr 1e-4`、`n_steps 2048`、
  `batch_size 128`、`n_epochs 10`、`gamma 0.99`、`gae_lambda 0.95`、`clip_range 0.2`、`clip_range_vf null`、
  `ent_coef 0.0`、`vf_coef 0.5`、`max_grad_norm 0.5`、`share_features_extractor true`、`net_arch [12288]`。
  **禁止**引用 `configs/unified_cfg/LSTM.yaml`。
- `env_base` 逐字一致（无 wrappers）；`stage1.env.shape: stage1`、`stage2.env.shape: stage2`。
- battle 调度：`battle_step 1e7`、`battle_cycles 30`、`threshold_timesteps 2e7`、`update_freq_episodes 1000`、
  `n_episodes_eval 50`、`eval_freq 50000`、`num_cpu 30`（决定 rollout 2048×30，不可改）。
- `seed:` 是死参数（`grep -n seed src/training/train.py` 为空）；重复实验直接再跑一遍、换 `--exp_name _r2`。

## 6. 时间预算与当前实测

| 项 | 数值 |
|---|---|
| 单臂独占跑（stage1 + 30 cycles） | ≈ 20 h（42 min/10M 步，实测 BDC-MLP 20.4 h / BC 18.6 h） |
| **本次两臂并发**（各 num_cpu=30，共约 60 env 进程） | 实测 ≈ 133k（A）/ 111k（B）步/分钟 → **约 75–90 min/cycle → 每个 run ≈ 37–45 h**，两臂并行约 1.6–1.9 天 |
| stage1（300k 步） | ≈ 3 分钟（含后置评估） |
| 冒烟（20k 步 + 1 cycle） | ≈ 4 分钟 |

（并发会互相争抢 CPU，总时长与顺序跑接近；若更在意单个结果先出来，可改为顺序跑。）

## 7. 监控与留档

```bash
bash scripts/ablation/monitor.sh experiments/20260924_225859_selfplay_init
bash scripts/ablation/monitor.sh experiments/20260924_225859_weakinit_nogoalpoint
```

注意：stage 级 logger 是 SB3 `configure(["csv","tensorboard"])`，其 `logger.info` 不写 stdout，
因此日志里**没有** “BATTLE-TRAINING Cycle N” 横幅；判进度要看
(a) 池名册 `md5sum <run>/opponent_pool/*/best_model.zip`、(b) `<run>/stage2/<ts>_cycle_N/` 目录数、
(c) stdout 日志中的 `课程更新 -> goal_point_prob` 与 `对手池未满/完整复制到/正在用新模型替换` 行、
(d) `<run>/tensorboard/battle/PPO_0/`（每 cycle 落一个 event 文件）。

每个 run 目录已写入 `arm_spec.yaml`（臂定义、种子 md5、代码版本、改动说明、日志与监控命令）。

## 8. 暂缓项

1. **用同一代码版本重训对照 `fullinit`**（BDC-MLP 设定，不加 stage 级覆盖 → 行为与旧 run 等价）。
2. **中立 held-out 评测集**：现有 `opponent_pool/test_pool` 的 1/2/3 = 对照的受保护对手、11/16/17/23 = `pool2`、
   616 = Baseline 本体，对这三档并不中立，不能用来下结论。
3. 第三臂「4 空槽 / 0 protected」：当前 `_get_opponent_action` 在无模型时返回随机动作，需先定回退语义。
4. 每臂重复 2 次（`_r2`）以估计方差。

## 9. 文件清单

```
configs/ablation/
  selfplay_init.yaml              # A 臂（stage1 + battle，battle 池 = @self_stage1，whitelist: []）
  weakinit_nogoalpoint.yaml       # B 臂（battle 池 = weakinit 1/2/3，whitelist: ['1','2','3']）
  smoke/selfplay_init_smoke.yaml          # ★ 冒烟版
  smoke/weakinit_nogoalpoint_smoke.yaml   # ★ 冒烟版
  README.md                       # 本文件
scripts/ablation/monitor.sh       # 监控脚本
base_opponent_model_weakinit/{1,2,3}   # B 臂池来源（md5 已校验）
src/training/train.py             # 未提交改动：stage 级池覆盖 + @self_stage1
```
