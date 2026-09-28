# Benchmark 评测指标的调整（回应"eval 指标包含在奖励里"的意见）

本文件记录 2026-09-28 对 benchmark 评测章节指标展示方式的改动：去掉与训练奖励重合的指标，
换成四个与奖励无关、可测量、可复现的指标，并给出复现命令与论文改动位置。

## 1. 原指标与奖励项的重合审计

被评估策略的最终奖励 = `jsb_env/jsbgym_m/task_tracking.py` 的 `shape=stage2` 五项
（`make_assessor`，行 382–438）+ 终局项（`_reward_terminal_override`，行 1000–1015）。

| 原 eval 指标 | 对应奖励/机制 | 代码位置 | 判定 |
|---|---|---|---|
| Mean track angle $\theta_{tra}$ | `relative_position`、`gunsnap_blue` 的自变量 | 385–406 | **就是奖励自变量，必须去掉** |
| Gun opportunity ratio | 与 `_update_HP` 的伤害判定完全同一条件；与 `gunsnap_blue` 作用区间重合 | 534–537 / 962 | **等于奖励/伤害机制的指示函数** |
| Total damage to opponent | 终局奖励 `(3-HP_oppo)*10` 的单调函数 | 1000–1015 | **就是终局奖励项** |
| Mean specific energy | 奖励无显式 $E_s$ 项，但 $h$ 出现在 `deck` 项、$V$ 以 $V_c$ 出现在 `closure_rate`；且正文 535 行原文承认奖励在塑造能量策略 | 391–421 | 间接相关，换成相对量 $\Delta E_s$ |
| Win/Draw/Loss、$\Delta$HP | 终局判定 | 1000–1015 | 任务目标本身，保留（不含在"能力指标"里） |

附带发现：
- `evaluator.py` 中 `adverse_angle_mean(_oppo)`、`damage_rate(_oppo)` 只声明未 `append`，记录恒为 0，不可用。
- 论文 `tab:reward_terminal` 的 $r_{hp}=w_{hp}(\mathrm{HP}-\mathrm{HP}_{oppo})$ 与代码 `(self.HP - HP_oppo)*10`（自身 HP 取常数 3）不一致。
- 论文正文 Baseline 胜率写 0.42%，而 `plot_eval_results.py` 文本块为 0.39%（新 bench8_v4 复跑会覆盖这两个数）。

## 2. 新指标（论文最终只报告前两个，均不出现在任何奖励项中）

在**与 benchmark 胜负统计相同的协议**下测量（`goal_point_prob=0`、随机初始态势、每对阵 150 局）：

| 指标 | 定义 | 为什么与奖励无关 | 论文是否使用 |
|---|---|---|---|
| $\Delta E_s$ (ft) | $\mathrm{mean}(E_s^{ego}-E_s^{oppo})$，$E_s=h+V_T^2/(2g)$ | 奖励只比较角度与距离，从不比较双方能量 | **使用** |
| LOS $R^2$ | 视线方位角与**带符号**转弯率的平方相关系数（符号取坡度方向） | 奖励中无转弯率项，无法直接优化 | **使用** |
| Turn rate (deg/s) | 协调转弯的 aerodynamic turn rate $\omega=g\sqrt{n^2-1}/V$，$n$ 取 JSBSim `forces/load-factor` | 奖励中无转弯率、无法向过载 | 已按要求删除（数据保留） |
| Corner-speed adherence | $\mathrm{mean}\exp(-\lambda\lvert V-V_c\rvert/V_c)$，$\lambda=5$，仅在机动段（$n\ge2$）；$V_c=\sqrt{2n_{\max}W/(\rho S C_{L,\max})}$，$n_{\max}=9$、$C_{L,\max}=1.6$（JSBSim F-16 模型的操纵/气动限制），$W,\rho$ 为逐步实测 | 奖励中完全没有空速项（closure rate 用的是距离变化率） | 已按要求删除（数据保留） |

结果（`experiments/capability_profile/paper/summary.md`，对手集 = `bench8_v4`）：

| Model | Win | $\Delta E_s$ (ft) | LOS $R^2$ | (Turn rate) | (Corner adherence) | mean TAS (ft/s) |
|---|---|---|---|---|---|---|
| Baseline | 2.2% | −2941 | 0.24 | (7.86) | (0.449) | 509 |
| BC | 22.4% | +1526 | 0.22 | (10.49) | (0.415) | 720 |
| BD | 38.8% | +1815 | 0.24 | (10.90) | (0.372) | 742 |
| BDC-MLP | 40.2% | +3758 | 0.35 | (9.96) | (0.334) | 838 |
| BDC-LSTM | 61.6% | +4679 | 0.36 | (10.12) | (0.270) | 875 |

要点：
- $\Delta E_s$ 与胜率单调一致；BDC-LSTM 是对 8 个对手**全部** $\Delta E_s>0$ 的唯一模型（最小 +2955 ft）；Baseline 有 6/8 为负。
- LOS $R^2$ 把 BDC 两个变体（0.35/0.36）与其余三个（0.22–0.24）分开。
- 括号内的 turn rate / corner adherence 仅作记录，不再出现在论文中。其中 corner adherence 是**反向**的（Baseline 0.449 最高、LSTM 0.270 最低），原因是它实际度量"飞得快不快"：$V_c\approx660$ ft/s，Baseline 平均 509 ft/s、LSTM 875 ft/s；这也正是它不适合作为能力指标的原因。
- 若需要与航向角速率口径的 $R^2$ 对照，同一份数据里另有 `react_r2`（Baseline 0.29 / BC 0.24 / BD 0.27 / MLP 0.33 / LSTM 0.25）；论文用的是带符号气动转弯率口径 `react_r2_phys`。

## 3. 产物与复现

代码（均为新增，未修改 `jsb_env/**`、`src/**`、`scripts/opponent_behavior/**`）：

- `scripts/ablation/measure_capability_profile.py`
  在标准 benchmark 协议下逐 step 读取 JSBSim 量（`forces/load-factor`、`rho`、`weight`、`vc-kts`、
  `vtrue-fps`）并聚合四个指标。
  ```bash
  for i in 0 1 2 3; do OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
    python scripts/ablation/measure_capability_profile.py --group paper --episodes 150 \
    --shards 4 --shard $i --bench-file /tmp/bench_new_eval_1000/bench8_v4.json & done; wait
  python scripts/ablation/measure_capability_profile.py --group paper --merge
  ```
  （`--group ablation` 为对手池初始化消融两臂；`--bench-file` 可指向任意对手集 JSON，
  默认用 `opponent_pool/test_pool` 的 8 个老对手。）
- `scripts/ablation/make_paper_tables_eval.py` → 生成 `configs/ablation/tables_eval_metrics.tex`（论文表 7/8 的数据源）。
- `scripts/ablation/run_model_behaviour.py`
  复用对手行为研究（`scripts/opponent_behavior/`）的**无奖励自由飞行**协议测量 5 个被测模型，
  产物 `experiments/model_behavior/`；用于交叉验证（自由飞行下 turn rate / R² / corner adherence 的量级一致）。
- `scripts/ablation/measure_turn_performance.py`
  JSBSim 级探针，用于确认 `forces/load-factor`、`vc-kts`、CL 等的量纲与合理性（记录在 `experiments/model_behavior/turn/`）。

数据：`experiments/capability_profile/{paper,ablation}/merged.json|summary.md`、`experiments/model_behavior/`。

## 4. 论文（`小论文-终稿/sn-article.tex`）改动清单

备份：`sn-article.tex.bak_before_eval_metrics`（首次改动前）。

1. §4.3.1：删除原 `Fig. combat_metrics`（`graph/model_capabilities.png` 四联图：gun opportunity / specific energy / track angle / damage），
   替换为 **Table `tab:reward_free_metrics`**（只含 $\Delta E_s$ 与 LOS $R^2$），并新增 **Figure `fig:capability_metrics`**
   （同一组数据的双栏柱状图，脚本已重写，见下）。
2. §4.3.1 正文：删除"奖励函数与指标重合"的解释段落（不放在正文），改为在指标定义段开头用一句话说明表/图展示了什么；
   删除奖励项细节说明句、turn rate 相关句、corner-speed 段落与公式 (21)。
3. §4.5：`tab:init_ablation_results` 删除 Gun opportunity 与 Turn rate 列，保留 LOS $R^2$；正文相应改写。
4. §4.3.2：把"better energy management"改为"larger energy margin ... in agreement with Table 7"。
5. Conclusion："in terms of win rate and energy management" → "win rate and of the energy margin that it holds over the opponent"。
6. 绘图脚本：`src/evaluation/metric/plot_eval_results.py` 的 `plot_capability_metrics` 重写为读取
   `experiments/capability_profile/paper/merged.json`，绘制 $\Delta E_s$ 与 LOS $R^2$ 的双栏柱状图
   （`model_win_rates.png` 部分未改动）。生成后拷贝到 `小论文-终稿/graph/model_capabilities.png`：
   ```bash
   MPLCONFIGDIR=/tmp/mplcache MPLBACKEND=Agg conda run --no-capture-output -n js python \
       src/evaluation/metric/plot_eval_results.py   # 需要 seaborn：用 js 环境
   cp src/evaluation/plot/model_capabilities.png 小论文-终稿/graph/model_capabilities.png
   ```
7. 编译通过：`latexmk -pdf sn-article.tex`（exit 0，无 undefined reference）；表 7 在 p.28、图 9 在 p.29、表 8 在 p.34。

## 5. 待决 / 注意

- **对手集变更**：环境中已在跑 `bench8_v4`（新 8 对手）的 5 模型 × 8 对手 × 1000 局复跑
  （输出 `/tmp/bench_new_eval_1000/`，脚本 `/tmp/bench_new_eval_1000/run_eval.py`）。
  表 7 的四个指标用的就是这一对手集；等该复跑完成，图 8 的胜率与正文数字会一起更新。
- **消融表口径**：`tab:init_ablation_results` 的胜负/$\Delta$HP 仍来自旧 bench8（1000 局/臂），
  新增的两列来自 bench8_v4（150 局/臂），表注已声明；若希望完全统一，需要按新对手集重跑两臂。
- **LOS $R^2$ 估计量**：论文 §4.3.1 中对手行为表的 $R^2\le0.11$ 用的是航向角速率估计量，
  新表用的是带符号气动转弯率估计量（已说明）；若要严格可比，需要用同一定义重算 8 个对手。
- `adverse_angle_mean`、`damage_rate` 在 `src/evaluation/evaluator.py` 中恒为 0（未 append），任何表都不应使用。
