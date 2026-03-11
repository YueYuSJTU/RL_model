



### 第一部分：PDF核心空战技巧总结

该文件主要围绕**1对1、同类型战机（Similar Aircraft）、仅使用机炮（Guns Only）**的空战场景展开。核心技巧可以归纳为以下几个维度：

#### 1. 追踪曲线（Pursuit Curves）
*   **前置追踪（Lead Pursuit）**：机头指向目标前方，用于快速拉近距离（增加Closure），但会导致夹角（AOT）增大。
*   **纯追踪（Pure Pursuit）**：机头直接指向目标，产生中等接近率，并向目标展示最小的截面积。
*   **滞后追踪（Lag Pursuit）**：机头指向目标后方，用于减缓或停止接近，防止超越（Overshoot），同时保持或减小AOT。

#### 2. 空间机动与能量管理（Out-of-Plane Maneuvers）
*   **高低Yo-Yo（High/Low Yo-Yo）**：
    *   **High Yo-Yo**：当接近率过高且面临超越危险时，拉起机头用动能换取势能（减速），防止超越目标，随后在顶部切入改变追踪角度。
    *   **Low Yo-Yo**：当距离较远、自身盘旋能力不足以拉出前置量时，向下俯冲用势能换取动能（加速）和重力辅助，缩小转弯半径以获得前置量。
*   **滞后滚转（Lag Displacement Rolls / Barrel-roll attack）**：在近距离、高接近率时，通过垂直方向的拉起并做桶滚，以消耗前进速度，防止超越目标。

#### 3. 转弯与交汇战术（Turn Tactics）
*   **提前转弯（Lead Turn）**：在双方交汇（Pass）前提前开始转向对手，利用飞行路径分离（Flight-Path Separation）在交汇后瞬间获得角度优势。
*   **头对头（Nose-to-Nose）与头对尾（Nose-to-Tail）盘旋**：
    *   *Nose-to-Nose*：更看重**转弯半径（Turn Radius）**，速度较慢、半径较小的一方占优。
    *   *Nose-to-Tail*：更看重**转弯率（Turn Rate）**，角速度快的一方占优。

#### 4. 经典剪刀机动（Scissors）
*   **水平剪刀（Flat Scissors）**：一系列水平面内的“头对头”交汇与超越，核心在于**减速能力**，谁能最慢、最快地改变机头指向谁就赢。
*   **滚转剪刀（Rolling Scissors）**：当发生高速超越时，双方在三维空间中互相围绕做桶滚。核心在于**能量管理**和向下的垂直速度控制。
*   **防御性螺旋（Defensive Spiral）**：被咬尾时进行的极小半径、近乎垂直向下的滚转剪刀，利用机动改变平面来破坏敌方的机炮瞄准线（Spoil guns-tracking solution）。

#### 5. 两大战略流派（Angles Fight vs. Energy Fight）
*   **角度战（Angles Fight）**：不顾能量损耗，以最快速度获得开火位置。经常使用最大过载（Max-G）转弯，寻求机炮的“急促射击（Snapshot）”。
*   **能量战（Energy Fight）**：保持最佳盘旋速度（Corner Speed）或进行爬升，消耗敌机能量（速度/高度），在建立足够的**比能量（Specific Energy）**优势后，再将其转化为射击优势。

---

### 第二部分：强化学习模型评估指标与公式计算

为了评测你的RL模型，我们需要将上述战术直觉转化为客观的标量。假设我们的智能体为 $A$ (Agent)，对手为 $B$ (Bogey)。

定义基础状态向量：
*   位置： $\vec{P}_A, \vec{P}_B \in \mathbb{R}^3$
*   速度向量： $\vec{V}_A, \vec{V}_B \in \mathbb{R}^3$ （速率记为 $V_A = \|\vec{V}_A\|$）
*   相对距离矢量： $\vec{R} = \vec{P}_B - \vec{P}_A$ （标量距离为 $R = \|\vec{R}\|$）

#### 指标1：综合角度优势得分（Angular Advantage Score, AAS）
**目的**：评估模型是否掌握了“角度战（Angles Fight）”中获取阵位优势的能力。
**理论基础**：文中强调要将机头指向敌机（减小AON），同时迫使敌机尾部对准自己（减小AOT）。
**公式**：
首先计算天线偏角 $\text{ATA}$ (Angle Off Nose) 和 目标尾部偏角 $\text{AOT}$ (Angle Off Tail，注意：AOT=0表示完美的六点钟正后方咬尾)：
$$ \text{ATA}_A = \arccos\left( \frac{\vec{V}_A \cdot \vec{R}}{V_A \cdot R} \right) \in [0, \pi] $$
$$ \text{AOT}_B = \arccos\left( \frac{\vec{V}_B \cdot \vec{R}}{V_B \cdot R} \right) \in [0, \pi] $$
**角度优势得分**可以归一化至 $[-1, 1]$ 之间：
$$ \text{AAS} = \left( 1 - \frac{\text{ATA}_A}{\pi/2} \right) + \left( 1 - \frac{\text{AOT}_B}{\pi/2} \right) - 1 $$
*注：当 $A$ 完美咬尾 $B$ 时（$\text{ATA}=0, \text{AOT}=0$），得分为 1；反之被完美咬尾得分为 -1。*

#### 指标2：能量优势与管理能力（Energy Advantage, $\Delta E_s$）
**目的**：评估模型是否掌握了“能量战（Energy Fight）”的精髓（避免过度使用Max-G导致失速）。
**理论基础**：书中指出，特定过剩功率（$P_s$）和比能量（Specific Energy）决定了垂直机动和剪刀机动的胜负。
**公式**：
计算双方的单位重量比能量（包含了动能和势能，以高度为单位）：
$$ E_{s, A} = h_A + \frac{V_A^2}{2g} $$
$$ E_{s, B} = h_B + \frac{V_B^2}{2g} $$
**能量优势指标**：
$$ \Delta E_s = E_{s, A} - E_{s, B} $$
在评测中，你可以绘制对战全程的 $\Delta E_s$ 变化曲线。一个优秀的能量战模型，应在不丢失过多AAS的前提下，稳步提升 $\Delta E_s$。

#### 指标3：最佳盘旋速度保持率（Corner Speed Adherence）
**目的**：文中多次强调，转弯性能（无论是转弯半径还是转弯率）在“Corner Speed（最佳盘旋速度，记为 $V_c$）”附近达到最优。模型应当学会在持续盘旋中保持该速度。
**公式**：
计算模型在交战（尤其在 $\text{ATA} > 30^\circ$ 的盘旋阶段）中的速度偏离惩罚：
$$ M_{corner} = \frac{1}{T} \int_0^T \exp\left( - \lambda \left| \frac{V_A(t) - V_c}{V_c} \right| \right) dt $$
*注：$\lambda$ 为缩放系数（如取 5），该指标越接近 1，说明智能体的速度控制能力越好。*

#### 指标4：机炮开火窗口驻留时间（Guns Firing Solution Time）
**目的**：检验模型最终将位置转化为有效射击（Snapshot 或 Tracking）的能力。
**理论基础**：书中提到，机炮射击需要同时满足距离限制、目标位于前置视野中。
**公式**：
定义机炮有效开火窗口为一个布尔函数 $F(t)$，必须同时满足：
1. 距离限制： $R_{min} \le R(t) \le R_{max}$ （如 150m 到 800m）
2. 视角限制（机炮漏斗区）： $\text{ATA}_A(t) \le \theta_{max}$ （如 $5^\circ$ 到 $15^\circ$以内，取决于前置角计算）
3. 角度限制： $\text{AOT}_B(t) \le 60^\circ$ （确保目标不会有极高的横穿角率）

**评估指标**（有效开火秒数）：
$$ T_{shoot} = \int_0^T \mathbb{I}\Big( R \in [R_{min}, R_{max}] \land \text{ATA}_A < \theta \land \text{AOT}_B < 60^\circ \Big) dt $$
这里可以不按这个公式，直接按照我们环境中计算出的机炮范围即可

#### 指标5：致命超越失误率（Overshoot Penalty / 3-9 Line Violation）
**目的**：评测模型是否学会了滞后追求（Lag Pursuit）、High Yo-Yo 等防止超越的动作。
**理论基础**：在狗斗中，冲到敌机的前方（即越过敌机的 3-9 线）是最致命的失误（见文中 Flat Scissors 和 Lag Displacement Rolls 章节）。
**公式**：
目标 $B$ 的 3-9 线平面是垂直于 $\vec{V}_B$ 并穿过 $\vec{P}_B$ 的平面。
判断 $A$ 是否越过 $B$ 的 3-9 线（即 $A$ 在 $B$ 前方）：
$$ \text{Overshoot\_Condition}: \quad \vec{R} \cdot \vec{V}_B < 0 $$
（因为 $\vec{R}$ 是从 A 指向 B，如果两者的点积小于0，说明 A 已经跑到 B 速度向量的正前方了）。
你可以统计一场Episode中，A 从“在3-9线后方”变成“在3-9线前方”的**次数**，或者越线后的**持续时间**作为负面评测指标。

#### 指标6：航径分离度与提前转弯（Flight-Path Separation & Lead Turn Index）
**目的**：评价模型是否掌握了文中第76页提到的高阶战术：“Lead Turn”。即在双方迎头交汇前，创造横向距离并提前转向。
**公式**：
当双方处于迎头交汇（Nose-to-Nose, $R$ 正在减小，且双方航向角之差 $> 120^\circ$）时，计算 $A$ 到 $B$ 速度延长线的垂直距离（Flight-path separation）：
$$ D_{sep} = \frac{\|\vec{R} \times \vec{V}_B\|}{V_B} $$
**Lead Turn 能力评估**：如果在 $R$ 达到极小值（Pass点）前 $\Delta t$ 秒内，A 已经开始向 B 产生高角加速度（机头提前转向），则表明模型学会了 Lead Turn。

### 总结建议
在你的RL评测报告中，建议不要仅看“胜率（Win Rate）”。应当通过**AAS指标（位置优势）**和**$\Delta E_s$（能量优势）**的二维相图，分析你的模型属于“Angles Fighter”（喜欢通过剧烈掉速换取开火窗口）还是“Energy Fighter”（喜欢打能量压制、滚转剪刀）。通过分析**指标5（超越次数）**，你可以评估模型是否还存在“贪枪”导致冲到敌机前方的低级策略。