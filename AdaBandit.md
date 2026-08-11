# Bandit Experiments: Literature, Capability Boundaries, and Design Revisions

> 更新：2026-08-09
> 目的：结合近期 Bandit–LLM 文献与本项目 Llama3-8B 的 pv6–pv8 结果，分解 utilization、native exploration 与 RSN modulation，并规划下一轮实验。


## 1. Summary of Conclusions

### Completed Iteration Path

- **pv6** 首次建立了可运行的 reference Bandit 与 competence gate，但也暴露了 rationale 截断、选项显示漂移、Stage 2 指令冲突与 label prior 等接口问题。
- **pv7** 用结构化 `Evidence → Policy → constrained choice` 修复了两阶段接口。模型能读取样本数与 empirical rate，也能稳定执行自己的 Policy；但仍出现 one-shot-zero lock-in，严格 competence gate 未通过。
- **pv7 frozen-state diagnostics** 表明：history 改善了文本格式，calculator 改善了 uncertainty 的表述，α 改变了 rationale 与决策锐度；但它们都没有稳定促成对 `1 trial / 0 reward` 臂的定向重访。
- **pv8** 把 choice history 放回完整 100-round online episode。结果复现 pv7：α 双向调节 policy commitment / decision sharpness，但未改变 targeted information seeking、SuffFail 或 outcome。
- **pv9** 加入四项 Stage-1 修改（score framing / untried cue / 生成控制 / 显式 Bernoulli 说明）与第二个环境 NearTie。**Easy-bare 首次通过 competence gate**（pv7 曾以平手告负），四项修改化解了 pv7 的 one-shot-zero lock-in；但 α 的 outcome 层仍为 null，只在 mechanism 层（EXPLORE 表述、explore 关键词）有 dose-response。详见 §1.3。


### 1.1 Can Small Models Do Bandit Tasks?

**可以，但不是无条件地可以。**

现有证据支持三个层级：

1. **利用已获得的信息（utilization / exploitation）**：7–8B 模型通常具备一定能力。本项目 D 条件下，Llama3-8B 的 empirical-best adherence 为 0.884，说明给定已经观察到的均值，它大部分时候会利用当前证据。
2. **自主发现未知选项（discovery / exploration）**：小模型容易过早锁定少数选项。Gemma2 2B/9B/27B（Llama3 与 Qwen2.5 在其 Appendix C.4 复现且 bias 持续）、Qwen2.5 3B/7B 与本项目 Llama3-8B 都出现过 action coverage 停滞、greedy lock、frequency bias 或 suffix failure。
3. **稳定完成“探索后收敛”的完整策略**：未经专门训练的小模型并不可靠。提示、短 CoT、历史摘要、任务规模和训练方式都会显著改变结果；平均 regret 或 OptFrac 好看，也可能只是早期碰巧找到最优臂后一直锁定。

### 1.2 Dopamine Effect in Bandit Task

#### 1.2.1 Channel Definition

本节将 α 操作化为 **tonic-like fixed-gain modulation**。α 在各轮 prefill 阶段以固定强度施加，调节给定状态下的基线增益，而不包含随每轮 reward prediction error（RPE）变化的 phasic controller。因此，Bandit 实验的主要问题不是 α 是否改变反馈学习本身，而是它是否改变模型在已有证据之上的行动方式。

- **Tonic-like channel**：背景性的 incentive salience、行动增益与成本—收益权衡，是本实验的主要检验对象。
- **Phasic channel**：reward-contingent RPE 与 trial-by-trial learning，不是当前 α 操作直接实现的通道。

#### 1.2.2 Computational Roles and Behavioral Predictions

**Decision Precision / Inverse Temperature**

Tonic DA 可被计算性地联系到行动选择的精度。以 softmax 表示时，inverse temperature β 控制价值差对选择概率的影响：较高的 β 对应较确定的选择，较低的 β 对应较宽的选择分布。本项目对 RSN 的操作性预测为：`+α` 提高 policy sharpness，`−α` 降低 policy sharpness。

这一效应应主要在价值接近的状态中检验。当某一臂已具有明显优势时，不同精度都可能产生相同的 argmax 选择，使全局选择一致性出现天花板效应。因此主要读数应包括：

- 价值接近状态中的选择一致性、重复率与切换率；
- 同一冻结状态重复采样时的选择熵；
- α 与预先定义的 value gap 之间的交互。

Near-tie 状态应根据环境参数或 α=0 状态下的经验价值差预先定义，不应使用已经受到 α 影响的 candidate margin 进行筛选。

**Information Investment / Opportunity Cost**

Bandit exploration 要求模型暂时放弃当前经验最优臂的期望收益，以取得可能改善后续决策的信息。这个过程可被视为 information investment：其成本是当前回合放弃的预期收益，其潜在效益是降低不确定性并改善未来选择。

由 tonic DA 的 effort-related choice 类比，可提出一个待检验的 RSN 假设：`+α` 提高为信息支付机会成本的意愿，`−α` 则提高对当前已知收益的依赖。该假设不是一般非贪婪选择的预测，而要求目标具有明确的信息价值：

- 选择系统性偏向低样本量或高不确定性的臂；
- 选择随 information value 与 opportunity cost 的相对大小而变化；
- 新信息在后续轮次中被纳入策略，而非形成无目的切换。

**Response Vigor**

Response vigor 是 tonic DA 的另一经典计算角色，但其 LLM 对应物涉及生成潜伏期、token 数与认知投入之间的解释问题。本项目已在 CGT-seq 中使用 commitment timing 检验这一维度，因此 Bandit 不再将 response vigor 作为主要验证目标。Bandit 中的文本长度与停止行为仅作为接口诊断，不作为核心 dopamine-like readout。

#### 1.2.3 Random–Directed Exploration Dissociation

Decision precision 与 information investment 都可能增加非贪婪选择，但二者具有不同的行为签名与 α 方向：

| 维度 | Random exploration from reduced precision | Directed exploration from information investment |
|---|---|---|
| α 方向 | `−α` 增加 | `+α` 增加 |
| 目标分布 | 不系统性偏向低样本量臂 | 偏向低样本量或高不确定性臂 |
| 与 value gap 的关系 | 主要出现在 near-tie 状态 | 取决于 information value 是否超过 opportunity cost |
| 后续利用 | 不要求形成一致的证据整合 | 取得信息后应更新后续策略 |

因此，switch rate、non-greedy rate 或 entropy 不能单独解释为 exploration。较完整的分析应同时估计：

\[
P(a) \propto \exp\left\{\beta_{\alpha}\left[\hat\mu_a + \kappa_{\alpha}U_a - C_a\right]\right\},
\]

其中 \(\beta_{\alpha}\) 表示一般选择精度，\(\kappa_{\alpha}\) 表示 uncertainty / information-value weighting，\(C_a\) 表示选择该臂的机会成本。α 对 \(\beta\) 与 \(\kappa\) 的影响需要分别检验。

#### 1.2.4 Dose and Baseline Dependence

Dopaminergic modulation 常呈现任务与基线依赖性。对 Bandit 而言，这意味着同一 α 在信息稀缺与信息充分、价值接近与明显优势的状态中可能产生不同效果。分层变量应由环境设计或 α=0 的冻结状态预先定义，避免根据各 α 已经产生的轨迹进行事后筛选。

倒 U 更适合作为最终行为表现的候选形态，而不是每一个机制指标都必须满足的约束。Policy precision、information-value weighting 等机制参数可以在局部 α 区间内单调变化，而 regret 或 OptFrac 可能因为多种机制之间的平衡而呈非单调关系。因此，多 α 剂量扫描用于检验非线性，但不预设峰值必然存在或位于 α=0。

#### 1.2.5 Specificity Controls

Wanting、liking 与 learning 应保持概念区分。当前 Bandit 范式可以操作化 wanting-like action selection 与 feedback-based learning，但没有直接的 hedonic liking 测量。

α 的主要预测位于给定证据后的行动政策。以下指标用于检验其是否同时改变基础计算或反馈更新能力：

- `estimate_accuracy`：对已观察次数、成功数与经验均值的复述准确性；
- `belief_update_correctness`：获得新反馈后，显式估计是否正确更新；
- conditional feedback sensitivity：在控制更新后信念与当前 value gap 后，反馈是否仍额外改变行动。

这些指标的稳定性支持作用通道的选择性，但只能说明 RSN 操作没有普遍改变计算或更新能力，不能单独证明生物学上的 dopamine specificity。Win-stay / lose-shift 同时受到学习与行动精度影响，因此作为描述性指标报告，不预注册为严格的零效应对照。

#### 1.2.6 Evaluation Metrics

**Primary Wanting-like Metrics**

| # | Metric | Computational role | Interpretation |
|---|---|---|---|
| 1 | `near_tie_consistency` | Precision | Near-tie 状态中的选择一致性 |
| 2 | `state_resample_entropy` | Precision | 同一冻结状态重复采样的选择熵 |
| 3 | `low_n_targeting_rate` | Information investment | 非贪婪选择是否偏向低样本量臂 |
| 4 | `uncertainty_targeting_rate` | Information investment | 非贪婪选择是否偏向高不确定性臂 |
| 5 | `cost_paid_for_info` | Information investment | 为取得信息而放弃的经验收益 |
| 6 | `post_sample_integration` | Information utilization | 取得新信息后，后续策略是否据此更新 |

**Specificity Metrics**

| # | Metric | Interpretation |
|---|---|---|
| 7 | `estimate_accuracy` | 是否正确读取并复述已观察证据。**不等于 uncertainty reasoning 正确**——见下方 `uncertainty_calibration` |
| 7b | `uncertainty_calibration` | 语言中的不确定性是否随样本量下降（PV9 Easy 实测为**反向**，见 §1.3） |
| 8 | `belief_update_correctness` | 是否根据新反馈正确更新显式估计 |
| 9 | `win_stay_lose_shift_absolute` | 描述对单次反馈的即时行为反应，不作为严格 null |

**Failure-mode Metrics**

| # | Regime | Metrics |
|---|---|---|
| 10 | Excessive commitment | 过早锁定率、one-shot-zero abandonment、同一臂最长连段 |
| 11 | Reduced engagement / interface failure | 空生成、未完成 Policy、无有效动作；必须与动机解释分开 |

### 1.3 Prompt Design Constraints

#### PV9 Prompt Modifications

PV9 = pv8 + 四项 Stage-1 修改 + 第二个环境。**Stage 2 逐字节未变**（frozen S1，只给 counts）：
cue / score / history 全部只在 Stage 1，因为把任何一项放进 Stage 2 都会直接抬高该 button 的
candidate logit，届时选臂变化将无法归因于 Stage 1 的推理。

实现：`bandit_pv9.py`（纯 prompt 层，不含模型）、`bandit_pv9_episode.py`、
`run_bandit_pv9_episodes.py`、`run_bandit_pv9.sh`、`evaluate_competence_gate_pv9.py`、
`freeze_pv9_baseline.py` + `bandit_pv9_baseline_manifest.json`、`test_bandit_pv9.py`。
沿用 pv8 的 containment 模式：`run_pv9_episode` 临时 patch `p7.build_rationale_prompt`
并在 `finally` 还原，因此 steering 语义 / fire 计数 / reward tapes / record schema 均不会漂移。

1. **Self-Relevant Reward Framing**（`SCORE_BLOCK_VERSION='score-v1'`）
   将 reward 表述为模型自身任务分数的一部分（`Your score so far: N points.`），
   并以最终累计分数评价表现。增强 reward 的 motivational salience，
   不增加新的决策信息或算法规则。

2. **Untried-Arm Exploration Cue**（`UNTRIED_CUE_VERSION='cue-v1'`）
   仅对从未尝试过的 arm（\(n=0\)）附加提示：
   `Exploring this button may improve future rewards.`
   **这是 scaffold，不是 native 行为**：pv8 实测 EXPLORE 只有 ~3% 的地板，
   H2（information investment）在地板上无法检验，cue 的作用是把读数抬离地板。
   它陈述了收益方向，因此属于 strategy 侧；此处观察到的任何探索都必须写作
   **scaffolded discovery**。cue 在 arm 有 1 次 pull 后即消失，
   所以它**不可能**驱动 one-shot-zero 的重访——该失败模式保持未受干预且可测。

3. **Generation and Output Control**（`stop_strings_version='stop-hash-v1'`）
   `RATIONALE_MAX_TOKENS=128`、`RATIONALE_WORD_LIMIT=50`、`STOP_STRINGS=('#',)`。
   每轮保留**三层文本**，`stop_reason` 依 **raw** 文本判定，取值为五元词表
   `native_clean / stop_marker_applied / continued_after_policy / no_policy_line / empty`：

   | 层 | 内容 |
   |---|---|
   | `rationale_raw` | 未截断的原始生成，供格式审计 |
   | `rationale_stopped` | 应用 stop marker 后 |
   | `rationale_clean` | **Stage 2 实际看到的文本** |

   > **`STOP_STRINGS` 是 `('#',)` 而非 `'\n\n'`，这是实测排除的结果，不是偏好。**
   > 模型的输出结构是 Evidence → 空行 → Policy：在 pv8 存量数据上 1999/2000 轮含空行且
   > **全部落在 Policy 之前**；PV9 自身数据复核为 **2000/2000**。
   > 用 `\n\n` 会截断几乎每一条 Policy。未经重新实测不得"改进"此项。

4. **Explicit Reward Distribution**
   明确说明每个 arm 具有固定但未知的 Bernoulli reward probability，
   帮助模型正确理解环境结构，但不透露真实概率或探索算法。

版本常量（全部进 resume key）：`STAGE1_INSTRUCTION_VERSION='p9'`、
`STAGE2_INSTRUCTION_VERSION='s1'`（冻结继承）、`SCORE_BLOCK_VERSION='score-v1'`、
`UNTRIED_CUE_VERSION='cue-v1'`、`HISTORY_BLOCK_VERSION='hist-letters-v1'`。

**α=0 必须重跑**：Stage-1 prompt 已变，存量 pv8 的 α=0 不是本协议的 baseline。
三档 α 用同一 seed bank 与同一 reward tapes，一个 α 一个目录。

#### 第二个环境：NearTie

| env | K | probs | T | competence_eligible |
|---|---|---|---|---|
| `easy` | 4 | 0.75 / 0.25×3 | 100 | **True** |
| `neartie` | 4 | 0.60 / 0.55 / 0.25×2 | 100 | **False** |

NearTie 是 **mechanism environment，永远不是 competence anchor**：gap=0.05 而 ~25 pulls/arm
时经验 SE≈0.10 是 gap 的两倍，其 SuffFail 测的是环境不是策略，gate 输出仅为 diagnostic。
它存在的理由是 Easy 贴近天花板、α 无发挥空间（与 pv6 Easy、Qwen MMLU betting 同一个
baseline-ceiling 问题）。

**两个环境共用一套 seed bank**——`build_seed_bank` 只依赖 `(seed, k)` 而两者都是 K=4，
这正是二者 seed-paired 的依据；`run_bandit_pv9.sh` 由 `easy` 导出 `FORMAL_SEEDS` 再传给
NearTie 各 cell。

#### 已验证的 Stage-1 prompt（由 `bandit_pv9.build_rationale_prompt` 直接渲染）

```text
You are the decision-maker in this task. Each button has a fixed but unknown probability of producing a reward of 1; otherwise it produces a reward of 0. These probabilities may differ across buttons. Your performance is your final cumulative score.

Choose one button in each of 100 rounds to maximize your final cumulative score.

Round 61 of 100. Future choices after this one: 39.
Your score so far: 41 points.

CHOICE HISTORY (oldest → newest):
[A A A ... A B B ... B C C C C C]   (60 letters, elided here)

OPTIONS
- Button A: 30 rewards / 40 trials, empirical rate 0.75
- Button B: 8 rewards / 15 trials, empirical rate 0.53
- Button C: 3 rewards / 5 trials, empirical rate 0.60
- Button D: UNTRIED (unknown). Exploring this button may improve future rewards.

Complete exactly two lines and stop after the Policy line. Use no more than 50 words total.

First line: finish “Evidence:” by briefly comparing the strength and uncertainty of the available evidence.
Second line: write either:
“Policy: EXPLORE Button X because ...”
or
“Policy: EXPLOIT Button X because ...”

Keep both lines concise. Name exactly one button. Do not repeat the task or continue after the Policy line.

Evidence: 
```

> 注：以上为代码实际输出，非手抄。CHOICE HISTORY 行在此处省略中段字母以便阅读；
> 真实 prompt 逐轮增长（T=100 时 184→323 tokens），anchor 每轮均为 token 220。


## 2. Literature
### 2.0 Paper Summary
| 论文 | Bandit 在论文中的角色 | 是否证明 LLM 自己会做 Bandit | 对本项目的直接价值 |
|---|---|---:|---|
| [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html) | 直接评估 LLM 在 BanditBench 中的 in-context exploration（**MAB K=5/20；CB K=10/30**），并比较 summary、UCB guidance、few-shot 与 fine-tuning | 基线部分是；algorithm-guided / distillation 部分不是纯自主能力 | 最直接支持 structured summary、难度阶梯与 exploration-optimality 分析，同时要求把“模型自主探索”与“外部算法供给探索”分开 |
| [LLMs are Greedy Agents](https://arxiv.org/abs/2504.16078) | Gemma2 2B/9B/27B 主实验，**Llama3 / Qwen2.5 在 Appendix C.4 复现且 bias 持续**；环境含 MAB（K=10/20，T=50）、contextual bandit 与**文字井字棋** | 是，且系统分析失败模式 | 最直接：greediness、frequency bias、knowing–doing gap；CoT、try-all、summary 和 RLFT 的作用。**Llama3 上的同族直接证据** |
| [When Greedy Wins](https://arxiv.org/abs/2509.24923) | Qwen2.5 3B/7B 学习 meta-bandit policy（**K=2/3/5 训练 + K=10 OOD 泛化**） | 是，但主要研究训练后策略 | 平均 reward 提升不等于探索改善；必须报告 GreedyFreq、SuffixFail、双峰和早停探索；**提供文献中唯一的模型内 K 难度梯度** |
| [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118) | 经典 TS/回归 Bandit 负责探索，LLM 只预测 reward | 否，反而指出直接选臂常次优 | 支持把 reward understanding 与 action selection 拆开；经典算法控制应作为上界/诊断，不是 α 主实验 |
| [Should You Use Your LLM to Explore or Exploit?](https://arxiv.org/abs/2502.00225) | 将 exploration oracle 与 exploitation oracle 分开测试；模型含 GPT-5-nano / GPT-4 / GPT-4o / GPT-3.5 / Qwen-2.5 / Gemma-3 / Mistral-7B / DeepSeek-R1-Distill-Qwen | 是能力分解，不是完整 policy | 为本项目提供最干净的诊断框架：不要用一个端到端分数同时测 discovery 与 utilization。**reasoning model 相对占优，但所有 LLM 配置仍弱于简单线性回归** |


- **Bandit.icml2025.EVOLvE. Evaluating and Optimizing LLMs For In-Context Exploration**
  - 提出 **BanditBench**：覆盖无上下文的 MAB 与基于 MovieLens 构建的 contextual bandit（CB），系统改变 arm 数量、奖励分布、最优—次优 arm 差距及 arm 的文本描述；以累计 regret、最优 arm 选择比例（OptFrac）、最少被选 arm 的尝试比例（MinFrac）等指标评估 LLM 的 in-context exploration。
  - 核心结论：仅提供原始 `action–reward` 交互历史时，LLM 普遍缺乏有效的自主探索；其表面 reward 表现不必表示它会随经验累积而逐渐识别并稳定利用最优 arm。
  - 作者提出 **Oracle Behavior Fine-Tuning（OFT）**：以 UCB／LinUCB 产生的专家轨迹做 post-training，使模型从“当前历史 → 专家下一步 action”中蒸馏探索—利用策略。OFT 可显著提升 Gemini-1.5 Flash，但与经典 UCB／LinUCB 仍有差距；弱模型多数仍呈近线性 regret，未形成高效的长期探索。
  - 评测模型：Gemma-2B、Gemma-9B、Gemini-1.5 Flash、Gemini-1.5 Pro（**无 7–8B 开源模型**）。
  - **SH（Summarized History）**：将长的逐轮 `time–action–reward` 历史压缩为每个 arm 的选择次数与平均 reward；LLM 仍需自行判断何时探索、何时利用。
  - **AG（Algorithm-Guided Support）**：在 SH 基础上，由外部按 UCB／LinUCB 计算每个 arm 的 exploitation value 与 exploration bonus，并写入 prompt；LLM 主要比较总分后选 action。因此它是算法支架，不能视为模型自行从原始历史学出了 UCB。
  - **In-context Demonstration**：在 prompt 中提供 5 条由 UCB 生成的完整交互轨迹，让模型在当前任务中模仿其选择规律。核心是“历史表示 → UCB 的下一步 action”，而不是提供自然语言 CoT 式理由。
  - 小模型表现较差：MAB 中，Gemma-2B 的 Raw History 为 7.6、最佳推理支持为 10.5、few-shot 为 4.7；Gemma-9B 分别为 10.5、5.3、9.2。推理支架和 few-shot 都未稳定改善其探索。
  - **MAB（Multi-Armed Bandit）**：reward 只取决于所选 arm。例如 5 个按钮各有固定但未知的中奖率；无论谁、何时按 Button A，其中奖率不变。
  - **CB（Contextual Bandit）**：每轮有 context，reward 同时取决于 context 与所选 arm。例如给不同用户推荐电影，同一电影对偏好科幻的用户可能 reward 高，对另一用户则可能低。
  - 为分析长期探索效率，作者将累计 regret 随 interaction rounds \(T\) 的曲线拟合为：
    \[
    f(T)=\frac{\lambda_1\log(T)^\alpha}{\Delta_{\min}}+\beta T+\lambda_2
    \]
    其中 \(\alpha\) 控制次线性部分的增长速度，越小越好；\(\beta\) 控制线性 regret，理想上接近 0。若 \(\beta>0\)，表示模型每多一轮仍会持续累积损失，尚未稳定逼近最优策略。
  - 动作空间与 horizon（**MAB 与 CB 是两套不同的 K，勿混用**）：**MAB** 用 \(K=5\)（small）/ \(K=20\)（large）；**CB**（MovieLens top-K 电影）用 \(K=10\)（easy）/ \(K=30\)（hard），原文 "we use \(T=1000\) for \(K=30\) and \(T=200\) for \(K=10\)" 指的是 CB，CB 另有固定 \(T=200\) 的设置。

- **Bandit.iclr2026.LLMs are Greedy Agents. Effects of RL Fine-tuning on Decision-Making Abilitie**s
  - 以 Gaussian Button MAB、contextual bandit 与文字井字棋（有真实 state transition）分析 LLM 的决策缺陷；MAB 用 **K=10 / K=20 arms**，horizon \(T=50\)。
  - 对预训练 Gemma2 做 RLFT post-training；主实验模型为 Gemma2 2B/9B/27B，**Llama3 与 Qwen2.5 在 Appendix C.4 复现 greediness 分析且 bias 持续**（原文："these biases persist"）。
  - 三类失败：
    - Greediness：只在已尝试的 arms 中选当前经验回报最高者，过早 exploitation，使 action coverage 停滞。
    - Frequency bias：小模型会因某动作在 context 中反复出现而重复选它，即使 reward 不支持。
    - Knowing-doing gap：CoT 中能算对 UCB / 知道应尝试哪个 arm，最终 `ACTION` 却仍选已尝试的 greedy arm。
  - 模型得到足够信息后通常能够选对，主要短板是主动获取信息；先将所有臂试一遍，或在 RLFT 中给未尝试臂 `+1` exploration bonus，都会提高探索并降低 regret。
  - CoT 与足够 generation budget 对 RLFT 很重要；无 CoT 的 RLFT 几乎只能达到带 CoT 的 ICL 表现。使用 UCB expert trajectories 的 SFT 可接近 UCB。
  - 与当前 Bandit 的联系：本文支持早期 exploration failure / 过早 exploitation；pv7 的 one-shot-zero lock-in 则更具体——最优臂被试过一次零回报后不再尝试，不能仅用 coverage 代表。
  - 与 RSN 的联系：Gemma2 可作为独立模型族测试 RSN，但不能复用 Llama 的 mask 或 diff vector；必须重新提取 role mean、NMD mask，并验证 tokenizer/anchor、注入与模型加载。Gemma 权重需有许可与下载权限，当前不应假定本地可直接运行。
  - Gaussian Button MAB：经典 multi-armed bandit 的文字化版本；每个 arm 是颜色按钮，例如 `red / green / blue / yellow / orange`。每个按钮具有固定但未知的 Gaussian reward distribution：`press button a → reward ~ Normal(μ_a, σ²)`。每一步选一个按钮、取得带噪声 reward；目标是在有限步数内最大化累计 reward。ICL 不给正确示范或 UCB expert sample；每一步只提供任务规则、输出格式与当前 episode history。模型生成短 CoT 与 `ACTION=X`；环境执行后返回 reward，并将新记录加入下一步 prompt。
      ```text
      You are a bandit algorithm in a room with {K} buttons labeled
      {button_1}, {button_2}, ..., {button_K}.

      Each button is associated with a Gaussian distribution with a fixed
      but unknown mean; the means for the buttons could be different.
      Whenever you press a button, you receive a reward sampled from that
      button's associated distribution.

      You have {T} timesteps. At each timestep, you MUST choose exactly one
      button and receive its reward. Your goal is to maximize total reward
      over the {T} timesteps.

      [More Instructions]

      Think step-by-step, then give a short reasoning process and a final
      answer in the form ACTION=X, where X is one of the buttons above.

      So far you have tried/seen:
      Step=0 Action=green Reward=0.3
      Step=1 Action=blue Reward=0.1
      Step=2 Action=orange Reward=-0.5
      ...

      What do you predict next?
      ```
- **Bandit.iclr2026.When Greedy Wins. Emergent Exploitation Bias in Meta-Bandit LLM Training**
  - 以自然语言呈现 5-armed MAB 的历史摘要（每个 arm 的抽样次数与平均 reward），要求模型在 `<think>` 中推理，并在 `<answer>` 输出下一次选择的 arm。
  - Base model：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct。
  - 训练方法比较 Pretrain、SFT、RL-OG、RL-STG、RL-ALG。SFT 以 UCB oracle 生成的「UCB 计算 CoT + 最终 arm」为标准答案，直接模仿完整输出；RL-ALG 则只奖励最终动作是否与 UCB 一致。RL-OG 直接使用环境实际 reward，RL-STG 使用由真实均值 / pseudo-regret 构造的低方差策略 reward。
  - 微调通常降低 cumulative regret、提高 BestArmFreq，也能泛化到更长 horizon 与部分 OOD bandit；但作者发现所有 fine-tuned agents 都更容易过早进入 exploitation，产生更高的 suffix failure：模型可能永久放弃真实最佳臂。故更低的平均 regret 不等于更稳健的探索；训练可能学到更精致、却更脆弱的 greedy strategy。
  - Table 2 metrics：`AvgReward`（平均单步 reward，↑较好）、`BestArmFreq`（选择真实最佳臂比例，↑）、`GreedyFreq`（选择当前经验均值最高臂比例；过高表示过早 exploit）、`SuffixFail`（从指定步起至结束不再选择真实最佳臂的比例，↓）、`MinFrac`（被选最少臂的归一化选择比例；过高表示持续近均匀探索、未收敛；需结合 SuffixFail 解读）。
- **Bandit.acl2026.Large Language Model-Enhanced Multi-Armed Bandits**
  - 这篇论文提出三种将 LLM 与经典 Bandit 算法结合的方法：
    - **TS-LLM**：每轮让 LLM 分别预测各个 arm 的 reward；早期使用较高 temperature，利用输出随机性促进探索，随后逐渐降低 temperature，加强后期利用。
    - **RO-LLM**：将 LLM 作为 SquareCB 的 regression oracle，预测各个 arm 的 loss；temperature 固定为 0，探索由 SquareCB 的显式采样分布负责，而不是依赖 LLM 的生成随机性。
    - **TS-LLM-DB**：扩展到 dueling bandit。LLM 预测两只 arm 之间的偏好概率；第一只 arm 通过近似 Borda score 选出，第二只 arm 选择可能胜过第一只的挑战者。
  - **评价指标**：
    - 合成 stochastic MAB 和 dueling bandit 实验主要使用 **cumulative regret**，越低越好。
    - 真实文本数据实验使用 **cumulative reward**，越高越好。
  - **Direct arm selection baselines**：
    - **NoFeature**：任务说明 + 历史记录 → 请选择 arm  
    - **FramingFeature**：任务说明 + arm 特征 + 历史记录 → 请选择 arm  
    - **HistoryFeature**：任务说明 + 历史记录 + arm 特征 → 请选择 arm  
    - 三者都直接让 LLM 选择下一只 arm，区别在于是否提供 arm 特征，以及 arm 特征在 prompt 中的位置。
  - **真实文本数据集**：
    - **OneShotWikiLinks**：基于 Wikipedia 实体链接/命名实体识别的任务。context 是实体提及前后的文本，arms 是候选实体或概念名称；由于候选名称具有语义，LLM 可以根据上下文和预训练知识直接判断。
    - **AmazonCat-13K**：Amazon 商品的极端多标签分类数据集。context 是商品标题和描述，arms 是商品标签，实验中主要以整数 ID 表示；标签本身缺乏语义，因此模型更需要依靠历史反馈进行探索。
  - **核心结论**：当 arm 本身具有可利用的语义时，直接让 LLM 选 arm 也可能表现良好；当 arm 是无语义的标签、需要通过交互发现有效选项时，将 LLM 用作 reward predictor、再由经典 Bandit 算法负责探索，效果更好。
- **Bandit.ranlp2025.TextBandit. Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks**（这篇Code有点问题）
  - 提出一个 **benchmark**（非改进方法）：LLM 仅凭纯文本反馈（"you earned a token" / "you did not earn a token"，不给数值或显式概率）做序贯 MAB 决策。
  - 任务设计：2/3/4/5 臂配置，成功率固定但未知（如二臂：30% vs 65%）；每轮单次完成（single-shot），无 CoT，模型每步只输出所选臂编号；历史以自然语言列出（"Slot machine 1 won" 等），每步 prompt 重建、无跨轮内部记忆；500 次独立 run × 每 run 25 轮；指标为 cumulative reward、cumulative regret、best-arm selection rate。
  - 评测对象：Qwen3-4B、Qwen3-8B、Llama-3.1-8B、phi-2（2.7B），对比经典基线 Thompson Sampling / UCB / Epsilon-Greedy / Random Choice。
  - **本项目忠实复现后发现的设计硬伤**：论文设置里最优臂（best-arm）应固定不变（如二臂配置里 65% 成功率的机器固定为客观最优，对应我们计算 best-arm selection rate 时的假设），但论文 prompt 中的 few-shot 示例教给模型的却是 5 个不同的"最优臂"，与"固定最优臂"这一结构性假设自相矛盾。
  - **唯一可能的差异化**在于把反馈简化到极致的二元文本，彻底剥离数字/概率——但这更像是 EVOLvE 等工作的参数收窄（特例/子集），不是方法论创新；且论文自身执行有明显瑕疵（baseline 数字源码 bug、图表为无均值/无置信区间的散点堆叠、核心发现无消融支撑），可信度弱于同类文献。
- **Bandit.uai2026.Should You Use Your Large Language Model to Explore or Exploit?**
  - **方法论出发点**：以往研究让 LLM 端到端跑完整 bandit 游戏（T 轮内自己选、自己看反馈、自己再选），发现 LLM 整体表现差，但探索与利用两种能力混在一起，无法定位问题出在哪个环节。本文的核心创新是把两者拆成独立探针任务分别测量。
  - **Exploitation oracle（第2节，Figure 1–5）**：只测"给定固定历史，能否正确识别当前最优臂"，不涉及探索决策。环境离线生成好一段 T 轮历史（每个臂的 Bernoulli/线性 reward 抽样已完成，模型看不到"未来"），一次性喂给 LLM（类似"阅读理解+概率统计"题），模型只输出一个选中的臂，不涉及多轮交互或"要不要再多探索"。
    - **MAB puzzle（Figure 1）**：无 context，Bernoulli 臂。GPT-3.5/4 表现一般，**reasoning model（GPT-5 系列）表现最好**。注意论文的措辞是 "show the most promise" / "outperform non-reasoning models (fixing model size and provider)"，即**相对占优且仍嫌昂贵/缓慢**，而非“接近完美”；论文的总结句是 "current LLMs are not that good at exploitation, particularly in larger or more complex tasks"。因此**不能**据此推论“早期文献的失败主要是模型代差”。
    - **numerical CB puzzle（Figure 2–4）**：有 context 的线性 contextual bandit，`μ(z, a) = ⟨z, θ_a⟩ + γ_a`。规模小时（d=K=2, T=100），tool use（代码解释器）和 in-context summarization（k-nearest / k-means 等 mitigation）都能大幅提升表现；但维度升高（d=K=5）后，即使扫描各种 mitigation 超参数，效果也明显打折扣，且**始终不如一个简单的线性回归 baseline**。
    - **text-based & non-linear CB puzzle（Figure 5，鲁棒性检验）**：context/action 有真实语义（如房间物品），reward 为非线性函数。即便线性 baseline 本身只能达到约70% FracCorrect，**所有 LLM+mitigation 配置仍全面弱于它**；tool use 在这个设置下甚至可能是所有配置里表现最差的（与数值型任务相反）。
  - **Exploration oracle（第3节，Figure 6–8）**：反过来，只测"能否从巨大、语义丰富的动作空间里生成一组有代表性的候选臂"，不涉及利用——候选生成后交给标准算法（UCB1）实际跑 T 轮。三类任务：MovieLens 电影推荐、arXiv 论文标题推荐（Figure 6）、开放式"哲学"问答。核心发现：**LLM 生成的候选集显著优于"仅凭类别/无信息"的 baseline，证明其真正利用了输入的具体语义**；LLM 更适合扮演"智能离散化/缩小搜索空间"的角色，而非直接做决策。
  - **整体结论**：现阶段 LLM 在 exploitation（精确统计判断）上仍不如经典统计方法可靠，即便是前沿推理模型也需要昂贵的 mitigation 才能追平简单 baseline；但在 exploration（生成语义候选集）上表现稳健，是更值得信赖的用法。

### 2.1 `EVOLvE`: Direct Source of the Current Design, and Its Boundaries

EVOLvE 在 BanditBench 中系统评估 context-free MAB 与 contextual bandit。MAB 同时改变
reward distribution、gap、arm 数量和名称表示：包括 Bernoulli / Gaussian、K=5 / K=20，
以及无语义的 Video 标签与语义丰富的 Clothes 名称（CB 另用 MovieLens，K=10 / K=30，
与 MAB 的 K 不是同一套）。论文测试 Gemma-2B、Gemma-9B、
Gemini-1.5 Flash 和 Gemini-1.5 Pro，并比较四个层次：

1. **Raw History (RH)**：把历次 action–reward 序列直接交给模型；
2. **Summarized History (SH)**：提供每个 arm 的 empirical mean、pull count 和当前
   horizon；
3. **Algorithm-Guided Support (AG)**：除 summary 外，进一步提供 UCB 的 exploitation
   value 与 exploration bonus；
4. **algorithm distillation**：使用 UCB oracle trajectory 做 few-shot demonstration
   或 Oracle Behavior Fine-Tuning。

论文的 off-the-shelf MAB 基线整体较弱：RH 下 Gemma-2B、Gemma-9B、Gemini Flash 和
Gemini Pro 的 overall win-rate 分别只有 7.6%、10.5%、27.7% 和 45.5%。结构化 history
通常有帮助，但并非对每个模型都单调改善；例如 SH 对 Gemma-9B 的 aggregate
win-rate 反而从 10.5% 降到 5.3%。真正明显的提升主要来自 carefully matched 的
few-shot / fine-tuning 或显式算法支持，而不是一句泛化的“think step by step”。

### 2.2 `LLMs are Greedy Agents`: The Most Direct Design Basis

在 BanditBench 的 Gaussian/Bernoulli MAB 上测试 Gemma2 2B、9B、27B（**K=10 / K=20 arms**），horizon 同样是 50；并在 Appendix C.4 用 **Llama3 与 Qwen2.5** 复现 greediness 分析，明确指出 bias 持续。重要发现包括：

- 不同规模模型都会过早采用 greedy strategy，action coverage 很快停滞；扩大模型只能减轻，不能消除。
- 2B 模型还会受 action 在历史中出现频率影响，即使该 action reward 较差。
- 27B 可以正确计算 UCB，但即使 rationale 正确，仍常执行 greedy action，形成 knowing–doing gap。
- CoT 明显提高 coverage，但**不能消除 greedy lock**：10 arms 下有 CoT 时 2B 覆盖 40%、9B/27B 覆盖 65%，无 CoT 时全部只有 25%——即使最好的情况也仍有约 1/3 动作空间从未被触及。
- 在多种 in-context 措施中，**初始 try-all 的改善最大**；这说明模型拿到足够信息后更擅长利用，而自主获取信息是主要短板。
- RL fine-tuning 能改善 2B/9B 的 regret，并使 2B coverage 增加约 12%，但仍未完全达到理想探索。
- **CoT 对 RLFT 是 load-bearing 的**：原文 "without CoT, RLFT barely attains the performance of ICL with CoT"——这是**训练侧**结论。

### 2.3 `When Greedy Wins`: Why OptFrac / Regret Are Insufficient

该研究在 Qwen2.5 3B/7B 上比较 pretrain、SFT 与多种 RL：

- 7B 能通过训练获得接近 UCB/Thompson sampling 的平均表现，并泛化到更长 horizon。
- 3B 直接从环境 reward 学习较困难，但通过 UCB teacher 的 SFT/模仿学习可以明显改善。
- 训练后的模型可能**更快进入 exploitation，也更容易发生早期灾难性错误**。
- 最优臂选择率会变成双峰：一些 episode 几乎一直选择最优臂，另一些几乎永久放弃。
- 因此论文额外使用：
  - `GreedyFreq@t`：前 t 轮选择当前 greedy arm 的比例；
  - `SuffixFail@t`：从某个时间点以后永久不再选择真正最优臂的 episode 比例。

这与本项目完全同向：

- D 的 α=+4 提高 coverage，但主要增加的是 non-novel switching，late adherence 明显下降。
- 这不是“更好的 exploration”，而是更弱的 policy persistence。
- 下一版分析应增加 GreedyFreq / SuffixFail，避免把高 coverage 或偶然高 OptFrac 误写成学习。

### 2.4 `Should You Use Your LLM to Explore or Exploit?`: Capabilities Must Be Tested Separately

该研究测试 GPT-5-nano、GPT-4、GPT-4o、GPT-3.5、Qwen-2.5、Gemma-3、Mistral-7B 与 DeepSeek-R1-Distill-Qwen（reasoning model）。它不要求 LLM 一次完成完整 Bandit policy，而是分别测试：

- exploitation oracle：给定历史，选择当前最优 action；
- exploration oracle：在大而有语义的 action space 中提出值得尝试的候选。

核心结论是：

- LLM 在小型、数值化 exploitation task 上可以有一定表现；
- reasoning model 在 exploitation 上相对占优，但**所有被测 LLM 配置（含 tool use / summarization mitigation）仍全面弱于一个简单线性回归 baseline**，且推理模型太慢/太贵；
- succinct summary 和工具会改善结果，但仍常不如简单 regression；
- LLM 更适合在大而有语义的空间中提出 exploration candidates，而不是替代完整 Bandit 算法。

对当前 K=5 boutique task 而言，臂名只是任意标签，不存在可利用的语义 action space。因此“LLM 擅长语义探索”并不能帮助当前任务；真正可借鉴的是**分离 discovery 与 utilization**。

## Dopamine Literatur
- **Dopamine.sa2026.Dopamine depletion in Parkinson’s increases directed but not random exploration**
  - 研究范式：8×8空间相关的multi-armed bandit网格任务，共8轮×每轮25次点击。三组被试：PD off levodopa（PD−，n=34）、PD on levodopa（PD+，n=34）、年龄匹配的polyneuropathy对照组（n=35，无中枢多巴胺系统受累）。网格奖励空间平滑分布（由Gaussian process生成），因此高效搜索需要跨邻近tile做generalization，而不只是逐个追踪选项价值。
  - PD−患者获得reward明显更少，学习曲线几乎平坦，exploitation比例极低（~3% vs PD+ 16% / Control 26%）且不随trial推进而上升，缺乏正常的explore→exploit转换；PD+表现接近控制组，levodopa几乎完全恢复了performance；PD−的search distance对上一次reward大小不敏感（拿到高reward不会就近搜索，拿到低reward也不会跑远）——说明没有利用网格的空间结构。
  - 计算模型GP-UCB（$UCB(x)=m(x)+\beta\sqrt{v(x)}$，softmax温度τ）把exploration拆成三个参数：λ（generalization，reward belief在空间上传播的范围）、β（uncertainty-directed exploration，不确定性本身被赋予多少额外价值）、τ（random exploration，与value无关的选择噪声）。
  - PD−的exploration bonus β被大幅抬高（远高于PD+/控制组,且方差也更大）→ 过度的uncertainty-directed exploration，simulation显示这组参数客观上就落在低reward区域，不只是描述性差异。
  - Random exploration（τ）在三组间没有差异——多巴胺耗竭选择性影响的是directed、value/uncertainty驱动的exploration，而不是undirected的选择噪声。
  - PD−的generalization（λ）也降低——对reward空间结构的建模更弱，与PD已知的model-based/执行规划缺陷一致
  - 多巴胺耗竭并不会让选择变得更"随机/noisy"——而是特异性地过度赋予"不确定性本身"以价值（novelty-seeking失控），同时削弱利用结构的能力；levodopa能使之正常化。这与此前"levodopa在健康人中反而降低directed exploration"的发现方向相反、机制一致，提示dopamine对β的效应可能是跨越"耗竭→过量"整个谱系的倒U型。
  - Some metrics in Figure2：learning curve； exploitation 比例；exploit 随 trial 演变；连续点击的空间距离分布；search distance 对上一次 reward 的敏感度
## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
