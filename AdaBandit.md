# Bandit Experiments: Literature, Capability Boundaries, and Design Revisions

> 更新：2026-08-09
> 目的：结合近期 Bandit–LLM 文献与本项目 Llama3-8B 的 pv6–pv8 结果，分解 utilization、native exploration 与 RSN modulation，并规划下一轮实验。


## 1. Summary of Conclusions

### Completed Iteration Path

- **pv6** 首次建立了可运行的 reference Bandit 与 competence gate，但也暴露了 rationale 截断、选项显示漂移、Stage 2 指令冲突与 label prior 等接口问题。
- **pv7** 用结构化 `Evidence → Policy → constrained choice` 修复了两阶段接口。模型能读取样本数与 empirical rate，也能稳定执行自己的 Policy；但仍出现 one-shot-zero lock-in，严格 competence gate 未通过。
- **pv7 frozen-state diagnostics** 表明：history 改善了文本格式，calculator 改善了 uncertainty 的表述，α 改变了 rationale 与决策锐度；但它们都没有稳定促成对 `1 trial / 0 reward` 臂的定向重访。
- **pv8** 把 choice history 放回完整 100-round online episode。结果复现 pv7：α 双向调节 policy commitment / decision sharpness，但未改变 targeted information seeking、SuffFail 或 outcome。
- **pv9** 加入四项 Stage-1 修改（score framing / untried cue / 生成控制 / 显式 Bernoulli 说明）与第二个环境 NearTie。**Easy-bare 首次通过 competence gate**，四项修改化解了 pv7 的 one-shot-zero lock-in；但 α 的 outcome 层仍为 null，只在 mechanism 层（EXPLORE 表述、explore 关键词）有 dose-response。详见 §1.3。

### 1.1 Can Small Models Do Bandit Tasks?

**可以，但不是无条件地可以。**

现有证据支持三个层级：

1. **利用已获得的信息（utilization / exploitation）**：7–8B 模型通常具备一定能力。本项目 D 条件下，Llama3-8B 的 empirical-best adherence 为 0.884，说明给定已经观察到的均值，它大部分时候会利用当前证据。
2. **自主发现未知选项（discovery / exploration）**：小模型容易过早锁定少数选项。Gemma2 2B/9B/27B（Llama3 与 Qwen2.5 在其 Appendix C.4 复现且 bias 持续）、Qwen2.5 3B/7B 与本项目 Llama3-8B 都出现过 action coverage 停滞、greedy lock、frequency bias 或 suffix failure。
3. **稳定完成“探索后收敛”的完整策略**：未经专门训练的小模型并不可靠。提示、短 CoT、历史摘要、任务规模和训练方式都会显著改变结果；平均 regret 或 OptFrac 好看，也可能只是早期碰巧找到最优臂后一直锁定。


### 1.2 Prompt Design Constraints

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

版本常量：`STAGE1_INSTRUCTION_VERSION='p9'`、
`STAGE2_INSTRUCTION_VERSION='s1'`、`SCORE_BLOCK_VERSION='score-v1'`、
`UNTRIED_CUE_VERSION='cue-v1'`、`HISTORY_BLOCK_VERSION='hist-letters-v1'`。

#### NearTie

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

####  Stage-1 prompt（`bandit_pv9.build_rationale_prompt`）

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

- **Bandit.nips2024.Krishnamurthy et al. Can Large Language Models Explore In-Context?**
  - 系统研究"LLM 能否 in-context explore"的第一篇论文（Microsoft Research, NeurIPS 2024），与本项目 pv6–pv9 的设计路径高度重合，`SuffFailFreq` / `MinFrac` / bimodal 分析这套 surrogate statistics 就直接来自这篇。
  - **任务**：纯文字 Bernoulli MAB，hard instance K=5/Δ=0.2（主实验）+ easy instance K=4/Δ=0.5（对照），T=100（主）/200/500（robustness check）。评测 GPT-3.5、GPT-4、Llama2，baseline 用 UCB 与 Thompson Sampling（TS）。
    - **UCB**：给每个 arm 算"经验均值 + 不确定性 bonus `√(log T / n_a)`"，每轮选 index 最高的 arm；试的次数越少 bonus 越大，随试验增多逐渐收敛到真实最优臂——确定性、结构化的探索（optimism under uncertainty）。
    - **TS**：给每个 arm 维护 Beta-Bernoulli 后验分布，每轮各抽一个样本、选样本值最大的 arm；不确定性越大后验越宽，越容易被抽中试探——探索通过随机采样方差自动实现。
  - **Prompt 设计空间**：5 个独立二元开关、`2^5=32` 种组合（Figure 2 是一张自顶向下遍历生成 prompt 的决策图）：
    1. **scenario**：buttons（按按钮）vs advertisements（选广告投放）——只是换故事包装，任务本质不变。
    2. **framing**：neutral（中性描述规则）vs **suggestive**（明确提示"要平衡 exploration/exploitation"）。Adverts 场景下 suggestive framing 的具体写法："A good strategy to optimize for clicks in these situations requires balancing exploration and exploitation. **You need to explore to try out all of the options and find those with high click rates, but you also have to exploit the information that you have to accumulate clicks.**" Neutral 版本则只说 "You are in a room with 5 buttons labeled blue, green, red, yellow, purple. Each button is associated with a Bernoulli distribution..."，不含任何探索提示。
    3. **history 呈现**：raw（逐轮列出 "green button, reward 1"）vs **summarized**（汇总成"每个 arm 试了几次、平均 reward 多少"）。
    4. **输出形式**：return action（只输出一个 arm）vs return distribution over actions（输出跨 arm 概率分布）。
    5. **CoT**：reply-only（直接给答案）vs chain-of-thought / **reinforced CoT**（GPT-4 只在 system prompt 提醒一次 CoT 不够稳定，需在 user prompt 末尾再提醒一次）。
    - 用五字母编码命名配置，如 **BSSC̃0** = Buttons + Suggestive framing + Summarized history + reinforced CoT（C 上加波浪号）+ temperature 0；基础配置叫 **BNRN0**（Buttons + Neutral + Raw history + No CoT + temp 0）。
  - **核心指标（surrogate statistics）**，用于在中等规模实验下就能可靠检测长期探索失败：
    - **`SuffFailFreq(t)`**：`SuffFail(t,R) = 1` 当且仅当 replicate R 从时间步 t 到 T 这段"后缀"区间里**一次都没选过最优臂**；对所有 replicate 取平均即得 `SuffFailFreq(t)`。该曲线随 t 单调不减；早期快速爬升并趋平，意味着大部分 replicate 很早就永久放弃了最优臂——这是**不可逆的长期探索失败**（suffix failure）。
    - **`K·MinFrac(t)`**：`MinFrac(t,R)` = 被选最少的那个 arm在 `[1,t]` 里的占比；乘以 K 归一化后，好算法应随 t 增大而单调下降（逐渐把预算集中到最优臂），若长期维持高位不降，说明模型一直近似均匀摇摆、从未真正收敛——**uniform-like failure**，与 suffix failure 是两种不同的病理，且互不蕴含（可以同时避开 suffix failure 却仍是 uniform-like failure，见下）。
    - **`GreedyFrac`**：某配置的逐轮决策与纯 Greedy 算法重合的比例，用来判断"是不是几乎完全在模仿 Greedy"。
  - `SuffFailFreq`、双峰 lock-correct/lock-wrong 计数、`K·MinFrac` 这套分析框架也应直接迁移到 PV10 的结果分析中。
  - 文章的配置考量：
    - 第一层（L1）— scenario：buttons scenario（按按钮）vs advertisements scenario（选广告投放）。这只是换个"故事包装"，任务本质相同（都是 5-arm Bernoulli bandit），用来检验结果是否对 framing 的具体措辞敏感。
    - 第二层（L2）— framing：neutral framing（中性描述任务规则）vs suggestive framing（明确提示"要平衡 exploration 和 exploitation"，就是我们上一条讨论的那句话）。
    - 中间的固定层：MAB problem description（不是开关，是所有配置共享的固定说明——arm 的 Bernoulli 分布、时间步数、目标）。
    - 第三层（L3）— history 呈现方式：raw history（逐轮列出 "green button, reward 1"）vs summarized history（汇总成"每个 arm 试了几次、平均 reward 多少"）。
    - 第四层（L4，图里标"return"）— 输出形式：return action（只输出一个 arm）vs return distribution over actions（输出一个跨 arm 的概率分布，格式类似 "blue:a,green:b,..."）。
    - 第五层（L5，图里标"final prompt"）— 是否 CoT：reply-only（直接给答案，不解释）vs chain-of-thought（先"let's think step by step"再给答案）。
  - 唯一成功：BSSC̃0 = Buttons 场景 + Suggestive framing + Summarized history + Reinforced CoT + temperature 0（GPT-4）


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

## 3. Dopamine Literatur
- **Dopamine.sa2026.Dopamine depletion in Parkinson’s increases directed but not random exploration**
  - 研究范式：8×8空间相关的multi-armed bandit网格任务，共8轮×每轮25次点击。三组被试：PD off levodopa（PD−，n=34）、PD on levodopa（PD+，n=34）、年龄匹配的polyneuropathy对照组（n=35，无中枢多巴胺系统受累）。网格奖励空间平滑分布（由Gaussian process生成），因此高效搜索需要跨邻近tile做generalization，而不只是逐个追踪选项价值。
  - PD−患者获得reward明显更少，学习曲线几乎平坦，exploitation比例极低（~3% vs PD+ 16% / Control 26%）且不随trial推进而上升，缺乏正常的explore→exploit转换；PD+表现接近控制组，levodopa几乎完全恢复了performance；PD−的search distance对上一次reward大小不敏感（拿到高reward不会就近搜索，拿到低reward也不会跑远）——说明没有利用网格的空间结构。
  - 计算模型GP-UCB（$UCB(x)=m(x)+\beta\sqrt{v(x)}$，softmax温度τ）把exploration拆成三个参数：λ（generalization，reward belief在空间上传播的范围）、β（uncertainty-directed exploration，不确定性本身被赋予多少额外价值）、τ（random exploration，与value无关的选择噪声）。
  - PD−的exploration bonus β被大幅抬高（远高于PD+/控制组,且方差也更大）→ 过度的uncertainty-directed exploration，simulation显示这组参数客观上就落在低reward区域，不只是描述性差异。
  - Random exploration（τ）在三组间没有差异——多巴胺耗竭选择性影响的是directed、value/uncertainty驱动的exploration，而不是undirected的选择噪声。
  - PD−的generalization（λ）也降低——对reward空间结构的建模更弱，与PD已知的model-based/执行规划缺陷一致
  - 多巴胺耗竭并不会让选择变得更"随机/noisy"——而是特异性地过度赋予"不确定性本身"以价值（novelty-seeking失控），同时削弱利用结构的能力；levodopa能使之正常化。这与此前"levodopa在健康人中反而降低directed exploration"的发现方向相反、机制一致，提示dopamine对β的效应可能是跨越"耗竭→过量"整个谱系的倒U型。
  - Some metrics in Figure2：learning curve； exploitation 比例；exploit 随 trial 演变；连续点击的空间距离分布；search distance 对上一次 reward 的敏感度

## 4. Result

PV9 六格在线扫描（Easy × NearTie，`rationale_alpha ∈ {−4, 0, +4}`，`action_alpha=0`），每格 20 seeds × 100 rounds。
数据源 `analyze_bandit_pv9.py`（冻结分析器，`--part all`）。统计单位固定为 **seed（n=20）**，同环境内按 seed 配对（共用冻结 bank 与 reward tape），paired Wilcoxon + seed-cluster bootstrap 95% CI；pooled round 计数仅作描述。`*` = raw p<.05。

**口径提示**
- `late_opt_frac` / `final_score` 在两个环境下均为**严格双峰**（0 或 1，无中间值），均值仅描述性，主读数为 lock-correct 计数 + exact McNemar。
- Beta(1,1) 并列为结构性（`0/1`→1/3，未试→1/2），故 uncertainty targeting 报 **BAND**（unique-max 下界 .. tie-inclusive 上界）。
- σ 与 n 在 PV9 不可分离（corr(SD, log n) ≈ −.99），交互项命名 `b_information`，**不是** GP-UCB 的 β。
- NearTie `competence_eligible=False`，其 gate 输出仅为 diagnostic。
- host 未记录于 JSON，标记为 `pending`。

### 4.0 Validity

| 项目 | Easy | NearTie |
|---|---|---|
| cells × seeds × rounds | 3 × 20 × 100 | 3 × 20 × 100 |
| `steering_fires` 与预期一致 | 60/60 episodes | 60/60 episodes |
| Stage 2 `action` fires | 0（三格） | 0（三格） |
| config / mask SHA256 / tape / arm_order 跨格一致 | ✅ | ✅ |
| `invalid_rate` | 0.000（三格） | 0.000（三格） |
| `policy_target_source` = `policy_first_clause` | 2000/2000（每格） | 2000/2000（每格） |
| executor: `action == argmax(candidate_scores)` | 2000/2000（每格） | 2000/2000（每格） |
| round-1 label 分布 | Button A 19/20（三格相同） | — |
| round-1 display position 分布 | {1:10, 2:4, 3:5, 4:1} | — |
| host | pending | pending |

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| `policy_parse_rate` | Easy | .985 | .994 | .991 | .480 | .414 |
| | NearTie | .996 | .997 | .998 | .854 | .180 |
| `action_follows_policy` | Easy | .994 | .997 | .999 | .167 | .257 |
| | NearTie | .995 | .997 | 1.000 | .715 | .068 |
| format flags（4 项） | 两环境 | ≤.001 | ≤.001 | ≤.001 | ns | ns |

`stop_reason`（pooled rounds）

| 环境 | α | `stop_marker_applied` | `continued_after_policy` | `native_clean` |
|---|---|---|---|---|
| Easy | −4 / 0 / +4 | .793 / .783 / .798 | .114 / .127 / .144 | .093 / .090 / .058 |
| NearTie | −4 / 0 / +4 | .759 / .821 / .870 | .129 / .114 / .085 | .113 / .066 / .045 |

### 4.1 Discovery（cue-scaffolded）

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| `arms_discovered` | Easy | 3.95 | 3.95 | 3.95 | 1.000 | n/a |
| | NearTie | 4.00 | 4.00 | 4.00 | n/a | n/a |
| `best_never_tried` | Easy | 0/20 | 0/20 | 0/20 | n/a | n/a |
| | NearTie | 0/20 | 0/20 | 0/20 | n/a | n/a |
| incomplete scan | Easy | .050 | .050 | .050 | 1.000 | n/a |
| | NearTie | .000 | .000 | .000 | n/a | n/a |
| first pull of best arm | Easy | 2.55 | 2.85 | 2.65 | .131 | .414 |
| | NearTie | 2.55 | 2.85 | 2.60 | .131 | .339 |
| round of full coverage | Easy | 9.79 | 10.42 | 9.37 | .271 | .720 |
| | NearTie | 12.20 | 15.30 | 7.95 | .396 | .098 |

pooled（分母为 eligible states）

| 指标 | 环境 | α=−4 | α=0 | α=+4 |
|---|---|---|---|---|
| untried targeting（r≥6，存在未试臂） | Easy | 11/184 = .0598 | 17/187 = .0909 | 12/174 = .0690 |
| | NearTie | 12/142 = .0845 | 16/198 = .0808 | 12/58 = .2069 |
| one-shot-zero revisit | Easy | 7/1947 = .0036 | 8/1944 = .0041 | 6/1948 = .0031 |
| | NearTie | 7/1947 = .0036 | 11/1946 = .0057 | 8/1950 = .0041 |

#### `untried targeting（r≥6，存在未试臂）`

例如 Easy α=0：`17/187 = .0909`

这里是 **20 个 episode（seeds）× 每个 100 轮 = 2000 个总轮次**，不是只算 20 轮。

`17/187` 的计算是：

- 从 2000 轮中筛出第 6 轮以后、且当时仍存在未试 arm 的轮次，共 **187 轮**。
- 其中 **17 轮**模型选择了未试 arm。
- 剩余 **170 轮**模型选择了已经试过的 arm。

在这 187 个合格轮次内，“选择未试臂”和“没有选择未试臂”是互斥的。
即：有 187 次补齐探索的机会，模型只有 17 次选择未试臂。

#### `one-shot-zero revisit`

例如 Easy α=0：`8/1944 = .0041`

- **分母 1944**：存在至少一个“此前只被选择过一次，而且那次 reward=0”的 arm 的轮次数。
- **分子 8**：在这些轮次中，模型实际重新选择该 one-shot-zero arm 的次数。

即：有 1944 次可以复查“一次失败 arm”的机会，模型只有 8 次真的回去尝试。 -> 某个 arm 只被选择过一次，而且这次 reward=0 后，模型几乎不会再选择它。-> 模型很容易把一次负反馈当成充分证据，随后长期忽略该 arm。

### 4.2 Directed exploration（PRIMARY）

**4.2.1 `policy_uncertainty_targeting_rate`（BAND）**

| 口径 | 环境 | α=−4 | α=0 | α=+4 |
|---|---|---|---|---|
| unique-max（下界，pooled） | Easy | **0/98 = .0000** | **0/265 = .0000** | **0/119 = .0000** |
| | NearTie | **0/164 = .0000** | **0/246 = .0000** | **0/190 = .0000** |
| tie-inclusive（上界，pooled） | Easy | 5/1254 = .0040 | 4/1306 = .0031 | 5/1321 = .0038 |
| | NearTie | 4/1257 = .0032 | 4/1081 = .0037 | 2/1382 = .0014 |
| action 层（executor check） | Easy | .004 | .005 | .005 |
| | NearTie | .003 | .004 | .002 |

per-seed median eligible denominator = **0**（两环境）→ pooled 为主读数。
> 模型虽然会在 cue 推动下尝试未探索的 arm，但此后几乎不会主动选择低样本量、高不确定性或仅尝试一次便获得 0 的 arm -> 模型缺乏自主的 uncertainty-directed exploration，容易在少量证据后转入 greedy lock-in。

#### Posterior Bernoulli Reward

PV9 是 Bernoulli reward（0/1），因此使用 Beta–Bernoulli posterior 计算每个 arm 的不确定性。
假设先验为：`p_i ~ Beta(1, 1)`
某个 arm 已获得：

- \(s\) 次 reward=1
- \(f\) 次 reward=0

则 posterior 为：`p_i | data ~ Beta(a, b)`, where `a = 1 + s` and `b = 1 + f`.

Posterior mean 是：`E[p_i] = a / (a + b)`

Posterior variance 是：`Var(p_i) = ab / [(a + b)^2 (a + b + 1)]`

例如：

| 观察 | Posterior | Mean | Variance |
|---|---|---:|---:|
| 从未尝试 | Beta(1,1) | .500 | .0833 |
| 1次，reward=0 | Beta(1,2) | .333 | .0556 |
| 1次，reward=1 | Beta(2,1) | .667 | .0556 |
| 10次，7个1、3个0 | Beta(8,4) | .667 | .0171 |

>通常采样次数越多，posterior variance 越小。因此“选择 posterior variance 最大的 arm”表示模型选择当前证据最少、估计最不确定的选项。需要注意：这个 variance 是**分析阶段根据完整 action–reward history 计算的**，并没有在 prompt 中直接告诉模型。在 PV9 里它与 arm 的采样次数高度相关，所以该指标更准确地衡量“低样本量／高不确定性 targeting”，不能声称已经把纯粹的不确定性偏好与低样本量偏好完全分开。

**4.2.2 `policy_weighting_model`：`α × low-n/uncertainty`（`b_information`）**

主参数化 `w = 1/√n`

其中 `mu` 是根据该轮之前的 reward history 计算的 posterior mean：`mu = (1 + s) / (2 + s + f)`；`s` 和 `f` 分别为该 arm 已获得的 reward=1 与 reward=0 次数。它表示基于现有证据估计的当前价值，而非隐藏的真实奖励概率。

| term | Easy coef [95% CI] | NearTie coef [95% CI] |
|---|---|---|
| `mu` | 6.753 [5.328, 10.921] * | 9.270 [7.533, 15.028] * |
| `w` | −2.671 [−5.130, 1.528] | −3.539 [−8.012, −0.574] * |
| `a_mu` | 0.213 [−0.504, 0.503] | 0.314 [−0.593, 0.887] |
| **`a_info`** | **0.121 [−0.124, 0.418]** | **−0.004 [−0.124, 0.190]** |
| `pos` | 0.229 [−0.089, 0.566] | 0.225 [−0.148, 0.469] |
| `rec` | 0.984 [0.562, 1.165] * | 1.261 [0.883, 1.449] * |
| `logn` | 0.829 [0.321, 1.861] * | 0.767 [0.007, 1.435] * |
| `lab_B` | −1.155 [−2.160, −0.127] * | −0.345 [−1.180, 0.493] |
| `lab_C` | −0.617 [−1.387, 0.569] | 0.147 [−0.651, 1.167] |
| `lab_D` | −1.439 [−2.383, −0.064] * | −0.501 [−1.391, 0.349] |
| n decisions | 5138（converged） | 5291（converged） |

> 模型选择某个 arm，是因为它当前看起来收益高、样本少、最近选过、采样次数多，还是因为按钮标签或位置？
- `mu`：该 arm 当前估计的收益。系数显著为正，说明模型强烈偏向当前价值高的 arm。
- `w = 1/√n`：低样本量指标；越大表示采样越少、越不确定。Easy 的 CI 包含 0；NearTie 的系数显著为负，说明在价值接近、较需要补充信息的环境中，模型反而较少选择低样本量 arm。这是跨 α 的总体选择倾向，不是 α 效应。
- `a_mu`：α 是否改变模型对预期收益 `mu` 的权重。两环境均未检出。
- `a_info`：核心指标，α 是否改变模型对低样本量／高不确定性 arm 的权重。Easy 和 NearTie 的 CI 都包含 0，因此未检出 α 效应。
- `pos`：展示位置效应，不显著。
- `rec`：近因效应。系数显著为正，模型偏向最近选择过的 arm。
- `logn`：采样次数效应。显著为正，模型偏向已经尝试很多次的 arm，符合 incumbent/greedy 惯性。
- `lab_B/C/D`：相对于 Button A 的标签效应。Easy 中 B、D 明显弱于 A，说明存在标签先验。

> 核心结论是：模型主要根据当前估计收益、选择历史和既有采样量作决定。NearTie 进一步显示模型总体上回避低样本／高不确定 arm；但 `a_info` 仍跨 0，因此没有证据表明 α 改变了这种权重。

`posterior SD` 敏感性分析只是把 `1/√n` 换成另一种不确定性表示，检查结论是否依赖具体公式。系数是模型中的相对权重，不是直接的选择概率。

敏感性参数化 `w = posterior SD`

| term | Easy coef [95% CI] | NearTie coef [95% CI] |
|---|---|---|
| `mu` | 6.593 [5.233, 10.661] * | 10.670 [8.568, 14.041] * |
| `w` | −6.284 [−42.843, 6.058] | −49.256 [−72.505, −22.606] * |
| **`a_info`** | **0.639 [−0.201, 2.117]** | **0.131 [−0.506, 1.133]** |

>把低样本指标 `w=1/√n` 换成更正式的 `posterior SD = √variance`，检查结论是否依赖不确定性的计算方式。

- `mu` 仍显著为正：两个环境中，模型仍明显偏向 posterior mean 较高的 arm。
- `w`：
  - Easy 的 CI 很宽且包含 0，未检出可靠效应。
  - NearTie 显著为负，表示控制其他因素后，模型反而更少选择 posterior SD 较高的 arm。
- `a_info` 是关键的 α 交互项：
  - Easy：`0.639 [−0.201, 2.117]`
  - NearTie：`0.131 [−0.506, 1.133]`
  - 两个 CI 都包含 0，因此 α 没有可靠改变模型对不确定性的权重。
> 所以敏感性分析与主分析结论一致：模型偏向当前估计价值较高的 arm，但没有证据表明 α 增加或降低了 uncertainty weighting。

>需要注意，posterior SD 的数值尺度与 `1/√n` 不同，因此两张表的系数大小不能直接比较；它又与采样次数高度相关，所以 NearTie 的负系数更适合作为模型表现出的回避不确定性，而不是一个完全独立的机制参数。

**4.2.3 low-n targeting（r≥6，全臂已试，pooled）**

| 环境 | α=−4 | α=0 | α=+4 |
|---|---|---|---|
| Easy | 5/1696 = .0029 | 8/1693 = .0047 | 7/1706 = .0041 |
| NearTie | 7/1738 = .0040 | 9/1682 = .0054 | 8/1822 = .0044 |

> 开局扫描结束后（r≥6），且所有臂都至少试过一次时，模型是否主动选择当前采样次数很少的臂。

- 分母：满足上述条件的全部轮次，例如 Easy α=0 有 1693 轮。
- 分子：模型选择低样本量臂（按分析器冻结的 low-n 定义）的轮次，例如只有 8 轮。
- 因为所有臂均已试过，所以这里排除了 `UNTRIED` cue 的直接影响，更接近自主的信息补采样。

结果只有 **0.29%–0.59%**，且没有清楚的 α 剂量反应。说明模型完成初始覆盖后，几乎不会回头补采样证据不足的臂；`±α` 也没有改变这种行为。这与前面的 uncertainty-targeting 近零结果一致。

**4.2.4 Non-greedy decomposition（overlap-aware，flags 非互斥；n=0 目标另计）**

| 环境 | α | n(non-greedy) | `unc_only` | `unc_AND_best` | `best_only` | `lowN_not_unc` | `costly_error` | cue(n=0) |
|---|---|---|---|---|---|---|---|---|
| Easy | −4 | 240 | 4 (.017) | 1 (.004) | 111 (.463) | 0 | 124 (.517) | 79 |
| | 0 | 230 | 1 (.004) | 1 (.004) | 194 (.843) | 0 | 34 (.148) | 80 |
| | +4 | 296 | 0 (.000) | 1 (.003) | 283 (.956) | 0 | 12 (.041) | 79 |
| NearTie | −4 | 213 | 3 (.014) | 1 (.005) | 119 (.559) | 0 | 90 (.423) | 80 |
| | 0 | 277 | 2 (.007) | 2 (.007) | 237 (.856) | 0 | 36 (.130) | 82 |
| | +4 | 270 | 1 (.004) | 1 (.004) | 242 (.896) | 0 | 26 (.096) | 80 |

**稳健性（剔除各格自身 lock-wrong episodes，`late_opt_frac < .2`）** — Easy：`costly_error` .517/.148/.041 → **.073/.049/.034**，`best_only` → .902/.951/.966；per-seed median `costly_error` = **0.000（三格）**，paired p=.317。

> 这个表把“非贪婪选择”进一步拆开，避免把所有偏离经验最优臂的行为都误称为探索。
- `unc_only`：选择最不确定的臂，但它不是真实最优臂——最接近真正的信息探索。
- `unc_AND_best`：目标既最不确定、又是真实最优臂，探索与正确坚持重叠。
- `best_only`：目标不再是经验最优，但它是真实最优臂——属于对真实最优臂的正确坚持，而非探索。
- `lowN_not_unc`：选择低样本量臂，但它不是最不确定臂。
- `costly_error`：既不符合不确定性探索，也不是真实最优臂，更像错误选择或错误锁定。
- `cue(n=0)`：选择未尝试臂，受 prompt cue 影响，独立统计，不算自主探索证据。

>1. 真正的不确定性探索几乎不存在: `unc_only` 只占非贪婪选择的 0–1.7%，与前面的 uncertainty-targeting 地板一致。
>2. `+4` 增加的非贪婪选择主要不是探索: Easy 中，+4 的 296 次非贪婪选择里，283 次（95.6%）选择的其实是真实最优臂。也就是说，模型是在经验均值暂时落后时继续选择真实最优臂，表现为抗短期噪声的 persistence，而不是寻找信息。
>3. −4 的高 `costly_error` 主要由少数错误锁定 episode 驱动: Easy 表面上是 `.517/.148/.041`，但剔除各格错误锁定的 episode 后变为 `.073/.049/.034`，且配对检验 `p=.317`。因此不能声称 −4 普遍增加 costly errors；这是少数 seed 翻面的离散失败。

> PV9 中的 non-greedy behavior 主要由“继续选择暂时被低估的真实最优臂”和少数错误锁定构成，而非 uncertainty-directed exploration。尤其不能把 +4 较高的 non-greedy 次数解释为探索增加。

> +4 可能增强了对既有目标的坚持，使模型在该臂因短期噪声暂时失去经验第一名时，仍继续选择它。这是一种 resistance to empirical-greedy switching 或 correct persistence，而不是 information-seeking exploration。

### 4.3 Precision / randomness（analogue）

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| switch rate | Easy | .072 | .062 | .063 | .112 | .937 |
| | NearTie | .079 | .077 | .071 | .675 | .410 |
| non-novel churn | Easy | .042 | .032 | .033 | .133 | .844 |
| | NearTie | .049 | .046 | .040 | .807 | .480 |
| max single-arm share | Easy | .939 | .949 | .950 | .698 | .779 |
| | NearTie | .905 | .910 | .923 | .925 | .083 |
| `near_tie_choice_sharpness`（margin） | Easy | 4.177 | 4.268 | 4.699 | .231 | **.019\*** |
| | NearTie | 4.005 | 4.250 | 4.404 | .388 | .133 |
| `candidate_distribution_entropy` | Easy | .124 | .119 | .089 | .216 | .090 |
| | NearTie | .140 | .121 | .107 | .154 | .076 |

**NearTie `α × posterior_gap`**（连续；bins 仅供显示。m = Stage-2 margin，g = policy 指向经验最优臂）

> α 是否改变选择的随机性／锐度，可以分成行为层和分布层。
- `switch rate`：相邻轮次更换 arm 的比例。
- `non-novel churn`：排除首次探索后，在已尝试臂之间切换的比例。
- `max single-arm share`：每个 episode 中选择最频繁 arm 所占比例，越高表示越集中。

>三项都没有显著 α 效应。因此，不能说 +4 在实际行为上显著减少切换或增强集中度。

>分布层出现方向一致的描述性信号：
- `margin`：最高候选分数与次高分数的差距；越大表示选择越锐利。
- `entropy`：候选分布的不确定程度；越低表示越集中。
从 −4 → 0 → +4：
- Easy margin：`4.177 → 4.268 → 4.699`，+4 raw `p=.019`
- NearTie entropy：`.140 → .121 → .107`，+4 raw `p=.076`

>两环境方向一致：**+4 的候选分布更尖锐、更确定；−4 相对更平坦。** 但显著性没有在同一指标上跨环境复现；Holm 校正后存活的唯一 primary 是 NearTie 的 `stance_align_broad`，不是 margin、entropy 或 directed-exploration 指标。

> α 对候选分布的 decision-precision analogue 呈现一致但较弱的方向性变化：+4 提高分布锐度，−4 降低锐度；然而该变化没有稳定转化为 switching、churn 或选择集中度的行为差异。也就是：**logit 层似乎更果断，但实际选择轨迹基本不变。**

| gap bin | α=−4 | α=0 | α=+4 |
|---|---|---|---|
| [0.00, 0.05) | n=636 m=4.17 g=.91 | n=579 m=4.41 g=.97 | n=558 m=4.82 g=.87 |
| [0.05, 0.10) | n=208 m=3.62 g=.50 | n=319 m=4.52 g=.70 | n=252 m=4.28 g=.84 |
| [0.10, 0.20) | n=381 m=3.79 g=.95 | n=355 m=3.57 g=.71 | n=539 m=3.94 g=.77 |
| [0.20, 1.00) | n=531 m=4.09 g=.98 | n=441 m=4.67 g=.98 | n=492 m=4.63 g=1.00 |

per-seed slope of margin on gap：−4 = +0.165，0 = +4.445，+4 = +0.658。
`α × gap`：−4 d=−4.280 [−12.075, +1.830] p=.798；+4 d=−3.788 [−10.260, +0.474] p=.374。

这张表检验的是：**α 对决策锐度的影响，是否特别集中在价值接近的状态。**

- `gap`：经验／posterior 最优与次优 arm 的价值差。
- `n`：落入该区间的轮次数。
- `m`：这些轮次的平均 Stage-2 margin。
- `g`：选择经验最优臂的比例（greedy alignment）。

> 如果 precision 假设成立，通常预期：gap 越大，margin 越大；而 +4 应增强这种关系，或者至少在 near-tie 状态下明显提高 margin。

但实际结果不稳定：

- α=0 的 slope 为 `+4.445`，符合 gap 越大、margin 越大的方向。
- −4 为 `+0.165`，接近零。
- +4 为 `+0.658`。
- 但是两个 `α × gap` 交互的置信区间都包含0：
  - −4：`−4.280 [−12.075, +1.830]`
  - +4：`−3.788 [−10.260, +0.474]`

因此，**没有统计证据证明 α 改变了 margin 对 value gap 的敏感性**。

还有一个值得注意的例子：在最小 gap `[0,.05)` 中，+4 的 margin 最高（4.82），但 greedy alignment 只有 `.87`，低于 α=0 的 `.97`。这说明“候选分布更尖锐”不等于“更准确地选择经验最优臂”；它可能只是对某个目标更加确信。
> 最简结论： +4 的整体 margin 较高，但这种锐化没有随 value gap 系统变化。因此目前只能说它可能提高一般性的输出锐度，不能说它选择性增强了 near-tie 状态下的 decision precision。

### 4.4 Utilization and specificity

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| policy → empirical-best | Easy | .870 | .876 | .837 | .656 | .575 |
| | NearTie | .899 | .875 | .867 | .507 | .594 |
| action → empirical-best | Easy | .870 | .874 | .833 | .388 | .374 |
| | NearTie | .902 | .874 | .867 | .463 | .480 |
| policy → posterior-mean-best | Easy | .751 | .780 | .779 | .248 | .767 |
| | NearTie | .701 | .675 | .754 | .272 | .109 |
| post-discovery adherence | Easy | .862 | .866 | .829 | .844 | .556 |
| | NearTie | .878 | .840 | .844 | .572 | .426 |
| win-stay（描述性） | Easy | .953 | .966 | .964 | .093 | .859 |
| | NearTie | .953 | .951 | .959 | .917 | .158 |
| lose-shift（描述性） | Easy | .138 | .124 | .122 | .407 | .807 |
| | NearTie | .114 | .107 | .106 | .826 | .826 |

**Estimate claims（严格邻接解析；宽松模式会把 "79 trials of Button C … 21 trials of OTHER buttons" 误绑而读出 ~.80）**

这一节检验两个问题：模型是否利用已有证据，以及 α 是否改变 reward 后的即时行为反应。

各指标：

- `policy/action → empirical-best`：是否选择当前经验成功率最高的臂。
- `policy → posterior-mean-best`：是否选择 Beta posterior mean 最高的臂。
- `post-discovery adherence`：发现真实最优臂后，后续是否持续选择它。
- `win-stay`：上一轮获得1后，下一轮是否继续选择同一臂。
- `lose-shift`：上一轮获得0后，下一轮是否换臂。

主要结果：
1. 模型具有很强的利用倾向: empirical-best alignment 大约 `.79–.90`，win-stay 约 `.95`，说明模型通常选择当前看来最好的臂，并在获得奖励后继续选择。
2. α 没有显著改变证据利用: 所有配对检验均不显著。虽然 +4 的 empirical-best alignment 和 adherence 数值较低，但不能解释为稳定效应。
3. Policy 与 action 几乎相同: 两组 empirical-best 数字高度接近，再次说明 Stage 2 主要忠实执行 Stage 1 Policy，并没有独立修正策略。
4. Trial-by-trial feedback response 不受 α 调节 : `win-stay` 和 `lose-shift` 均无显著差异。这符合当前的 specificity 预测：α 没有改变即时 reward 更新或 phasic-like feedback sensitivity。

另外，+4 较低的 empirical-best alignment 不一定表示更差。结合上一节，它有时可能继续选择**暂时被经验噪声低估的真实最优臂**，从而被记为不跟随 empirical-best。整体应写成：

> 模型稳定利用已有证据，但 α 未显著改变证据利用、最优臂坚持或即时反馈反应；+4 的分布锐化没有转化成更高的 empirical-best choice accuracy。

| 环境 | α | rate claims | count claims | 引用 rate 的轮次覆盖率 |
|---|---|---|---|---|
| Easy | −4 | 520/520 = **1.0000** | 145/154 = .9416 | 440/2000 = .220 |
| | 0 | 393/393 = **1.0000** | 233/234 = .9957 | 319/2000 = .160 |
| | +4 | 470/470 = **1.0000** | 133/135 = .9852 | 365/2000 = .182 |
| NearTie | −4 | 311/311 = **1.0000** | 131/135 = .9704 | 283/2000 = .141 |
| | 0 | 152/152 = **1.0000** | 143/144 = .9931 | 131/2000 = .066 |
| | +4 | 234/234 = **1.0000** | 101/101 = **1.0000** | 205/2000 = .102 |

这张表检验的是模型能否正确读取并复述历史统计，而不是是否正确理解不确定性。

- `rate claims`：模型明确引用某个 arm 的经验成功率时，该数字是否正确。
- `count claims`：模型引用某个 arm 的 trials/rewards 数量时，数字是否正确。
- `引用 rate 的轮次覆盖率`：2000轮中，有多少轮至少引用了一次经验成功率。

例如 Easy −4：

- 模型共提出520个可严格匹配的 rate claim，`520/520` 全部正确；
- 154个 count claim 中145个正确；
- 440/2000轮至少引用了一次 rate，即覆盖22%。

主要结论：

1. **经验率读取完全正确**: 六格的 `rate claims` 准确率全部为100%。因此，后续的不确定性错误不能归因于模型看错或算错经验率。

2. **计数读取也非常准确**:准确率为94.2%–100%。少量错误可能来自复杂句法或真实复述错误，但整体不是主要缺陷。

3. **模型并非每轮都引用数字**:rate 引用覆盖率只有6.6%–22%。因此，100%准确表示“在可解析且确实引用 rate 的轮次中全部正确”，不表示每轮都显式计算或表达了 rate。

4. **引用频率没有清楚的 α 单调效应**:两个环境都是 −4 最高，但0和+4的关系不一致；在没有相应统计检验时，只能作描述性报告。

> 模型能够准确读取已有统计证据，但这不保证它能正确判断证据的不确定性。`estimate_accuracy` 与接下来的 `uncertainty_calibration` 是两个不同能力。

**`uncertainty_calibration`（正 ρ = 反向校准；描述性）**

| 环境 | α | ρ(hedge, n) | p | ρ(hedge, posterior SD) | hedge@n>50 |
|---|---|---|---|---|---|
| Easy | −4 | **−.002** | .94 | −.003 | .906 |
| | 0 | **+.178** | 4e−15 | −.183 | .974 |
| | +4 | **+.163** | 6.9e−13 | −.162 | .974 |
| NearTie | −4 | **+.057** | .013 | −.077 | .937 |
| | 0 | **+.189** | 6.7e−17 | −.183 | .967 |
| | +4 | **+.159** | 2.7e−12 | −.140 | .957 |

这里检验的是：模型使用 “uncertain / limited evidence / small sample”等 hedge 语言时，是否真的对应统计不确定性。

正确校准应当表现为：

- 样本量 `n` 越大，hedge 越少：`ρ(hedge,n) < 0`
- posterior SD 越大，hedge 越多：`ρ(hedge,SD) > 0`

但结果基本相反。

### α=0 和 +4：明显反向校准

以 Easy α=0 为例：

- `ρ(hedge,n)=+.178`：样本越多，反而越常表达不确定。
- `ρ(hedge,SD)=−.183`：统计不确定性越高，反而越少 hedge。
- 当 `n>50` 时，97.4%的文本仍使用 hedge。

NearTie 也复现相同方向。因此，这不是环境特有现象。

### −4：Easy 中消除了相关性，但没有形成正确校准

Easy −4：

- `ρ(hedge,n)=−.002`
- `ρ(hedge,SD)=−.003`

两者都接近零。说明 −4 消除了明显的反向关系，但没有得到理论上正确的“样本越多越确定”关系。

NearTie −4 仍有轻微反向校准：`ρ=+.057`，但比0和+4弱得多。

### `hedge@n>50`

表示目标臂已经被采样超过50次时，文本仍然使用 hedge 的比例。六格均高达90.6%–97.4%，说明模型几乎习惯性地表达“不确定”，没有随证据积累减少。

最准确的结论是：

> 模型可以准确复述经验率，却不能正确校准不确定性语言。α=0和+4表现出稳定的反向校准；−4削弱了这种反向关系，但没有恢复正确校准。

这是一个有趣的**文本层次级发现**，但不是探索行为证据，而且目前采用 pooled correlation 和词汇解析，适合作描述性结果，不宜作为主要 α 效应。

### 4.5 Persistence and failure modes

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| GreedyFrac r1–4 | Easy | .317 | .417 | .300 | .058 | **.047\*** |
| | NearTie | .283 | .367 | .267 | .096 | .071 |
| GreedyFrac r5–9 | Easy | .770 | .750 | .770 | .434 | .674 |
| | NearTie | .790 | .710 | .760 | .059 | .372 |
| GreedyFrac r10–49 | Easy | .844 | .855 | .819 | .928 | .859 |
| | NearTie | .855 | .809 | .785 | .806 | .928 |
| GreedyFrac r50–100 | Easy | .893 | .886 | .852 | .893 | .518 |
| | NearTie | .915 | .883 | .912 | .270 | .249 |
| longest same-arm run | Easy | 74.8 | 79.8 | 80.3 | .368 | .529 |
| | NearTie | 74.2 | 75.5 | 76.8 | .619 | .706 |
| `suffix_failure` | Easy | .300 | .200 | .200 | .157 | n/a |
| | NearTie | .500 | .350 | .400 | .083 | .317 |
| premature lock | Easy | .350 | .300 | .350 | .655 | .564 |
| | NearTie | .300 | .350 | .300 | .564 | .564 |
| wrong-arm lock | Easy | .150 | .050 | .050 | .157 | n/a |
| | NearTie | .150 | .150 | .150 | n/a | n/a |

**Abandonment（pooled）**

| 指标 | 环境 | α=−4 | α=0 | α=+4 |
|---|---|---|---|---|
| one-shot-ZERO abandoned | Easy | 44/52 = .846 | 43/52 = .827 | 45/52 = .865 |
| | NearTie | 43/52 = .827 | 40/52 = .769 | 43/52 = .827 |
| one-shot-POSITIVE abandoned | Easy | 2/27 = .074 | 1/27 = .037 | 1/27 = .037 |
| | NearTie | 1/28 = .036 | 2/28 = .071 | 1/28 = .036 |

**双峰 outcome：lock 计数 + exact McNemar（主读数）**

| 环境 | α | lock-correct (>.8) | lock-wrong (<.2) | mid | mean | vs α=0: better / worse / same | McNemar p |
|---|---|---|---|---|---|---|---|
| Easy | −4 | 14/20 | 6 | 0 | .694 | 0 / 2 / 18 | .500 |
| | 0 | 16/20 | 4 | 0 | .796 | — | — |
| | +4 | 16/20 | 4 | 0 | .793 | 0 / 0 / 20 | n/a |
| NearTie | −4 | 8/20 | 12 | 0 | .401 | 0 / 3 / 17 | .250 |
| | 0 | 11/20 | 9 | 0 | .556 | — | — |
| | +4 | 10/20 | 9 | 1 | .535 | 0 / 0 / 20 | n/a |

**NearTie arm-tier shares**（`.55` 非严重失败，故严格 `late_opt_frac` 低估表现）

| α | true_best (.60) | near_best (.55) | top2 | inferior (.25) |
|---|---|---|---|---|
| −4 | .396 | .329 | .725 | .275 |
| 0 | .526 | .195 | .721 | .279 |
| +4 | .514 | .266 | .781 | .219 |

### 4.6 Text and language–behaviour dissociation

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| `stance=explore` | Easy | .080 | .049 | .043 | **.015\*** | **.027\*** |
| | NearTie | .097 | .065 | .046 | **.012\*** | **.003\*** |
| `stance=exploit` | Easy | .906 | .945 | .948 | **.010\*** | .205 |
| | NearTie | .899 | .933 | .953 | **.021\*** | **.001\*** |
| `stance=both` | Easy | .015 | .007 | .009 | .680 | .414 |
| | NearTie | .004 | .003 | .002 | .854 | .180 |
| kw: `explor*` | Easy | .095 | .056 | .052 | **.011\*** | .232 |
| | NearTie | .101 | .068 | .047 | **.016\*** | **.001\*** |
| kw: most trials | Easy | .067 | .058 | .077 | .501 | .248 |
| | NearTie | .125 | .076 | .058 | **.031\*** | .816 |
| kw: highest rate | Easy | .912 | .916 | .925 | .977 | .510 |
| | NearTie | .860 | .871 | .907 | .495 | .277 |
| kw: future/information | Easy | .047 | .023 | .020 | .151 | .598 |
| | NearTie | .063 | .026 | .027 | .070 | .888 |
| kw: hedge | Easy | .889 | .931 | .911 | .312 | .129 |
| | NearTie | .910 | .916 | .910 | .869 | .904 |
| kw: confident | Easy | .444 | .326 | .309 | .053 | .952 |
| | NearTie | .398 | .185 | .214 | **<.001\*** | .729 |
| rationale chars | Easy | 269.6 | 267.2 | 264.0 | .571 | .812 |
| | NearTie | 286.6 | 259.8 | 248.3 | **.011\*** | .083 |

**`stance_behavior_alignment_broad` 与分解**（broad = non-greedy ∪ untried）

| 环境 | α | EXPLORE n | broad 兑现 | →untried (cue) | →unc-max | →true-best persist | →low-n | →costly error | →empirical-greedy |
|---|---|---|---|---|---|---|---|---|---|
| Easy | −4 | 160 | 95/160 = .594 | **77** | 5 | 0 | 0 | 13 | 65 |
| | 0 | 98 | 81/98 = .827 | **79** | 2 | 0 | 0 | 0 | 17 |
| | +4 | 86 | 82/86 = .953 | **79** | 1 | 0 | 0 | 2 | 4 |
| NearTie | −4 | 194 | 86/194 = .443 | **79** | 4 | 0 | 0 | 3 | 108 |
| | 0 | 129 | 84/129 = .651 | **80** | 4 | 0 | 0 | 0 | 45 |
| | +4 | 91 | 83/91 = .912 | **80** | 2 | 0 | 0 | 1 | 8 |

**Matched-state 衰减链**（与 α=0 共享相同 `(choices, feedbacks)` 前缀的轮次）

| 环境 | α | matched n | text | numbers | certainty | stance | target | action |
|---|---|---|---|---|---|---|---|---|
| Easy | −4 | 346 | .688 | .488 | .098 | .101 | .049 | .052 |
| | +4 | 574 | .589 | .453 | .139 | .023 | .024 | .026 |
| NearTie | −4 | 442 | .781 | .581 | .075 | .138 | .036 | .038 |
| | +4 | 498 | .582 | .371 | .120 | .052 | .030 | .032 |

### 4.7 Outcome

| 指标 | 环境 | α=−4 | α=0 | α=+4 | p(−4) | p(+4) |
|---|---|---|---|---|---|---|
| mean reward | Easy | .574 | .617 | .614 | .339 | .446 |
| | NearTie | .462 | .469 | .483 | .839 | .172 |
| `opt_frac` | Easy | .667 | .764 | .761 | .184 | .600 |
| | NearTie | .396 | .526 | .514 | .284 | .865 |
| `late_opt_frac`（双峰） | Easy | .694 | .796 | .793 | .068 | .180 |
| | NearTie | .401 | .556 | .535 | .058 | .056 |
| `cum_regret` | Easy | 16.6 | 11.8 | 11.9 | .167 | .527 |
| | NearTie | 11.3 | 10.7 | 9.0 | .807 | .386 |
| `k_min_frac` | Easy | .038 | .038 | .038 | 1.000 | n/a |
| | NearTie | .040 | .040 | .040 | n/a | n/a |
| `greedy_frac` | Easy | .849 | .853 | .818 | .655 | .575 |
| | NearTie | .865 | .829 | .833 | .656 | .366 |

**Final score（总分 = T=100 轮 reward 之和；即 Stage-1 prompt 中 `Your score so far: N points.` 的量）**

| 环境 | α | mean | sd | median | range | Δ vs α=0 [95% CI] | p |
|---|---|---|---|---|---|---|---|
| Easy | −4 | 57.35 | 21.16 | 67.0 | [22, 82] | −4.40 [−11.15, +0.25] | .370 |
| | 0 | **61.75** | 19.39 | 69.0 | [21, 80] | — | — |
| | +4 | 61.40 | 19.64 | 67.5 | [21, 82] | −0.35 [−1.35, +0.50] | .496 |
| NearTie | −4 | 46.15 | 14.07 | 50.5 | [20, 62] | −0.70 [−2.40, +0.60] | .839 |
| | 0 | 46.85 | 14.48 | 51.5 | [20, 62] | — | — |
| | +4 | **48.35** | 13.83 | 52.0 | [20, 62] | +1.50 [−0.25, +4.00] | .172 |

**与冻结算法基线对照**（manifest 仅存 `cum_regret`，故两侧同以 `E[score] = 100·p* − cum_regret` 推导；实得与期望差 1.0–2.6 分 = tape 采样噪声）

| 环境 | 策略 | E[score] | 95% CI |
|---|---|---|---|
| Easy（p\*=.75，天花板 75） | ORACLE | 75.00 | — |
| | **model α=0** | **63.20** | [53.90, 70.45] |
| | **model α=+4** | **63.08** | [53.85, 70.38] |
| | GREEDY | 60.83 | [51.38, 68.35] |
| | **model α=−4** | **58.38** | [48.90, 67.72] |
| | RANDOM | 37.55 | [36.67, 38.40] |
| NearTie（p\*=.60，天花板 60） | ORACLE | 60.00 | — |
| | **model α=+4** | **50.99** | [44.94, 56.08] |
| | **model α=0** | **49.26** | [42.88, 54.91] |
| | GREEDY | 48.79 | [42.55, 54.27] |
| | **model α=−4** | **48.71** | [42.52, 54.24] |
| | RANDOM | 41.58 | [41.06, 42.17] |

**Per-block mean reward（Holm 校正 10 个 block）**

| 环境 | α | 1–10 | 11–20 | 21–30 | 31–40 | 41–50 | 51–60 | 61–70 | 71–80 | 81–90 | 91–100 | raw p<.05 | Holm 存活 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Easy | −4 | .485 | .525 | .590 | .595 | .585 | .585 | .590 | .595 | .590 | .595 | blocks 4, 7 | **NONE** |
| | 0 | .470 | .545 | .645 | .660 | .645 | .620 | .655 | .630 | .640 | .665 | — | — |
| | +4 | .500 | .555 | .645 | .635 | .600 | .635 | .635 | .620 | .660 | .655 | — | **NONE** |
| NearTie | −4 | .415 | .430 | .420 | .490 | .460 | .530 | .485 | .475 | .440 | .470 | — | **NONE** |
| | 0 | .390 | .455 | .425 | .520 | .460 | .520 | .485 | .495 | .460 | .475 | — | — |
| | +4 | .420 | .455 | .480 | .520 | .495 | .540 | .480 | .505 | .460 | .480 | — | **NONE** |

### 4.8 Primary readouts（Holm，单一 family，16 tests）

| # | 指标 | 环境 | α=−4 | α=0 | α=+4 | raw p(−4) | raw p(+4) | Holm adj |
|---|---|---|---|---|---|---|---|---|
| 1 | `policy_uncertainty_targeting`（unique-max） | Easy | .000 | .000 | .000 | n/a | n/a | — |
| 1 | | NearTie | .000 | .000 | .000 | n/a | n/a | — |
| 2 | `b_information` | Easy | — | — | — | CI [−.124, +.418] 含 0 | | — |
| 2 | | NearTie | — | — | — | CI [−.124, +.190] 含 0 | | — |
| 3 | `α × posterior_gap` | NearTie | — | — | — | .798 | .374 | — |
| 4 | `stance_align_broad` | Easy | .810 | .897 | .971 | .0421 | .0422 | .2944 / .2944 |
| 4 | | NearTie | .712 | .790 | .944 | .1355 | **.0050** | .5420 / **.0400\*** |
| 5 | `suffix_failure` | Easy | .300 | .200 | .200 | .1573 | n/a | .5420 |
| 5 | | NearTie | .500 | .350 | .400 | .0833 | .3173 | .4163 / .5420 |
| 5 | `wrong-arm lock` | Easy | .150 | .050 | .050 | .1573 | n/a | .5420 |
| 5 | | NearTie | .150 | .150 | .150 | n/a | n/a | — |

**Holm survivor: NearTie `stance_align_broad`, +4 vs 0**（adjusted `p=.0400`）。

该结果表示 +4 提高了“模型自报 `EXPLORE` 时，其目标至少属于 non-greedy 或 untried”的一致性，而不是增加探索行为本身：NearTie 中 `EXPLORE` 的绝对兑现次数几乎不变（−4/0/+4 = **86/84/83**），其中 +4 的 83 次有 **80 次**是 cue 驱动的 untried-arm 初始覆盖，指向 uncertainty-max 的只有 **2 次**。对齐率从 `.790` 升至 `.944`，主要因为 +4 减少了未兑现的 `EXPLORE` 表述，而不是产生更多 uncertainty-directed sampling。

因此，更新后的 primary 结论不是“所有指标均 null”，而是：**α 对政策标签与行为的一致性存在一个条件性效应，但定向探索的数量、uncertainty weighting、错误锁定与结果指标仍未显示可靠改善。**

### 4.9 Environment dependence（DiD）

DiD = (NearTie_α − NearTie_0) − (Easy_α − Easy_0)，seeds 两环境共有 n=20。DiD 为二阶差分，方差约为一阶的两倍；CI 含 0 记作 **未检出**。

| 指标 | α | Δ Easy | Δ NearTie | DiD | 95% CI | p |
|---|---|---|---|---|---|---|
| `policy_uncertainty_targeting` | −4 | .000 | .000 | n/a | — | n/a |
| | +4 | .000 | .000 | n/a | — | n/a |
| `stance_behavior_alignment_broad` | −4 | −.087 | −.077 | +.010 | [−.080, +.115] | .833 |
| | +4 | +.074 | +.154 | +.081 | [+.029, +.139] | **.021\*** |
| `EXPLORE stance rate` | −4 | +.031 | +.032 | +.001 | [−.018, +.023] | .878 |
| | +4 | −.006 | −.019 | −.013 | [−.026, −.003] | **.007\*** |
| `suffix_failure` | −4 | +.100 | +.150 | +.050 | [+.000, +.150] | .317 |
| | +4 | .000 | +.050 | +.050 | [+.000, +.150] | .317 |
| `wrong-arm lock` | −4 | +.100 | .000 | −.100 | [−.250, +.000] | .157 |
| | +4 | .000 | .000 | .000 | [+.000, +.000] | n/a |
| `late_opt_frac`（双峰） | −4 | −.102 | −.155 | −.053 | [−.150, +.004] | .436 |
| | +4 | −.003 | −.021 | −.018 | [−.040, +.000] | .204 |
| `Stage-2 margin` | −4 | −.091 | −.245 | −.154 | [−.564, +.206] | .795 |
| | +4 | +.431 | +.154 | −.277 | [−.692, +.001] | .122 |

两项 +4 的未校正 DiD 置信区间不含 0：

- `stance_behavior_alignment_broad`：NearTie 相较 Easy 的 +4 对齐效应更强（DiD `+.081`, `p=.021`）。
- `EXPLORE stance rate`：+4 在 NearTie 中更明显地减少 `EXPLORE` 表述（DiD `−.013`, `p=.007`）。

这两项共同指向**环境依赖的政策表达／语义—行为对齐**：在价值更接近的 NearTie 中，+4 更少使用 `EXPLORE` 标签，但一旦使用便更少落回 empirical-greedy 行为。它们不表示更多探索，因为 uncertainty targeting 仍为 0、兑现的绝对次数几乎恒定，且 final score / regret / lock outcomes 均未改善。由于本节同时检验多个 DiD 对比而未另作 family-wise 校正，这两项应标为 exploratory environment interactions，而非新的定向探索主证据。

### Overall interpretation

更新后的 NearTie +4 不再支持“α 完全只改文字、所有 primary 均为 null”的最强表述。更准确的结论是：

1. **模型具有稳定的 greedy / uncertainty-avoidant 基线。** NearTie 中 `w<0`，而 uncertainty-max targeting 仍为 0；模型会准确读取经验率，却不会系统性购买信息。
2. **α 改变政策表达及其一致性。** −4 更常自报 `EXPLORE`；+4 更少自报，但在 NearTie 中自报与广义行为类别的对齐更高。候选分布锐化仅为次级、未校正信号。
3. **PV9 未检出 α 对 directed exploration 的调节。** `a_info` 两环境均跨 0，low-n／uncertainty targeting 仍在地板，唯一存活的对齐指标主要由 cue-supported coverage 和分母变化构成。
4. **α 未带来可靠的任务改善。** reward、regret、late optimal choice、suffix failure 与 wrong-arm lock 均无稳健差异。

因此，PV9 仍应定位为 RSN–dopamine 类比的**作用边界证据**，但措辞需要从“α 只有文字效应”收紧为：**α 调节 policy stance、commitment-related sharpness，以及困难环境中的 stance–behavior consistency；在这个协议中，它没有表现为 uncertainty-directed exploration controller，也没有改善 Bandit outcome。**

# Best Arm Identification (BAI) / Pure Exploration）Bandit

## 1. Motivation

PV9 要求模型在 100 轮内最大化累计奖励，属于 **reward-maximizing Bandit**。在这种设定下，探索会牺牲即时收益，因此 PV9 检验的是模型是否愿意承担机会成本以获取信息。

PV10 将目标改为：经过 100 轮采样后，正确识别 reward probability 最高的 arm。任务因此转变为 **best-arm identification / pure exploration**：

- 前期采样用于降低不同 arm 的不确定性；
- 重复选择当前最佳 arm 本身不会直接提高最终得分；
- 模型需要在信息收集结束后，最终 commit 到一个 arm。

这可以检验 PV9 的 α-null 是否源于累计奖励目标本身鼓励模型过早进入 exploitation。

### Competing Predictions

| Potential mechanism | −α prediction | +α prediction |
|---|---|---|
| Decision precision | 判断较摇摆，采样较分散 | 更快形成明确判断 |
| Goal motivation / information investment | 信息投入不足或随机采样 | 系统性采样低证据或高不确定性 arm |
| Commitment / persistence | 延迟承诺 | 可能过早锁定当前领先 arm |
| Final performance | 可能因证据整合不足而下降 | 适量可能提高；过高可能因 premature commitment 下降 |

因此，+α 存在两种竞争性解释：

1. **Motivational account**：+α 提高“正确识别最优臂”这一终局目标的 salience，促进定向信息获取并提高最终识别率。
2. **Commitment account**：+α 放大当前领先 arm 的吸引力，导致过早锁定并减少验证性探索。

### Experimental Design

实验分为两个明确阶段：

```text
Exploration phase:
You have 100 trials to gather evidence. Rewards during these trials do not
directly determine your score.

Final decision:
After 100 trials, select the button with the highest unknown reward
probability. You receive one final score for identifying it correctly.
```

最小实验配置：

- Bernoulli reward probabilities：`[0.60, 0.50, 0.40, 0.30]`
- α：`−4 / 0 / +4`
- 20 个配对 seeds，arm labels 随机排列
- 100 个 evidence-gathering rounds
- 最后独立输出 `FINAL: Button X`
- 不提供真实 reward probabilities
- 不累积中间 task score
- 不提供额外 exploration cue
- α 仅作用于 Stage 1，Stage 2 保持 unsteered

### Primary Metrics

1. 最终最优臂识别率  
2. Posterior-uncertainty-max targeting rate  
3. 各 arm 的采样分配与最低采样数  
4. Premature commitment：过早集中选择当前 empirical-best arm  

### Interpretation

PV10 是对 PV9 的机制诊断，而不是为了寻找显著结果：

- 若 α 仍不改变 uncertainty targeting，说明 α 可能不是 directed-exploration controller。
- 若 +α 增加定向采样并提高识别率，支持 **goal-dependent wanting**。
- 若 +α 造成更早锁定且表现不升，支持 **commitment/persistence account**。

PV9 测量的是是否愿意牺牲即时奖励购买信息；PV10 测量的是为了最终识别目标如何分配采样资源。二者结合可以判断 α 的作用是否取决于任务的 **goal framing**。

## 2. Design
### PV10-A: Fixed-budget Sampling

模型获得固定的采样预算，例如 100 次；采样结束后必须选择最佳 arm。

```text
Rounds 1–100:
SAMPLE Button X

After Round 100:
FINAL: Button X
```

它主要测量：

- 如何在 arms 之间分配有限样本；
- 是否偏向高不确定性或 top-two candidate arms；
- 是否过早集中于当前领先 arm；
- 最终 best-arm identification accuracy。

这个版本最干净，适合先跑，因为所有 α 条件拥有相同信息预算。

### PV10-B: Self-paced Sampling and Commitment

模型每轮可以继续采样，也可以随时做最终承诺：

```text
Policy: SAMPLE Button X
```

或：

```text
Policy: COMMIT Button X
```

但必须同时设置：

- 一个最大上限，例如 `T_max = 100`；
- 或者每次采样具有明确成本，例如最终得分为：

\[
\text{Score}=\mathbb{1}(\text{correct identification})-\lambda N_{\text{samples}}
\]

否则“什么时候停止”没有代价，也就无法判断模型是否采样过多。

这个版本测的是另一种机制：

- `sampling allocation`：采样哪个 arm；
- `stopping threshold`：什么时候认为证据已经足够；
- `commitment accuracy`：承诺是否正确；
- `premature commitment` 与 `over-sampling`。

因此，PV10-A 对应 **pure exploration under a fixed budget**；PV10-B 对应 **active information sampling and optimal stopping**。建议先完成 A，再决定是否追加 B。

**参照定义：fixed-confidence BAI 的标准 stopping rule**（算法侧参照，非模型侧规则）

经典 fixed-confidence BAI（如 LUCB、Track-and-Stop）不预设采样预算，而是要求算法在有限步内以概率 `≥ 1−δ` 输出正确的最优臂：

- **δ-correct**：停止时间 τ（随机）与输出 arm `k̂` 满足 `P(k̂ = k*) ≥ 1 − δ`，其中 `k*` 是真实最优臂。
- **LUCB-style stopping rule**：每轮维护每个 arm 的置信区间 `[L_k(t), U_k(t)]`；令 `b(t) = argmax_k θ̂_k(t)`（当前经验最优臂），`c(t) = argmax_{k≠b(t)} U_k(t)`（除 b(t) 外置信上界最高的挑战者）。算法停止的条件是

  \[
  L_{b(t)}(t) \;>\; \max_{k \neq b(t)} U_k(t)
  \]

  即当前最优臂的置信下界已经高于所有其他 arm 的置信上界——不确定性区间不再重叠，才允许 commit。

这套判据来自经典统计 BAI 文献（如 Kaufmann & Kalyanakrishnan 2013; Jamieson et al. 2014 的 LUCB；也是本节 §3 引用的 MIT 论文 `PP-LUCB` 用的框架），本质是把"何时停止采样"变成一个**可证明正确率**的条件，而不是靠固定 T_max 或人为设置的 λ·N_samples 惩罚项。

**与 PV10-B 的关系**：PV10-B 目前设计是让模型自己输出 `SAMPLE` / `COMMIT`，停止决策是**模型的自主行为**，用 T_max 或采样成本 λ 只是防止"无限采样不用付出代价"这一退化解。上面这套 LUCB stopping rule 不是要模型学会计算置信区间，而是可以用作**算法侧的对照 baseline**——即用经典 δ-correct 判据跑一个"理性 agent 应该在什么时候停"的参照轨迹，再对比模型的实际 COMMIT 时机是过早（premature commitment）还是过晚（over-sampling）。这与 §4 已有的 GREEDY / ORACLE / RANDOM baseline 对照是同一思路的延伸。

**Stage 2 可以取消**

Stage 1 可以直接输出一个结构化决策：

```text
Evidence: ...
Policy: SAMPLE Button C
```

或：

```text
Evidence: ...
Policy: COMMIT Button A
```

然后由 code parser 直接执行。

这样测到的是 α 对 `SAMPLE/COMMIT` 决策本身的影响，而不是“Stage 1 先说、Stage 2 再执行”的文本传导。若保留 Stage 2，它更适合作为独立的 executor-fidelity control，不应成为主任务必需环节。

## 3. Literature


## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).

