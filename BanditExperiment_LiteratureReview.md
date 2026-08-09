# Bandit 实验：相关文献、模型能力边界与改进方案

> 更新：2026-07-31
> 目的：结合近期 Bandit–LLM 文献与本项目 Llama3-8B 的 C / D / D2 结果，判断小模型能否完成 Bandit，并确定下一版实验应该修什么、保留什么。

## 1. 结论先行

### 1.1 小模型能不能做 Bandit？

**可以，但不是无条件地可以。**

现有证据支持三个层级：

1. **利用已获得的信息（utilization / exploitation）**：7–8B 模型通常具备一定能力。本项目 D 条件下，Llama3-8B 的 empirical-best adherence 为 0.884，说明给定已经观察到的均值，它大部分时候会利用当前证据。
2. **自主发现未知选项（discovery / exploration）**：小模型容易过早锁定少数选项。Gemma2 2B/9B/27B（Llama3 与 Qwen2.5 在其 Appendix C.4 复现且 bias 持续）、Qwen2.5 3B/7B 与本项目 Llama3-8B 都出现过 action coverage 停滞、greedy lock、frequency bias 或 suffix failure。
3. **稳定完成“探索后收敛”的完整策略**：未经专门训练的小模型并不可靠。提示、短 CoT、历史摘要、任务规模和训练方式都会显著改变结果；平均 regret 或 OptFrac 好看，也可能只是早期碰巧找到最优臂后一直锁定。

因此，当前 Bandit 不应被简单判断为“模型会”或“模型不会”。更准确的研究问题是：

> RSN α 改变的是未知选项的信息寻求、已知证据的利用，还是一般性的策略持续性？

### 1.2 对当前实验最重要的改进

优先级从高到低：

1. **把 capability validation 与 α 主实验分开**：用外部提供的平衡初始证据验证 utilization；主实验仍保留自主 exploration。
2. **增加 short-CoT 接口对照**：当前 `Choice: ` anchor 要求立即输出臂名，几乎不给模型显式整合样本量、均值和剩余轮次的空间；文献显示 CoT 对小模型的 Bandit coverage 和训练效果是 load-bearing。
3. **使用难度阶梯**：先 K=2，再 K=3，最后才是当前 K=5；不能用 K=5 的失败直接推断模型完全没有 Bandit 能力。依据是 EVOLvE 的任务参数扫描（MAB K=5/20、reward 分布、\(\Delta_{\min}\)）与 When Greedy Wins 的模型内梯度（7B 在 K=2/3/5 随迭代提升且 K 越小越准，3B 停滞）——两者都是论文内部比较，**不依赖已弃用的 TextBandit 跨模型差异**。
4. **保留简洁 summary，不再增加解释段落**：将 tried / untried 分开，并显式给出 `successes / trials / rate`；不要继续在 prompt 中堆采样理论。
5. **指标按 discovery / utilization / stability / outcome 分层**：coverage 和 OptFrac 都不能单独作为主结论；加入 GreedyFreq、SuffixFail 和 matched-history divergence。

## 2. Paper
### Paper Summary
| 论文 | Bandit 在论文中的角色 | 是否证明 LLM 自己会做 Bandit | 对本项目的直接价值 |
|---|---|---:|---|
| [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html) | 直接评估 LLM 在 BanditBench 中的 in-context exploration（**MAB K=5/20；CB K=10/30**），并比较 summary、UCB guidance、few-shot 与 fine-tuning | 基线部分是；algorithm-guided / distillation 部分不是纯自主能力 | 最直接支持 structured summary、难度阶梯与 exploration-optimality 分析，同时要求把“模型自主探索”与“外部算法供给探索”分开 |
| [LLMs are Greedy Agents](https://arxiv.org/abs/2504.16078) | Gemma2 2B/9B/27B 主实验，**Llama3 / Qwen2.5 在 Appendix C.4 复现且 bias 持续**；环境含 MAB（K=10/20，T=50）、contextual bandit 与**文字井字棋** | 是，且系统分析失败模式 | 最直接：greediness、frequency bias、knowing–doing gap；CoT、try-all、summary 和 RLFT 的作用。**Llama3 上的同族直接证据** |
| [When Greedy Wins](https://arxiv.org/abs/2509.24923) | Qwen2.5 3B/7B 学习 meta-bandit policy（**K=2/3/5 训练 + K=10 OOD 泛化**） | 是，但主要研究训练后策略 | 平均 reward 提升不等于探索改善；必须报告 GreedyFreq、SuffixFail、双峰和早停探索；**提供文献中唯一的模型内 K 难度梯度** |
| [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118) | 经典 TS/回归 Bandit 负责探索，LLM 只预测 reward | 否，反而指出直接选臂常次优 | 支持把 reward understanding 与 action selection 拆开；经典算法控制应作为上界/诊断，不是 α 主实验 |
| ~~[TextBandit](https://arxiv.org/abs/2510.13878)~~ | ~~开源 LLM 直接根据语言化 0/1 feedback 选臂~~ | — | **不采用**：few-shot 教了 5 个不同最优臂，与固定最优臂的结构假设自相矛盾（详见 §2.4）。其跨模型差异不构成证据 |
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


### 2.1 `EVOLvE`：当前实验设计的直接来源与边界

EVOLvE 在 BanditBench 中系统评估 context-free MAB 与 contextual bandit。MAB 同时改变
reward distribution、gap、arm 数量和名称表示：包括 Bernoulli / Gaussian、K=5 / K=20，
以及无语义的 Video 标签与语义丰富的 Clothes 名称（CB 另用 MovieLens，K=10 / K=30，
与 MAB 的 K 不是同一套）。论文测试 Gemma-2B、Gemma-9B、
Gemini-1.5 Flash 和 Gemini-1.5 Pro（**均非 7–8B 开源模型**），并比较四个层次：

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

这对本项目有四个直接含义：

1. 当前 `TRIED / UNTRIED + rewards / trials / rate` 不是任意 prompt engineering，
   而是与论文的 sufficient-statistics contextualization 同方向；它减少读取历史的
   负担，但不保证模型自动获得 exploration policy。
2. 若把 UCB bonus 或“现在应该探索哪个 arm”直接写进 prompt，属于 **AG 条件**：
   可以作为能力上界或 scaffolded-policy 对照，但不能再解释为 Llama3-8B 自主产生
   exploration，也不适合作为 RSN α 的唯一主实验。
3. 论文用随时间变化的 `MinFrac` 与 `OptFrac` 区分早期广泛探索和后期利用，并明确
   提醒高 OptFrac 可能只是偶然先选中最优臂。这直接支持本项目把 novel exploration、
   lock / persistence、late adherence 与 regret 分开报告。
4. 论文显示 smaller model 经 UCB trajectory distillation 后可以超过较大但未优化的
   模型，因此“小模型做不好”更准确地表示**off-the-shelf in-context policy 不可靠**，
   而不是不存在可训练或可 scaffold 的 Bandit 能力。

需要保留的边界是：EVOLvE 的 improved condition 大量依赖外部计算的 UCB 信息、
oracle demonstrations 或参数微调；它不能证明仅靠增加自然语言解释，Llama3-8B 就能
稳定完成当前 K=5 自主探索。相反，它支持将 E-direct 作为自主行为主条件，并把更强的
algorithm-guided prompt 单独命名为诊断对照。

### 2.2 `LLMs are Greedy Agents`：最直接的设计依据

该研究在 BanditBench 的 Gaussian/Bernoulli MAB 上测试 Gemma2 2B、9B、27B（**K=10 / K=20 arms**），horizon 同样是 50；并在 Appendix C.4 用 **Llama3 与 Qwen2.5** 复现 greediness 分析，明确指出 bias 持续。重要发现包括：

- 不同规模模型都会过早采用 greedy strategy，action coverage 很快停滞；扩大模型只能减轻，不能消除。
- 2B 模型还会受 action 在历史中出现频率影响，即使该 action reward 较差。
- 27B 可以正确计算 UCB，但即使 rationale 正确，仍常执行 greedy action，形成 knowing–doing gap。
- CoT 明显提高 coverage；没有 CoT 时，各规模模型探索都更少。
- 在多种 in-context 措施中，**初始 try-all 的改善最大**；这说明模型拿到足够信息后更擅长利用，而自主获取信息是主要短板。
- RL fine-tuning 能改善 2B/9B 的 regret，并使 2B coverage 增加约 12%，但仍未完全达到理想探索。

对本项目的含义：

1. 当前 Llama3-8B 的 greedy lock 不是异常，也不必优先归因于 code bug——**该论文已在 Llama3 族上直接复现同一 bias，这不再是跨模型外推**。
2. **允许短 CoT 是一个尚未被当前 C/D/D2 正式测试的重要接口维度。**
3. forced initialization 可以验证“获得信息后是否会用”，但不能作为自主 wanting/exploration 的主实验。

### 2.3 `When Greedy Wins`：为什么 OptFrac / regret 不够

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

### 2.4 `TextBandit`：**不采用**（设定自相矛盾，仅存档）

TextBandit 使用语言化反馈（“earned a token” / “did not earn a token”），每轮重建包含完整历史的 prompt，不使用 CoT，测试 Qwen3-4B、Qwen3-8B、Llama3.1-8B 与 Phi-2。论文报告：

- Qwen3-4B 的 best-arm selection rate 达 89.2%；
- Llama3.1-8B 为 31.6%，其他多数模型也低于经典算法；
- 模型规模与结果并非单调关系；
- 任务从 2 到 5 arms，简单 2-arm 条件更容易出现正结果。

这个结果只能说明：

> 小模型在某些非常简单、特定 prompt / few-shot / arm-label 条件下能够形成 Bandit-like adaptation。

**但本项目已决定不引用该论文的任何结论。** 其 few-shot 示例教给模型 5 个不同的“最优臂”，与“最优臂固定”这一结构性假设（也是 best-arm selection rate 的计算前提）自相矛盾；加上 baseline 数字的源码 bug、无均值/无置信区间的图表、核心发现无消融支撑，其跨模型差异（Qwen3-4B 好 / Llama3.1-8B 差）说明不了任何事。**难度阶梯的正当性改由 EVOLvE 的任务参数扫描与 When Greedy Wins 的模型内 K=2/3/5 梯度支撑**（两者都是同一论文内部的比较，比跨论文比模型干净）。本节仅作存档。

### 2.5 `Should You Use Your LLM to Explore or Exploit?`：能力应拆开测

该研究测试 GPT-5-nano、GPT-4、GPT-4o、GPT-3.5、Qwen-2.5、Gemma-3、Mistral-7B 与 DeepSeek-R1-Distill-Qwen（reasoning model）。它不要求 LLM 一次完成完整 Bandit policy，而是分别测试：

- exploitation oracle：给定历史，选择当前最优 action；
- exploration oracle：在大而有语义的 action space 中提出值得尝试的候选。

核心结论是：

- LLM 在小型、数值化 exploitation task 上可以有一定表现；
- reasoning model 在 exploitation 上相对占优，但**所有被测 LLM 配置（含 tool use / summarization mitigation）仍全面弱于一个简单线性回归 baseline**，且推理模型太慢/太贵；
- succinct summary 和工具会改善结果，但仍常不如简单 regression；
- LLM 更适合在大而有语义的空间中提出 exploration candidates，而不是替代完整 Bandit 算法。

对当前 K=5 boutique task 而言，臂名只是任意标签，不存在可利用的语义 action space。因此“LLM 擅长语义探索”并不能帮助当前任务；真正可借鉴的是**分离 discovery 与 utilization**。

### 2.6 三篇“Bandit 调度 LLM”的论文：只能作为方法类比

`OPTS`、`BanditSpec` 与 `LLM-Enhanced MAB` 的共同结构是：

```text
经典 Bandit 算法负责探索/调度
        ↓
LLM 负责生成、预测或执行局部子任务
```

它们说明外部 TS/UCB/EXP3 controller 可以稳定处理 exploration–exploitation，但不说明 LLM 本身学会了该策略。

对本项目可以提供两类对照：

1. **algorithmic upper bound**：同一组 reward seeds 下跑 UCB / TS，确认环境和 gap 足以产生学习；
2. **reward-prediction diagnostic**：让 LLM 只判断哪个 option 的证据最强，不让它决定是否 exploration。

但不能把 external controller 用于 α 主实验，否则 exploration timing 已被程序决定，正好删除待测行为。

## 3. PLAN

### 3.1 研究目标与证据边界

Llama3-8B 位于怎样的 Bandit capability boundary；RSN α 改变的是目标性 information seeking、exploration stopping、policy persistence，还是只让行为在 greedy lock 与 uniform flailing 之间移动？

不预设 Llama3-8B 必须完成 Reference-Hard，但 **α 的 capability-effect 解释必须建立在至少一个 native competence anchor 上**：模型需要先在不含外部 uncertainty / policy guidance 的 Reference-Easy（K=4）或 Reference-Hard（K=5）中表现出最低限度的 discovery→utilization。K=2 只作为最小 stochastic adaptation 诊断，不足以建立主实验的 competence anchor。若 K=4/K=5 native 条件都未达到门槛，α 仍可用于描述 greedy lock、uniform flailing 与 policy persistence 的变化，但不能称为 information-seeking improvement 或 capability rescue。

后续结论必须区分三种证据：

1. **Native capability**：模型只看到环境说明与 externally summarized sufficient statistics，自行决定 explore / exploit。
2. **Calculator / uncertainty scaffold**：Python 进一步提供平滑概率、uncertainty 或 credible interval；模型仍做选择，但状态表示已受到外部算法帮助。
3. **Algorithm-guided policy**：warm-start、UCB/TS guidance 或 oracle demonstrations 已提供探索信息或策略，只能作为 utilization control、upper bound 或训练条件。

新实验使用独立 protocol version、输出目录和 resume key。

### 3.2 文献对齐的 reference environment

第一阶段采用 [Krishnamurthy et al.（NeurIPS 2024）](https://arxiv.org/abs/2403.15371) 的两组 Bernoulli MAB，建立可与既有研究对照的能力坐标：

| 条件 | K | Reward probabilities | T | 用途 |
|---|---:|---|---:|---|
| **Reference-Easy** | 4 | `0.75 / 0.25 / 0.25 / 0.25` | 100 | 能力下界；确认模型在大 gap 下是否至少能脱离 greedy failure |
| **Reference-Hard** | 5 | `0.60 / 0.40 / 0.40 / 0.40 / 0.40` | 100 | 主能力与 α 实验；直接测试小 gap 下的自主探索与收敛 |

两组环境均遵守：

- neutral arm labels，不利用服饰语义；
- 使用实验前冻结的 counterbalanced seed bank；每个 run 仍由 seed 独立随机化 name→probability mapping 与 display order；
- 同一个 run 的全部 100 rounds 中，臂名称、显示位置与真实概率保持固定；
- 不同 α、LLM 接口与 algorithmic baselines 使用相同 seeds、arm mapping、per-arm reward tapes 和生成种子；
- N=20 runs；开发阶段可以先做 N=3 smoke test，但 smoke test 不产生研究结论。**smoke seeds 独立于正式 seed bank**：在查看任何模型行为之前生成并冻结，与 N=20 bank 不重合，使正式 20 个 run 保持完全未被观察；smoke 尽量覆盖不同 best-arm identity/position，但 N=3 不得作为 counterbalance 证据。

Reference 环境的次优臂概率相同，因此不再使用 `WorstFrac` 作为主指标；它只适用于原来的 graded reward vector。

若 Reference-Easy 仍未形成可解释的 native policy，Track C 可增加一个非文献 reference 的 **Native-Floor**：K=2、`.70/.30`、T=50。它只回答模型是否存在最小 stochastic adaptation，不进入 competence gate、不作为 α 主实验环境，结果也不得外推为 K=4/K=5 exploration 能力。

### 3.3 Native reference interface

主接口采用文献中唯一成功配置的核心元素：**suggestive framing + externally summarized history + reinforced CoT**，但加入 two-stage constrained final choice，以消除本项目已经观察到的格式污染，并冻结每轮只在真正的 action decision 上施加一次 α。

每轮状态只显示：

```text
Round 21 of 100; 80 rounds remain.

TRIED OPTIONS
- Button A: 3 rewards / 5 trials, empirical rate 0.60
- Button C: 1 reward / 4 trials, empirical rate 0.25

UNTRIED OPTIONS
- Button B
- Button D
- Button E

Each button has a fixed but unknown probability of reward 1.
Balance exploration and exploitation to maximize total reward over all 100 rounds.

Briefly reason about the amount of evidence, observed rewards, uncertainty,
and remaining rounds in at most two sentences. Do not state a final choice yet.
```

接口规则：

- Python 只计算 `successes / trials / empirical rate`；不提供 Beta smoothing、credible interval、UCB bonus 或推荐臂。
- `UNTRIED` 只表示 unknown，不编码为 `0.00`。
- reinforced CoT reminder 同时出现在任务说明与每轮 user query；第一阶段只生成最多两句 rationale，并完整保存。
- 第二阶段把同一轮完整状态、第一阶段 rationale 与 `Choice: Button` 组成 action prompt，只计算合法 suffix（`A/B/C/D/E` 中当前环境存在的 K 个）的条件 log-probability，并以 deterministic argmax 选择臂。
- **Rationale sanitization 必须冻结**（否则 α 的注入点会随生成内容漂移，破坏 §3.3 "α 只施加在 action prompt 最后一个 prefill token" 的语义）。规则如下，不使用字符或句子截断——长度由 token cap 控制，字符切分会破坏小数、臂名与推理内容：
  1. rationale 生成固定 `max_new_tokens=64`；
  2. 原样保存 `rationale_raw`；
  3. 删除所有包含 `Choice:` 的**完整行**（大小写不敏感），防止模型提前 commit 使 action prompt 出现两个 anchor；
  4. 对剩余文本只做首尾空白清理，不做任何语义改写、重排或摘要；
  5. 保存为 `rationale_clean`；
  6. 最后固定追加 `\nChoice: Button`，并对最终 prefill token 做审核（应为 `Button` 后的空格或等价 token，而非 chat control token）；
  7. α 在此 action pass 的最后一个 prefill token 注入。**（2026-08-05 更新）** 此处描述的是 `--steering_scope action`，它现已降为**机制 ablation**；B1 主实验使用 `--steering_scope both`，即 rationale pass 与 action pass **各自**在自己的最后一个 prefill token 注入一次（decode 均不注入）。两种 scope 在 α=0 时都不注册 hook，故 Track A 的 α=0 cell 对两者通用。详见 `AdaDopamine.md` §3.2.5。
- action 候选必须经过 tokenizer audit：优先保证共同前缀后的候选 suffix 均为单 token；若无法保证，则对每个完整候选字符串计算 sequence log-probability，并记录 tokenization，不得只比较首 token。
- 最终 choice 在结构上必为合法臂，`invalid_rate=0`；不再经过自由文本 parser，也不使用 random fallback 改写 trajectory。
- 每轮保存全部 candidate log-scores、top-1/top-2 margin、选中臂、rationale 与两阶段 prompt attestation，便于区分接近决策边界与稳定 policy。
- constrained decoding 只解决输出合法性，不视为 exploration 能力的来源；相关实现先例来自 [Monea et al.（2024）](https://arxiv.org/abs/2410.05362) 的 contextual-bandit classification，不能直接当作当前 MAB exploration 的正面证据。

**Steering 语义必须冻结：**rationale pass 不注入 α；α 只在第二阶段 action prompt 的最后一个 prefill token 上施加一次，随后进行 candidate-only scoring。这样 α 直接作用于实际选臂而不是只通过 rationale 间接作用，同时仍保持每轮一次 prefill steering。两阶段都使用 temperature=0；不得让 E-direct、E-CoT 与 constrained 版本共享结果目录。

### 3.4 Temperature 与 chat-format 决策

- **Primary capability 与 α experiment：temperature=0。** 这隔离模型的 deliberate exploration，避免把 sampling-induced switching 误写为主动探索。
- **Secondary robustness：temperature=1。** 只在 primary 完成后运行，用于判断外部采样随机性是解除 lock-in，还是制造 uniform flailing；不能与 temperature=0 合并统计。
- 不采用“capability 用 T=0、α 只用 T=1”的唯一设计，因为那会让 baseline capability 与 intervention 落在不同 policy regime。

文献 reference interface 使用 system/user chat，而 NMD mask 来自 bare-string activation distribution。因此不根据开发结果二选一，而是固定两者的不同职责：

1. **reference-chat，α=0**：复现文献支持的最强 native interface，作为 capability comparator；
2. **reference-bare，α=0**：与 NMD mask activation distribution 对齐，作为 RSN capability baseline。

B1 α 主实验只能使用通过 competence gate 的 `reference-bare` K=4/K=5 条件。若只有 chat 通过，结论是“模型在文献接口下存在 capability，但当前 RSN-aligned bare interface 没有 competence anchor”；此时 chat α 最多作为 cross-format secondary analysis，chat 下 α null 不能直接解释为 RSN 方向无效。

接口集合在实验前冻结为上述 `reference-chat` 与 wording 等价的 `reference-bare`；只允许修复格式、tokenization 或实现错误，不依据 α=0/α≠0 的行为效果继续改写 prompt，也不把两者之间表现较好的一项事后重新定义为唯一主接口。

### 3.5 Baselines 与主要指标

所有 reference environments 使用相同 reward structure 与 seeds 跑：

- Random；
- Greedy（见下方冻结定义）；
- UCB1；
- Thompson Sampling；
- Oracle（见下方冻结定义；只作结果上限，不作可比策略）。

**Greedy 冻结定义。** Greedy 是 competence gate 规则 1 的比较基准，而 Reference 环境的次优臂概率完全相同（Easy `0.25×3`、Hard `0.40×4`），初始化后每臂 `n=1` 时并列极其频繁，因此 tie-break 规则会实质改变 `SuffFailFreq_Greedy(T/2)`，必须在实现前写死：

- 前 `K` 轮按 display order 每臂各拉一次作为初始化；**这 K 轮计入 `T`，并计入全部指标**（不作为 warm-start 从分母中排除）；
- 其后选择 empirical mean 最高的臂；并列时在**并列臂集合内** uniform random，不使用 first-index；
- tie-break 使用独立的 `tie_rng = Random(seed + TIE_RNG_OFFSET)`，**不得消费 per-arm reward tape**，也不与 fallback / 生成种子共用流；
- 结果中保存 tie-break policy 名称与 RNG version，使该 baseline 可被精确重放。

不使用 first-index tie-break 的原因：它会让 Greedy 的选择与 display position 耦合，正好污染 §3.5 精心平衡的 best-arm position counterbalance。

**Oracle 冻结定义。** 从第一轮开始始终选择真实最优臂，使用与其他 policy 相同的 per-arm reward tape；仅作为 reward / regret 的上限，不作为可比策略（它不面对任何探索问题）。

#### Counterbalancing 与 paired reward tape

- 每个环境在运行前冻结 N=20 seed bank，不在看到模型结果后换 seed。
- K=4 中最优臂 display position 各出现 5 次；K=5 中各出现 4 次。
- 最优臂 identity 同样精确平衡，并检查 identity × position 不出现明显集中；完整 seed、arm mapping 与 display order 写入结果。
- 每个 seed 为每个臂预生成独立 Bernoulli uniform tape；该臂第 `n` 次被选择时使用 tape 的第 `n` 个 draw。
- 所有 α、reference-chat/reference-bare、Random、Greedy、UCB1、TS 与 Oracle 共用同一 seed 对应的 tapes，使不同 policy 对同一臂的第 `n` 次 pull 面对相同潜在 reward。

主要指标按以下顺序解释：

#### A. Persistent failure

- `SuffFailFreq(T/2)`：在 rounds `[T/2, T]` 中一次也未选择真实最优臂的 run 比例；主读数为 `SuffFailFreq(50)`。
- `SuffixFail` time curve：不能只报一个终点，需确认 failure 是否持续。

#### B. Uniform-like failure

- 对每个 run 定义 `MinFrac(T) = min_a n_a(T)/T`；报告跨 runs 的 `K × MinFrac(T)`。**跨 run 聚合固定为 arithmetic mean**（与 Krishnamurthy 一致），competence gate 规则 2 与 §3.7 的 α 判定均使用该 mean；median 与 IQR 只作为分布补充报告，不参与任何 gate。
- 同时报告 `K × MinFrac(t)` time curve。值长期接近 1 表示各臂近似均匀选择，属于 flailing，不是成功探索。

#### C. Greedy / discovery / utilization

- `GreedyFrac`；
- coverage、best-never-tried、first-best index、last novel trial；
- novel vs. non-novel switches；
- empirical-best adherence、late adherence；
- longest same-arm streak、choice entropy。

#### D. Outcome

- OptFrac、early / late OptFrac；
- cumulative regret 与 per-round regret；
- reward trajectory。

成功不能由单一 OptFrac 或 coverage 决定：

| 形态 | SuffFailFreq | K×MinFrac | 解释 |
|---|---:|---:|---|
| Greedy lock / suffix failure | 高 | 低 | 过早锁定，部分 runs 永久放弃最优臂 |
| Uniform flailing | 低 | 高且不随时间下降 | 持续乱试，没有形成 exploitation |
| 有效 explore→exploit | 低，接近 UCB/TS | 随时间下降 | 早期收集信息，后期集中到较优臂 |

### 3.6 Track A — α=0 capability boundary 与 competence gate

固定判读顺序：

1. N=3 smoke：分别确认 reference-chat/reference-bare prompt、两阶段 constrained choice、candidate tokenization、arm counterbalancing、reward tape 与存储 schema；不看效果。
2. Reference-Easy，α=0，T=100，N=20：chat 与 bare 分别报告，建立文献接口 comparator 与 RSN-aligned 能力下界。
3. Reference-Hard，α=0，T=100，N=20：chat 与 bare 分别和 Greedy / UCB / TS 比较，确定模型落在 suffix-failure、uniform-failure 或有效学习区域。

**Task-validity gate** 只要求输出合法、轨迹无 parser / RNG 污染且 longitudinal metrics 可解释；它决定实验是否有效，但不等于模型具备 Bandit competence。

**Native competence gate** 只应用于 `reference-bare` K=4/K=5，并要求某个条件同时满足以下预注册规则：

1. discovery 优于 Greedy lock：`SuffFailFreq_model(T/2) < SuffFailFreq_Greedy(T/2)`，且差异不是由 non-novel churn 造成；
2. 不属于 uniform flailing：`K×MinFrac_model(T) < K×MinFrac_Random(T)`，且 `K×MinFrac_model(T) < K×MinFrac_model(T/2)`；
3. 找到较优臂后能够利用：post-discovery late empirical-best adherence `> 1/K`；
4. 产生最低限度的行为收益：late OptFrac `> 1/K`。

四项使用 point estimate 作机械 gate，同时报告 paired bootstrap interval 表示不确定性；不因单个 p 值或单个 OptFrac 事后改变规则。通过 gate 的最难 `reference-bare` 条件定义为后续的 **competence anchor**。reference-chat 结果独立报告，不进入 RSN anchor 选择。

Track A 的 headline 允许是 capability boundary：如果 Llama3-8B 在文献最强 native interface 下仍接近 Greedy，这本身是结果，不继续无上限调 prompt。

### 3.7 Track B — α 是否移动 capability boundary

#### B1. Competence-anchor α 主实验

若 Track A 建立了 competence anchor，主实验在“通过 gate 的最难 reference-bare 条件”上运行，优先级为 `Reference-Hard > Reference-Easy`：

```text
α ∈ {−4, 0, +4}
temperature = 0
T = 100
N = 20 paired seeds
```

α=0 复用 Track A 的同一 cell，不重新定义 prompt 或环境。B1 回答在已经存在最低 Bandit competence 时，α 改变的是 discovery、exploration stopping、utilization 还是 policy persistence。

#### B2. Reference-Hard boundary stress test

无论 competence anchor 位于 Easy 还是 Hard，reference-bare Reference-Hard 都保留 `−4/0/+4` 三点测试，用于判断 α 能否把 capability boundary 推向更困难环境。若 B1 的 anchor 已是 Reference-Hard，B1 与 B2 是同一组实验，不重复运行。reference-chat α 不进入 B1/B2 主结果；如运行，只能单列为 cross-format secondary analysis。

若 reference-bare K=4/K=5 均未通过 competence gate，Reference-Hard 的三点 α 仍可作为 **failure-mode characterization**，但不得使用 capability-effect / rescue / information-seeking improvement 的措辞。此时只判断 α 是否改变 greedy lock、uniform flailing、non-novel churn 或 persistence，并优先进入 Track C 定位缺失能力。

所有 α=0 均复用 Track A 的同一 cell。先完成三点实验，再决定是否扩大到 `−8…+8`；不得直接恢复旧 Bandit dose curve。

只有 α 使原本失败的目标环境跨过 competence gate，并同时满足以下条件，才能称为 capability rescue；若 anchor 本来已通过 gate，则写成 capability modulation / improvement：

1. `SuffFailFreq(T/2)` 向 UCB/TS 方向下降（Reference 的 T=100 对应 50）；
2. `K×MinFrac` 没有升成 uniform-like failure，并随时间合理下降；
3. novel exploration 增加而不是 non-novel churn 增加；
4. late adherence、OptFrac 或 regret 至少一项同步改善；
5. invalid / fallback 不参与解释。

若 α 降低 suffix failure、却提高 `K×MinFrac`、choice entropy 与 non-novel switching，应写成 **policy destabilization / lock-to-flail tradeoff**，不能写成 exploration improvement。若只改变 empirical-best adherence 而不改变 discovery，则解释为 policy persistence，而不是 information seeking。

temperature=1 的三点 α 仅作为后续 robustness；必须单独报告，不能与 temperature=0 拼成一条 dose curve。

### 3.8 Track C — 机制与支架诊断

Track C 只在 Track A/B 已给出 frozen verdict 后启动，不承担 K=4/K=5 native exploration 主证据。

| 诊断条件 | 改动 | 回答的问题 | 证据边界 |
|---|---|---|---|
| **C1 Native-Floor** | K=2，`.70/.30`，T=50，α=0 | 模型是否存在最小 stochastic adaptation | diagnostic floor；不是 competence anchor，不跑 B1 α |
| **C2 Uncertainty scaffold** | 提供 Beta-smoothed mean + credible interval / uncertainty | failure 是否来自 `1/1=1.0` 等小样本过度确信 | calculator-assisted，不是纯 native |
| **C3 Warm-start** | 每臂先提供 2–5 个平衡 observations | 给定信息后能否排序并维持较优选择 | utilization-only |
| **C4 Policy checklist** | 显式先判 `EXPLORE / EXPLOIT` 再选臂 | 模型是否知道策略却执行失败 | prompt-scaffolded policy |
| **C5 UCB/TS guidance** | 提供 bonus、posterior sample 或推荐臂 | state estimation 与 action policy 哪一层失败 | algorithm-guided upper bound |
| **C6 Distillation / RLFT** | 用 counterbalanced UCB/TS trajectories 训练 | 专门训练后能否获得稳定 Bandit policy | trained capability |

Uncertainty scaffold 可使用：

\[
\hat p_i = \frac{s_i+1}{n_i+2}
\]

但必须明确标记为外部 Bayesian state representation。Few-shot / distillation demonstrations 必须来自独立 runs，充分 counterbalance 最优臂名称与位置，并与测试时“run 间最优臂可变、run 内固定”的生成结构一致，不能重现 TextBandit 的示例—测试矛盾。

### 3.9 实现范围与文件组织

不复制整份 `get_answer_bandit.py` 建立平行实现。新协议沿用现有 steering、layer indexing、paired RNG 与存储 schema，在原入口中增加：

- `--reference_environment {easy,hard,native_floor}`。**`graded` 不作为新环境选项实现**：旧的 `.7/.5/.4/.3/.1` 保留在 legacy path 上（不传该 flag、不使用 `F-reference` 时行为完全不变），仅供旧协议复现，不得进入新的 competence gate，避免出现第三种未经 gate 的环境；
- `--temperature`；
- 新的 `F-reference / pv6` prompt variant；
- two-stage rationale + candidate-only action scoring mode；
- frozen counterbalanced seed banks 与可复用的 per-arm reward tapes；
- `SuffFailFreq`、`K×MinFrac` 与 `GreedyFrac` 所需的逐轮字段。

新增独立 launcher，例如 `run_bandit_reference.sh`；输出目录、protocol version 与 resume key 全部与 E/C/D 系列隔离。结果额外保存 candidate log-scores/margin、rationale、tokenization audit、reward-tape id 与 prompt attestation。`run_bandit_algorithmic_baseline.py` 增加 Greedy，并支持 reference reward vectors 与同一 reward tapes。实现前先冻结 prompt、temperature、environment、choice mode、seed banks 与 metrics；之后每次只改变一个实验维度。

**共享模块 `bandit_reference.py`。** 拆分的理由不是 `get_answer_bandit.py` 的长度（当前 1883 行），而是以下组件会被 **两个入口共同使用**——若各自实现，`run_bandit_algorithmic_baseline.py` 会再次实现出一套与 LLM 入口不完全一致的环境，正是 paired reward tape 设计要防止的事：

- reference environment specifications（Easy / Hard / Native-Floor 的 K、概率向量、T）；
- counterbalanced seed banks（含 smoke bank）；
- per-arm reward tapes；
- `F-reference` prompt construction；
- rationale sanitization；
- candidate / tokenization utilities；
- reference metrics（`SuffFailFreq`、`K×MinFrac`、`GreedyFrac`）。

职责边界：`get_answer_bandit.py` 继续负责模型加载、RSN hook、episode orchestration 与结果写入；`run_bandit_algorithmic_baseline.py` 复用同一 environment / tape 实现。该拆分不复制 steering，也不触碰 pv1–pv5 的任何分支。

### 3.10 最终执行顺序

1. 实现并验证 reference environment、two-stage candidate scoring、counterbalanced seed banks、per-arm reward tapes 和新 schema。
2. 用 N=3 smoke 分别验证 reference-chat/reference-bare 的 prompt attestation、candidate tokenization、合法 choice 与 α 的注入位置（`action` 或 `both`，由 `--steering_scope` 决定，以 `steering_fires` site counter 实测核对）；不依据效果选择接口。
3. 跑 α=0 Reference-Easy 与 Reference-Hard：chat/bare 分别报告，competence gate 只判 reference-bare。
4. 在通过 gate 的最难 reference-bare condition 上跑 B1 `−4/0/+4`；reference-bare Reference-Hard 作为 B2 boundary stress test，若与 B1 重合则不重复。
5. 用预注册 competence gate、`SuffFailFreq × K×MinFrac` 与 discovery / utilization / stability 指标判断 α 是 improvement、rescue、无效还是 lock-to-flail tradeoff。
6. 若没有 reference-bare competence anchor，α 结果只作为 failure-mode characterization；随后先跑 K=2 Native-Floor，再依次使用 uncertainty scaffold、warm-start 与 algorithm-guided controls 定位失败层级。
7. 需要时才跑 temperature=1 robustness；需要时才单列 chat α cross-format analysis。
8. 只有在三点 α 给出稳定、可解释方向后，才考虑更宽 dose sweep 或跨模型复现。

## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
