# Bandit 实验：相关文献、模型能力边界与改进方案

> 更新：2026-08-09
> 目的：结合近期 Bandit–LLM 文献与本项目 Llama3-8B 的 pv6–pv8 结果，分解 utilization、native exploration 与 RSN modulation，并规划下一轮实验。

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
2. **保留并重新定位 short-CoT 接口对照**：pv7/pv8 的 Stage 1 已经要求模型先写 `Evidence`，再写 `Policy: EXPLORE/EXPLOIT Button X`，因此当前问题不是“没有思考空间”，而是模型虽然能描述 uncertainty，却很少把它转化为 information-seeking action。下一轮应比较 direct choice、native short-CoT 与显式 uncertainty-to-action policy scaffold；后者属于外部策略帮助，不能作为自主 exploration 证据。文献中的 prompt CoT、SFT 的 UCB-CoT target 与 RL 训练效果必须分开表述。
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

### 3.1 已完成的迭代路径

- **pv6** 首次建立了可运行的 reference Bandit 与 competence gate，但也暴露了 rationale 截断、选项显示漂移、Stage 2 指令冲突与 label prior 等接口问题。
- **pv7** 用结构化 `Evidence → Policy → constrained choice` 修复了两阶段接口。模型能读取样本数与 empirical rate，也能稳定执行自己的 Policy；但仍出现 one-shot-zero lock-in，严格 competence gate 未通过。
- **pv7 frozen-state diagnostics** 表明：history 改善了文本格式，calculator 改善了 uncertainty 的表述，α 改变了 rationale 与决策锐度；但它们都没有稳定促成对 `1 trial / 0 reward` 臂的定向重访。
- **pv8** 把 choice history 放回完整 100-round online episode。结果复现 pv7：α 双向调节 policy commitment / decision sharpness，但未改变 targeted information seeking、SuffFail 或 outcome。

因此，pv6–pv8 作为已完成的接口与机制迭代保留，不再在其上无上限追加 prompt patch。新一轮实验从 capability decomposition 重新规划。

### 3.2 新一轮的三个问题

1. **Utilization**：给定外部平衡、足量的证据时，Llama3-8B 能否识别并稳定利用较优臂？
2. **Native exploration**：不提供 uncertainty bonus、UCB/TS 推荐或 oracle demonstration 时，模型能否随着反馈自主形成 explore→exploit？该能力在 K=2/3/5 的边界在哪里？
3. **RSN modulation**：在一个行为可解释的 online condition 中，α 改变的是 targeted discovery、utilization、exploration stopping 还是 policy commitment？

Capability validation 与 α 主实验必须分开。如果 α=0 条件未建立 native competence anchor，α 仍可用于 failure-mode characterization，但不能称为 capability improvement 或 rescue。

### 3.3 实验路径

#### Phase A — 先分解 capability

1. **Utilization probe**：由程序提供每臂平衡的初始 observations，单独测试模型是否会读取样本数、均值并选择证据支持的较优臂。这是 utilization control，不计作自主 exploration。
2. **Native online ladder**：在同一 summary、decoding、seed/tape 与指标口径下，依次跑 K=2 → K=3 → K=5。K=2 是最小 stochastic-adaptation floor，K=3 是主要 competence-anchor 候选，K=5 是 boundary stress test。不用 K=5 失败直接推论模型完全没有 Bandit 能力。
3. **α=0 接口对照**：在最低成本的环境上比较：
   - direct constrained choice；
   - native short-CoT：先评估 evidence / uncertainty / remaining horizon，再自行决定 `EXPLORE` 或 `EXPLOIT`；
   - policy-scaffolded short-CoT：显式提醒小样本证据弱，并要求考虑一次新观测对后续决策的信息价值。

当前 pv7/pv8 Stage 1 已经属于 short-CoT。新对照不应再要求模型“更彻底地 exploitation”：这会强化已观察到的 one-shot-zero lock-in。native short-CoT 不提供 UCB/TS 计算或推荐臂；policy-scaffolded 条件必须单列，不能冒充 native capability。

#### Phase B — 再跑 α

1. 先冻结 Phase A 选出的 environment、prompt、temperature、seed bank、reward tape 与 analysis parser。
2. 优先在“最简单但已呈现可解释 Bandit-like adaptation”的 native online condition 上跑 `α ∈ {−4, 0, +4}`；先不扩大 dose range。
3. Stage 1 rationale/policy 是当前的主要 steering 候选位置；Stage 2 保持为可审核的 constrained executor。若日后改变作用点，必须作为独立 mechanism ablation，不与主协议混合。
4. α 的结论按 `validity → discovery → utilization → stability → outcome` 解释；只有 novel/targeted exploration 与后期利用同时改善，才能写 information-seeking improvement。只改变 margin/entropy、switching 或 non-novel churn，仍只是 commitment/persistence modulation。

#### Phase C — 支架与训练对照

- summary/counts 属于状态表示；
- balanced warm-start 属于 utilization control；
- Beta/credible interval 属于 calculator-assisted representation；
- UCB/TS score、recommendation 或 demonstration 属于 algorithm-guided policy；
- UCB-CoT imitation 是 SFT target，RL-OG / RL-STG / RL-ALG 则分别是不同 reward signal 的训练条件。

这些条件用于定位“表示、决策还是训练”哪一层缺失，不与 native α 主实验合并。

### 3.4 固定的分析顺序

1. **Validity**：prompt/tokenizer/steering attestation、legal action、paired seeds/tapes。
2. **Discovery**：arms tried、novel pulls、one-shot-zero revisit、SuffixFail。
3. **Utilization**：empirical-best adherence、GreedyFreq、warm-start exploitation accuracy。
4. **Stability**：exploration stopping、non-novel churn、longest streak、matched-history divergence。
5. **Outcome**：late OptFrac、reward 与 regret，最后报告。

`K×MinFrac(t)` 只作为与 SuffixFail 配对的 anti-flailing 诊断：它的后期下降能排除持续均匀乱试，但“整体很低”不能单独证明有效 exploration，也可能是过早 lock-in。

### 3.5 下一步

暂不直接写新 episode code。先冻结以下规格：

1. K=2/3/5 的 reward probabilities、horizon 与进阶/停止条件；
2. utilization probe 的平衡初始证据；
3. direct / native short-CoT / policy-scaffolded 三个 prompt 的精确文本；
4. competence gate 是否沿用，以及 K=2 与 K=3 各自允许的结论强度；
5. 新 protocol version、counterbalanced seed bank 与预注册 analysis parser。

上述五项确定后，再开始新一轮实现与实验。

## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
