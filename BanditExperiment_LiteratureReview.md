# Bandit 实验：相关文献、模型能力边界与改进方案

> 更新：2026-07-31
> 目的：结合近期 Bandit–LLM 文献与本项目 Llama3-8B 的 C / D / D2 结果，判断小模型能否完成 Bandit，并确定下一版实验应该修什么、保留什么。

## 1. 结论先行

### 1.1 小模型能不能做 Bandit？

**可以，但不是无条件地可以。**

现有证据支持三个层级：

1. **利用已获得的信息（utilization / exploitation）**：7–8B 模型通常具备一定能力。本项目 D 条件下，Llama3-8B 的 empirical-best adherence 为 0.884，说明给定已经观察到的均值，它大部分时候会利用当前证据。
2. **自主发现未知选项（discovery / exploration）**：小模型容易过早锁定少数选项。Gemma2 2B/9B/27B、Qwen2.5 3B/7B 与本项目 Llama3-8B 都出现过 action coverage 停滞、greedy lock、frequency bias 或 suffix failure。
3. **稳定完成“探索后收敛”的完整策略**：未经专门训练的小模型并不可靠。提示、短 CoT、历史摘要、任务规模和训练方式都会显著改变结果；平均 regret 或 OptFrac 好看，也可能只是早期碰巧找到最优臂后一直锁定。

因此，当前 Bandit 不应被简单判断为“模型会”或“模型不会”。更准确的研究问题是：

> RSN α 改变的是未知选项的信息寻求、已知证据的利用，还是一般性的策略持续性？

### 1.2 对当前实验最重要的改进

优先级从高到低：

1. **把 capability validation 与 α 主实验分开**：用外部提供的平衡初始证据验证 utilization；主实验仍保留自主 exploration。
2. **增加 short-CoT 接口对照**：当前 `Choice: ` anchor 要求立即输出臂名，几乎不给模型显式整合样本量、均值和剩余轮次的空间；文献显示 CoT 对小模型的 Bandit coverage 和训练效果是 load-bearing。
3. **使用难度阶梯**：先 K=2，再 K=3，最后才是当前 K=5；不能用 K=5 的失败直接推断模型完全没有 Bandit 能力。
4. **保留简洁 summary，不再增加解释段落**：将 tried / untried 分开，并显式给出 `successes / trials / rate`；不要继续在 prompt 中堆采样理论。
5. **指标按 discovery / utilization / stability / outcome 分层**：coverage 和 OptFrac 都不能单独作为主结论；加入 GreedyFreq、SuffixFail 和 matched-history divergence。

## 2. 八篇论文分别说明什么

| 论文 | Bandit 在论文中的角色 | 是否证明 LLM 自己会做 Bandit | 对本项目的直接价值 |
|---|---|---:|---|
| [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html) | 直接评估 LLM 在 BanditBench 中的 in-context exploration，并比较 summary、UCB guidance、few-shot 与 fine-tuning | 基线部分是；algorithm-guided / distillation 部分不是纯自主能力 | 最直接支持 structured summary、难度阶梯与 exploration-optimality 分析，同时要求把“模型自主探索”与“外部算法供给探索”分开 |
| [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://arxiv.org/abs/2503.01163) | Thompson sampling 在外部选择 prompt-design strategy | 否 | 说明不同 prompt 策略不应一次全部叠加；但不能用 prompt optimizer 自动优化本实验，否则会选择性改变待测行为 |
| [LLMs are Greedy Agents](https://arxiv.org/abs/2504.16078) | Gemma2 2B/9B/27B 直接进行 MAB/CB 决策 | 是，且系统分析失败模式 | 最直接：greediness、frequency bias、knowing–doing gap；CoT、try-all、summary 和 RLFT 的作用 |
| [When Greedy Wins](https://arxiv.org/abs/2509.24923) | Qwen2.5 3B/7B 学习 meta-bandit policy | 是，但主要研究训练后策略 | 平均 reward 提升不等于探索改善；必须报告 GreedyFreq、SuffixFail、双峰和早停探索 |
| [BanditSpec](https://arxiv.org/abs/2505.15141) | UCB/EXP3 在外部调度 speculative decoding 配置 | 否 | 只支持“经典算法可以稳定做在线调度”；不支持 Llama/Qwen 自主 exploration 能力 |
| [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118) | 经典 TS/回归 Bandit 负责探索，LLM 只预测 reward | 否，反而指出直接选臂常次优 | 支持把 reward understanding 与 action selection 拆开；经典算法控制应作为上界/诊断，不是 α 主实验 |
| [TextBandit](https://arxiv.org/abs/2510.13878) | 开源 LLM 直接根据语言化 0/1 feedback 选臂 | 部分支持 | Qwen3-4B 在简单设置中可表现很好，但 Llama3.1-8B 较差；结果强烈依赖模型、K、few-shot 和标签格式 |
| [Should You Use Your LLM to Explore or Exploit?](https://arxiv.org/abs/2502.00225) | 将 exploration oracle 与 exploitation oracle 分开测试 | 是能力分解，不是完整 policy | 为本项目提供最干净的诊断框架：不要用一个端到端分数同时测 discovery 与 utilization |

### 2.1 `EVOLvE`：当前实验设计的直接来源与边界

EVOLvE 在 BanditBench 中系统评估 context-free MAB 与 contextual bandit。MAB 同时改变
reward distribution、gap、arm 数量和名称表示：包括 Bernoulli / Gaussian、K=5 / K=20，
以及无语义的 Video 标签与语义丰富的 Clothes 名称。论文测试 Gemma-2B、Gemma-9B、
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

该研究在 BanditBench 的 Gaussian/Bernoulli MAB 上测试 Gemma2 2B、9B、27B，horizon 同样是 50。重要发现包括：

- 不同规模模型都会过早采用 greedy strategy，action coverage 很快停滞；扩大模型只能减轻，不能消除。
- 2B 模型还会受 action 在历史中出现频率影响，即使该 action reward 较差。
- 27B 可以正确计算 UCB，但即使 rationale 正确，仍常执行 greedy action，形成 knowing–doing gap。
- CoT 明显提高 coverage；没有 CoT 时，各规模模型探索都更少。
- 在多种 in-context 措施中，**初始 try-all 的改善最大**；这说明模型拿到足够信息后更擅长利用，而自主获取信息是主要短板。
- RL fine-tuning 能改善 2B/9B 的 regret，并使 2B coverage 增加约 12%，但仍未完全达到理想探索。

对本项目的含义：

1. 当前 Llama3-8B 的 greedy lock 不是异常，也不必优先归因于 code bug。
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

### 2.4 `TextBandit`：小模型正结果存在，但不可直接外推

TextBandit 使用语言化反馈（“earned a token” / “did not earn a token”），每轮重建包含完整历史的 prompt，不使用 CoT，测试 Qwen3-4B、Qwen3-8B、Llama3.1-8B 与 Phi-2。论文报告：

- Qwen3-4B 的 best-arm selection rate 达 89.2%；
- Llama3.1-8B 为 31.6%，其他多数模型也低于经典算法；
- 模型规模与结果并非单调关系；
- 任务从 2 到 5 arms，简单 2-arm 条件更容易出现正结果。

这个结果只能说明：

> 小模型在某些非常简单、特定 prompt / few-shot / arm-label 条件下能够形成 Bandit-like adaptation。

它不能说明任意 4B/8B 模型都能在当前 K=5 环境中自主探索。尤其本项目已经记录 TextBandit 实现中固定数字标签、few-shot 与最优臂结构等设计问题，因此它适合作为“能力存在的边界案例”，不应作为当前协议的金标准。

### 2.5 `Should You Use Your LLM to Explore or Exploit?`：能力应拆开测

该研究不要求 LLM 一次完成完整 Bandit policy，而是分别测试：

- exploitation oracle：给定历史，选择当前最优 action；
- exploration oracle：在大而有语义的 action space 中提出值得尝试的候选。

核心结论是：

- LLM 在小型、数值化 exploitation task 上可以有一定表现；
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

新实验不再预设“找到一个 prompt 后 Llama3-8B 就能完成 Bandit”。主问题改为：

> 在文献支持的最强原生接口下，Llama3-8B 位于怎样的 Bandit capability boundary；RSN α 改变的是目标性 information seeking、exploration stopping、policy persistence，还是只让行为在 greedy lock 与 uniform flailing 之间移动？

后续结论必须区分三种证据：

1. **Native capability**：模型只看到环境说明与 externally summarized sufficient statistics，自行决定 explore / exploit。
2. **Calculator / uncertainty scaffold**：Python 进一步提供平滑概率、uncertainty 或 credible interval；模型仍做选择，但状态表示已受到外部算法帮助。
3. **Algorithm-guided policy**：warm-start、UCB/TS guidance 或 oracle demonstrations 已提供探索信息或策略，只能作为 utilization control、upper bound 或训练条件。

旧 C / D / D2 与 2026-07-28 前 Bandit 数据只保留为失败分析，不进入新主结果。新实验使用独立 protocol version、输出目录和 resume key。

### 3.2 文献对齐的 reference environment

第一阶段采用 [Krishnamurthy et al.（NeurIPS 2024）](https://arxiv.org/abs/2403.15371) 的两组 Bernoulli MAB，建立可与既有研究对照的能力坐标：

| 条件 | K | Reward probabilities | T | 用途 |
|---|---:|---|---:|---|
| **Reference-Easy** | 4 | `0.75 / 0.25 / 0.25 / 0.25` | 100 | 能力下界；确认模型在大 gap 下是否至少能脱离 greedy failure |
| **Reference-Hard** | 5 | `0.60 / 0.40 / 0.40 / 0.40 / 0.40` | 100 | 主能力与 α 实验；直接测试小 gap 下的自主探索与收敛 |

两组环境均遵守：

- neutral arm labels，不利用服饰语义；
- 每个 run 独立随机化 name→probability mapping 与 display order；
- 同一个 run 的全部 100 rounds 中，臂名称、显示位置与真实概率保持固定；
- 不同 α 使用相同 seeds、arm mapping、reward draws 和生成种子；
- N=20 runs；开发阶段可以先做 N=3 smoke test，但 smoke test 不产生研究结论。

Reference 环境的次优臂概率相同，因此不再使用 `WorstFrac` 作为主指标；它只适用于原来的 graded reward vector。

### 3.3 Native reference interface

主接口采用文献中唯一成功配置的核心元素：**suggestive framing + externally summarized history + reinforced CoT**，但加入 constrained final choice 以消除本项目已经观察到的格式污染。

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
and remaining rounds. Then choose exactly one button.
```

接口规则：

- Python 只计算 `successes / trials / empirical rate`；不提供 Beta smoothing、credible interval、UCB bonus 或推荐臂。
- `UNTRIED` 只表示 unknown，不编码为 `0.00`。
- reinforced CoT reminder 同时出现在任务说明与每轮 user query；rationale 限制为最多两句并完整保存。
- 最终 choice 只能从 K 个合法臂名中产生，`invalid_rate` 在结构上应为 0；不再使用 random fallback 改写 trajectory。
- constrained decoding 只解决输出合法性，不视为 exploration 能力的来源；相关实现先例来自 [Monea et al.（2024）](https://arxiv.org/abs/2410.05362) 的 contextual-bandit classification，不能直接当作当前 MAB exploration 的正面证据。

**Steering 语义必须冻结：**首选单次 decision generation，在同一次生成中先产生 rationale、再约束最终臂名，使每轮只发生一次 prefill steering。若实现必须分成两次 forward，则 α 只施加在第一次 decision prompt；第二次只做 deterministic legal-choice readout，不得再次注入 α。不得让 E-direct、E-CoT 与 constrained 版本共享结果目录。

### 3.4 Temperature 与 chat-format 决策

- **Primary capability 与 α experiment：temperature=0。** 这隔离模型的 deliberate exploration，避免把 sampling-induced switching 误写为主动探索。
- **Secondary robustness：temperature=1。** 只在 primary 完成后运行，用于判断外部采样随机性是解除 lock-in，还是制造 uniform flailing；不能与 temperature=0 合并统计。
- 不采用“capability 用 T=0、α 只用 T=1”的唯一设计，因为那会让 baseline capability 与 intervention 落在不同 policy regime。

文献 reference interface 使用 system/user chat，而 NMD mask 来自 bare-string activation distribution。因此先在 α=0 下对照：

1. reference-chat；
2. wording 等价的 reference-bare。

只有通过 task-validity 的接口进入 α 主实验。若只有 chat 通过，可以在 chat 下运行 α，但必须把结果限定为 cross-format steering；chat 下 α null 不能直接解释为 RSN 方向无效。若两者都通过，优先使用与 NMD mask 对齐的 bare-string 接口，并把 chat 作为 robustness。

### 3.5 Baselines 与主要指标

所有 reference environments 使用相同 reward structure 与 seeds 跑：

- Random；
- Greedy（每臂一次初始化后，始终选择 empirical-best）；
- UCB1；
- Thompson Sampling；
- Oracle（只作结果上限，不作可比策略）。

主要指标按以下顺序解释：

#### A. Persistent failure

- `SuffFailFreq(T/2)`：在 rounds `[T/2, T]` 中一次也未选择真实最优臂的 run 比例；主读数为 `SuffFailFreq(50)`。
- `SuffixFail` time curve：不能只报一个终点，需确认 failure 是否持续。

#### B. Uniform-like failure

- 对每个 run 定义 `MinFrac(T) = min_a n_a(T)/T`；报告跨 runs 的 `K × MinFrac(T)`。
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

### 3.6 Track A — α=0 capability boundary

固定判读顺序：

1. N=3 smoke：确认 prompt、constrained choice、arm counterbalancing、reward RNG 与存储 schema；不看效果。
2. Reference-Easy，α=0，T=100，N=20：建立能力下界。
3. Reference-Hard，α=0，T=100，N=20：与 Greedy / UCB / TS 比较，确定模型落在 suffix-failure、uniform-failure 或有效学习区域。
4. 只有输出合法且 longitudinal metrics 可解释，才进入 α；不以“必须成功”作为 task-validity gate。

Track A 的 headline 允许是 capability boundary：如果 Llama3-8B 在文献最强 native interface 下仍接近 Greedy，这本身是结果，不继续无上限调 prompt。

### 3.7 Track B — α 是否移动 capability boundary

主实验只在 **Reference-Hard** 上运行：

```text
α ∈ {−4, 0, +4}
temperature = 0
T = 100
N = 20 paired seeds
```

α=0 复用 Track A 的同一 cell，不重新定义 prompt 或环境。先完成三点实验，再决定是否扩大到 `−8…+8`；不得直接恢复旧 Bandit dose curve。

只有同时满足以下条件，才能称为 α 改善或 rescue：

1. `SuffFailFreq(50)` 向 UCB/TS 方向下降；
2. `K×MinFrac` 没有升成 uniform-like failure，并随时间合理下降；
3. novel exploration 增加而不是 non-novel churn 增加；
4. late adherence、OptFrac 或 regret 至少一项同步改善；
5. invalid / fallback 不参与解释。

若 α 降低 suffix failure、却提高 `K×MinFrac`、choice entropy 与 non-novel switching，应写成 **policy destabilization / lock-to-flail tradeoff**，不能写成 exploration improvement。若只改变 empirical-best adherence 而不改变 discovery，则解释为 policy persistence，而不是 information seeking。

temperature=1 的三点 α 仅作为后续 robustness；必须单独报告，不能与 temperature=0 拼成一条 dose curve。

### 3.8 Track C — 失败后的机制与支架诊断

Track C 只在 Track A/B 已给出 frozen verdict 后启动，不承担 native exploration 主证据。

| 诊断条件 | 改动 | 回答的问题 | 证据边界 |
|---|---|---|---|
| **C1 Difficulty floor** | K=2，`.70/.30`，T=50 | 小模型是否连最简单 stochastic MAB 也无法适应 | capability floor，不外推到 K=5 |
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

- `--environment {reference_easy,reference_hard,graded}`；
- `--temperature`；
- 新的 `F-reference / pv6` prompt variant；
- constrained legal-choice mode；
- `SuffFailFreq`、`K×MinFrac` 与 `GreedyFrac` 所需的逐轮字段。

新增独立 launcher，例如 `run_bandit_reference.sh`；输出目录、protocol version 与 resume key 全部与 E/C/D 系列隔离。`run_bandit_algorithmic_baseline.py` 增加 Greedy，并支持 reference reward vectors。实现前先冻结 prompt、temperature、environment、choice mode 与 metrics；之后每次只改变一个实验维度。

### 3.10 最终执行顺序

1. 实现并验证 reference environment、constrained choice 和新 schema。
2. 跑 α=0 Reference-Easy 与 Reference-Hard capability boundary。
3. 在 Reference-Hard 跑 temperature=0 的 `−4/0/+4`。
4. 用 `SuffFailFreq × K×MinFrac` 判断 α 是 rescue、无效还是 lock-to-flail tradeoff。
5. 需要时才跑 temperature=1 robustness。
6. 若 native interface 失败，再依次使用 uncertainty scaffold、warm-start 与 algorithm-guided controls 定位失败层级。
7. 只有在三点 α 给出稳定、可解释方向后，才考虑更宽 dose sweep 或跨模型复现。

## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
