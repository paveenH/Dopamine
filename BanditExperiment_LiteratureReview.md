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

## 3. 当前 C / D / D2 到底说明了什么

### 3.1 C：untried 被错误编码为 0.00

C 的 summary 将未试臂显示为 `0 times, average reward 0.00`。这会把 unknown 写成 tied-worst：

- 20 seeds 中有 6 个从未碰到最优臂；
- 碰到最优臂的 14 个 run 平均 OptFrac 为 0.434；
- 未碰到的 6 个全部是 0。

因此 C 可以支持 utilization 观察，但 discovery 结论被 prompt 编码污染。

### 3.2 D：修复 discovery 后，α=+4 主要造成 churn

D 修复 `UNTRIED ≠ 0`，并说明 reward probability 固定但未知。20-seed 配对结果：

| 指标 | α=0 | α=+4 | 解释 |
|---|---:|---:|---|
| coverage | 3.75 | 4.60 | 接触的臂更多 |
| novel switches | 2.70 | 3.45 | 新臂探索仅小幅增加 |
| non-novel switches | 6.85 | 13.25 | 已试臂之间 churn 大幅增加 |
| adherence | 0.884 | 0.680 | 更少遵循当前证据 |
| late adherence | 0.807 | 0.532 | 后期仍未稳定 exploit |
| OptFrac | 0.533 | 0.456 | 额外切换未转化为收益 |

所以 D 当前最可靠的读法是：

> α=+4 降低 policy persistence、扩大 action sampling，但没有证据显示新增 exploration 是目标性或 reward-efficient 的。

α=−4 的 invalid rate 为 9.9%，且主要是正确臂名前多出轮次编号；随机 fallback 改写了后续 history，因此该 cell 不能用于剂量方向判断。

### 3.3 D2：格式修好，但任务塌成 pure greedy

D2 增加采样不确定性解释，并把 summary 改成 `successes / trials / rate`：

| 指标（α=0） | D | D2 |
|---|---:|---:|
| invalid | 0.009 | 0.001 |
| coverage | 3.75 | 2.35 |
| switch rate | 0.197 | 0.032 |
| entropy | 0.336 | 0.090 |
| adherence | 0.884 | 1.000 |
| best never tried | 1/20 | 7/20 |

D2 成功消除了编号格式错误，却让三个 α 基本都退化为纯 greedy；α=−4 的 18/20 runs 与 α=0 选择序列相同。

这不能归因于某一句话，因为 D→D2 同时改变了：

- 采样不确定性说明；
- summary 顺序；
- round / remaining 表示；
- exploration–exploitation 句子；
- 输出结尾。

因此 D2 应保留为一个**负面 prompt ablation**，不再作为 α 主协议，也不应继续在其上追加更多解释。

## 4. 推荐的新实验结构

### Calculator-assisted Deliberate Bandit

先建立一个更简单的新协议：

- K=2，概率为 `0.70 / 0.30`
- T=30 或 40
- 使用短、无语义偏好的臂名，例如 `Option A / Option B`
- 不同 run 随机化最优臂，run 内始终固定
- Python 计算统计量，模型负责最终 explore/exploit 决策
- 先允许短思考，再通过 constrained choice 输出合法臂名

每轮提供：

```text
Round 9 of 30; 21 rounds remain.

Option A
- rewards: 1 / 1
- estimated probability: 0.67
- uncertainty: HIGH

Option B
- rewards: 5 / 8
- estimated probability: 0.60
- uncertainty: MEDIUM

An option with fewer trials has a less reliable estimate.

Briefly decide:
1. Is more information still valuable?
2. Should you explore an uncertain option or exploit the best-supported option?

Use at most two short sentences.
Choice: <exact option name>
```

这里的 `0.67` 最好不是原始 `1/1=1.00`，而是由程序计算的 Beta-smoothed posterior mean，例如：

\[
\hat p_i = \frac{s_i+1}{n_i+2}
\]

同时由程序提供 posterior uncertainty 或 credible interval。这样能直接削弱当前“第一次成功 → 经验率 1.0 → 永久锁定”的 confirmation loop。

重要边界：只提供历史数据计算出的 estimate 和 uncertainty，**不能展示真实的 0.7/0.3**。

### 为什么比普通 CoT 更合适

文献支持 CoT 会提高 coverage，但当前 E-CoT 的问题是：模型有时推理正确，却没有严格输出最终 `Choice:`，随后 random fallback 改写整条 trajectory。

所以不应只是增加：

```text
Think step by step.
```

更适合采用“两阶段输出”：

1. 模型自由生成至多两句 rationale。
2. 第二阶段只允许从合法臂名中选择，不能生成解释。

可以通过 constrained decoding，或者在 rationale 后重新追加 `Choice:`，只对候选臂计算/采样概率。这样能保留思考，同时消除格式失败和 random fallback。文献综述中的 short-CoT 方向见 [`BanditExperiment_LiteratureReview.md:270`](/Users/paveenhuang/Downloads/Dopamine/BanditExperiment_LiteratureReview.md:270)。

### 难度阶梯

建议按以下顺序增加难度：

| 阶段 | 设置 | 主要目的 |
|---|---|---|
| F1 | K=2，`.70/.30`，structured summary + constrained CoT | 最低 Bandit 能力 |
| F2 | K=3，`.70/.45/.20` | 测 discovery 与排序 |
| F3 | K=5，`.70/.50/.40/.30/.10` | 接近当前主任务 |
| F4 | K=5，小 gap，例如 `.60/.50/...` | 真正困难的 exploration |

只有前一级表现出“早期探索、后期收敛”，才进入下一级。这个阶梯也符合综述的建议，[`BanditExperiment_LiteratureReview.md:477`](/Users/paveenhuang/Downloads/Dopamine/BanditExperiment_LiteratureReview.md:477)。

### 可以进一步加的支架

按对模型自主性的影响由小到大排列：

1. **提供统计计算**

   提供 rewards/trials、smoothed estimate、uncertainty，但不告诉模型选哪个。这仍可称为 calculator-assisted autonomous choice。

2. **Policy checklist**

   要求模型先判断：

   ```text
   EXPLORE: evidence is insufficient and rounds remain
   EXPLOIT: evidence is sufficient and one option is best-supported
   ```

   然后选择臂。它让策略显式化，但会轻度诱导行为。

3. **Try-all / warm-start**

   每个臂先由程序采样 2–5 次，再让模型决策。这很可能让 Llama3 完成任务，因为当前结果已表明它拿到平衡信息后能够 exploit。

   但这只能证明 conditional utilization，不能证明自主 exploration。

4. **Few-shot policy demonstrations**

   给出少量“早期探索、后期利用”的完整示例。示例必须：

   - 来自独立 run；
   - 最优臂的名称和位置充分 counterbalance；
   - 与测试时“每个 run 最优臂可变、run 内固定”的结构完全一致；
   - 不能重现 TextBandit 那种 few-shot 与测试结构矛盾。

   这应命名为 few-shot scaffold，而不是原生能力。

5. **UCB/Thompson guidance**

   Python 直接提供 posterior sample、UCB exploration bonus，甚至推荐臂。这最容易成功，但 exploration policy 已由算法提供，只能作为 algorithm-guided condition。

6. **UCB trajectory distillation / RLFT**

   用大量 counterbalanced UCB/TS trajectory 对 Llama3 做 LoRA/SFT。文献显示这是让小模型稳定学习 Bandit policy 最有力的方法之一，但结论会变成“经过 Bandit policy training 后能完成”，而非 off-the-shelf 能力。

### 我的具体建议

优先实现一个新的 **F1 calculator-assisted、constrained-deliberation K=2 protocol**：

- Python 提供 smoothed probability 和 uncertainty；
- 模型在选择前进行最多两句推理；
- 最终臂名采用 constrained decoding，不允许 random fallback；
- run 间随机化、run 内固定；
- 先只跑 α=0、20 个配对 seeds；
- 通过标准不是仅看 OptFrac，而是：
  - 两个臂都能被发现；
  - late OptFrac > early OptFrac；
  - late adherence 高；
  - SuffixFail 低；
  - 不是第一次 reward=1 后永久锁定。

如果它通过，再升到 K=3；如果 K=2 仍失败，再使用 try-all 或 UCB-guided。这样可以清楚判断：究竟是**概率计算负担、思考接口、不确定性表示，还是自主探索策略本身**导致失败。

## References

1. Nie et al. (2025). [EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration](https://proceedings.mlr.press/v267/nie25b.html). ICML 2025.
2. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
3. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
4. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
5. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
6. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
7. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
8. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
