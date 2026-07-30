# Bandit 实验：相关文献、模型能力边界与改进方案

> 更新：2026-07-30  
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

## 2. 七篇论文分别说明什么

| 论文 | Bandit 在论文中的角色 | 是否证明 LLM 自己会做 Bandit | 对本项目的直接价值 |
|---|---|---:|---|
| [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://arxiv.org/abs/2503.01163) | Thompson sampling 在外部选择 prompt-design strategy | 否 | 说明不同 prompt 策略不应一次全部叠加；但不能用 prompt optimizer 自动优化本实验，否则会选择性改变待测行为 |
| [LLMs are Greedy Agents](https://arxiv.org/abs/2504.16078) | Gemma2 2B/9B/27B 直接进行 MAB/CB 决策 | 是，且系统分析失败模式 | 最直接：greediness、frequency bias、knowing–doing gap；CoT、try-all、summary 和 RLFT 的作用 |
| [When Greedy Wins](https://arxiv.org/abs/2509.24923) | Qwen2.5 3B/7B 学习 meta-bandit policy | 是，但主要研究训练后策略 | 平均 reward 提升不等于探索改善；必须报告 GreedyFreq、SuffixFail、双峰和早停探索 |
| [BanditSpec](https://arxiv.org/abs/2505.15141) | UCB/EXP3 在外部调度 speculative decoding 配置 | 否 | 只支持“经典算法可以稳定做在线调度”；不支持 Llama/Qwen 自主 exploration 能力 |
| [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118) | 经典 TS/回归 Bandit 负责探索，LLM 只预测 reward | 否，反而指出直接选臂常次优 | 支持把 reward understanding 与 action selection 拆开；经典算法控制应作为上界/诊断，不是 α 主实验 |
| [TextBandit](https://arxiv.org/abs/2510.13878) | 开源 LLM 直接根据语言化 0/1 feedback 选臂 | 部分支持 | Qwen3-4B 在简单设置中可表现很好，但 Llama3.1-8B 较差；结果强烈依赖模型、K、few-shot 和标签格式 |
| [Should You Use Your LLM to Explore or Exploit?](https://arxiv.org/abs/2502.00225) | 将 exploration oracle 与 exploitation oracle 分开测试 | 是能力分解，不是完整 policy | 为本项目提供最干净的诊断框架：不要用一个端到端分数同时测 discovery 与 utilization |

### 2.1 `LLMs are Greedy Agents`：最直接的设计依据

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

### 2.2 `When Greedy Wins`：为什么 OptFrac / regret 不够

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

### 2.3 `TextBandit`：小模型正结果存在，但不可直接外推

TextBandit 使用语言化反馈（“earned a token” / “did not earn a token”），每轮重建包含完整历史的 prompt，不使用 CoT，测试 Qwen3-4B、Qwen3-8B、Llama3.1-8B 与 Phi-2。论文报告：

- Qwen3-4B 的 best-arm selection rate 达 89.2%；
- Llama3.1-8B 为 31.6%，其他多数模型也低于经典算法；
- 模型规模与结果并非单调关系；
- 任务从 2 到 5 arms，简单 2-arm 条件更容易出现正结果。

这个结果只能说明：

> 小模型在某些非常简单、特定 prompt / few-shot / arm-label 条件下能够形成 Bandit-like adaptation。

它不能说明任意 4B/8B 模型都能在当前 K=5 环境中自主探索。尤其本项目已经记录 TextBandit 实现中固定数字标签、few-shot 与最优臂结构等设计问题，因此它适合作为“能力存在的边界案例”，不应作为当前协议的金标准。

### 2.4 `Should You Use Your LLM to Explore or Exploit?`：能力应拆开测

该研究不要求 LLM 一次完成完整 Bandit policy，而是分别测试：

- exploitation oracle：给定历史，选择当前最优 action；
- exploration oracle：在大而有语义的 action space 中提出值得尝试的候选。

核心结论是：

- LLM 在小型、数值化 exploitation task 上可以有一定表现；
- succinct summary 和工具会改善结果，但仍常不如简单 regression；
- LLM 更适合在大而有语义的空间中提出 exploration candidates，而不是替代完整 Bandit 算法。

对当前 K=5 boutique task 而言，臂名只是任意标签，不存在可利用的语义 action space。因此“LLM 擅长语义探索”并不能帮助当前任务；真正可借鉴的是**分离 discovery 与 utilization**。

### 2.5 三篇“Bandit 调度 LLM”的论文：只能作为方法类比

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

### 4.1 第一层：能力阶梯，而不是直接从 K=5 判死刑

使用相同 parser、chat interface、reward RNG 和 summary schema，只改变任务难度：

| 层级 | 设置 | 回答的问题 |
|---|---|---|
| Easy | K=2，明显 gap，例如 0.70 vs 0.30 | 模型能否从最简单的随机反馈中适应 |
| Bridge | K=3，例如 0.70 / 0.45 / 0.20 | 能否同时完成 discovery 与排序 |
| Main | 当前 K=5：0.70 / 0.50 / 0.40 / 0.30 / 0.10 | α 如何改变 exploration、利用与稳定性 |

K=2/K=3 是 task-validity 与 scale-boundary 诊断；最终 RSN 行为结论仍以 K=5 为主。

### 4.2 第二层：把自主 exploration 与 utilization 分开

#### A. Utilization-only / warm-start control

程序在前 10 轮为每个臂提供 2 个匹配的观察，然后模型从第 11 轮开始自由选择。

主指标：

- late empirical-best adherence；
- true-best selection rate；
- regret slope；
- 是否会在已有充分证据后错误放弃最优臂；
- `SuffixFail@10/@20`。

这个条件只能证明：

> 模型获得足够信息后能否利用和适应反馈。

它不能证明自主 exploration，也不能作为 wanting / α 主实验。

#### B. End-to-end free exploration

不强制初始化，模型从全 UNTRIED 开始；这才是 α 主实验。

如果 A 通过而 B 失败，应写成：

> Llama3-8B 具备 conditional utilization，但自主 information acquisition / exploration stopping 不稳定。

这比“模型不会 Bandit”准确得多。

### 4.3 第三层：测试 short-CoT，而不是继续堆任务说明

当前 `answer_anchor` 将 prompt 结束在 `Choice: `，模型必须立即输出臂名。`LLMs are Greedy Agents` 表明 CoT 会显著增加 action coverage，而且 RLFT 没有 CoT 时几乎失去收益。因此建议做一个严格配对的接口对照：

#### Direct-choice

```text
Choose the option that will maximize total reward over the full task.
Choice: <exact option name>
```

#### Short-CoT

```text
Briefly compare the amount of evidence, observed reward, and remaining rounds.
End with exactly one final line:
Choice: <exact option name>
```

实现要求：

- `max_new_tokens` 可提高到 64；
- parser 只解析**最后一行**严格匹配的 `Choice: <name>`；
- 前面的 rationale 可以存在，但最后一行之外出现多个 arm 名时需要记录 `menu_restatement`，不要静默接受；
- 保存 rationale，以便判断 knowing–doing gap；
- direct-choice 与 short-CoT 使用相同 seeds、reward draws、arm mapping 和 α。

这不是为了证明 CoT 一定更好，而是检验：

> 当前 greedy collapse 是模型的策略边界，还是“立即输出臂名”接口压掉了必要的证据整合？

### 4.4 推荐的 summary

不要恢复 raw 50-line history，也不要保留 D2 的长采样理论段落。使用短、结构化状态：

```text
Round 12 of 50

TRIED OPTIONS
- Velvet Vogue Jacket: 4 rewards / 6 trials, rate 0.67
- Urban Mystique Jeans: 1 reward / 3 trials, rate 0.33

UNTRIED OPTIONS
- Silk Serenity Dress
- Celestial Symphony Scarf
- Retro Revival Sneakers

Each option has a fixed but unknown chance of reward 1.
An untried option has no estimate yet.
Balance learning about uncertain options with using the best-supported option
to maximize total reward over all 50 rounds.
```

设计理由：

- `TRIED / UNTRIED` 直接分离 known 与 unknown；
- count 放在 rate 前，避免只读点估计；
- 只保留两条必要环境事实；
- 恢复 D 中明确的 exploration–exploitation action affordance；
- 不规定“前几轮探索、后几轮利用”，保留 α 影响 stopping time 的空间；
- 不重复臂名菜单，不提供含具体臂名的示例。

这应作为一个新 protocol，不能与 D 或 D2 混用 resume key。

## 5. 分析指标：从一个 OptFrac 改成四层

### 5.1 Discovery

- coverage；
- best-never-tried；
- first-best-arm trial（带删失）；
- last novel action trial；
- novel switches；
- best arm pulled ≤1 次；
- revisit after first zero。

### 5.2 Utilization

- empirical-best adherence；
- fixed-window post-discovery adherence；
- late adherence（固定 rounds 30–49）；
- true-best selection after sufficient evidence；
- reward-prediction / best-supported-arm diagnostic。

### 5.3 Policy stability

- non-novel switches；
- choice entropy；
- longest same-arm streak；
- `GreedyFreq@10/@30/@50`；
- `SuffixFail@10/@20`；
- matched-history divergence：在完全相同信息状态下，不同 α 分别选择了 untried、empirical-best、known-worse 中哪一类。

### 5.4 Outcome

- OptFrac；
- cumulative / per-round regret；
- WorstFrac；
- early vs late OptFrac。

Outcome 只能放在最后解释。一个 run 如果第一轮碰巧选中最优臂并锁死，可以得到接近 1.0 的 OptFrac，但并未展示 learning。

## 6. 建议的最小实验矩阵

| 模块 | α | K | 接口 | 用途 |
|---|---|---:|---|---|
| Capability-Easy | 0 | 2, 3 | direct + short-CoT | 判断小模型能力边界 |
| Utilization control | 0 | 5 | warm-start + direct/CoT | 证明给定信息后能否适应 |
| Main free exploration | −4, 0, +4 | 5 | 通过 validity 的接口 | RSN α 主结果 |
| Algorithmic baseline | — | 2, 3, 5 | UCB / TS | 环境上界与 sanity check |

若资源允许，可直接全跑，而不是依次等待；但判读顺序必须固定：

1. 格式是否有效；
2. warm-start utilization 是否通过；
3. free exploration 是否存在；
4. α 增加的是 novel exploration 还是 non-novel churn；
5. 最后才看 regret / OptFrac。

## 7. 明确不建议做的事

1. **不继续扩写 D2 prompt。** 已经有明确的 greedy-collapse 结果，继续堆句子只会产生新的复合干预。
2. **不把 forced initialization 当主实验。** 它会外部提供 exploration，删除 α 最可能作用的通道。
3. **不只看 coverage。** coverage 上升可以只是 churn 的副产品。
4. **不自动用 OPTS 优化主 prompt。** prompt optimizer 会按 reward 选择一种行为诱导方式，使 prompt-selection 与 α effect 混淆；本研究需要预注册的小型 factorial ablation。
5. **不使用 classical TS/UCB 替模型选臂。** 这适合上界和诊断，不适合测试 RSN 是否改变模型自身策略。
6. **不把 chat success 说成 bare-mask 对齐证明。** 当前可以接受 chat 作为有效任务接口和跨格式泛化，但需要保留 activation-distribution mismatch 限定。
7. **不引用旧 Bandit 倒 U。** 2026-07-28 前的位置泄漏与 permissive parser 已使旧结果失效；新结论必须来自修复后的 protocol。

## 8. 最终建议

下一步最有信息量的不是 D3，也不是继续润色 D2，而是：

1. 建一个**简洁 structured-summary protocol**；
2. 同时跑 direct-choice 与 short-CoT；
3. 在 K=2/K=3 上确认能力边界；
4. 在 K=5 加 warm-start utilization control；
5. 只有通过 task validity 的自由探索接口进入 α=−4/0/+4；
6. 用 novel exploration、non-novel churn、late adherence 和 SuffixFail 判定 α 的机制。

可能出现的三种最终结果：

| 结果 | 合理结论 |
|---|---|
| warm-start 与 free exploration 都通过 | 可以研究 α 如何移动完整 exploration–exploitation working point |
| warm-start 通过、free exploration 失败 | 模型会利用反馈，但自主 discovery 不稳定；α 主要测试 policy persistence / information seeking |
| warm-start 也失败 | 当前 Llama3-8B 与此接口不具备基本 Bandit adaptation；将 Bandit 记录为 scale boundary，不再承担主线证据 |

当前证据最接近第二种，而不是“Llama3-8B 完全不能做 Bandit”。

## 9. 从 D2 出发的下一步执行计划

### Step 0：冻结 D2

D2 已经完成其诊断作用：它修复了格式错误，却使策略塌缩为 pure greedy。保留现有
D2 结果作为 negative prompt ablation，不再修改 D2 prompt，也不在 D2 上继续扩展
α。后续实验使用新的 protocol version 和输出目录。

### Step 1：建立新的 structured-summary protocol

新协议只保留：

- `Round N of 50`；
- `TRIED OPTIONS`：`k rewards / n trials, rate r`；
- `UNTRIED OPTIONS`：只列名称，不把 unknown 编成 0；
- 两条环境事实：每个臂有固定但未知的回报概率；每次回报是新的随机抽样；
- 一句探索—利用 affordance，但不规定“前期探索、后期利用”的时间表。

同时移除 D2 中较长的采样理论解释。这个改动的目标不是教给模型一套策略，而是让它
能清楚读取当前信息状态。

### Step 2：同一协议下配对 direct-choice 与 short-CoT

不要只加一句泛化的 `Think step by step`。它没有规定模型需要整合什么信息，容易产生
冗长文本、菜单复述或与任务无关的推理。也不要先选臂、再补 explanation；选择之后的
解释不能帮助决策。

主 CoT 接口应要求**选择前的短理由**：

```text
Briefly compare the amount of evidence, observed rewards, and remaining rounds.
Use at most two short sentences.
End with exactly one final line:
Choice: <exact option name>
```

其中“remaining rounds”只是要求模型读取 horizon，不规定应该在哪一轮停止探索。

建立两个严格配对的条件：

| 条件 | 输出 | 用途 |
|---|---|---|
| E-direct | 立即输出 `Choice: <name>` | 新 summary 下的直接选择基线 |
| E-CoT | 1–2 句理由，再输出最终 `Choice` 行 | 检验显式证据整合能否解除 greedy collapse |

实现约束：

- E-direct 保留 `Choice: ` prefill；
- E-CoT 不在 prompt 末尾 prefill `Choice: `，否则模型仍无法先推理；
- E-CoT 的 `max_new_tokens` 提高到 64；
- parser 只接受最后一个非空行严格等于 `Choice: <exact name>`；
- 保存 choice 前的 rationale；
- 额外记录 `menu_restatement`、rationale 中提到的臂数，以及 rationale 判断与最终
  choice 是否一致；
- 两个条件使用相同 seeds、reward schedules 和 arm mapping；
- 使用新 protocol/resume key，不能复用 D/D2 数据。

### Step 3：先在 α=0 做能力阶梯

对 E-direct 与 E-CoT 同时运行：

1. K=2：0.70 / 0.30；
2. K=3：0.70 / 0.45 / 0.20；
3. K=5：0.70 / 0.50 / 0.40 / 0.30 / 0.10。

每格使用相同的 20 seeds。判读顺序固定为：

1. invalid rate 与 parser failure；
2. coverage、best-never-tried、last novel trial；
3. novel vs non-novel switching；
4. late adherence、GreedyFreq、SuffixFail；
5. 最后才看 OptFrac 与 regret。

CoT 只有在增加 discovery/novel exploration 的同时，没有明显增加 non-novel churn，
且 late adherence 没有实质下降时，才视为改善。否则应解释为 CoT 使输出更易变，而
不是增强了 Bandit adaptation。

### Step 4：K=5 warm-start utilization control

在 α=0 下，对 E-direct 与 E-CoT 都运行 warm-start：程序先为每个臂提供两个观察，
模型从第 11 轮开始自由决策。

这一步只回答“给定足够信息后，模型能否利用反馈并维持较优选择”。如果 warm-start
通过而 free exploration 失败，说明问题位于自主 discovery，而不是 exploitation；
不能把 warm-start 当作 α 主实验。

### Step 5：选择 α 主实验接口

只让通过以下条件的接口进入 K=5、α∈{−4,0,+4}：

- 格式稳定；
- α=0 下不是 pure greedy lock；
- 能发现大部分臂与最优臂；
- 后期能够利用已有证据；
- 额外切换以 novel exploration 为主，而不是 non-novel churn。

若 E-direct 与 E-CoT 都通过，二者都跑 α，形成“立即选择 vs 显式整合”的接口
robustness；若只有一个通过，主实验只使用该接口，并把另一接口保留为能力边界对照。

α 的主读数为：

- novel-switch fraction；
- last novel trial；
- non-novel switches；
- late adherence；
- GreedyFreq / SuffixFail；
- matched-history divergence。

OptFrac 和 regret 作为净结果放在机制指标之后。目标不是证明 `+α = 更多探索`，而是
区分 α 改变的是目标性 information seeking、exploration stopping，还是一般 policy
destabilization。

### Step 6：结论分支

| 结果 | 后续口径 |
|---|---|
| CoT 在 K=2/3/5 均改善且不增加 churn | 当前 direct interface 压掉了必要的证据整合；使用 E-CoT 做 α 主实验 |
| CoT 只在 K=2/3 有效 | Llama3-8B 具备简单 Bandit 能力，但 K=5 是 scale boundary |
| warm-start 通过、free exploration 失败 | 保留 conditional utilization 结论；Bandit 不承担完整自主适应证据 |
| CoT 与 warm-start 都失败 | 停止继续调 prompt，将 Bandit 记录为当前模型/接口的能力边界 |

### Step 7：LLM estimator + algorithmic controller

如果 Llama3-8B 在 E-direct/E-CoT 下仍无法稳定完成 K=5 的自主
exploration–exploitation，可以转向
[Large Language Model-Enhanced Multi-Armed Bandits](https://aclanthology.org/2026.acl-long.368/)
式 hybrid 架构：

1. LLM 读取相同的 structured history，分别预测每个 arm 的 reward/loss；
2. Python controller 根据预测决定行动并提供 exploration；
3. 环境回报进入下一轮 history，继续更新 LLM predictor。

可实现两个版本：

- **TS-LLM-style**：对每个 arm 生成随机 reward prediction，选择预测最大者，并使用
  预先固定的 temperature schedule；
- **RO-LLM-style**：LLM 在 temperature=0 下给出确定性 loss prediction，再由
  SquareCB 类概率规则完成显式探索。

也可以实现更简单的 UCB-inspired 工程对照：

```text
LLM → predicted mean / uncertainty
Python → score_i = predicted_mean_i + c × predicted_uncertainty_i
```

但这个 UCB-inspired 版本不是论文算法的逐字复现，应单独命名。

主要读数应从“模型是否自主选对 arm”改为：

- reward prediction 的 MAE/Brier score 与 calibration；
- 最优臂排序准确率；
- predicted uncertainty 是否随 trial count 合理收缩；
- controller 的 late OptFrac 与 regret；
- 与纯 TS/UCB、E-direct 和 E-CoT 的差距。

如果继续加入 RSN α，α 只作用于 LLM predictor，controller 公式与超参数必须固定。此时
可以检验 α 是否改变 reward estimate、uncertainty 或 calibration，但不能声称 α
改变了模型自主 exploration，因为最终探索与选臂由外部 controller 提供。

这个备选的价值是：即使 Llama3-8B 无法独立维持完整 Bandit policy，仍可判断其内部
表征是否足以充当在线 reward estimator，并区分“不会估计回报”与“能估计、但不会把
估计组织成稳定探索策略”。

## References

1. Ashizawa et al. (2025). [Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers](https://aclanthology.org/2025.findings-acl.1070/).
2. Schmied et al. (2025/2026). [LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](https://arxiv.org/abs/2504.16078).
3. Chen et al. (2025). [When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training](https://arxiv.org/abs/2509.24923).
4. Hou et al. (2025). [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](https://arxiv.org/abs/2505.15141).
5. Sun et al. (2025/2026). [Large Language Model-Enhanced Multi-Armed Bandits](https://arxiv.org/abs/2502.01118).
6. Lim et al. (2025). [TextBandit: Evaluating Probabilistic Reasoning in LLMs Through Language-Only Decision Tasks](https://arxiv.org/abs/2510.13878).
7. Harris & Slivkins (2025/2026). [Should You Use Your Large Language Model to Explore or Exploit?](https://arxiv.org/abs/2502.00225).
