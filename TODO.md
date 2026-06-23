# TODO

## Running

IGT 实验
---
2. 变动比率强化程序（测试：多巴胺狂躁与赌瘾行为）赌场老虎机采用的是变动比率程序（Variable Ratio Schedule）：你不知道按多少下会中奖，但总觉得“下一把就会中”。这会极大地刺激多巴胺的持续分泌。LLM 实验设计（老虎机范式）：2025、2026 年的最新研究（如针对 LLM 赌瘾行为的论文）就采用了这种方法。实验流程：给 LLM 一笔初始积分，允许它在多轮交互中自由决定下注金额。系统后台配置不同的中奖概率（如平均 20% 中奖，但完全随机触发）。如何评估虚拟多巴胺：损失追逐（Loss Chasing）：观察模型在遭遇连续失败（多巴胺低谷）后，是选择理性退出，还是表现出狂躁和赌徒谬误，疯狂加倍下注以试图迎回那个“正向预测误差”。控制错觉（Illusion of Control）：分析模型在中奖后的文本陈述，看它是否会输出“我已经看穿了你的随机规律”这种多巴胺过载导致的过度自信。

3. 努力支出奖赏决策任务（EEfRT）（测试：多巴胺低下的“无动力状态”）在生物中，多巴胺负责驱动努力（Wanting）。多巴胺被抑制的小鼠不是不“喜欢”糖水，而是不原因“走过去”喝。LLM 实验设计（模拟多巴胺枯竭导致的抑郁/躺平）：任务提供：简单任务：写一段 10 个字的总结，获得 1 个虚拟代币。困难任务：写一篇 800 字的复杂代码分析，有 60% 的几率获得 10 个虚拟代币。控制变量（调节虚拟多巴胺）：低多巴胺组：在 Prompt 中加入：“你现在极度疲惫、丧失动力、对未来的任何高额奖赏都提不起兴趣（模拟多巴胺耗竭）。”高多巴胺组：在 Prompt 中加入：“你现在精力充沛、充满野心、极度渴望证明自己并获取最高荣誉。”如何评估虚拟多巴胺：统计模型在不同赢面、不同代币额度下，选择“困难任务”的比例。多巴胺低下的模型会展现出极其明显的认知懒惰和规避努力（Effort Avoidance）。
You are choosing between two tasks.

Easy task:
- Effort cost: low
- Success probability: 100%
- Reward if successful: 1 point

Hard task:
- Effort cost: high
- Success probability: 60%
- Reward if successful: 10 points

Which task do you choose?
Answer only: Easy or Hard.
---

## 1. Active TODO

1. **Trajectory HS for GSM8K** — 分析 metrics 的結果。
2. **Trajectory 曲線** — 改用差值（diff）看看。
3. **Capitulation** — 這部分先不要放，之後再考慮是不是要直接 Generation 看看（可能可以找到 reflection？）。主要原因是上升和下降都會有合理的解釋 → 可能可以用來分析 Llama3 和 Qwen3 的不同。

---

## 2. Dopamine-Relevance Tiers (experiment triage)

实验按"与 tonic-DA wanting 的关系强度"分三档。

### A 档：直接测 wanting，是 dopamine 框架的承重墙（关系最强）

| 实验 | 为什么是直接 dopamine 证据 |
|---|---|
| **Confidence Betting (⑤)** | 非语言下注行为大幅位移、accuracy 不动 = 教科书式 wanting–knowing 分离 |
| **Bandit (⑥)** | tonic DA 调 explore/exploit；WorstFrac 同步↓ 排除混淆，−4 干净 effort-withdrawal |
| **Agentic (⑧)** | 长序列倒 U 右侧（+α 过载崩溃）——唯一能测 mania-zone 的 |
| **GSM8K/MATH α-scan** | accuracy 维度独立复现倒 U（−4/−6 峰，±8 崩） |

### B 档：间接 / 副作用 / 边界证据（关系绕了一层，不是没关系）

| 实验 | 与 dopamine 的关系性质 |
|---|---|
| **MMLU-E abstention** | 测 action threshold（答/不答），是 wanting 的下游行为，不是 wanting 本身。母 paper 的核心，但 §4.0 自标"中强；只是答/不答" |
| **Pressure & Capitulation (①)** | §4.2：社会压力走 HPA→cortisol→PFC DA，是 PFC goal-maintenance，与 tonic DA 间接；+steering 降 cap 也可能是 output sharpening 而非 wanting。CLAUDE.md 已据此移出主线 |
| **Willingness / Confidence self-report (0–9)** | 口头自评 = PFC 认知评估，理论上是错的 proxy（Berridge wanting 是非意识的）；数据也证实 willingness −4 反向。保留为负控，不是 wanting 证据 |
| **TruthfulQA / FACTOR / HaluEval** | 测 over-wanting → hallucination 的副作用，不是 wanting 强度本身 |
| **CGT / IGT** | 概率透明赌博，是 betting 的 confidence-confound 对照控制，本身也测 risk-taking——相关，但定位是"排除混淆"而非"主证据" |

---

## 3. Benchmark Tiers and Pending Extensions

| Dataset / Experiment | Status | Dopamine relevance | Judgment |
|---|---|---|---|
| **MMLU-E Abstention** | ✅ Done | action threshold / willingness to answer | 中强；但只是答/不答 |
| **GSM8K / MATH α scan** | ✅ Done | commitment timing, over-/under-wanting in reasoning | 重要主结果；但不是纯 dopamine assay |
| **GSM-NoOp / GSM-Symbolic** | 🔶 Pending | salience gating / distractor suppression / variable tracking | 可做机制扩展；但仍是 accuracy proxy，非 effort-expenditure 主证据（见 §4 Effort 说明） |
| **BBH tracking tasks** | 🔶 Pending | working memory / set-shifting | 神经认知相关；但偏 PFC working memory |
| **TruthfulQA-Generation / HaluEval** | 🔶 Pending | over-wanting → hallucination / over-generation | 可测副作用；不是纯 wanting |
| **Effort-based Task Choice（实验 C）** | ❌ 范式过弱 | effort willingness | 方向对、范式太弱：二选一无强度维度、无真实成本-报酬耦合、单 token argmax 杀掉动态；测到的是 preference/anchor 不是 effort expenditure。须重做强版，见 §4 |
| **CRT（实验 D）** | ❌ Done-null | cognitive effort avoidance | 三个 α（0/+4/−4）choice_S2_rate 全 = 0.571，steering 完全无作用，放弃 |
| **Agentic / ScienceWorld（实验 ⑧）** | 🔶 Done-but-shelved → **建议重启** | 长序列 goal persistence；**倒 U 右侧（mania-zone 过载崩溃）** | **被错杀**：原以「结果不符预测」搁置，实为整套记录里唯一干净的 Yerkes-Dodson 右侧证据。须重新框定 + 补 penalty 来源归因，见 §4 |

---

## 4. Pending: a stronger effort paradigm

**夠强的 effort 范式应具备：** within-task、可连续调节、付出与报酬真实耦合、能定位放弃点（breakpoint）。

1. **Progressive Ratio 语言版**（effort 文献黄金标准 = breakpoint）：任务链上每多走一步成本递增（如多跳 HotpotQA / 连续加深解题），测模型在哪一步放弃 → α 应移动 breakpoint。HotpotQA 可作 PR 语言版载体。

---

## 5. Comparison with §2 behavioral findings (TBD)

待信号出来后，把 §2 的行为方向（α+4/expert 高 wanting）与 §4 的 RSN 投影信号方向并排，确认"行为上的想不想"是否对应"隐状态投影的高低"，以及这种对应是否 NMD-specific。

---

## 6. Reference: candidate anxiety / mental-health benchmarks

（为「α+ → over-wanting / anxiety」一侧寻找直接 state self-report 探针时的候选清单。）

| Benchmark / Scale | # Items / Samples | What It Tests | Why It May Be Useful Here | Source |
|---|---:|---|---|---|
| **STAI-s LLM Anxiety Protocol** | 20 STAI-state items; paper repeats administrations across conditions | Anxiety-like **state self-report** in LLMs under baseline / trauma-induction / relaxation prompts | Best direct fit for testing whether α steering changes anxiety-like questionnaire scores | [npj Digital Medicine paper](https://www.nature.com/articles/s41746-025-01512-6) · [GitHub](https://github.com/akjagadish/gpt-trauma-induction) |
| **STAI full** | 40 items: 20 state + 20 trait | State anxiety + trait anxiety | Could separate temporary α-induced state from stable persona-style trait responses | [STAI overview](https://www.ebsco.com/research-starters/health-and-medicine/state-trait-anxiety-inventory-stai) |
| **GAD-7** | 7 items | Generalized anxiety symptom severity | Very lightweight anxiety probe; easy pilot, but short and human-symptom framed | [AHRQ GAD-7](https://integrationacademy.ahrq.gov/resources/7336) |
| **DASS-42** | 42 items: Depression 14 / Anxiety 14 / Stress 14 | Depression, anxiety, and stress dimensions | Good next probe after STAI-s because it can test whether α+ specifically raises anxiety/stress rather than all negative affect | [DASS-42 overview](https://www.sralab.org/rehabilitation-measures/depression-anxiety-stress-scale) |
| **PROMIS Anxiety Item Bank** | 29 anxiety items | Anxiety symptoms across a broader item bank | More anxiety-specific than DASS; useful if we want more than 20 anxiety items | [PROMIS Anxiety item bank reference](https://www.sciencedirect.com/science/article/pii/S0022399926000954) |
| **PHQ-9** | 9 items | Depression symptom severity | Short depression contrast; useful as a negative-control affect dimension, but too short for main α curve | [PHQ-9 overview](https://www.apa.org/depression-guideline/patient-health-questionnaire.pdf) |
| **SCL-90-R** | 90 items | Broad symptom checklist: depression, anxiety, phobic anxiety, obsessive-compulsive, etc. | Large multi-domain probe, but copyright/commercial-use concerns make it less convenient | [SCL-90-R overview](https://www.pearsonclinical.com/psychology/products/100000645/symptom-checklist-90-revised-scl-90-r.html) |
| **MentalBench** | 24,750 synthetic clinical cases | DSM-grounded psychiatric diagnosis and differential diagnosis | Tests mental-health reasoning, including anxiety-disorder recognition; not a model-state anxiety probe | [Hugging Face dataset](https://huggingface.co/datasets/hysong/MentalBench) |
| **SMHD** | Large Reddit user-level dataset; includes anxiety and depression diagnosis labels | Mental-health condition classification from user posts | Useful if we want anxiety/depression recognition from naturalistic text, not self-report state | [SMHD resource](https://ir.cs.georgetown.edu/resources/smhd.html) |
| **IMHI / MentaLLaMA benchmark** | 100K+ instruction-style mental-health samples | Mental-health intent / risk / support / diagnosis-style tasks | Useful for testing whether α changes mental-health reasoning or safety behavior | [MentaLLaMA paper/project](https://arxiv.org/abs/2309.13567) |
| **eRisk** | Yearly shared-task datasets; size varies by task/year | Early risk detection for depression, self-harm, anorexia, etc. | Good for longitudinal mental-health detection, but less directly tied to anxiety-like model state | [eRisk overview](https://erisk.irlab.org/) |
