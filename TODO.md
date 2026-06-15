# TODO

## Running

- **EVOLvE ClothesShopping alpha scan** (@182) — No-Role, −8→+8, layers 11–20, 30 runs × 50 rounds. + 需要做统计分析 + STD可以作为一个观测指标



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
