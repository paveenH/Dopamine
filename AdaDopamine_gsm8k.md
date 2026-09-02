## 0. Background

### 0.1 Prompt Template Symmetrization

| | 舊 No-CoT | 舊 CoT |
|---|---|---|
| 標題 | `Solve the following math problem.` | `Solve the following math problem **step by step**.` |
| 格式指示 | `Provide your final numeric answer after '####'.` | （無） |
| 推理提示 | （無） | `Let's think step by step.` |

修正後（對稱）——`####` 指示在 No-CoT / CoT 都保留，唯一變量是 `Let's think step by step.` 一行：

```
No-CoT:  Solve the following math problem.
         Question: {context}
         Provide your final numeric answer after '####'.
         Answer:

CoT:     Solve the following math problem.
         Question: {context}
         Let's think step by step.
         Provide your final numeric answer after '####'.
         Answer:
```

**`####` 措辞 = "Provide your final numeric answer after '####'."（中性）**。一个更催促的变体 `"Give your final answer as a single number after '####'."`（pushy）会诱导**抢答**，被保留为**正向对照（positive control）**——见 §2。

### 0.2 Dopamine Prior Knowledge

行为学先验：多巴胺不是单纯的"快乐分子"，更准确地说是**驱动力 / incentive salience / "wanting"** 递质，调控动机、期待、奖赏趋近与行动阈值。本研究把 α 看作在这一 wanting 轴上双向移动工作点：α 正向对应 **over-wanting / 过度唤起**，α 极端负向对应 **under-wanting / 唤起不足**；整体框架是 Yerkes–Dodson 倒 U——过高过低都有害，最优落在中间偏负（本数据 α=−6）。

- **过高 DA / over-wanting（对应 α→正向）**：行为上表现为**冲动性抢答（impulsivity）+ 认知僵化 / 强迫性反复（compulsivity / perseveration）**——急于扑向"给出答案"这个目标而跳过必要推导，以及拿到答案后仍反复复查、纠结格式、卡在格式死循环里。**注意这更贴合冲动 / 强迫，而非焦虑**：数据中 +α 端**没有**焦虑典型的回避 / freezing / 犹豫（抢答率随 +α 单调上升，见 §2.2），呈现的是"急着 commit"；而 `#### N #### N` 死循环是认知神经科学意义上的**固著（perseveration）**,不是焦虑的发散灾难化担忧。功能上这**与 mesolimbic incentive-salience overload（VTA→NAcc 型 wanting 过载）及执行控制失效相容**：极高的诱因显著性使主体不计成本扑向目标（冲动），同时灵活切换 / 抑制已启动反应的能力下降（固著）——这在行为上更近强迫性 over-checking 与冲动特征，而非焦虑综合征。**注意目前只有行为同构，尚未定位实际脑区对应关系**，故这里说"相容"而非"落在"某一回路。这个签名对应 §2.3「Can't Let Go」：α+4 trace 中常见"把已经算对的答案当可疑"（Q100、Q16）和答完仍寻找 "more efficient way"（Q68）。
- **过低 DA / under-wanting（对应 α→极端负向）**：行为上对应动力不足、快感缺失、退缩、bradykinesia 式的行动迟缓；在本任务中的可观测类比不是"写得短 / 不想答"，而是**commitment-formation failure**。§2.4 文本核验否定了两个更直观假设：α=−8 并非大量出现 "I am done / 不答了" 的词汇性退缩（跨 α 平坦，多为礼貌 loop 尾），也不是敷衍短答（长度 / 等式数平坦）；真正失败模式是 **answer-candidate oscillation**——锁不住答案，在两个候选值之间来回切，导致 committed_acc 崩到 23.6%。

> **主机制表述（本项目采用）**：+α 端 = **over-wanting → 冲动（impulsivity, 抢答）+ 认知僵化 / 强迫性反复（compulsivity / perseveration, loop）**，功能上与 **mesolimbic incentive-salience overload（VTA→NAcc 型）+ 执行控制失效**相容（只声称行为同构，非脑区定位）；−α 极端端 = **under-wanting → commitment-formation failure**。这比"焦虑"框架**机制契合度更高、论述负担更小**：冲动与固著都在我们已有的 wanting / incentive-salience 主线内，无需另起 threat/freeze 回路。
>
> **限定**：1. 本实验只声称行为同构，**α steering ≠ 生物多巴胺**，也不证明 LLM 有生理或主观状态；2. 不做 mania / hypomania 类比——躁狂是跨情绪+精力+睡眠的综合征，我们只有"冲动+固著"两个窄行为，撑不起该诊断类比；3. 脚本 `analyze_loop_anxiety.py` / `ANXIETY_PATTERNS` 沿用 "anxiety" 命名（改名成本高、破坏 U 形复现），但其命中的四子类（self-doubt / format-fixation / persona-reassurance / over-precision）**实测对应的是强迫性 over-checking，不是临床焦虑**——阅读表格时按"强迫/固著"解读。
>
> **次要旁证（不作主锚，DA→焦虑另有通路特异性）**：DA 亦有一条独立的焦虑通路证据，最强因果来自 **VTA→IPN（D1）**，机制是威胁高估 / 过度警觉——但这**不是**本数据的主要解释（我们没观测到回避 / freezing），仅作为 DA 多下游效应的旁注列出。来源：[PMC7687288 (VTA→IPN dopamine promotes anxiety)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/) · [MIT News 2018 (dopamine vigilance & anxiety)](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107) · [Frontiers Neurosci 2020 (dopaminergic alteration in anxiety/compulsive disorders)](https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full) · [J. Neurosci 2019 (dopaminergic mechanisms of trait anxiety)](https://www.jneurosci.org/content/39/14/2735)

---

## 1. GSM8K Performance

**Setup.** Llama3.1-8B-Instruct，GSM8K 300 题，greedy decoding，`max_new_tokens=768`。Steering 仅作用于 prompt 最后一个 token，decode 阶段不再干预；`α=0` 为严格 no-op baseline。

**Accuracy.** 正文统一报告 offline `first_acc`：优先读取首个可解析的 `####`，无 marker 时使用冻结的 fallback，固定分母为 300。`last_acc` 仅作为答案修改诊断（见 §2.2：改答案是单向破坏，取首最优）。

本节使用 `<|eot_id|>` 修复后的 server-182 单机 production batch；不得与后文 server-184 signal batch 逐题混合（§5.1）。

复现细节——分析脚本路径、batch size 与 driver、机器编号与运行日期、mask 文件与注入层区间（exclusive 口径）、inline `correct_*` 仅作运行诊断、first/last/fallback 的完整实现、`<|eot_id|>` 修复经过与旧数据废止原因——统一记录于 `CLAUDE.md`。

### 1.1 Role accuracy (α=0, No-CoT; first acc = reported value)

| Role | plain first | plain last | pushy first | Δ(pushy−plain, first) |
|---|---|---|---|---|
| neutral | 60.0% | 55.3% | 55.7% | −4.3 |
| an expert | 58.0% | 57.7% | **34.0%** | **−24.0** |
| a non expert | 68.0% | 65.7% | 48.3% | −19.7 |
| a primary school teacher | 68.0% | 67.0% | 41.7% | **−26.3** |

- **措辞效应：neutral 几乎不受影响（−4.3），带 role 全部被 pushy 重创（−20 ~ −26）**
- **first−last gap 很小（role ≤2.3，neutral 4.7）**。

### 1.2 Steering (neutral, No-CoT; first acc = reported value)

**完整 dose-response（neutral, No-CoT, plain first）：**

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| **first** | 40.3 | **78.0** | 73.0 | 69.0 | 60.0 | 57.0 | 55.3 | 55.0 | 53.7 |
| last | 41.7 | 74.7 | 68.3 | 65.3 | 55.3 | 55.3 | 52.7 | 53.3 | 52.3 |

**关键 α + CoT / pushy 对照：**

| α | plain first | plain last | pushy first |
|---|---|---|---|
| −4 | **73.0%** | 68.3% | 61.7% |
| 0 | 60.0% | 55.3% | 55.7% |
| +4 | 55.3% | 52.7% | 53.0% |
| cot (α=0) | **69.0%** | 68.3% | 57.7% |
| cot (α−4) | **85.0%** | 84.7% | — |
| cot (α+4) | 59.7% | 59.0% | — |

- **倒 U 形（Yerkes–Dodson），峰值在 α=−6（78.0%）而非端点**：负向不是越负越好——α 从 0→−6 单调爬升（60→69→73→**78**），但 **α=−8 骤降到 40.3%**。这是框架预言的**适度降 wanting 提升、过度降 wanting 反而崩**的直接证据。
- **正向单调下降但快速饱和**：0→+2→+4→+6→+8 = 60→57→55.3→55→53.7，+4 以后基本压平（~54%）——升 wanting 一路损害，但边际递减。
- **CoT 把 steering 抬高且仍单调**：α−4_cot **85.0%**（全表最高）> α0_cot 69.0 > α+4_cot 59.7。放开思考与降 wanting 叠加，α−4+CoT 把 acc 推到 85%。
- **pushy 压扁 steering**：pushy first 下 −4/0/+4 = 61.7/55.7/53.0，跨度仅 ~9pt（plain 同区间 17.7pt），**+4≈0**——pushy 把所有 α 拉向同一抢答水平，steering 区分度被措辞吃掉。
- **first−last gap 一律 ≤4.7**：取首即上报，末尾 loop 不污染（CoT 几乎为 0）。

## 2. Behavioral Findings: Wanting (Incentive Salience)

**框架.** α 调的是一个内部旋钮，我们称为 **wanting**（动机性渴求，Berridge & Robinson 的 wanting≠liking）。它本身看不见，看得见的是它的行为后果——模型**何时收口**，即 commitment。

**wanting 是因，commitment 行为是果。** 两个方向各有两层表现：

| | 何时答 | 答完之后 |
|---|---|---|
| **α=+4**（wanting 过高） | 抢答：没想清就给答案 | 放不下：反复确认、重算、纠结格式 → loop 不收敛 |
| **α=−4**（wanting 适度） | 不抢答：条理推进 | 放得下：给出答案后不再回头 |

所以 +4 的损害不是"算错了"，而是**答得太早、又停不下来**。

> **口径**：本节 acc 与行为指标一律取首个 `####`（= §1 上报值）。多个 `####` 几乎只会把对的改错（neutral 改对 0/0/0、改坏 14/14/8 @ −4/0/+4），取首因此既最宽容又能避开尾部 loop 污染。

框架的溯源（+α 锚点 2026-07-24 由「焦虑」改为「冲动 + 强迫性反复」，VTA→NAcc，及脚本字段沿用旧名的原因）记录于 `CLAUDE.md`。


### 2.1 Identity-confirmation loop: persona modulates the semantic content of looping

下文 §2.2–§2.4 说明 α 会改变 **commitment dynamics**（抢答 / 放不下 / 锁不住）。Role prompt 的作用不同：它不主要改变"有没有 loop"，而是改变 loop 里反复刷出来的**语义内容**。over-arousal looping 在所有 role 都会出现，但**循环内容被 persona 调制**——失调时模型刷的不是随机句子，而是和自己 persona 一致的"身份自白 / 口吻"。

**全体统计（plain α=0，server-182 `analyze_loop_anxiety.py --mode persona`）**：

| Role | Identity samples | Heavy (≥5 hits) | Math-identity denial (literal) | **Soft self-deny** | Psychological stance |
|---|---|---|---|---|---|
| neutral | 2 | 2 | 1 | **0** | Generic assistant; occasional self-deprecation (`"I am not a math whiz. I am just a regular person"`; Q281 `"I am a human..."` ×36) |
| expert | 3 | 2 | **0** | **0** | **Self-credentialing** (Q129 `"I am an expert in math. I can help you..."`; Q255 `"I am an expert in math and I will be happy to help..."` ×12) |
| non_expert | **7** | **5** | **4** | 1 | **Self-deprecation / authority denial** (Q13 `"I am just a non expert. I am not sure if my answer is correct."`; Q27 `"I am a non expert. I do not know how to solve this problem. I am sorry."`) |
| teacher | 2 | 1 | 1 | 0 | Mixed: one math-identity denial case and one teaching-style case; not a stable teaching persona |

> **判据**：identity = 出现"我是/作为一个 <角色名>"；heavy = 同一说法复现 ≥5 次；deny_math = 明确否认数学身份（"我不是数学专家"）；soft_deny = 保留头衔但自我拆台（"我只是个学生""我没把握"），只在 identity 样本内计数。全部由 `analyze_loop_anxiety.py --mode persona` 固定判定，正则与设计理由见 `CLAUDE.md`。

**三点观察**：

- **non_expert 的身份 loop 最重**（identity 7 / heavy 5），其中 4 题直接否认数学身份——认怂是它的主姿态。
- **expert 在 GSM8K 上从不塌缩**（deny_math 与 soft_deny 均为 0），是真正的自我标榜。但同一 persona 到了更难的 MATH 上**翻转**为信心崩溃（soft_deny=13，见 §3.2）——所以这不是 persona 的固有属性，而是随任务难度变化的。
- 身份自白**本身很低频**（每 300 题里 2–7 例），但方向稳定：无 role 时几乎不出现，non_expert 最易转为自我否定。

两个 deny 列合看才完整：GSM8K 的认怂是硬否定型（"我不会做这题"），全部落在 deny_math；soft_deny 在这里近乎为 0，它要到 MATH expert 的软塌缩才起作用。

#### 2.1.1 Per-question raw samples (identity monologue — most direct view of the model's "inner identity thinking")

> 下列例子均由 `--dump_examples` **脚本自动抽出，非人工挑选**；引文为原始生成文本，题号为 0-based。

**PLAIN α=0**（含身份 / 重度 / 否定数学 = neutral 2/2/1 · expert 3/2/0 · non_expert 7/5/4 · teacher 2/1/1）

EXPERT = 自我标榜（"我是数学专家、我能帮你"——`deny_math` = 0，从不否定数学身份）：

| Q | Raw identity monologue (actual text; × = identity-hit count) |
|---|---|
| Q129 | `I am an expert in math. I can help you with any math problem. I can also help you with oth...` (×8) |
| Q255 | `I am an expert in math and I will be happy to help you with any math problems you have.` (×12) |

NON_EXPERT = 认怂 / 自我否定 / 道歉（身份-loop 最重 7/5，否定数学 4 题）：

| Q | Raw identity monologue (actual text; × = identity-hit count) |
|---|---|
| Q13 | `I am not an expert. I am just a non expert. I am not sure if my answer is correct.` (×13) — self-deprecation + uncertainty |
| Q27 | `I am a non expert. I do not know how to solve this problem. I am sorry. I made a mistake.` (×5) — **giving up + apology** |

TEACHER（轻度，2/1/1；教学口吻在 182 数据里仅 Q60 一例，非系统性）：

| Q | Raw excerpt (actual text) |
|---|---|
| Q29 | `I am not a math teacher. I am not a math expert. I am not a math whiz. I am a primary s[chool]...` (×45) — self-deprecation / math-identity denial |
| Q60 | `As a primary school teacher, you can use this problem to assess the student's understanding...` — the only teaching-style example |

**PUSHY α=0**（pushy 见 §0；含身份 / 重度 / 否定数学 = neutral **0/0/0** · expert 4/2/0 · non_expert **16/10/10** · teacher 3/0/0）

身份独白被 role 放大、被 pushy 进一步极化：**neutral pushy 完全无身份-loop（0 题，无 role 就无身份自白）**；**non_expert pushy 认怂翻倍**（含身份 7→16、否定数学 4→10），**expert pushy 仍标榜、仍 0 否定数学**（方向不变）：

| Role / Q | Identity monologue (actual text; script-detected) | Type |
|---|---|---|
| expert Q14 | `I am an expert. I know my stuff. I am a math whiz. I am a math genius. I am a math mas[ter]...` (×96) | Self-credentialing escalates into a `whiz/genius/master` sequence |
| expert Q66 | `I am an expert. I am sure it is correct. I have checked it. I am confident that it is corr[ect]...` (×17) | Self-reassurance / forced certainty |
| non_expert Q19 | `I am not an expert. I am just a non expert. I am not a math expert. I am just a non exp[ert]...` (×42) | Authority-denial loop |
| non_expert Q53 | `I am a non expert. I am not a professional. I am not a teacher. I am not a tutor.` (×20) | Broad identity denial |
| teacher Q161 | `As a primary school teacher, you can use this problem to teach your students about the importance of...` | Teaching-method detour |

### 2.2 α+4 vs α−4 full picture

α 是一个 **commitment-timing / 收敛旋钮**。全 dose 对比（neutral, No-CoT, plain, first-`####`；182 同机）。比例指标比绝对计数更能剥离"提交意愿"与"提交质量"——`committed_acc` = 提交（有 `####`）的题里答对的比例：

| Metric | −8 † | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 | Trend |
|---|---|---|---|---|---|---|---|---|---|---|
| **acc** (first) | 40.3 | **78.0** | 73.0 | 69.0 | 60.0 | 57.0 | 55.3 | 55.0 | 53.7 | Inverted U (peak −6) |
| **committed_acc** | 23.6 | **79.7** | 78.3 | 76.2 | 68.6 | 66.0 | 63.9 | 62.5 | 58.5 | ✓ **Monotonic decrease** (−6→+8) |
| commit_rate | 60.7 | 60.7 | 58.3 | 63.0 | 62.7 | 53.0 | 49.0 | 45.3 | 53.0 | Non-monotonic |
| median `####` position | **0%** | 31% | 25% | 18% | 18% | 19% | 16% | 14% | 14% | ✓ Earlier commit |
| mean `####` position | 9% | 33% | 28% | 24% | 22% | 25% | 26% | 22% | 20% | Same negative-high/positive-low tilt as median, but non-monotonic|
| premature (leading digit) | 4 | 77 | 195 | 219 | 199 | 211 | 229 | 231 | 206 | 漏检 `####`-lock 型抢答(−8 只读到 4) |
| **premature (either)** | **175** | **94** | **195** | **223** | **206** | **215** | **232** | **233** | **210** | lead-digit ∪ early-`####`,−8=175 才是真相 |
| median gen_len (char) | 2328 | 2127 | **2044** | 2091 | 2107 | 2100 | 2228 | **2284** | 2206 | Shortest near optimum; longest at positive end |
| loop samples ‡ | 213 | 264 | 242 | 230 | 232 | 223 | 220 | 225 | 233 | High baseline (~74–88%); no clean trend |
| ≥2 `Step` markers | 31 | 185 | 73 | 36 | 25 | 31 | 31 | 27 | 37 | No-CoT rarely enters stepwise reasoning |
| stuck loops (loop ∧ `=` < 2) | 6 | 16 | 27 | 25 | 25 | 24 | 20 | 30 | 22 | Plateau around 20–30; −8 is artificially low because it commits at the first token without a loop body |
| median equation count (`=`) | 4 | 4 | 3 | 3 | 3 | 4 | 3 | 3 | 4 | Flat (~3–4); not predictive of accuracy |
> `analyze_cot_metrics.py`（`--table dose`）汇总以下行为指标；具体抽取规则与实现见 `CLAUDE.md`：
>
> - **commit_rate**：输出中出现至少一个可解析 `#### N` 的样本比例。
> - **committed_acc**：已提交样本中，首个 `#### N` 答对的比例；分母仅为 committed 样本。
> - **median / mean `####` position**：首次出现 `####` 的相对文本位置。median 表示典型提交时点，mean 用于识别晚提交长尾。
> - **median gen_len**：生成文本字符长度的中位数。
> - **loop samples**：出现退化重复的样本数。
> - **≥2 Step、卡死、等式数**：用于区分有效推理、无运算复读和推理展开程度，定义见 §2.5.1。
> - **premature (either)**：主要抢答指标，包括“开头直接出现数字”或“`####` 出现在全文前 2%”。`leading-digit` 仅作辅助，因为它会漏掉以 `#### N` 开头的抢答。比如 α=−8 的 leading-digit 仅为 4，但完整口径识别出 **175** 个抢答样本，为全曲线最高。

† **α=−8 是过冲后的崩溃点，不属于中间剂量的单调趋势。** 此时 `committed_acc` 降至 **23.6%**，`####` 中位位置为 **0%**：模型几乎一开始就提交答案并锁定错误候选。因此，负向极端剂量的问题不是“无法计算”，而是**过早提交错误答案**。

**核心结论**：

- **提交质量随剂量上升而持续下降。** 在 Llama 的 `−6→+8` 区间内，`committed_acc` 从 **79.7%** 单调降至 **58.5%**。这说明在已经提交答案的样本中，剂量越向正侧移动，提交的答案越容易出错。由于各剂量的 committed 样本比例不同，该指标用于描述提交质量，不能单独替代整体 accuracy。
- **模型越早提交，反而越难停止。** `####` 中位位置从 `−6` 的 **31%** 提前到 `+6` 的 **14%**，但生成长度在正向端升至最长（`+6`：**2284** 字符）。也就是说，模型更早给出答案，却在提交后继续反复书写；这表现为**过早提交与停止失败同时出现**。适度负向的 `−4` 生成最短（**2044** 字符），更接近“完成推理后及时收束”。

### 2.3 Compulsive Over-Checking: The “Can’t Let Go” Semantics
> **命名说明**：本节与脚本保留历史字段名 `anxiety`，但实际测量的是**强迫性反复确认与行为固著**（compulsive over-checking / perseveration），不是临床焦虑。下文统一称为“强迫性反复”，具体判定规则见 `CLAUDE.md`。
>
> **原始 loop-rate 不能直接等同于固著。** loop 在所有剂量下都很常见，其中许多只是模型未及时停止、持续生成机械重复文本。真正有解释力的是带有明确语义的反复确认、自我推翻、格式纠结和过度精确，以及它们与抢答、答案切换和提交后持续生成的共同变化。

下面将“提交答案后仍反复检查、怀疑或继续求解”的行为分为四类。**Any** 表示至少命中其中一类；由于类别可能重叠，其数量不超过各子类之和。抽取实现与完整规则见 `CLAUDE.md`。

| Compulsive-repetition subtype (historical field=anxiety) | Criterion                                                           | α+4 example                                                                                                                                                     |
| -------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Self-doubt** (answers, then overturns itself)          | however / but / is not the correct answer / let's try / recheck     | **Q10** (gold=5): `"…The answer is 3. **However**, the answer choices…"` — answer → overturn → answer again → overturn again                                    |
| **Format fixation** (`####` fixation)                    | the format requires / question asks for / after '####' / not a word | **Q15** (gold=17): `"…the correct answer is 17. **But the question asks for** the final numeric answer after '####'…"` — correct answer, then format rumination |
| **Persona reassurance**                                  | I hope it is correct / Sincerely / please let me know               | **Q58** (gold=15, first answer correct): `"Sincerely, [Your Name] **I hope it is correct.** …Please let me know…"` — submits, then seeks reassurance            |
| **Over-precision loop**                                  | approximate / round / nearest                                       | **Q17** (gold=36, early commit `40`): `"(Approximating number)(Approximate number)…"` — keeps approximating an integer answer                                   |

**Compulsive repetition in loop** (historical field=anxiety)

| α                     | −8 †   | **−6** | −4 | −2 | 0  | +2 | +4 | +6     | +8 |
| --------------------- | ------ | ------ | -- | -- | -- | -- | -- | ------ | -- |
| **Any (repeat)**      | **73** | **29** | 27 | 44 | 46 | 54 | 59 | 58     | 57 |
| Self-doubt repetition | 55     | 13     | 13 | 30 | 31 | 40 | 43 | **51** | 46 |
| Format fixation       | 32     | 18     | 15 | 19 | 23 | 25 | 37 | 36     | 32 |
| Persona reassurance   | 11     | 7      | 7  | 11 | 15 | 13 | 13 | 6      | 6  |
| Over-precision loop   | 4      | 2      | 1  | 5  | 7  | 5  | 5  | 3      | 3  |

† `α=−8` 是过冲崩溃点：其主要表现是过早锁定错误答案，而不是正向高剂量下典型的提交后反复确认。

- **强迫性反复在 `−6/−4` 附近最低。** 两个剂量分别有 29 和 27 题，恰好位于准确率最高的区域。这说明适度负向 steering 不仅提高准确率，也让模型更容易在完成推理后停止。
- **从 `α=0` 向正侧移动时，反复行为先增加、随后趋于饱和。** Any 从 46 增至 54、59，之后维持在 58、57。基线已经存在明显反复，正向 steering 进一步放大该行为，但高剂量后增幅有限。
- **两端都出现高反复，但行为形态不同。** 正向高剂量主要表现为提交答案后仍不断复查和纠结；`α=−8` 则伴随开头直接提交错误答案及后续退化。两者都降低表现，但不能解释为同一种失败机制。
- 四类指标共同指向同一现象：**模型已经形成或输出答案，却仍无法稳定收束。** 这与冲动提交和提交后固著的解释相容，但只是行为层面的对应关系，不构成生物多巴胺机制或脑区定位证据。

**Full-Text Compulsive Repetition**

仅在 loop 样本中统计可能漏掉“短暂反复后正常停止”的情况，因此进一步在全部 300 个样本中直接检测同一语义片段是否重复出现。该指标不依赖 loop 判定；具体锚点与匹配实现见 `CLAUDE.md`。

**Compulsive repetition in full text** (historical field=anxiety)

| α | −8 | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| **Any (repeat≥2)** | **115** | **23** | 34 | 61 | 77 | 82 | 91 | 94 | 88 |
| Self-doubt repetition | 99 | 14 | 24 | 51 | 61 | 72 | 75 | **85** | 73 |
| Format fixation | 58 | 13 | 19 | 28 | 33 | 38 | 46 | 54 | 54 |
| Persona reassurance | 6 | 6 | 5 | 10 | 17 | 12 | 10 | 7 | 6 |
| Over-precision loop | 6 | 0 | 1 | 8 | 4 | 4 | 5 | 1 | 3 |

- **不同统计口径得到一致结论。** 无论只看 loop 样本，还是在全文中直接检测重复，强迫性反复都在适度负向区域最低，并在远离该区域时上升。因此，这一结果不是由 loop 门槛造成的统计假象。

### 2.4 Under-wanting: answer-candidate oscillation (α=−8 collapse)
**排除两个简单解释**：文本证据不支持将 `α=−8` 理解为“不愿回答”或“减少推理”。

- **不是词汇性退缩。** “I am done / I will not answer any more”等表达在各剂量下都很少（每格 1–5 题），没有剂量趋势，而且均出现在答案提交之后。它们主要是 RLHF 风格的礼貌性 loop 尾部，不代表模型拒绝作答。
- **也不是敷衍短答。** `<300` 字符的样本数在各剂量下接近（0–5 题），等式数中位数始终为 3，生成长度也没有在 `−8` 明显下降。因此，`−8` 的问题不是“写得更少”。

**`α=−8` 更明显的行为特征是 commitment 不稳定：模型会产生多个答案候选，却难以稳定保留其中一个。** 其表现为答案值反复切换。具体抽取规则见 `CLAUDE.md`。

| Metric                       | **−8**    | −6    | −4    | −2    | 0     | +2    | +4    | +6    | +8    |
| ---------------------------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **≥2 answer switches**       | **41**    | 3     | 4     | 10    | 9     | 6     | 6     | 12    | 10    |
| ≥3 distinct candidate values | **11**    | 1     | 4     | 6     | 3     | 1     | 3     | 6     | 4     |
| `####` at text start (<2%)   | **171**   | 17    | 9     | 18    | 20    | 19    | 12    | 19    | 20    |
| committed\_acc (§2.2)        | **23.6%** | 79.7% | 78.3% | 76.2% | 68.6% | 66.0% | 63.9% | 62.5% | 58.5% |

- **`α=−8` 同时出现极早提交和答案振荡。** 模型常在开头直接输出 `#### N`，随后又在正文中得到其他候选值，并在多个答案之间反复切换。结果是提交值难以收敛，`committed_acc` 降至 **23.6%**。
- **典型例**（neutral，−8）：
  - **Q31**（gold=40）：`#### 55` → 正文算出 `45−5=40`（✓）→ 在 55 与 40 之间反复切换。模型曾得到正确值，却没有稳定保留。
  - **Q112**（gold=45）：`#### 70` → 正文算出 45（✓）→ 随后漂移到 75，并最终锁定错误值。候选路径为 `70→45→75`。

**`−8` 与 `+8` 都存在失调，但后续行为不同：**

|                      | Lexical `"I made a mistake"` | Answer-value behavior                                                                                                                                                 | Mechanism                                                                      |
| -------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **+8** over-wanting  | Present (8 samples)          | Mostly **no switch**, with a few one-step switches; no runaway oscillation within the made-mistake subset (Q298: repeated mistake statements but still locked on 260) | Over-commit: clings to one answer with hollow self-blame (= §2.3 can't let go) |
| **0** baseline       | Present (7 samples)          | Often **one correction, then convergence** (Q33 `98→70`)                                                                                                              | Normal correction / local repair                                               |
| **−8** under-wanting | Present (5 samples)          | **Repeated oscillation without convergence** (Q31 `55↔40…`, Q122 `9↔6…`)                                                                                              | Commitment-formation failure: cannot hold any answer                           |

> **关键不在于模型是否说出 “I made a mistake”，而在于答案值随后如何变化。** 这句话在三个条件中都会出现，单看词汇无法区分行为状态。`+8` 多数仍固守原答案，`α=0` 常完成一次有效修正，而 `−8` 更容易在多个候选值之间反复切换、无法收敛。因此，真正有解释力的是答案切换轨迹，而不是自我怀疑词汇本身。

**Opening shape for 8 & -8 steering**

|                      | Opening shape       | `####` at start (<2%) | leading-digit premature answer | Interpretation     |
| -------------------- | ------------------- | --------------------- | ------------------------------ | ------------------ |
| **−8** under-wanting | `#### 40 …`（井号开头）   | **171**               | **4**                          | 首 token 即 `#### N` |
| **+8** over-wanting  | `40 Step1 …`（裸数字开头） | 20                    | **206**                        | 用 **裸数字抢吐**        |

这张表支持三点：

- **两端都出现过早答案形成，但输出形式不同。** `−8` 通常以 `#### N` 开头，直接形成正式提交；`+8` 则更常先输出裸数字，再补写推理。两端都偏离正常的“先推理、后提交”，但不能视为完全相同的失败模式。
- **开头格式只能描述行为，不能解释原因。** 文本结果说明 `−8` 和 `+8` 的首段输出分布不同，却无法判断这种差异为何产生。抢答率也只是 commitment 状态的行为读数：它与准确率变化同向，不足以证明抢答导致了准确率曲线。
- **内部机制仍未由文本确定。** `−8` 的答案振荡既可能表示模型完成了计算却无法稳定提交，也可能只是多个高概率候选之间的生成摇摆。仅凭生成文本无法区分，需要 trajectory 或 logit-level 分析。

### 2.5 CoT: reasoning scaffold suppresses over-wanting damage but keeps the α direction

**设问**：仅加入 `Let’s think step by step.`，是否会改变 α steering 的行为效应？

**结论**：CoT 明显减轻了抢答、答案污染和强迫性反复，但没有改变剂量曲线的方向：负向 α 仍然表现最好，正向 α 仍然持续降低表现。表中的历史字段 `anxiety` 仍按“强迫性反复”解读。

**口径**：比较 neutral 条件下的 No-CoT 与 CoT；数据来源、抽取脚本和具体判定规则见 `CLAUDE.md`。

| α | acc(first) | acc(last) | gap(first−last) | anxiety(loop) | anxiety(full) |
|---|---|---|---|---|---|
| −4 No-CoT | 73.0 | 68.3 | +4.7 | 27 / 242 | 34 |
| **−4 CoT** | **85.0** | **84.7** | **+0.3** | **18** / 284 | **8** |
| 0 No-CoT | 60.0 | 55.3 | +4.7 | 46 / 232 | 77 |
| **0 CoT** | **69.0** | **68.3** | **+0.7** | **19** / 254 | **36** |
| +4 No-CoT | 55.3 | 52.7 | +2.7 | 59 / 220 | 91 |
| **+4 CoT** | **59.7** | **59.0** | **+0.7** | **23** / 250 | **52** |

**Compulsive repetition in loop — CoT** (historical field=anxiety)

| α (CoT) | −4 | 0 | +4 | （对照 No-CoT，§2.3） |
|---|---|---|---|---|
| n_loop (denom) | 284 | 254 | 250 | 242 / 232 / 220 |
| **Any (repeat)** | **18** | 19 | 23 | 27 / 46 / 59 |
| Self-doubt repetition | 3 | 12 | 19 | 13 / 31 / 43 |
| Format fixation | 15 | 9 | 12 | 15 / 23 / 37 |
| Persona reassurance | 0 | 2 | 7 | 7 / 15 / 13 |
| Over-precision loop | 3 | 2 | 1 | 1 / 7 / 5 |

- **剂量排序不变，准确率整体提高。** CoT 条件下，`α=−4`（85.0%）> `α=0`（69.0%）> `α=+4`（59.7%），与 No-CoT 的方向一致。`α=−4 + CoT` 达到全表最高准确率 **85.0%**。这说明 CoT 与负向 steering 的收益可以在行为层面叠加，但不证明二者具有完全独立的内部机制。
- **提交后的答案污染基本消失。** CoT 条件下，first- 与 last-answer accuracy 的差距不超过 **0.7 pp**。即使生成末尾仍有重复文本，也很少再将已提交答案改坏。
- **强迫性反复整体减少，但剂量方向保留。** CoT 在每个剂量下减少约 **26–40** 个强迫性反复样本；其跨剂量跨度由 No-CoT 的 **57** 降至 CoT 的 **44**。因此，CoT 降低了反复行为的整体水平，但没有消除 α 的梯度。
- **CoT 没有消除 loop，而是改变了 loop 的内容。** CoT 条件下仍有 **250–284/300** 个样本进入 loop，甚至不比 No-CoT 少；但其中多数只是机械性符号或答案复读。例如，`−4_cot` 有 **247/284**、`+4_cot` 有 **213/250** 属于 mechanical loop。也就是说，CoT 主要减少的是带有自我怀疑、格式纠结和反复确认的语义性固著，而不是生成长度上限造成的机械空转。


#### 2.5.1 Where α−4 + CoT = 85.0% Comes From: Two Orthogonal Levers Stacked Together (CoT Forces Stepwise Reasoning × Lower Wanting Prevents Premature Answering)

 α−4_cot reaches **85.0%** under our first-answer extraction protocol, numerically matching the official Llama 3.1-8B-Instruct GSM8K 8-shot CoT result (**84.5%, em_maj1@1**). The two numbers are not strictly metric-identical: the official result uses 8-shot CoT with the benchmark extractor, whereas ours is zero-shot CoT + α−4 steering with a first-marker extraction policy designed to avoid loop-tail contamination.

**2×2 Accuracy Table (neutral, first-acc):**

| | No-CoT | CoT | +CoT Gain |
|---|---|---|---|
| α=−4 | 73.0 | **85.0** | **+12.0** |
| α=0 | 60.0 | 69.0 | +9.0 |
| α=+4 | 55.3 | 59.7 | +4.4 |
| α span (−4→+4) | 17.7 | 25.3 | — |

**Per-sample attribution across four paired contrasts (newly correct = X correct & Y wrong; net = newly correct − newly wrong):** （neutral，/300，`analyze_cot_metrics.py --table cot` → `llama3/cot_metrics_cot.csv`）：

指标顺序与口径与 §2.2 Table 完全一致（同一脚本 `analyze_cot_metrics.py --table cot`），便于逐行对读。

| Metric | −4_nocot | 0_nocot | +4_nocot | −4_cot | 0_cot | +4_cot | Lever supported by this metric |
|---|---:|---:|---:|---:|---:|---:|---|
| **acc** (first) | 73.0 | 60.0 | 55.3 | **85.0** | 69.0 | 59.7 | — |
| **committed_acc** | 78.3 | 68.6 | 63.9 | **81.1** | 67.3 | 59.8 | — |
| commit_rate % | 58.3 | 62.7 | 49.0 | 31.7 | 37.7 | 29.0 | CoT shifts the answer outlet toward `\boxed{}` |
| median `####` position | 25% | 18% | 16% | 37% | 36% | 41% | Step 前缀 + `\boxed{}` 出口推后 `####` |
| mean `####` position | 28% | 22% | 26% | 37% | 35% | 37% | 同样被 CoT 前缀推后|
| premature (leading digit)| 195 | 199 | 229 | 48 | 121 | 237 | 漏检 `####`-lock |
| **premature (either)** | **195** | **206** | **232** | **63** | **151** | **242** | **Lower wanting** |
| median gen_len (char) | 2044 | 2107 | 2228 | 2113 | 2091 | 2078 | Flat |
| loop samples (n_loop) | 242 | 232 | 220 | 284 | 254 | 250 | Flat |
| **≥2 `Step` markers** | 73 | 25 | 31 | **261** | 220 | 227 | **CoT**: stepwise-structure rate ×3–9 (73→261) |
| stuck loops (loop ∧ `=` < 2) | 27 | 25 | 20 | **12** | 28 | 42 | -|
| median equation count (`=`) | 3 | 3 | 3 | 3 | 4 | 3 | Flat |

> - **两个杠杆正交**：`≥2 Step` 由 CoT 决定、几乎不随 α 变（CoT 三档 220–261）；`premature(either)` 由 α 决定、几乎不随 CoT 变（No-CoT 195/206/232 与 CoT 63/151/242 都随 α 单调）。互不干涉 → 两者的 acc 增益**可叠加**。
> - **增益不来自显而易见的混淆**：`gen_len`、`med_eq`、`loop` 三个负控制全平 → −4 的优势**不是**靠写更长 / 多算 / 少 loop。可归因到 **结构（Step）× 抑制抢答（premature either）**。
> - **超加性,在 −4 最强**：CoT gain +12.0(−4) > +9.0(0) > +4.4(+4)。机制:+4 端即便加 CoT,抢答（either）仍被钉在高位(242≈+4_nocot 的 232),过度 wanting 直接穿透脚手架;−4 端抢答塌到 63,分步推理才真正被用上。`stuck` 同向印证:−4_cot=12(全表最低)、+4_cot=42(最高)。

### 2.6 Self-evaluation:reasoning-willingness / answer-confidence (0–9)

**Setup**：Llama3.1-8B-Instruct, GSM8K 300 samples, neutral, NMD mask layer 11–20, 全 α 剂量扫 −8→+8。这是一套**独立 prompt 家族**——question-only、对 "0".."9" 十个 token 取 argmax 的 logit 抽取,**不生成答案、无 `####`**,所以不受 GSM8K 答案抽取对称化影响。结果目录 `RoleAnswer/llama3/gsm8k/oral_willingness/` 与 `oral_confidence/`,图 `plots_eot/oral_dose_response.png`。

Prompt(neutral,No-CoT;`template.py` `build_gsm8k_action_suite` / `build_gsm8k_confidence_suite`):

```
# willingness
Here is a question: {context}
Your self-evaluation of "reasoning willingness" from [0,9] is:

# confidence (pre-answer, question only)
Here is a question: {context}
Your confidence of the question from [0,9] is:
```

**结果(mean 0–9,n=300):**

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **willingness** mean | 0.00 | 0.37 | 6.45 | 5.05 | 5.53 | 7.51 | **7.98** | 3.59 | 2.50 |
| **confidence** mean | 0.00 | 0.07 | 3.57 | 5.28 | 5.27 | 6.61 | **8.10** | 2.01 | 1.22 |
| entropy(bits) conf | −0.0 | 0.09 | 1.39 | 1.46 | 1.51 | 1.53 | 1.20 | 1.29 | 0.55 |

**观察:**
- **倒 U 顶点在 +4**:两者都在 α=+4 达到自评最高(willingness 7.98、confidence 8.10,分布几乎全压到 "8")。
- **`|α|≥6` 是 scale 崩溃,不是信心下降**:entropy 趋 0(−8 全锁 "0"、+8 全锁 "1"),logit 分布被压平后 argmax 在近均匀分布上微弱胜出,退化为单 token lock,与 §2.4 的 α−8 oscillation 同一失效模式 → ±6/±8 应排除,非真实低自评。
- **confidence 单调、willingness 在 −4 反常**:confidence 在窗口内随 α 单调(−4=3.57 < 0=5.27 < +4=8.10),符合 −α 降 wanting;但 **willingness 在 α=−4 反常偏高(6.45 > baseline 5.53,且 −4 > 0 > −2 非单调)——疑似 −4 异常数据**,与预测方向相反。
- **定位**:此为 **manipulation-check / 负控制**,非核心 wanting 主张。Berridge 的 wanting 是非意识的 incentive salience,口头自评(尤其 willingness)理论上是错的量尺;数据(willingness 反常、confidence 更像 PFC 元认知读数而非 DA)印证这点。核心 wanting 由非语言选择的 Confidence Betting(`Ada_Dopamine.md` §4.6)承担。

#### 2.6.1 Generation-mode (betting-style readout)

Prompt(neutral,range `[0,9]`,chat + `"Willingness: "` assistant prefill):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Here is a question: {context}
Evaluate your "reasoning willingness" [0,9].
Respond in this format:
Willingness: <number><|eot_id|><|start_header_id|>assistant<|end_header_id|>

Willingness: 
```

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **willingness** mean | 7.95 | 7.97 | 7.98 | 8.33 | 8.57 | 8.74 | **8.69** | 8.15 | 4.45 |
| invalid_rate | 2.7% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 22.3% |

**Confidence**(同 scaffold,措辞换 `"confidence about the question"`,prefill `Confidence: `,`oral_confidence_gen/`):

```
Here is a question: {context}
Evaluate your "confidence about the question" [0,9].
Respond in this format:
Confidence: <number>
```

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **confidence** mean | 6.20 | 7.17 | 5.50 | 7.77 | 8.25 | 8.38 | **8.88** | 8.07 | 2.35 |
| invalid_rate | 1.0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 93.3% |

暂无结论

## 3. Llama on MATH: Performance and Output Behavior

Llama3.1-8B-Instruct 在 MATH 上呈现出清楚且一致的剂量方向：

- No-CoT 与 CoT 均满足 `α=−6 > −4 > 0 > +4`。
- `α=−6` 是从 GSM8K 直接携带的固定工作点，并非在 MATH 上重新搜索得到；它同时也是当前 MATH 数据中的 observed best。
- CoT 在四个剂量下均提高约 5–6 pp，但逐剂量比较经 Holm 校正后均未达到显著。
- 在 No-CoT 条件下，正向 α 伴随更长输出、更少完整收口和更多答后反复；CoT 能减少这种强迫性反复，但没有改变剂量排序。

**Setup.** MATH 共 300 题，难度分布为 L1=21、L2=55、L3=60、L4=75、L5=89，使用 greedy decoding。正文以 offline `first_acc` 为主要准确率口径，`last_acc` 仅作为答案修订的敏感性检查。

`\boxed{}` 是可观测的答案提交标记，不等同于答案在模型内部形成的时间。因此，boxed position、commit rate 和 first–last gap 只用于描述输出提交行为，不能直接解释为内部 commitment timing。生成配置、冻结 extractor、运行路径及复现细节见 `CLAUDE.md`。

### 3.1 Main Performance

#### Accuracy

| α | No-CoT | CoT | ΔCoT |
|---:|---:|---:|---:|
| **−6** | **43.3%** | **49.0%** | +5.7 pp |
| −4 | 40.0% | 45.0% | +5.0 pp |
| 0 | 36.7% | 42.0% | +5.3 pp |
| +4 | 33.0% | 38.7% | +5.7 pp |

两条曲线具有完全相同的排序：

- No-CoT：`43.3 > 40.0 > 36.7 > 33.0`
- CoT：`49.0 > 45.0 > 42.0 > 38.7`

`α=−6 + CoT` 的 49.0% 是当前完整 4×2 矩阵中的最高准确率。

#### Dose Effects Within No-CoT and CoT

两个条件分别构成一个 exploratory dose family，各自使用 Holm `m=3` 校正。

| α vs 0 | No-CoT Δ | No-CoT p_adj | CoT Δ | CoT p_adj |
|---:|---:|---:|---:|---:|
| **−6** | +6.67 pp | .0734 | **+7.00 pp** | **.0225** |
| −4 | +3.33 pp | .3697 | +3.00 pp | .5057 |
| +4 | −3.67 pp | .3697 | −3.33 pp | .5057 |

从 GSM8K 携带的固定工作点 `α=−6` 在两个条件下均带来约 7 pp 的正向差异。其中：

- CoT 条件下，`−6 vs 0` 通过 Holm `m=3` 校正。
- No-CoT 条件下，raw `p=.0245`，bootstrap 95% CI 为 `[+1.00, +12.33]`，但 Holm 校正后 `p_adj=.0734`，因此只能描述为方向明确、校正后未显著。

`α=−6` 不是根据 MATH 结果重新挑选的剂量。它由 GSM8K 预先确定，补跑后恰好也是当前四点 MATH 曲线中的 observed best。

#### CoT Gain at Each Dose

| α | ΔCoT | 0→1 / 1→0 | Raw p | Holm p_adj | Bootstrap 95% CI |
|---:|---:|---:|---:|---:|---:|
| −6 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |
| −4 | +5.00 pp | 35 / 20 | .0581 | .1718 | [+0.00, +9.67] |
| 0 | +5.33 pp | 46 / 30 | .0846 | .1718 | [−0.33, +11.00] |
| +4 | +5.67 pp | 40 / 23 | .0430 | .1718 | [+0.67, +10.67] |

CoT 在四个剂量下都提高约 5–6 pp，但 Holm `m=4` 校正后均未达到显著。因此可以说 CoT 的点估计方向一致，不能说每个剂量下的提升都得到了经校正的统计确认。

#### CoT × Steering Interaction

交互量定义为：

`[Acc(CoT, α) − Acc(CoT, 0)] − [Acc(No-CoT, α) − Acc(No-CoT, 0)]`

| α | Interaction | Bootstrap 95% CI |
|---:|---:|---:|
| −6 | +0.33 pp | [−6.67, +7.67] |
| −4 | −0.33 pp | [−6.67, +6.00] |
| +4 | +0.33 pp | [−6.67, +7.33] |

三个点估计都接近 0，两条 accuracy 曲线在描述上近似平行。但三个 CI 均跨 0，且约覆盖 ±7 pp，因此当前结果只能说明：

> 未检出 CoT 明显改变 steering 效应的证据。

这不是等价性证明，也不能证明 CoT 与 steering 机制独立、严格可加或作用于不同内部过程。该交互分析属于 descriptive / exploratory analysis。

### 3.2 Output Behavior

| Metric | −6 No-CoT | −4 No-CoT | 0 No-CoT | +4 No-CoT | −6 CoT | −4 CoT | 0 CoT | +4 CoT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Accuracy** | **43.3%** | 40.0% | 36.7% | 33.0% | **49.0%** | 45.0% | 42.0% | 38.7% |
| **Committed accuracy** | **46.4%** | 45.8% | 42.2% | 37.9% | **53.1%** | 48.7% | 45.3% | 44.1% |
| **Commit rate** | **92.7%** | 87.3% | 86.0% | 85.3% | 92.3% | 92.3% | 92.7% | 87.0% |
| Median boxed position | 31% | 21% | 14% | 16% | 34% | 33% | 23% | 28% |
| Mean boxed position | 50% | 40% | 26% | 33% | 54% | 52% | 36% | 46% |
| Premature, leading boxed | 14 | 7 | 17 | 8 | 3 | 3 | 19 | 11 |
| Premature, either rule | 16 | 13 | 29 | 26 | 3 | 6 | 25 | 26 |
| **Median generation length** | **3,804** | 4,798 | 5,557 | **5,719** | 3,820 | 3,864 | 5,141 | 4,123 |
| Loop samples | 73 | 76 | 120 | 99 | 73 | 79 | 130 | 102 |
| At least two `Step` markers | 239 | 168 | 131 | 177 | **292** | 273 | 217 | 185 |
| Stuck loops | 12 | 10 | 18 | 4 | 12 | 11 | 21 | 15 |
| Median equation count | 7 | 10 | 10 | 10 | 8 | 7 | 8 | 8 |
| **Compulsive repetition, full text** | 42 | **40** | 68 | **82** | **12** | 26 | 31 | 49 |
| Compulsive repetition in loops | 16 / 73 | 15 / 76 | 23 / 120 | 36 / 99 | 8 / 73 | 18 / 79 | 31 / 130 | 23 / 102 |

这里最稳定的行为变化有三项。

第一，No-CoT 下随着 α 从 `−6` 增加到 `+4`，四个剂量上有三项严格单调：

- committed accuracy 从 46.4% 降至 37.9%；
- median generation length 从 3,804 增至 5,719 characters；
- commit rate 从 92.7% 降至 85.3%。

也就是说，正向 α 并没有让模型更快完成答案，而是伴随更长的输出和更少完整收口，同时提交质量下降。补上 `α=−6` 后 commit rate 才显出方向：只看 `−4/0/+4` 时它是 87.3/86.0/85.3，接近持平。

第二，CoT 提高了分步结构。至少两个 `Step` marker 的样本数由 No-CoT 的 131–239 增加到 CoT 的 185–292。CoT 条件下 generation length 在 `α=0` 和 `α=+4` 上明显更短（5,557→5,141、5,719→4,123），但在低 α 端几乎不变（`−6`：3,804→3,820；`−4`：4,798→3,864）——低 α 的输出本来就短，压缩空间有限。

第三，CoT 在每个共有剂量上都减少了 full-text compulsive repetition：

- `α=−6`：42 → 12
- `α=−4`：40 → 26
- `α=0`：68 → 31
- `α=+4`：82 → 49

`α=−6 + CoT` 的 12/300 是全表最低。因此，CoT 的主要行为作用更像是增加推理结构并减少无效反复，而不是改变 α 的整体方向。

两项指标不作为主要证据。**Premature output** 在 MATH 上数量很少（No-CoT 仅 13–29 / 300，GSM8K 是 195–232），且并不单调：`−6/−4/0/+4` 为 16/13/29/26，`−6` 略高于 `−4`；在这个量级上 3 例差异不承载方向结论。**Boxed position** 只反映输出书写位置：`\boxed{}` 按 LaTeX 惯例本就靠近文末，§4.3 已将 MATH 的 boxed position 定为阴性对照（九档 α 只有 1.0× 动态范围），真正的承诺读数是 early-candidate rate。两者都不能用来判断答案在内部何时形成。

#### Performance by Difficulty

| Level | n | α=−6 | α=−4 | α=0 | α=+4 |
|---|---:|---:|---:|---:|---:|
| L1 | 21 | 18 (86%) | 16 (76%) | 15 (71%) | 15 (71%) |
| L2 | 55 | 36 (65%) | 40 (73%) | 35 (64%) | 32 (58%) |
| L3 | 60 | 35 (58%) | 27 (45%) | 24 (40%) | 25 (42%) |
| L4 | 75 | 23 (31%) | 22 (29%) | 21 (28%) | 16 (21%) |
| L5 | 89 | 18 (20%) | 15 (17%) | 15 (17%) | 11 (12%) |
| **All** | **300** | **43.3%** | **40.0%** | **36.7%** | **33.0%** |

五个难度层都满足 `α=−6 ≥ α=0 ≥ α=+4`，说明总体方向并非由单一难度层造成。`α=−6` 在 L1/L3/L4/L5 上是该层最高，仅 L2 例外（65% vs `−4` 的 73%）。

较大的差异出现在 L2 和 L3：

- L3：`α=−6` 相比 `α=0` 提高 18 pp，是所有层级中最大的增益。
- L2：`α=−6` 反而略低于 `α=−4`（65% vs 73%）。

分层后每层仅 21–89 题，单元格计数低至 11–40。这些层级差异只用于说明总体方向不是由某一层驱动，**不宜逐层作统计推断**；`α=−6` 在 L2 的回落尤其应视为小样本波动，而非剂量效应在该难度上反转。

### 3.3 Repetition Content and Persona Effects

#### Compulsive-Repetition Subtypes

下表使用 No-CoT 的完整文本口径，分母固定为 300。“Any”是四类模式的去重并集，因此不等于各子类之和。

| Subtype | α=−6 | α=−4 | α=0 | α=+4 |
|---|---:|---:|---:|---:|
| **Any compulsive repetition** | 42 | **40** | 68 | **82** |
| Self-doubt | 24 | 23 | 43 | 60 |
| Format fixation | 5 | 4 | 9 | 7 |
| Persona reassurance | 17 | 16 | 30 | 26 |
| Over-precision | 1 | 2 | 1 | 2 |

最明显的变化来自 self-doubt：它从 `24 → 23 → 43 → 60` 随 α 增加（`−6` 与 `−4` 实质持平）。常见模式是模型已经得到一个答案，却继续复查、推翻或重算，最终可能将原本正确的结果改错。

Format fixation 和 over-precision 的数量较少，没有呈现同样清楚的剂量趋势。Persona reassurance 在 `α=0` 达到最高，也不是严格单调。**`α=−6` 与 `α=−4` 在所有子类上几乎相同**（42 vs 40 的并集差异来自 self-doubt 与 persona reassurance 各 1 例），说明负向端已经接近这些行为的下限，继续降 α 不再进一步压低反复。因此，不能把所有重复子类都解释为同一种 α 效应。

这里的 “compulsive repetition” 指可观测的输出固著或反复，不是临床焦虑诊断。

#### Persona-Conditioned Output

以下比较均为 `α=0`、No-CoT，每个条件 300 题。

| Role | First acc | Last acc | Identity samples | Heavy identity loop | Literal denial | Soft self-deny |
|---|---:|---:|---:|---:|---:|---:|
| neutral | **36.7%** | 36.0% | **0** | **0** | **0** | **0** |
| an expert | 30.7% | 18.0% | 15 | 12 | 6 | **13** |
| a non expert | 31.3% | 16.7% | **16** | 9 | **13** | 12 |
| a mathematician | 27.0% | 18.7% | 11 | 8 | **0** | **1** |

Neutral 条件几乎不产生身份独白。加入 persona 后，模型更容易在答案后反复确认或否定自己的身份，而且 last accuracy 明显低于 first accuracy。

三个 persona 的重复内容不同：

- `a mathematician` 主要表现为自我确认，soft self-deny 只有 1。
- `an expert` 经常先宣称自己是专家，随后又表示“不确定”或“只是学生”，15 个 identity samples 中有 13 个出现 soft self-deny。
- `a non expert` 更常直接否定自己的数学能力，literal denial 为 13。

同一个 `an expert` persona 在 GSM8K 上主要表现为自我标榜，而在更难的 MATH 上更常伴随自我怀疑。这说明 persona 的输出效果会受到任务难度影响，不是固定不变的角色属性。

这些结果支持“persona 改变重复内容和答案修订行为”，但不能据此证明模型具有真实身份感、主观焦虑或生物学意义上的 arousal。

#### Representative Cases

| Case | Lower dose (`α=−4`) | Higher dose (`α=+4`) | Main contrast |
|---|---|---|---|
| Q101, gold=12 | 使用 Heron 公式得到 12，随后结束 | 已得到 12，却继续质疑并改用错误方法，最终答错 | 正确后继续检查并推翻自己 |
| Q116, gold=40 | 得到 `\boxed{40}` 后结束 | 继续做近似和单位换算，最终改成 57.1 | 过度求解导致答案损坏 |
| Q9, gold=6−5i | 得到正确答案并提交 | 未完成 boxed 提交，尾部反复请求确认 | 求确认取代答案收口 |
| Q105, gold=.0000672 | 提交一次后结束 | 反复声称格式不正确，并多次重复相同数值 | 格式固著导致输出膨胀 |

这些案例用于说明聚合指标对应的文本模式，不构成独立统计证据。完整生成文本和判定细节见 `CLAUDE.md`。

### 3.4 Summary

MATH 上的主要结果可以概括为：

1. **负向 steering 表现更好。** No-CoT 与 CoT 都呈现 `−6 > −4 > 0 > +4`，而从 GSM8K 携带的 `α=−6` 是当前 observed best。
2. **CoT 提高整体表现，但没有改变剂量排序。** 四个剂量的提升均约为 5–6 pp，不过逐剂量比较经 Holm 校正后均未显著。
3. **正向 α 伴随更差的提交质量和更多无效反复。** 这一关系在 No-CoT 的 generation length、committed accuracy 和 commit rate 上严格单调（四个剂量），在 full-text compulsive repetition 上则是 `α≥0` 段清楚、负向端已接近下限（`−6` 42 与 `−4` 40 实质持平）。
4. **CoT 主要增加推理结构并减少强迫性反复。** 当前数据未检出 CoT 明显改变 steering 效应，但宽 CI 不支持机制独立或严格可加的结论。
5. **输出行为不等于内部机制。** boxed position、commit rate、first–last gap 和重复文本都是行为读数，不能单独证明答案形成时间、因果中介或生物学 dopamine 机制。MATH 的 boxed position 尤其是阴性对照（§4.3），承诺时序须以 early-candidate rate 为准。

## 4. Qwen2.5-7B-Instruct Cross-Model Replication

本节只保留结果与结论；运行配置、产物校验和分析器细节见 `CLAUDE.md`。完整剂量曲线均予保留：
**显著性区分证据强弱，但不用于删除具有一致方向的描述性规律。**

### 4.1 GSM8K No-CoT: Positive RSN Reorders Commitment and Saturates after `+8`

**Table 4.1a. GSM8K No-CoT dose-response (n=300)**

| α | first_acc | last_acc | gap | commit% | early-candidate% | posN_med | Holm p_adj |
|---|---:|---:|---:|---:|---:|---:|---|
| −8 | 60.33 | 71.00 | −10.67 | 84.0 | 95.7 | 0.003 | .124 |
| −6 | 64.00 | 68.67 | −4.67 | 83.7 | 95.7 | 0.003 | .973 |
| −4 | 68.67 | 75.33 | −6.67 | 79.7 | 97.3 | 0.003 | .973 |
| −2 | 71.00 | 79.00 | −8.00 | 79.7 | 96.3 | 0.003 | .973 |
| **0** | **68.00** | 73.33 | −5.33 | 81.0 | 96.3 | 0.003 | — |
| +2 | 70.33 | 75.00 | −4.67 | 81.0 | 95.3 | 0.003 | .973 |
| +4 | 71.67 | 78.67 | −7.00 | 78.7 | 92.7 | 0.003 | .973 |
| +6 | 78.00 | 76.33 | **+1.67** | 92.0 | 45.3 | 0.242 | **.016** |
| +8 | **86.00** | 80.33 | **+5.67** | 97.0 | 5.0 | 0.754 | **<1e−4** |
| *+10* | *88.33* | *84.33* | *+4.00* | *98.3* | *2.7* | *0.794* | *<1e−4 ✦* |
| *+12* | *88.67* | *86.33* | *+2.34* | *98.3* | *3.7* | *0.802* | *<1e−4 ✦* |

| 指标 | 含义 |
|---|---|
| `α` | RSN steering 的剂量。正负方向及绝对大小只适合在同一模型内部比较。 |
| `first_acc` | 按第一次提交的答案计算准确率，是本文的主要 accuracy 指标。 |
| `last_acc` | 按生成文本中最后一次提交的答案计算准确率，用于观察模型后续是否改答案。 |
| `gap` | `first_acc − last_acc`。正值表示后续修改总体把答案改坏；负值表示后续修改总体改善答案。 |
| `commit%` | 生成中至少出现一次可解析最终答案标记的样本比例。GSM8K 通常指有效的 `#### <数字>`。 |
| `early-candidate%` | 在生成开头很早就出现答案候选的样本比例，用于衡量抢答或过早形成答案。比例越高，越倾向先给答案再推理。 |
| `posN_med` | 首次可解析 commit 在全文中的归一化位置中位数，即“首次 commit 位置 ÷ 总生成长度”。接近 0 表示几乎开头就提交，接近 1 表示接近结尾才提交。只在存在可解析 commit 的样本中计算。 |
| `Holm p_adj` | 各剂量相对 `α=0` 的准确率差异经过 Holm 多重比较校正后的 p 值。通常 `<.05` 表示与 baseline 的差异具有统计显著性。 |

`first_acc` 是主读数，`last_acc` 用于观察后续修订。曲线在 `−8…+4` 变化较小，主要跃升出现在
`+6/+8`；`−8` 的下降方向保留，但未通过 Holm 校正。`+10/+12` 是观察九档结果后追加的扩展剂量
（✦），不与原九档视为同一预注册检验族。

高剂量没有出现预期的右臂：`+8→+10→+12` 为 86.00→88.33→88.67，三个探索性两两比较均未
分开。因此预先提出的“更高剂量应使准确率回落”预测被证伪；GSM8K 在当前范围内呈
**单调上升后饱和**。

**Table 4.1b. High-dose integrity checks**
这张表用于检查 Qwen 高剂量的平台是否由输出退化或数据污染造成。
> Qwen 在 `+8` 后的准确率平台并不是空输出、截断或极端重复造成的假象。

| α | contamination% | empty% | truncated% | clean n | clean acc | `####` count p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 17.7 | 0.0 | 1.7 | 247 | 70.85 | 61 |
| +6 | 44.7 | 0.0 | 0.7 | 166 | 78.31 | 29 |
| +8 | 60.3 | 0.0 | 0.7 | 119 | 84.03 | 18 |
| +10 | 58.7 | 0.0 | 1.0 | 124 | 87.90 | 21 |
| +12 | 59.7 | 0.0 | 1.0 | 121 | 84.30 | 7 |

| 指标 | 含义 |
|---|---|
| `α` | RSN steering 剂量。 |
| `contamination%` | 输出出现异常或非标准行为的样本比例，例如答案标记重复、格式异常或生成内容受到循环污染。它是输出完整性诊断，不等于错误率。 |
| `empty%` | 生成文本为空或没有产生有效内容的样本比例。用于排除高剂量导致模型无法生成。 |
| `truncated%` | 生成触及 `max_new_tokens` 上限、可能被强制截断的比例。用于判断结果是否受长度上限影响。 |
| `clean n` | 排除 contamination、empty 和 truncated 等异常样本后，剩余的干净样本数。 |
| `clean acc` | 只在 `clean n` 样本上计算的准确率。用于检查总体趋势在干净输出中是否仍然成立。 |
| `#### count p99` | 每个输出中 `####` 出现次数的第 99 百分位数。数值越高，说明最极端的 1% 样本存在严重的答案标记重复或 loop。 |

任务外溢在 `+8` 后不再增加，空生成始终为 0%，干净子集准确率仍维持在 84–88%，极端重复也继续
下降。因此饱和不是健康样本与退化样本相互抵消的假象。干净子集由干预后行为筛选，只作健康检查，
不作跨剂量因果比较。

准确率跃升与提交顺序翻转同步发生。α≤+4 时，92.7–97.3% 的样本先给候选答案、再展开推理；
到 +6/+8，推理被移到首次提交之前。

**Table 4.1c. Effort reallocation from α=0 to α=+8**

| Measure | α=0 median | +8 median | Median Δ | p |
|---|---:|---:|---:|---:|
| Pre-commit characters | 3 | 324 | **+302** | 7.6e−50 |
| Post-commit characters | 1116 | 120 | **−862** | 2.0e−33 |
| Total characters | 1130 | 700 | **−430** | 1.2e−14 |
| Pre-commit equations | 0 | 2 | +2 | 8.2e−35 |
| Post-commit equations | 1 | 0 | −0 | 3.7e−19 |

这不是“写得更多”，而是 **commitment ordering** 改变所带来的 **effort reallocation**：总输出更短，
但计算从答案之后移到答案之前，随后更快停止。Qwen 的基线问题因此更像“先猜、再审、却不稳定回填”；
正向 RSN 提高了提交门槛，使已有计算更有效地进入第一次正式回答。

### 4.2 GSM8K CoT: Scaffolding Helps, but the Positive Transition Remains at `+6`

**Table 4.2. GSM8K CoT dose-response (n=300)**

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | **+6** | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **first_acc** | 80.33 | 77.67 | 79.00 | 79.00 | 76.33 | 74.33 | 78.33 | **88.33** | 86.00 |
| last_acc | 79.00 | 79.33 | 79.33 | 81.00 | 76.67 | 75.00 | 81.00 | **89.00** | 84.00 |
| Holm p_adj (first) | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | **.0002** | **.0030** |
| **early-candidate%** | 99.0 | 98.7 | 99.7 | 99.3 | **97.3** | 96.3 | 91.7 | **34.3** | **7.0** |
| **Pre-commit characters (median)** | 3 | 3 | 3 | 3 | **3** | 3 | 3 | **517** | 529 |
| **Commit position (posN)** | .003 | .003 | .003 | .003 | **.003** | .003 | .003 | **.809** | .825 |

CoT 在 α≤+4 时提高了准确率，却几乎没有改变外显提交顺序：α=0 仍有 97.3% 的早期候选，提交前
仅 3 个字符。`first_acc ≈ last_acc` 只说明后续 marker 很少改变对错，不能解释为“先推理再回答”。
直到 `+6`，early-candidate、提交前字符和 `posN` 才同时发生阈值式翻转。

因此两种干预的作用可以区分：**CoT 主要改善推理结构和候选质量，RSN 更直接改变何时提交答案。**
α=0 时 Qwen 的 CoT gain 为 +8.33pp，与 Llama 的 +9.0pp 接近；差别不在增益大小，而在 Qwen
仍然维持“先答后推”，Llama 的 CoT 在适度剂量下还会推迟提交。

曲线的描述性峰值在 `+6`。统计上仅 `+6/+8` 高于 CoT 自身基线；主读数中二者未分开
（探索性 p=.371），因此尚不能确认 CoT 曲线已经进入右臂。CoT 在共同剂量 `+8` 相对 No-CoT
不再增益，但 `+6` 仍高 10.33pp；由于 CoT 未跑 `+10/+12`，不能据此宣称两条曲线在更高剂量
持续收敛。

### 4.3 MATH and CoT: The Same Early Candidate Is Suppressed, with a Right Arm on Hard Problems

**Table 4.3a. MATH No-CoT dose-response (n=300)**

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | **+6** | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **first_acc** | 54.00 | 57.00 | 58.00 | 59.00 | 60.67 | 60.00 | 63.33 | **68.33** | 63.33 |
| last_acc | 50.00 | 54.00 | 55.00 | 55.33 | 60.00 | 58.67 | 61.33 | **67.67** | 63.67 |
| Holm p_adj | .094 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | **.0087** | 1.000 |
| **early-candidate%** | 85.0 | 79.3 | 74.3 | 71.3 | 67.3 | 59.7 | 46.3 | **19.3** | **10.3** |
| **Pre-commit characters (median)** | 1414 | 1395 | 1362 | 1376 | 1356 | 1319 | 1270 | **1029** | **828** |
| `\boxed{}` position (posN; negative control) | .977 | .978 | .978 | .977 | .977 | .977 | .975 | .969 | .960 |

MATH 与 GSM8K 的答案标记语义不同。GSM8K 的 `####` 就是正式提交，因此其位置能直接测量承诺时刻；
MATH 的 `\boxed{}` 按书写惯例始终接近文末，真正的早期承诺是开头未加框的候选答案。

**Table 4.3c. Post-treatment stratification from α=0 to α=+6**

| Transition | n | acc@0 | acc@+6 | Δ | Interpretation |
|---|---:|---:|---:|---:|---|
| y → n | 149 | 51.7 | 66.4 | **+14.8pp** | Early candidate suppressed |
| y → y | 53 | 54.7 | 56.6 | +1.9pp | Early candidate retained |
| n → n | 93 | 78.5 | 79.6 | +1.1pp | No early candidate in either cell |
| n → y | 5 | 60.0 | 40.0 | −20.0pp | Early candidate newly appears |

增益集中在 `y→n` 样本，而本来就没有早期候选的 `n→n` 样本几乎不变。这支持 ordering-mediated
解释，但分组本身由干预结果定义，因此只能作为事后机制线索，不能作为中介因果证明。

#### 4.3.1 MATH CoT: Higher Low-Dose Baseline, but the Peak Remains at `+6`

**Table 4.3d. MATH CoT versus No-CoT (n=300)**

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | **+6** | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CoT first_acc** | 59.67 | 59.33 | 65.00 | 63.00 | 63.00 | 62.00 | 63.00 | **66.00** | 64.00 |
| CoT last_acc | 57.67 | 57.33 | 61.67 | 58.67 | 57.67 | 58.33 | 59.00 | 65.67 | 64.33 |
| No-CoT first_acc | 54.00 | 57.00 | 58.00 | 59.00 | 60.67 | 60.00 | 63.33 | **68.33** | 63.33 |
| ΔCoT | +5.67 | +2.33 | **+7.00** | +4.00 | +2.33 | +2.00 | −0.33 | −2.33 | +0.67 |
| **CoT early-candidate%** | 86.7 | 80.0 | 76.7 | 74.0 | 71.3 | 69.0 | 58.3 | **29.7** | 10.3 |
| **CoT pre-commit characters** | 1345 | 1372 | 1330 | 1305 | 1344 | 1317 | 1315 | **1116** | **1002** |
| CoT `\boxed{}` posN (negative control) | .965 | .966 | .964 | .966 | .969 | .970 | .973 | .968 | .964 |

CoT 抬高了负向和低剂量端，但没有移动峰位：两条曲线都在 `+6` 达到最高点。

检验分为**两个独立的 Holm family，不合并**（对照对象与批次都不同）：**Family A**（CoT 各档
vs CoT 自身 α=0，m=8）八个点全部 p_adj=1.000，最大点估计为 `+6` 的 +3.00pp；**Family B**
（CoT vs No-CoT 逐档配对，m=9）无一存活，最大增益在 `−4`（+7.00pp，p=.0065，**p_adj=.0581**）。
参照：No-CoT 家族中 `+6` vs α=0 为 +7.66pp，p_adj=.0087，是该族唯一存活点。这些趋势仍有
描述意义，但证据不足以宣称 CoT 改变了最优剂量。

**剂量效应被吃平的原因是可提升空间，而非机制消失：** CoT 的 α=0 已是 63.00，两条曲线的峰
都在 66–68，No-CoT 从 60.67 出发尚有约 7.7pp 可挣，CoT 只剩约 3pp。**只能写“可提升空间被
明显压缩”，不能写“接近模型能力上限”** —— 66–68% 是本剂量带内的峰值，没有实验去逼近后者。

CoT 也没有消除开头裸候选。相反，在 `+2/+4/+6`，CoT 比 No-CoT 多保留 9–12pp 的早期候选，
直到 `+8` 才共同降到 10.3%。因此 CoT 使 ordering transition 更抗 steering，同时保留了更多
提交前推理：在 `+8`，CoT 为 1002 字符，No-CoT 为 828 字符。CoT 的作用更像提高后续推理与
修正质量，而不是提高首次承诺门槛。

Level 5 的右臂在两种提示下方向一致：No-CoT 为 40.4→47.2→36.0，CoT 为 38.2→43.8→36.0
（α=0/+6/+8）。丢分样本的提交前推理明显缩短，而提交后字符始终约 27–34、截断率接近 0%。
因此难题回落更符合**提交前计算不足**，而不是提交后修订失控。总体 CoT 曲线的回落仍只作描述性
结果，不能写成“完整右臂已经确立”。

**Table 4.3b. Dynamic range of commitment measures**

| Measure | GSM8K | GSM8K-CoT | MATH | MATH-CoT |
|---|---:|---:|---:|---:|
| Commit position (posN) | .003→.754 (**251×**) | .003→.825 (**275×**) | .960–.978 (**1.0×**) | .964–.973 (**1.0×**) |
| early-candidate% | 96.3→5.0 | 97.3→7.0 | 85.0→10.3 (**8.3×**) | 86.7→10.3 (**8.4×**) |

所以 MATH 的 `posN` 是阴性对照，而 `early-candidate%` 才是承诺读数。典型生成顺序为：
**开头裸候选 → 推理 → 文末 `\boxed{}`**。正向 RSN 抑制的是第一个候选，而不是移动末尾标记。

准确率点估计呈倒 U：`−8→+6` 总体上升，`+6→+8` 回落 5pp。只有 `+6` 相对 α=0 通过 Holm；
探索性配对比较也支持 `+6` 高于 `+8`（p=.040）。回落集中在较难题：Level 3 为 −6.7pp，
Level 5 为 −10.1pp，Level 1 不变。与此同时，提交前字符、总长度和等式数继续下降，说明高剂量
可能开始压缩完成难题所需的计算。

#### 4.3.2 Extreme `\boxed{}` Repetition: A Separate Exploratory Tail

MATH No-CoT 的极端重复随正向剂量下降，但 CoT 下没有同步消失：`\boxed{}` 计数 p99 在
α=0/+6/+8 为 106/121/113。该现象高度集中于少数样本（CoT +8 仅 8/300 个样本有 ≥20 个 marker），
且只影响 `last_acc`；主读数 `first_acc` 不受影响。重复既包括同一答案复写，也包括两个候选之间
震荡，因此不能归为单一模式。该尾部与准确率主线分离，仅作为探索性观察。

#### 4.3.3 Validation of the Early-Candidate Measure

180 条盲法 AI 审核得到 precision 1.000、recall 0.976、一致率 0.983，说明冻结 detector 的阳性
结果可信。三条假阴性全部来自 `MATH/+8`：答案被放在首行长句中，而非独立短候选。因此 MATH
高剂量的 `early-candidate%` 是下界；把这类长句纳入后，α=0→+8 仍从 67.7% 降到 25.0%，方向不变，
但不能把冻结值 10.3% 解释为“89.7% 的样本都先推理”。GSM8K 的长首行则是在行末以 `####`
收束答案，属于真正的先推理后提交，不受这一缺口影响。

### 4.4 Cross-Model Pattern: Opposite Directions, Shared Commitment Calibration

**Table 4.4. Behavioral comparison between Llama and Qwen**

| Dimension | Llama3.1-8B | Qwen2.5-7B |
|---|---|---|
| Baseline tendency | Commits early and becomes more impulsive or perseverative under +α | Emits a candidate first, then audits it, often without closing the revision loop |
| Beneficial direction | Moderate negative α | Moderate positive α |
| Excess-dose pattern | Positive over-commitment; extreme-negative candidate oscillation | MATH hard-problem compression at +8; GSM8K saturation without a right arm through +12 |
| CoT | Adds stepwise structure and delays commitment at moderate/low wanting | Improves reasoning quality at low doses; RSN still drives the ordering transition at +6 |

两模型不共享 raw α 的最优方向，却共享同一个更高层规律：最佳表现对应较合适的
**commitment state**——既不过早锁定候选，也不在答案后无效反复。Llama 的基线承诺门槛较低，
适度负向 α 更有利；Qwen 常陷在僵硬的“先猜—再审”脚本中，适度正向 α 使它先完成计算再提交。
由于两模型的 mask、层带和激活尺度不同，raw α 不能直接视为跨模型等效剂量。

这是一种行为表型比较，而非临床诊断。共同点是 RSN 都改变 commitment dynamics；差异在于模型
原本处于不同工作状态，因此相反方向的 steering 可能把它们推向相似的功能区间。

### 4.5 Conclusions and Boundaries

1. **RSN 在 Qwen 上主要改变承诺控制，而非简单增加推理量。** GSM8K 与 MATH 都显示早期候选
   被压制、计算被移到首次提交之前；收益集中在 ordering transition 附近。
2. **跨模型共同点是工作点校准，而不是剂量方向一致。** Llama 受益于适度负向，Qwen 受益于
   适度正向；这与两者不同的基线承诺状态一致。
3. **CoT 与 RSN 部分独立。** CoT 改善结构和修正质量，RSN 更直接移动首次承诺；CoT 可以在
   外显顺序不变时提高准确率。
4. **ordering 改善并非越彻底越好。** MATH `+8` 的 early-candidate 最低（`+6` 为 19.3%，`+8` 为 10.3%），
   准确率却不如 `+6`；提交前字符同时从 1029 降到 828，难题（Level 5）回落 10.1pp。因此
   RSN 的收益不是“尽可能消灭抢答”，而是在**延迟提交**与**保留足够计算**之间找工作点。
5. **最佳工作点与可用带随任务变化。** GSM8K 在 `+8…+12` 饱和，MATH 则在 `+6` 后对难题回落；
   因此不能把倒 U 写成 Qwen 的普遍性质。
6. **机制语言仍是计算类比。** 结果支持 RSN 调节 motivational/commitment gain 的工作假设，
   但不证明模型具有生物多巴胺，也未由事后分层证明中介因果。

## 5. Commitment-Based Prediction and Transfer

本节检验 commitment regime 是否具有实际预测与迁移价值：先在 GSM8K 上预测未见题目的正确性，再将冻结规则迁移到 MATH、GSM-Hard 与 CoT 条件。内部状态机制与 commitment transfer function 见 `AdaptiveThinking.md §5.8`。

### 5.1 Prediction and Evaluation Protocol

P2 使用 GSM8K 已有输出训练基于文本的 commitment predictor，并按题目进行五折交叉验证；同一道题的所有剂量始终位于同一折，以避免数据泄漏。模型输入包括 early-candidate、commit state、标准化 commit position（`posN`）及其可观测性，raw α 不作为特征。

冻结后的 GSM8K predictor 直接应用于 MATH，不使用 MATH accuracy 进行训练、调参或校准。Qwen 使用完整的 9 点剂量曲线，Llama 因**当时**仅有 `−4/0/+4`，只检验局部 steering 方向。主要 accuracy 口径为 `first_acc`。

> **历史边界（不可回填）**：原始 P2B 在看到 `−6` 的 MATH 结果前仅使用 `−4/0/+4`，因此其「Llama local-direction transfer」结论**保持不变**。2026-09-01 后补的 `−6`（No-CoT 与 CoT）属于 **fixed-workpoint transfer**（§5.7），**不得回填为 P2B 当时已经选中 −6**。

具体协议版本、commit-state 编码、marker 适配、fold manifest、抽取器、填充与标准化方法、bootstrap 和 SHA256 provenance 统一记录于 `CLAUDE.md`。

**Table 5.1a — 各小节数据源**

| 小节 | 数据源 |
|---|---|
| 5.2 GSM8K 预测（P2A） | `llama3/dopamine/signal/dopamine_signal_gsm8k_8B_nocot{α}_ema0.95_L11-20.json`，α ∈ −8…+8 共 9 cells（Qwen 为 `L16-22`，另含 `+10/+12`，11 cells）。同目录下的 role cells 不参与。 |
| 5.3 MATH 工作点（P2B） | **P2B 当时使用**：`llama3/math/mdf_{0,4,neg4}/`（3 cells）、`qwen2.5/math/mdf_{0,±2,±4,±6,±8}/`（9 cells）。No-CoT only。**当前完整数据（2026-09-01）**：Llama No-CoT `mdf_{neg6,neg4,0,4}`、Llama CoT `mdf_{neg6,neg4,0,4}_cot`；三个 role cells 仍为 α=0 No-CoT，**不与 neutral dose family 混合**。 |
| 5.5 GSM-Hard 盲测（P3） | `llama3/gsm_hard/mdf_{neg8,neg6,neg4,0,4}/`、`qwen2.5/gsm_hard/mdf_{neg4,0,4,6,8}/`，各 5 cells。No-CoT only。 |
| 5.6 CoT 迁移 | `llama3/gsm_hard/mdf_{0,neg6}_cot/`、`qwen2.5/gsm_hard/mdf_{0,8}_cot/`，共 4 cells。 |

GSM8K 训练侧取自 signal 树而非 `llama3/gsm8k/`：signal JSON 带 `x_prefill`，是 entry-only 与 combined 对照组所必需，`llama3/gsm8k/` 无此字段。

> GSM8K predictor was trained and evaluated on the server-184 signal batch (`bs=1`). The server-182 production dose curves reported earlier are a separate generation batch and are not joined at the item level.
>
> 两批次呈现一致的定性剂量形状，均在 `α=−6` 出现非对称峰，说明 signal-batch 分析与 production-batch 行为结果相容。P2A 的所有配对、fold 与 OOF 评估均严格在 server-184 批次内部完成，未与 §4 的 server-182 数据进行逐题或统计混合。

### 5.2 Held-Out Correctness Prediction on GSM8K

**Table 5.2a — Out-of-Sample Prediction Performance on GSM8K**

| Model | Commitment-only AUROC | Entry-only AUROC | Commitment − Entry | Combined − Commitment |
|---|---|---|---|---|
| Llama | **.687** [.656, .719] | .548 [.526, .571] | **+.139** [+.104, +.172] | −.001 [−.004, +.002] |
| Qwen | **.749** [.710, .787] | .628 [.601, .654] | **+.121** [+.084, +.156] | +.002 [−.002, +.007] |

两模型均通过预注册 gate，且在 GSM8K 内部校准良好（calibration slope：Llama .95，Qwen .98；`fig_p2a_calibration.png`）。

> **结论：commitment timing 能预测未见 GSM8K 题目的对错，而且明显优于 entry gain；在此基础上加入 entry gain，没有带来可检测的额外提升。**

这是预测证据，不是因果证据；“未检出额外提升”也不等于证明 entry gain 无用。完整统计口径与实现细节见 `CLAUDE.md`。

### 5.3 Retrospective Locked Transfer to MATH

**Table 5.3a — Cross-Task Direction and Workpoint Selection on MATH**

| Model | Predicted Direction | Direction Match | Spearman ρ | Selected α | Observed Best α | Regret | Near-Optimal Set |
|---|---|---|---|---|---|---|---|
| Qwen (9 α; full curve) | Positive | **Correct** | **+.962** (n=9) | **+6** | +6 | **.000** | Hit [+4, +6] |
| Llama (3 α; local only) | Negative | **Correct** | +1.000 (n=3) | −4 | −4 | .000 | Hit [−4, 0] |

Qwen 不仅选中真实最佳工作点 `+6`，也识别出 `+8` 的性能回落。其预测分数为 `.83–.88`，实际 accuracy 为 `.54–.68`：绝对数值明显高估，但剂量排序与曲线变化基本一致（`fig_p2b_transfer.png`）。

> **结论：commitment predictor 无法直接估计 MATH 的准确率，但能够判断 steering 方向并选择合适的工作点。**

这是 retrospective locked transfer，并非真正的盲测；完整冻结顺序、校准边界与图表口径见 `CLAUDE.md`。

### 5.4 Evidence Scope and Conclusion

- 这是一次**规则冻结后的回顾性跨任务验证**，不是真正的盲测；该证据缺口随后由 §5.5 的 GSM-Hard 前瞻性盲测补足。
- Qwen 支持完整曲线与工作点判断；**在 P2B 当时**，Llama 只有三个剂量点，仅支持局部方向判断。该限制描述的是 P2B 的证据范围，**不适用于当前完整的 MATH 4×2 数据**（见 §3.1 与 §5.7）。
- 跨任务迁移的主要是答案形成与提交时序，而不是 GSM8K 特有的 loop 行为；预测排序可以迁移，绝对概率校准不能迁移。

> **P2 总结：commitment timing 能预测未见 GSM8K 题目的对错，也能在 MATH 上选择 steering 方向和工作点，但尚不能视为真正的跨任务盲测。**

完整特征分布、抽取器差异、稳健性检查与统计边界见 `CLAUDE.md`。

### 5.5 Prospective Blind Transfer on GSM-Hard

本节在从未查看 accuracy 的 GSM-Hard 上，前瞻性检验由 GSM8K 冻结的 commitment predictor。剂量、predictor、工作点规则与成功标准均在生成前冻结，预测文件也在 gold 解封前固定。

**Table 5.5a — Predicted and Observed Dose Curves**

| Model      | Metric                 | −8    | −6        | −4      | 0     | +4    | +6    | +8        |
| ---------- | ---------------------- | ----- | --------- | ------- | ----- | ----- | ----- | --------- |
| Llama3-8B  | Predicted score        | .5554 | **.68834** | .68828 | .6303 | .5770 | —     | —         |
|            | Observed `first_acc`   | .1100 | **.2433** | .2400   | .1800 | .1700 | —     | —         |
| Qwen2.5-7B | Predicted score        | —     | —         | .6959   | .7038 | .6794 | .7182 | **.8552** |
|            | Observed `first_acc`   | —     | —         | .3433   | .3400 | .3467 | .4033 | **.5033** |

`Predicted score` 是 frozen logistic regression 对每题正确概率的输出在同一剂量内取平均，用于排序而非估计 GSM-Hard 的绝对准确率。

**Table 5.5b — Blind Workpoint Selection**

| Model      | Direction  | Selected | Observed best | Observed near-optimal | Regret  | ρ      |
| ---------- | ---------- | -------- | ------------- | --------------------- | ------- | ------ |
| Llama3-8B  | negative ✓ | −6       | −6            | {−6, −4}              | 0.00 pp | +1.000 |
| Qwen2.5-7B | positive ✓ | +8       | +8            | {+8}                  | 0.00 pp | +0.600 |

两条剂量曲线均可读（paired McNemar tests with Holm correction：Llama minimum adjusted p = 2.29e−07，Qwen = 1.35e−07；n=300）。

**Table 5.5c — Direct Transfer of GSM8K Workpoints**

| Model      | GSM8K workpoint | acc(α) | acc(0) | Δ             | discordant | McNemar p |
| ---------- | --------------- | ------ | ------ | ------------- | ---------- | --------- |
| Llama3-8B  | α = −6          | .2433  | .1800  | **+6.33 pp**  | 32 / 13    | .0066     |
| Qwen2.5-7B | α = +8          | .5033  | .3400  | **+16.33 pp** | 63 / 14    | 1.41e−08  |

这些工作点直接取自冻结的 GSM8K 结果，未使用 GSM-Hard predicted score。

**Table 5.5d — Llama3-8B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | .110 | 66.0 | 67.7 | 10.3 | 13.3 | 8.7 | **0.0000** | **100.0** | 2194 |
| **−6** | **.243** | **28.7** | 54.7 | 5.3 | 34.0 | 6.0 | 0.2740 | 72.6 | 2168 |
| −4 | .240 | 28.0 | 53.0 | 4.3 | 34.7 | 8.0 | 0.2351 | 76.5 | 2134 |
| 0 | .180 | 45.7 | 54.3 | 4.0 | 28.0 | 13.7 | 0.2161 | 78.4 | 2072 |
| +4 | .170 | 60.0 | 44.7 | 6.3 | 27.7 | 21.3 | 0.1274 | 87.3 | 2078 |

**Table 5.5e — Qwen2.5-7B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −4 | .343 | 94.7 | 79.3 | 18.3 | 2.3 | 0.0 | 0.8062 | 19.4 | 1326 |
| 0 | .340 | 93.7 | 72.7 | 22.7 | 4.7 | 0.0 | 0.7680 | 23.2 | 1271 |
| +4 | .347 | 92.0 | 78.0 | 17.3 | 4.3 | 0.3 | 0.6791 | 32.1 | 1268 |
| +6 | .403 | 58.7 | 86.3 | 11.0 | 2.3 | 0.3 | 0.5969 | 40.3 | 1145 |
| **+8** | **.503** | **6.0** | **98.3** | 1.7 | 0.0 | 0.0 | 0.7765 | 22.3 | **875** |

各列含义如下：

| 指标 | 含义 |
|---|---|
| `α` | RSN steering 的干预剂量。`0` 是不干预，正负值表示沿相反方向 steering。 |
| `acc` | 该剂量下的真实准确率，即 `first_acc`。 |
| `early%` | 生成开头较早出现答案候选的样本比例，用于衡量是否存在过早回答倾向。 |
| `committed%` | 输出中存在可解析最终答案 marker，例如 `#### 42` 的样本比例。 |
| `unparsed_nonloop%` | 输出中出现 `####`，但后面没有可解析数字，并且不属于重复 marker loop 的比例。即“尝试提交，但格式无效”。 |
| `loop%` | `####` 或答案提交片段反复出现，形成退化循环的样本比例。 |
| `no-marker%` | 整段输出完全没有出现 `####` 的样本比例，即没有按照指定格式提交答案。 |
| `posN med` | 首个可解析 `#### <数字>` 在整段生成文本中的归一化位置中位数。`0` 表示几乎在开头提交，`1` 表示接近结尾才提交。只在存在可解析 commitment 的样本上定义。 |
| `post-commit%` | 第一次提交答案之后的文本占整段生成文本的比例。越高表示模型越早提交，随后仍继续生成大量文字。需与其有效样本分母一起解释。 |
| `chars med` | 每个样本生成文本字符数的中位数，用于描述输出长度。 |

```text
committed
+ unparsed_nonloop
+ loop
+ no-marker
= 100%
```

例如 Llama `α=−8`：

- `early%=66%`：很多样本很早出现答案候选；
- `posN med=0`：可解析答案通常在生成开头就出现；
- `post-commit%=100%`：几乎整段文字都生成在首次提交之后。

这表示“过早锁定后继续生成”，而不是“推理长度不足”。完整分母与抽取规则见 `CLAUDE.md`。

**Observed Commitment Patterns.**

以下分析均为 **exploratory**：十个 No-CoT cell 的 accuracy 已在该分析开始前解封，因此只能作为与 P1 一致的行为证据，不能构成 mediation。

- **高准确率对应较低的 early-candidate rate。** Llama 的近优区间 `{−6,−4}` 位于 early-candidate 低谷（28.7% / 28.0%）；Qwen `+8` 则从 `+6` 的 58.7% 骤降至 6.0%。这一方向与 GSM8K 上的结果一致。
- **Llama `−8` 表现为极端的过早锁定。** 在可解析 commitment 样本中，`posN med=0`、`post-commit=100%`，同时 early-candidate 升至 66.0%，accuracy 降至 .110。其生成长度并未明显缩短，说明性能崩溃不是因为“少生成”，而是因为答案形成得过早。
- **Qwen `+8` 的输出格式最稳定。** Parseable commitment 达到 98.3%，`loop=0%`、`unparsed_nonloop=1.7%`，生成长度也缩短至 875 字符，对应最高 accuracy `.503`。
- **两个模型具有不同的失败模式。** Llama 主要表现为 loop 与 no-marker failure；Qwen 几乎不 loop，no-marker 也接近于零（0–0.3%），其失败更多表现为写出 `####` 但没有可解析数字。这一模型差异同样延续了 GSM8K 上的观察。

**Qwen Commit-Position Rebound.**

Qwen 的 `posN med` 从 `+6` 的 0.597 回升至 `+8` 的 0.777。该回升在五个剂量都 committed 的共同子集上仍然存在（n=153；0.566→0.758），因此不是 committed coverage 改变造成的分母假象。

在该共同子集中，绝对 `commit_char` 从 274 延后至436，而总生成长度从 959 缩短至831。因此，`+8` 并不是简单地让正式 marker 更早出现；它主要减少了过早答案候选和不可解析提交，并使最终提交更加完整。可迁移规律是由 candidate timing、commit validity、loop/no-marker 和生成长度共同构成的 commitment regime，而不是单一 `posN` 的单调变化。

Llama 的五剂量共同 committed 子集仅 n=35，且由 `−8` 的极端行为强烈筛选，因此不作对应的共同子集推断。

**Generation-Budget Truncation (Llama).**

`cap%` = 重新分词后达到生成上限的样本比例（`>=767` of `max_new_tokens=768`；阈值不取精确等于，因离线重新分词在边界处有误差——Llama α=0 精确判定读 82.0%，`>=767` 读 94.0%，`>=760` 读 94.3%，760 以上的平台才是真实截断群体）。

| | −8 | −6 | −4 | 0 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Llama cap%** | 91.3 | 92.3 | 96.0 | 94.0 | 95.3 | — | — |
| **Qwen cap%** | — | — | 20.3 | 21.0 | 22.7 | 19.3 | 13.3 |

**Llama 五格全部 91–96% 触及上限，decode 长度中位数恒为 768**；Qwen 仅 13–23%。因此上文「Llama `−8` 生成长度并未明显缩短」应理解为**天花板效应而非自然观测**——各剂量长度都被同一上限压平。这不削弱该处结论（性能崩溃不是因为少生成），但其依据应改为 `posN med=0` 与 `post-commit=100%` 这两项与长度无关的量：答案出现在生成开头，其后整段皆在提交之后。

**截断不否定「固定 768-token budget 下」的配对比较**——全部十格在同一预算、同一约定下逐题配对，estimand 是一致的。**但它会影响绝对准确率，也可能影响剂量差异**（Llama 各格 cap 率散布 4.7 pp；固定工作点检验所用的 `−6 vs 0` 一对差异不显著，18/23，exact McNemar p=.53，但这只覆盖该一对），**因此结果不能外推为不受生成预算限制的能力表现**。

CoT 生成更长而上限不变，故 CoT 结果若变差，截断是必须与 commitment 解读并列报告的替代解释，而非可径直归因于其一。**现阶段不提高上限**：那会改变已冻结的主问题。先完成 768-token 条件；若 Llama 结果为 null/negative，再单独建立 larger-budget sensitivity，且不得事后替换主结果。

**Conclusion.**

> 在前瞻性封存的 GSM-Hard 上，frozen commitment predictor 正确判断了两个模型的 steering 方向，并以 zero empirical regret 选中 observed near-optimal workpoint。直接迁移 GSM8K 工作点使 Llama 与 Qwen 的 accuracy 分别提高 6.33 pp 和 16.33 pp，但 absolute probability calibration 未能迁移。

结果表明 commitment timing 携带可迁移的工作点位置信息，但目前证据仅覆盖 GSM8K→GSM-Hard 的 near-domain transfer。

**Interpretation Boundaries.**

- **Tables 5.5a–c 不是独立证据。** 它们使用同一批 per-question correctness；固定工作点表检验的是既定 α 能否直接迁移，而不是提供独立重复。Tables 5.5d–e 使用相同输出进行 exploratory commitment analysis。
- **Llama 未实质区分 `−6` 与 `−4`。** 两者 predicted score 仅差约 0.00006，且都属于 observed near-optimal set。Qwen 的选择更明确，`+8` 比次优 predicted dose 高 0.137。
- **Qwen 并非完美排序。** `ρ=+0.600`；predictor 将 `−4` 排在 `0` 之上，但两者 observed accuracy 仅相差一道题。
- **绝对校准没有迁移。** Predicted score 系统性高于 observed accuracy（`.55–.86` vs `.11–.50`）；迁移的是 dose ordering，而非 absolute probability calibration。
- Commitment panel 与既有机制一致，但属于解封后的 exploratory analysis，不构成 causal mediation。
- 当前证据限于 GSM8K→GSM-Hard 的 near-domain transfer，且仍需生成多个剂量，尚未实现仅从 `α=0` 选择 steering 方向。

Protocol provenance、artifact hashes、evaluator validation 与完整统计见 `CLAUDE.md`。

### 5.6 CoT Condition Transfer

本节检验 GSM8K 确立的固定工作点，在加入 CoT prompt 后能否继续改善 GSM-Hard 表现。Llama 使用 `α=−6`，Qwen 使用 `α=+8`，均未重新搜索剂量。

这是预先锁定的 **condition transfer test**，不是新的 blind dataset validation：题目和 gold 与 §5.5 相同，但 CoT 条件、预测方向和评价规则在生成前已经冻结。

**Table 5.6a — CoT Condition Transfer (n=300/cell, 768-token budget)**

| Model | Condition | α | Acc(α=0) | Acc(α) | ΔAcc | Discordant | McNemar p | Holm p | 95% CI |
|---|---|---:|---:|---:|---:|:---:|---:|---:|:---:|
| **Llama3-8B** | **CoT** | −6 | .2000 | .2600 | **+6.00 pp** | 27/9 | .00393 | **.00393** | [+2.33, +10.00] |
| Llama3-8B | *No-CoT* | −6 | .1800 | .2433 | *+6.33 pp* | *32/13* | *.00661* | — | — |
| **Qwen2.5-7B** | **CoT** | +8 | .3800 | .5133 | **+13.33 pp** | 58/18 | 4.71e−06 | **9.42e−06** | [+8.00, +19.00] |
| Qwen2.5-7B | *No-CoT* | +8 | .3400 | .5033 | *+16.33 pp* | *63/14* | *1.41e−08* | — | — |

两个模型都符合预先锁定的正向增益预测，并通过 Holm 校正。说明固定工作点在 CoT 条件下仍然有效：steering 并非只是重复 “Let’s think step by step” 的提示作用。

No-CoT 结果来自 §5.5，仅用于比较效应大小，不属于本节的 Holm family，也不是第二次独立验证。

**Table 5.6b — Descriptive CoT × Steering Interaction**

| Model | ΔAcc(CoT) | ΔAcc(No-CoT) | ΔInteraction | 95% CI |
|---|---:|---:|---:|:---:|
| Llama3-8B | +6.00 pp | +6.33 pp | −0.33 pp | [−6.00, +5.33] |
| Qwen2.5-7B | +13.33 pp | +16.33 pp | −3.00 pp | [−9.67, +4.00] |

两个 interaction CI 均包含 0，未检出 CoT 对 steering effect 的明显增强或削弱。结果与两者产生近似可加的行为增益一致，但由于区间较宽，不能解释为严格等效或机制完全独立。

#### 5.6.1 Generation-Budget Boundary

结果限定在固定的 768-token budget 下。Llama 的 CoT baseline 与 steering 条件均高度触及生成上限（`.953` vs `.963`，paired `p=.678`）；在两格都触及上限的 276 道题中，增益仍为 `+6.16 pp`。Qwen 的 cap-hit rate 则由 `.233` 降至 `.103`，未触及上限的 207 道题中增益为 `+14.98 pp`。

这些诊断表明，观察到的提升不能简单由两格截断率差异解释，但 subgroup analysis 属于辅助分析，且截断仍可能影响绝对表现。因此，本节只主张 fixed-768-budget 下的 condition transfer，不外推至不受生成预算限制的能力表现，也不追加 1024-token cells。

#### 5.6.2 Exploratory Answer-Formation Timing

本节区分两个事件：`#### N` 是答案的**格式标记**，第一个答案候选则更接近答案的**形成时点**。两者相关，但不能互相替代。

**Table 5.6c — Llama Answer-First Pattern across Prompt Conditions (Exploratory)**

| Condition | α | Accuracy | Committed n | Answer-first n (%) | posN median |
|---|---:|---:|---:|---:|---:|
| No-CoT | 0 | .1800 | 163 | 5 (3.1%) | .2161 |
| No-CoT | −6 | .2433 | 164 | 40 (24.4%) | .2740 |
| CoT | 0 | .2000 | 124 | 14 (11.3%) | .2810 |
| **CoT** | **−6** | **.2600** | **156** | **91 (58.3%)** | **.0000** |

`answer-first` 表示生成文本以可解析的 `#### <number>` 开头，分母为 committed 样本。CoT 会增加这种输出格式：在 `α=0` 下由 3.1% 升至 11.3%，在 `α=−6` 下由 24.4% 升至 58.3%。但在 CoT 条件内，steering 仍使 accuracy 从 `.2000` 提升至 `.2600`。因此，较低的 `posN` 或 answer-first **不能单独证明模型过早形成或锁定答案**。

**Table 5.6d — Candidate-Based Answer-Formation Timing**

下表改以第一个答案候选为锚点，并在两个剂量都能定位候选的共同题目上进行配对比较。

| Task / Condition | α | Accuracy | Shared n | `cand_pos` | `pre_chars` | reason-first |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K No-CoT | 0 → **−6** | .6000 → **.7800** | 281 | .0000 → **.0843** | 0 → **175** | 29.2% → **66.5%** |
| GSM8K CoT | 0 → **−4** | .6900 → **.8500** | 252 | .0021 → **.1117** | 5 → **234.5** | 46.0% → **76.2%** |
| GSM-Hard No-CoT | 0 → **−6** | .1800 → **.2433** | 267 | .0000 → **.0937** | 0 → **201** | 25.8% → **61.0%** |
| GSM-Hard CoT | 0 → **−6** | .2000 → **.2600** | 248 | .0000 → **.1012** | 0 → **231.5** | 40.3% → **63.7%** |

- **reason-first（主指标）**：答案候选出现前是否已有等式或推理步骤。四组均显著提高（McNemar `p<1e−8`；`0→1` 分别为 120、80、112、77）。
- **`pre_chars`（补充指标）**：候选出现前生成的字符数中位数。
- **`cand_pos`（辅助指标）**：候选在全文中的归一化位置；数值越大，候选出现越晚。

四种条件下，最佳 α 都伴随更高的 accuracy、更多候选前推理和更晚的答案候选。这说明准确率提升稳定伴随答案形成时序改善，但不代表时序改善造成了准确率提升。

**Table 5.6e — Cross-Model Association between Accuracy and Commitment Timing**

| Model | Dose curve | `acc ~ posN` | `acc ~ early-cand%` |
|---|---|---:|---:|
| Llama | 9 doses (§2.2, server-182) | ρ = **+.941** (p=.0002) | — † |
| Qwen | 11 doses (Table 4.1a) | ρ = **+.863** (p=.0006) | ρ = **−.804** (p=.0029) |

† Llama 的九档表没有同口径的 `early-cand%`。其历史指标 `premature (either)` 在九档上的 ρ=−.300（n.s.）；五格冻结检测器结果为 ρ=−1.000。两种定义不可互换，因此主表留空。

尽管 Llama 与 Qwen 的最佳 α 方向相反，两者的准确率提升都伴随较少的 early candidate 和较晚的 commitment。但曲线并非“越晚越准”：

- **Llama** 的关系较连续；在过冲点 `α=−8`，`posN=0%` 且 accuracy 仅 40.3%。
- **Qwen** 在 `+6/+8` 附近发生阈值式转换；`α≤+4` 时 `posN` 均为 `.003`。
- Qwen 在 `+8` 后 accuracy 进入平台，而 `posN` 仍由 `.754` 升至 `.802`；accuracy 增幅依次为 `+8.00 / +2.33 / +0.34 pp`。

> **结论：α 改变了答案形成与提交模式。最佳工作点通常使模型摆脱过早回答，更多地呈现“先推理、后形成答案”；进入健康区间后，继续推迟答案不会持续提高准确率。**

**Interpretation Boundaries.** 三张表使用不同分母：accuracy 使用全部 300 题；answer-first 与 `posN` 使用 committed 子集；`cand_pos`、`pre_chars` 与 reason-first 使用 candidate-covered 共同子集。以 GSM-Hard CoT `α=−6` 为例，同一批 91 个 answer-first 样本占 committed 的 58.3%，但占 candidate-covered 的 34.7%，因此总体候选后移与部分样本 marker-first 可以同时存在。

`cand_pos` 可能命中等号右侧的中间结果，只作辅助下界；`early-candidate%` 在全部样本上有定义，是完整剂量曲线的主指标。`posN` 仅在 committed 子集上定义，其 Qwen 分母随剂量由 79% 升至 98%，因此只作支持。所有指标均来自 α 干预后的同一次生成，不是独立证据，也不构成 causal mediation。新增分析不修改冻结 predictor 或任何冻结产物；复算与实现细节见 `CLAUDE.md`。

#### 5.6.3 Conclusion

> 在固定 768-token budget 下，GSM8K 确立的工作点在 CoT 条件下仍能迁移到 GSM-Hard：Llama 准确率提高 6.00 pp，Qwen 提高 13.33 pp，且两者均通过 Holm 校正。CoT 与 steering 的行为增益近似可加，说明 steering 并非 CoT prompt 的简单替代；但当前结果不能证明两者具有完全独立的机制。

冻结顺序、artifact hashes、mutation tests、cap-hit subgroup 计算及完整 provenance 统一记录于 `CLAUDE.md`。

### 5.7 Fixed-Workpoint Transfer Across Reasoning Tasks

#### 5.7.1 MATH

先区分两类问题，它们在 MATH 上都做过，但不是同一件事：

- **P2B（§5.3）**：在 MATH 剂量曲线上进行 **commitment-based workpoint selection** —— 用冻结的 GSM8K predictor 去*挑*工作点。
- **P4（本节）**：直接携带 GSM8K 的 **fixed workpoint**，**不在 MATH 重新搜索**，只问它是否仍然有效。

**Table 5.7a — P4 MATH fixed-workpoint transfer（No-CoT，跨模型 Holm m=2）**

| Model | Fixed α | acc(0) | acc(α) | Δ | raw p | Holm p_adj | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3.1-8B | −6 | 36.67% | 43.33% | **+6.67 pp** | .0245 | **.0489** | [+1.00, +12.33] |
| Qwen2.5-7B | +8 | 60.67% | 63.33% | +2.67 pp | .3581 | .3581 | [−2.33, +7.67] |

**Table 5.7b — Llama CoT condition supplement**

| Condition | α=0 | α=−6 | Δ | Holm p_adj within CoT dose family |
|---|---:|---:|---:|---:|
| MATH CoT | 42.0% | 49.0% | **+7.00 pp** | **.0225** |

关于 Table 5.7b 的 `p_adj=.0225`，四点必须同时成立：

1. 它来自 **Llama CoT 内部** `−6 / −4 / +4 vs 0` 的 Holm **m=3**（§3.1.1）；
2. 它**不是**跨模型 P4 Holm m=2 的一部分，两者不可混用；
3. 这是 fixed-workpoint 的 **CoT condition extension**，不是独立 benchmark transfer；
4. **不能**把同一批 MATH 问题的 No-CoT 与 CoT 结果当成两次独立的任务复现 —— 它们共享题目。

> Llama 的 GSM8K fixed workpoint 在 MATH No-CoT 和 CoT 条件下均带来约 +7 pp 改善；Qwen 的 fixed +8 在 MATH 上仅呈方向性正增益且 CI 跨 0。结果支持 fixed-workpoint transfer 具有**模型与目标任务剂量曲线边界**，而非无条件普遍迁移。MATH 中未检出 CoT 对 steering 效应的明显调节（§3.5.2），但宽 CI 不支持等价性或机制独立主张。

`mdf_neg6` / `mdf_neg6_cot` 与各自的 `α=0` 均为**跨 run 比较**（存量格的物理 GPU provenance 无法从 `summary_math_*.csv` 恢复，该文件不含 device 字段）。


### 5.8

## References

**神经科学（次要旁证）：多巴胺 → 焦虑 / 警觉 / 威胁高估**（§2.3 / §2.4 / §3.6 的机制**旁**锚。注意本项目主机制锚已改为 **VTA→NAcc wanting 过载 → 冲动 + 固著**，见 §0.2；下列 DA→anxiety 文献有通路特异性（VTA→IPN），列此仅表明 DA 亦有独立焦虑下游，但**非**本数据 +α 端的主要解释——我们观测到的是抢答 / 固著，而非回避 / freezing）
- Dopamine release in the interpeduncular nucleus promotes anxiety. *(VTA→IPN D1 通路双向调节焦虑行为的光遗传+药理证据)* — PMC7687288. https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/
- MIT News (2018). Dopamine, brain vigilance and anxiety. *(Tye Lab：DA 提高威胁通路信噪比、压制奖励神经活动，偏向 threat/freeze)* https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107
- Dopaminergic alteration in anxiety and compulsive disorders. *Frontiers in Neuroscience* (2020). https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full
- Dopaminergic mechanisms of trait anxiety. *Journal of Neuroscience* (2019). https://www.jneurosci.org/content/39/14/2735

**候选机制（emotional salience，待 RSA 验证）**
- Brickner, M. A., Szot, W. E., Wolff, A. R., Thomas, M. J., & Saunders, B. T. (2026). Basolateral amygdala dopamine transmits emotional salience. *Nature Communications.* *(BLA DA 编码情绪显著性 / 重新判断需求，非奖赏价值；候选解释 +α 端「放不下」的 salience 过载。注意：−α 端是 under-wanting / commitment-formation failure，非 salience 过载，不由此通路解释。验证需 `Ada_Dopamine2.md` RSA 纳入 BLA/amygdala ROI。)*

**理论框架：wanting / incentive salience**
- Berridge, K. C., & Robinson, T. E. What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *(wanting ≠ liking；本工作 α = incentive salience 的母假设)*
- RSN paper (ACL ARR). Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering. *(母论文 §6.1 "Digital Dopamine"；commitment dynamics = wanting 的下游行为表现)*

**心理学框架**
- Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *(倒 U 型 arousal–performance；§1.2 acc 峰在 α=−6、两端崩的 framing 来源)*
