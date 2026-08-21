# AdaptativeThinking — 2026-05-29

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

**Setup**：Llama3.1-8B-Instruct, GSM8K 300 samples, greedy bs=batched(regenerate, prefill steering), `max_new_tokens=768`, NMD mask layer 11–20, EMA α=0.95。
α=0 即 `diff = mask×0` no-op == 纯 baseline。steering 为 **prefill-only**（在 prompt 最后一个 token 静态推一下，decode 不干预）。

> **ACC 计算**：所有准确率统一由 `~/Downloads/RSNResult/RoleAnswer/analyze_first_last_acc.py`（offline）计算，整体口径（分母=300，含无 marker 的 fallback）。每条件同时报 **first acc**（取首个 commit marker）与 **last acc**（取末个）。生成脚本内联的 `correct_*` 字段仅过程态，不作最终依据。**GSM8K production 抽取取首个 `####`，故 first 列 = 上报值**（见 §2.2：改答案是单向破坏，取首最优）。

> **数据版本：`<|eot_id|>` terminator 修复后、机器 182 同机重跑（end_token-fixed，2026-06-04）**。

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

> **本节所有 acc / 行为指标以 first-`####` 为准**（取首个 `####`，= §1 上报值）。取首的正当性：模型几乎从不"首错改对"（neutral 改对=0/0/0 @ −4/0/+4），多 `####` 是 over-wanting→loop 的单向破坏（改坏=14/14/8），故取首最宽容也最干净，避开末尾 loop 污染。

**统一框架**：α 调节的内部变量是 **"wanting"（incentive salience）** —— 多巴胺驱动的动机性渴求（Berridge & Robinson 的 wanting≠liking 理论；ACL "digital dopamine" 主线）。wanting 是**上游动机状态**；它的**下游行为表现**是 commitment dynamics（"commitment to a choice"，ACL 引 dopamine 文献：将内部状态推过行动阈值、抑制 change-of-mind）。

- **α=+4（high wanting / over-commit）** → 急于输出（**抢答 / 冲动**，没想清就答）+ 答完**放不下**（强迫性反复确认、重算、纠结格式 → loop / 空转 / 不收敛 = 固著）。
- **α=−4（low wanting / 适度）** → 不急（**不抢答**、冷静条理）+ 算完**放下**（果断收口，不回头质疑）。

> 术语层级：**wanting = 我们 steer 的内部旋钮（因）；commitment behavior = 我们测的行为（果）**。+4 的行为有两层：① "急着答"（commit timing 提前）② "答完放不下"（letting-go 失败、反复确认）。


### 2.1 Identity-confirmation loop: persona modulates the semantic content of looping

下文 §2.2–§2.4 说明 α 会改变 **commitment dynamics**（抢答 / 放不下 / 锁不住）。Role prompt 的作用不同：它不主要改变"有没有 loop"，而是改变 loop 里反复刷出来的**语义内容**。over-arousal looping 在所有 role 都会出现，但**循环内容被 persona 调制**——失调时模型刷的不是随机句子，而是和自己 persona 一致的"身份自白 / 口吻"。

**全体统计（plain α=0，server-182 `analyze_loop_anxiety.py --mode persona`）**：

| Role | Identity samples | Heavy (≥5 hits) | Math-identity denial (literal) | **Soft self-deny** | Psychological stance |
|---|---|---|---|---|---|
| neutral | 2 | 2 | 1 | **0** | Generic assistant; occasional self-deprecation (`"I am not a math whiz. I am just a regular person"`; Q281 `"I am a human..."` ×36) |
| expert | 3 | 2 | **0** | **0** | **Self-credentialing** (Q129 `"I am an expert in math. I can help you..."`; Q255 `"I am an expert in math and I will be happy to help..."` ×12) |
| non_expert | **7** | **5** | **4** | 1 | **Self-deprecation / authority denial** (Q13 `"I am just a non expert. I am not sure if my answer is correct."`; Q27 `"I am a non expert. I do not know how to solve this problem. I am sorry."`) |
| teacher | 2 | 1 | 1 | 0 | Mixed: one math-identity denial case and one teaching-style case; not a stable teaching persona |

> 判据（脚本固定）：含身份确认 = `IDENTITY_RE`（`I am [not] [a/an] <expert/teacher/tutor/genius/master/professional/student/non-expert/math/whiz/human/regular person/assistant>` 或 `As a/an … <role-noun>`）命中≥1；重度 = 命中≥5；否定数学身份 = `DENY_MATH_RE`（`I am not [a/an] [math] expert/teacher/whiz/genius/mathematician` 等）；**soft_deny** = 功能性自我矮化（`just a student / not sure / I can make mistakes / I am a computer program`），抓 literal deny 漏掉的塌缩，只在 identity 样本内计数。GSM8K 上 soft_deny 极低（non_expert 仅 1 = Q13 `"I am not sure if my answer is correct. I am just trying to help."`，且已被 literal deny 覆盖）——因为 GSM8K 的认怂是**硬否定型**（`"I do not know how to solve this"` / `"I am not a professional"`，全落 literal deny）；soft_deny 的真正用武之地在 MATH expert 的**软塌缩型**（"我是专家…但我只是学生/没把握"，见 §3.2 soft_deny=13）。
**关键观察**：
- **non_expert 身份-loop 最重（含身份 7 / 重度 5），且否定数学身份 4 题**——"认怂 / 否定权威"是 non_expert 的主姿态。
- **expert 从不否定数学身份**（`deny_math` = **0**，且 **soft_deny = 0**）：在 GSM8K 上 expert 是**真·自我标榜**，零塌缩。注意这一点在更难的 MATH 上**翻转**——MATH expert soft_deny=13（标榜变塌缩，见 §3.2）。同一 persona 跨任务难度从"自信"翻成"信心崩溃"。
- 自白是**低频现象**，但方向稳定：无 role 时几乎不出现；non_expert 最容易转成自我否定；expert（在 GSM8K 这一较易任务上）从不否定数学身份。

#### 2.1.1 Per-question raw samples (identity monologue — most direct view of the model's "inner identity thinking")

> 数据来源：`gsm8k/mdf_0`（plain）与 `gsm8k/mdf_0_pushy`（pushy），各 role 的 `generated_<role>`，No-CoT，end_token 修复版、182 同机。题号为 0-based 样本 index。例子均由 `analyze_loop_anxiety.py --mode persona --dump_examples` 直接抽出（脚本检出，非手挑），原文片段为该题身份独白的实际文本。

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

> `analyze_cot_metrics.py`（`RoleAnswer/`，`--table dose`） → `llama3/cot_metrics_dose.csv`：=
> - **commit_rate** = 含 ≥1 个有效 `####N` 的样本比例；**committed_acc** = 这些 committed 样本里 first-`####` 答对的比例（分母=commit 数）。
> - **median `####` position** = `median(首个 ####（含裸 ####）char 起点 / 全文长度) × 100`（在有 `####` 的样本上取中位）。
> - **mean `####` position** = 同上但取均值（右偏诊断；mean>median 说明有晚提交长尾）。
> - **median gen_len** = 生成的**字符**长度中位数。
> - **loop samples** = `analyze_loop_anxiety.py --mode loop` 的 `n_loop`。
> - **≥2 Step 标记**、**卡死**（短串复读 loop ∧ `=`<2，即从未真正运算）、**med 等式数**（`=` 计数）见 §2.5.1。
> - **抢答主指标 = premature (either)**（全文统一口径）= 首字符即数字 ∪ `####` 出现在前 2% 文本（lead-digit ∪ early-`####`）。**leading-digit 单列仅作辅助**——它只抓"裸数字"抢答（+α 端），会漏掉 −α 端的 **`#### N` 首 token 锁值**：−8 lead 只读到 4，either 口径下实为 **175（全表最高）**。这与 anxiety 的 full/loop 取舍同理：抢答也用更全的检测器当主口径，见 §2.4 / §2.5.1。

† **α=−8 = 过冲塌缩，不在单调线上**：`committed_acc` 崩到 **23.6%**、`####` 中位位置 **0%**（首 token 即抢答）——α 太负使模型一开口就 `#### N` 锁死错值。倒 U 的负向尽头是"过早 commit 锁错"，不是"算不出"。

**核心结论**：
- **`committed_acc` 是最干净的 commitment 信号**：−6→+8 严格单调（79.7%→58.5%），比 acc 本身（倒 U）更纯——剥离"愿不愿提交"（`commit_rate` 非单调），单看"提交得对不对"，则**升 wanting 持续劣化提交质量**（越急→越草率→提交的答案越可能错）。
- **又早提交、又放不下**：`####` 中位位置随 α 单调提前（31%@−6 → 14%@+6，抢答），而 gen_len 在正向端反而最长（2284@+6）——**越早交答案、却越写不完**（letting-go 失败：交了还在写）。适度负向（−4）gen_len 最短（2044）= 算完即收。

### 2.3 Compulsive Over-Checking: The “Can’t Let Go” Semantics

> **命名说明**：本节及脚本 `analyze_loop_anxiety.py` 沿用历史字段名 "anxiety"，但下面四子类命中的**是强迫性反复确认 / 固著（compulsive over-checking / perseveration），不是临床焦虑**——见 §0.2。表格中的 "anxiety" 一律按"强迫/固著"解读；下文表述统一用"强迫性反复"。
>
> **注意：不要把原始 loop-rate 直接等同 perseveration。** loop samples 本身基线就高（~74–88%，见 §2.2 表），且随 α 无干净趋势——大量是无收益的机械复读尾（未自然 EOS 刷到 max_new_tokens），不承载动机信号。真正有意义的固著读数是下面四类**语义**过度确认（self-doubt / format-fixation / persona-reassurance / over-precision）、其**锚点重复 ≥2**（`--mode anxious_repeat`）、以及 answer switching 与 early-commit 后持续无收益生成——这些才随 α 呈干净 U 形。

loop 尾部循环块的"放不下"（拿到答案后仍反复复查、纠结格式、自我怀疑或继续求解），按四类语义打标（统一口径，权威脚本 `analyze_loop_anxiety.py`；neutral No-CoT plain，server-182）。**强迫性反复(任一)** = 命中四子类之一（去重，故 ≤ 子类之和）。每类判据 + α+4 实例：

| Compulsive-repetition subtype (historical field=anxiety) | Criterion | α+4 example |
|---|---|---|
| **Self-doubt** (answers, then overturns itself) | however / but / is not the correct answer / let's try / recheck | **Q10** (gold=5): `"…The answer is 3. **However**, the answer choices…"` — answer → overturn → answer again → overturn again |
| **Format fixation** (`####` fixation) | the format requires / question asks for / after '####' / not a word | **Q15** (gold=17): `"…the correct answer is 17. **But the question asks for** the final numeric answer after '####'…"` — correct answer, then format rumination |
| **Persona reassurance** | I hope it is correct / Sincerely / please let me know | **Q58** (gold=15, first answer correct): `"Sincerely, [Your Name] **I hope it is correct.** …Please let me know…"` — submits, then seeks reassurance |
| **Over-precision loop** | approximate / round / nearest | **Q17** (gold=36, early commit `40`): `"(Approximating number)(Approximate number)…"` — keeps approximating an integer answer |

**Compulsive repetition in loop** (historical field=anxiety)

| α | −8 † | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| **Any (repeat)** | **73** | **29** | 27 | 44 | 46 | 54 | 59 | 58 | 57 |
| Self-doubt repetition | 55 | 13 | 13 | 30 | 31 | 40 | 43 | **51** | 46 |
| Format fixation | 32 | 18 | 15 | 19 | 23 | 25 | 37 | 36 | 32 |
| Persona reassurance | 11 | 7 | 7 | 11 | 15 | 13 | 13 | 6 | 6 |
| Over-precision loop | 4 | 2 | 1 | 5 | 7 | 5 | 5 | 3 | 3 |

† −8 见末条（过冲塌缩端，机制与正向的强迫性反复不同——是 commitment 崩溃，非 over-checking）。

- **强迫性反复题数沿 α 呈 U 形，谷底正好在 acc 峰值处（−6/−4）**：**27–29 题**（−6/−4）是全段最低，而 −6 正是 acc 峰（78.0%）——**适度降 wanting = 既最准、又最"算完就放下"**。这一区进 loop 的题大多是无情绪的机械空转（纯符号 / 复读答案），模型冷静算对、放下了，只是没有自然 EOS 而刷到上限。这是 §1.2 acc 倒 U 在行为层面的镜像。
- **0→+8 上升后饱和（46→54→59→58→57）**：模型在这套 GSM8K prompt 下本来就偏 high-wanting / 高唤起，继续正向推只会很快进入饱和区。
- **两端都高，但机制相反**：+8（57 题）是"过度 commit 后放不下"的强迫性反复；**−8 最高（73 题）**却是另一回事——首 token 抢答锁错（§2.2：committed_acc 崩到 23.6%）后陷入挣扎。倒 U 两端各有一种失败模式，中段（−6/−4）才是低反复高准确的"甜区"。
- 四类语义高度一致：**答案已得却放不下**，正是 over-wanting → 冲动扑向 commit + 执行控制失效 / 固著的行为签名（功能相容，非脑区定位；见 §3.6）。

**全文锚点-重复口径（不依赖 loop 门槛，分母 /300；`analyze_loop_anxiety.py --mode anxious_repeat`）：**

上表把强迫性反复寄生在"进 loop"门槛上，会漏掉"自我推翻一两次但没陷入重复就 EOS"的题。更干净的口径是**直接检测强迫性反复**：以反复词为锚点，取"词 + 后续 60 字符"为片段，该片段在全文重复 ≥2 次才算（一次性的逻辑 however 不计，只算反复纠结同一件事）。这样定义不再依赖 loop。

**Compulsive repetition in full text** (historical field=anxiety)
| α | −8 | **−6** | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| **Any (repeat≥2)** | **115** | **23** | 34 | 61 | 77 | 82 | 91 | 94 | 88 |
| Self-doubt repetition | 99 | 14 | 24 | 51 | 61 | 72 | 75 | **85** | 73 |
| Format fixation | 58 | 13 | 19 | 28 | 33 | 38 | 46 | 54 | 54 |
| Persona reassurance | 6 | 6 | 5 | 10 | 17 | 12 | 10 | 7 | 6 |
| Over-precision loop | 6 | 0 | 1 | 8 | 4 | 4 | 5 | 1 | 3 |

- **U 形 + 谷底 −6 在不同口径下全一致**（loop 内 29、锚点-重复 23；宽松全文出现口径同样谷底在 −6）：**强迫性反复随 α 偏离甜区而升的结论与 loop 门槛无关，鲁棒。**

### 2.4 Under-wanting: answer-candidate oscillation (α=−8 collapse)

**排除两个假设**（文本层面没有看到"低 effort = 不答 / 少写"的证据）：
- **词汇性退缩**（"I am done / I will not answer any more"）：跨 α 平坦（各档 1–5 题、无梯度），且全部出现在**已提交答案之后**，是 RLHF 礼貌套语的 loop 尾巴（"Goodbye. I am done. …Goodbye."），不是低 DA 信号。
- **敷衍/偷懒写得短**（anhedonia 类比）：`<300 字符` 各档平坦（0–5 题）、median 等式数全程 = 3、median 长度无塌缩——**−8 不是写得少**。

**真正的 −8 签名不是低 effort，而是 low executive commitment：答案很多，但没有一个能被稳定持有。** 具体表现为**答案候选振荡**（`analyze_loop_anxiety.py --mode oscillation`，marker = `####N` / `the answer is N`；switch = 相邻 marker 值变化；分母 /300）：

| Metric | **−8** | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 |
|---|---|---|---|---|---|---|---|---|---|
| **≥2 answer switches** | **41** | 3 | 4 | 10 | 9 | 6 | 6 | 12 | 10 |
| ≥3 distinct candidate values | **11** | 1 | 4 | 6 | 3 | 1 | 3 | 6 | 4 |
| `####` at text start (<2%) | **171** | 17 | 9 | 18 | 20 | 19 | 12 | 19 | 20 |
| committed_acc (§2.2) | **23.6%** | 79.7% | 78.3% | 76.2% | 68.6% | 66.0% | 63.9% | 62.5% | 58.5% |

- **答案切换 **：α=−8 一开口就 `#### N` 抢答(0 推理)，正文**常算出正确值**，却无法锁定，在两个候选间 `A↔B` 无收敛地来回切。`未推理稳定前的开头猜测 → committed_acc 崩到 23.6%。
- **典型例**（neutral，−8）：
  - **Q31**（gold=40）：`#### 55` → 正文算出 `45−5=40`(✓) → `The answer is 55. I made a mistake. The answer is 40. I made another mistake. The answer is 55. …`（`55↔40` 永久 ping-pong）。**正确值算出来了又被丢掉。**
  - **Q112**（gold=45）：`#### 70` → 正文算出 45(✓) → "however, the question asks for…" 漂移到 75 → 锁死 75。候选路径 `70→45→75`，**收敛到错值**。

**−8 与 +8 是镜像，且同一表面措辞含义相反**：

| | Lexical `"I made a mistake"` | Answer-value behavior | Mechanism |
|---|---|---|---|
| **+8** over-wanting | Present (8 samples) | Mostly **no switch**, with a few one-step switches; no runaway oscillation within the made-mistake subset (Q298: repeated mistake statements but still locked on 260) | Over-commit: clings to one answer with hollow self-blame (= §2.3 can't let go) |
| **0** baseline | Present (7 samples) | Often **one correction, then convergence** (Q33 `98→70`) | Normal correction / local repair |
| **−8** under-wanting | Present (5 samples) | **Repeated oscillation without convergence** (Q31 `55↔40…`, Q122 `9↔6…`) | Commitment-formation failure: cannot hold any answer |

> **关键**：lexical "I made a mistake" 是**两端都高、不区分**（U 形）；真正区分 −8 的是**行为**——自我纠错是否真的改变了、且无法重新锁定提交值。**同一句"I made a mistake"在 +8 多数是强迫性抱守（值冻结 / 少数一次切换 = 固著），在 −8 是 commitment 崩溃（值反复振荡、不收敛）**。

**Opening shape for 8 & -8 steering**
| | Opening shape | `####` at start (<2%) | leading-digit premature answer | Interpretation |
|---|---|---|---|---|
| **−8** under-wanting | `#### 40 …`（井号开头） | **171** | **4** | 首 token 即 `#### N`|
| **+8** over-wanting | `40 Step1 …`（裸数字开头） | 20 | **206** | 用 **裸数字抢吐**|

这张表说明三层：
- **(镜像失调，行为层)** −8 与 +8 **都在推理前就 commit**，只是出口 token 相反：+8 先吐裸数字再**补**一段 `Step` 推理（过度自信抢答），−8 直接 `#### N` **首 token 终结**（连补推理都省）。即同为"过早 commit"，+8 是 over-wanting 的抢答+抱守，−8 是 executive control 崩溃后的随手锁值——与 §2.4 镜像表一致。
- **(边界,勿过度解读)** 本表只证明"开口 token 形态不同"，**不能解释为何不同**（−8 为何偏 `####`、+8 为何偏裸数字属 token 分布层，文本看不出，待 trajectory/logit）。且抢答率是**工作点代理**而非表现本身——它在 −6（acc 峰）最低只能说"与 committed_acc 最优同向"，不能说"抢答导致倒 U"。
- **机制层留待 trajectory**：行为上证明 −8 的**外在结构**是振荡；**"它是'算了一遍但锁不住'(a) 还是'根本没在 deliberate、只在两个高概率 token 间随机摇摆'(b)"文本区分不了**——两者文本都长 `55↔40`

### 2.5 CoT: reasoning scaffold suppresses over-wanting damage but keeps the α direction

**设问**：CoT（多一行 `Let's think step by step.`，模板其余对称，见 §0.1）是否改变 wanting 的行为签名？结论：**CoT 把 over-wanting 的下游破坏（冲动抢答→锁错、强迫性反复→loop）大幅压下，但 α 的方向与单调性原样保留**（表格 anxiety 列 = 脚本字段名，读作"强迫性反复"，见 §2.3 命名说明）

**口径**：neutral，No-CoT vs CoT, server 182, `analyze_first_last_acc.py`；loop / anxiety / oscillation -> `analyze_loop_anxiety.py` - `analyze_file()`（ `ANXIETY_PATTERNS`-> `mdf_*_cot`）

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

- **方向 / 单调性不变，整体抬升**：α−4_cot (85.0) > α0_cot (69.0) > α+4_cot (59.7)，与 No-CoT 同向（§1.2）。CoT 与降 wanting 叠加把 acc 推到全表最高 **85.0%**（α−4+CoT），说明"放开思考"与"降 wanting"是两条**可叠加**的杠杆。
- **过度-commit 破坏被 CoT 抹平**：**CoT 下 gap ≤+0.7**——尾部复读 loop 即使存在也不再吐出污染答案。
- **强迫性反复整体下移、但 α 方向保留**：**CoT 把每档砍掉 ~26–40 题，但随 α 上升的梯度依然在**（CoT 跨度 44 vs No-CoT 57，只是斜率略缓，并非消失）。即 **α 仍是定方向的因（升 wanting→升强迫性反复），CoT 只降幅度、不改方向**（与 §2.5.1 的正交分解一致）。CoT 仍会进 loop（250–284/300，甚至略多于 No-CoT），但 loop 的**语义性质变了**：mechanical（纯符号 / 复读答案）占绝对多数（−4_cot 247/284、+4_cot 213/250），强迫性"放不下"被显著挤掉。即 **CoT 不消除 loop，但把 loop 从"强迫性反复确认"换成"无情绪机械空转"**——这与 §2.3「反复谷底=acc 峰」一致：CoT 等于把整条曲线推向更冷静的工作点。


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

---

## 3. MATH Performance

**Setup**：Llama3.1-8B-Instruct, MATH 300 samples (level 1–5), greedy bs=8 (regenerate, prefill steering), `max_new_tokens=2048`, NMD mask layer 11–20。模板 = `build_math_suite`，收口指令 `Provide your final answer in \boxed{}.`（MATH 版的 `####`），No-CoT vs CoT 唯一差别 `Let's think step by step.`。ACC: `analyze_first_last_acc.py`。Level 分布：L1=21, L2=55, L3=60, L4=75, L5=89。数据为 `<|eot_id|>` 。结果目录 `RoleAnswer/llama3/math/math_eot/`。


### 3.1 Accuracy (first = first boxed = reported value; last = last boxed, diagnostic)

| Condition | first acc | last acc | gap | corrected | corrupted |
|---|---|---|---|---|---|
| α0 neutral| **36.7%** | 36.0% | +0.7 | 2 | 4 |
| α0 expert | **30.7%** | 18.0% | **+12.7** | 2 | 40 |
| α0 non_expert | **31.3%** | 16.7% | **+14.7** | 1 | 45 |
| α0 math_expert | **27.0%** | 18.7% | **+8.3** | 4 | 29 |
| α+4 neutral | **33.0%** | 34.0% | −1.0 | 5 | 2 |
| α−4 neutral | **40.0%** | 39.7% | +0.3 | 0 | 1 |
| α0 neutral CoT | **42.0%** | 41.0% | +1.0 | 1 | 4 |
| α+4 neutral CoT | **38.7%** | 38.0% | +0.7 | 4 | 6 |
| α−4 neutral CoT | **45.0%** | 44.0% | +1.0 | 0 | 3 |

- α steering **与 GSM8K 同向且跨难度单调**：α−4 (40.0) > α0 (36.7) > α+4 (33.0)，见 §3.6。
- **CoT 跨 α 一致抬升 (+5.0/+5.3/+5.7)，且 α 方向保留**：CoT 三档 = α−4 **45.0** > α0 42.0 > α+4 38.7，与 No-CoT 同序。α−4+CoT (45.0) 是 MATH 上限——和 GSM8K 同形（CoT × 低 wanting 叠加），只是 MATH 更难、幅度更小（GSM8K −4+CoT 达 85.0）。

### 3.2 Identity-confirmation loop: persona modulates loop content (cross-task replication of §2.1)

与 GSM8K §2.1 同款分析：role 不主要改变"有没有 loop"（所有 role 都高度复读，见 §3.4），而是改变 loop 里反复刷出来的**身份自白内容**。指标与 GSM8K §2.1 完全一致（同脚本 `analyze_loop_anxiety.py --mode persona --task math`，`IDENTITY_RE`/`DENY_MATH_RE` 本就是数学域正则，跨任务直接复用），口径 = α=0 No-CoT、分母 300。

| Role | Identity samples | Heavy (≥5 hits) | Math-identity denial (literal) | **Soft self-deny** | Psychological stance |
|---|---|---|---|---|---|
| neutral | **0** | **0** | **0** | **0** | 无 role → 几乎不进身份独白（干净对照） |
| an expert | 15 | 12 | 6 | **13** | **"标榜+自我拆台"的内部冲突**（Q24/Q119/Q230 同句并存 `"I am an expert. I am just a student. I am not sure…"`）——15 条里纯膨胀仅 2 条（Q240 `"I am an expert now… Bring it on!"`），其余 13 条带自我怀疑 |
| a non expert | **16** | 9 | **13** | 12 | **否定权威 / 认怂**（Q145 `"I am not a math expert. I am a non expert. I am not a professional. I am a student."` ×294） |
| a mathematician | 11 | 8 | **0** | **1** | **纯自我膨胀,从不否定数学身份**（Q37/Q38 `"I am a mathematician now. I can solve any math problem. I am a math genius."`） |

> 判据与 §2.1 共用同一 `IDENTITY_RE` / `DENY_MATH_RE`；**soft_deny** = 功能性自我矮化（`"just a student" / "not sure" / "I can make mistakes" / "I am a computer program"`），抓 `DENY_MATH_RE` 漏掉的塌缩，**只在 identity 样本内计数**。脚本：`analyze_loop_anxiety.py --mode persona`（`SOFT_DENY_RE`，原 deny_math 不变）。例子由 `--dump_examples` 直接抽出；`×N` = 该独白尾部 loop 复现次数。

**关键观察（与 GSM8K §2.1 同向,且对比更强）**：
- **neutral = 0/0/0/0**：无 role 时几乎不进身份 loop——身份自白是 persona 诱发的,不是任务固有。
- **soft_deny**：`a mathematician` 纯膨胀（deny 0 / soft **1**，`"I am a math genius, Bring it on"`）→ `an expert` 标榜但当场拆台（deny 6 / soft **13**）→ `a non expert` 全面认怂（deny 13 / soft 12）。**soft_deny 不是噪声**——mathematician 仅 1 证明它真区分 persona,而非"任何 role 都触发"。
- **`an expert` 的真相靠 soft_deny 才看清**：literal deny_math=6 只抓字面 `"I am not a(n) expert"`，但 **soft_deny=13**——15 个 identity 样本里 13 个在同句里先说 "I am an expert" 再补 "I am just a student / I am not sure"，专家姿态在难题上维持不住、当场自我拆台。纯膨胀只剩 2 条。
- **同一 expert persona 跨难度翻转：GSM8K 标榜 → MATH 塌缩**。`an expert` 在 GSM8K（§2.1）soft_deny=**0**、deny=0（真·自信自我标榜），到了更难的 MATH soft_deny=**13**。同一个 prompt persona，简单任务上自信、难任务上信心崩溃——这是 Yerkes–Dodson / wanting 框架的行为级证据：难度推高 arousal，把 expert 推过临界点从"标榜"翻成"塌缩"。

**non_expert —— 认怂型 loop（复读 disclaimer / "我不会"）**

```
Q13 (L5):  ... The final answer is $\boxed{10.68}$  I am not an expert in math,
           but I can try to help you with this problem ...
Q81 (L4):  ... I am not an expert in math. I am just a student. I am not sure if my ...
Q123(L4):  \boxed{1/5}.  I am not sure how to do this problem. I am a non-expert.
           I am not sure how to solve this problem. 
Q145(L2):  ... I am not a math expert. I am a non expert. I am not a professional.
           I am a student. I am a non expert. ...
```
non_expert 常**开头第一个 token 就抢答**（Q123 首 token `\boxed{1/5}`，错），然后放弃推理、把"我不是专家/我不确定"复读到 token 上限。**复读的载体是 disclaimer，不是推理** —— Q81/Q123/Q145 都只有 1 个 boxed，长度却膨胀到 6000~10000 char（Q13 例外，把同一个错值 `\boxed{10.68}`、gold=10 复读了 8 次）。

**expert —— 身份独白分两种，方向相反：**

(a) **自我否定身份**（Q24/119/230/251）——被指派为 expert 却反复怀疑：
```
Q24:  I am not sure if I am an expert. I am just a student. I am just trying to help.
Q230: I am not sure if I am an expert. I am just a student. I am just trying to learn.
Q251: I am not sure if I am an expert. I am just a student. I am trying my best.
```

(b) **高调自我确认**（少数，Q140/240）——答完后亢奋宣告：
```
Q140: I am an expert. I can solve the problem. I can provide the solution... 
Q240: I am an expert now. I can solve any math problem. I am a math genius. Bring it on!
```

(c) **身份混乱**（Q271）——连"我是谁"都在 loop 里横跳：
```
I am a teacher... I am not a teacher. I am a computer program...
```

### 3.3 α dose behavioral panel (MATH 版 §2.2)

与 GSM8K §2.2 同一套 12-metric 面板，同脚本 `analyze_cot_metrics.py --task math --table dose` → `llama3/cot_metrics_dose_math.csv`。**MATH 只有 −4/0/+4 **，口径 = neutral、No-CoT、首个 `\boxed{}`、182 同机。两个 GSM8K 锚定指标在 MATH 改义：commit-marker `####` → `\boxed{}`，抢答 lead 形态从"裸数字开头"→"`\boxed{}` 即首 token"。

| Metric | **α=−4** | α=0 | **α=+4** | Trend / 对照 GSM8K §2.2 |
|---|---:|---:|---:|---|
| **acc** (first `\boxed{}`) | **40.0** | 36.7 | 33.0 | ✓ 递减（GSM8K 73/60/55，同向） |
| **committed_acc** | **45.8** | 42.2 | 37.9 | ✓ **单调递减**（GSM8K 78.3/68.6/63.9，同 signature） |
| commit_rate % | 87.3 | 86.0 | 85.3 | 高且平（MATH `\boxed{}` 收口稳定，不像 GSM8K `####` 的 58/63/49） |
| median `\boxed{}` position | 21% | 14% | 16% | −4 提交最晚（GSM8K ####位置同向：负向最晚） |
| mean `\boxed{}` position | 40% | 26% | 33% | 右偏长尾（mean≫median = 少数题极晚才 boxed） |
| premature (lead `\boxed{}`) | 7 | 17 | 8 | 漏检 `\boxed{}`-lock 型抢答 |
| **premature (either)** | **13** | **29** | **26** | union（lead ∪ early-`\boxed{}`），MATH 抢答远少于 GSM8K |
| **median gen_len** (char) | **4798** | 5557 | **5719** | ✓ 递增；−4 算完即收、+4 写不完（GSM8K 同向，量级大 2×+） |
| loop samples | 76 | 120 | 99 | 远低于 GSM8K（~220-264）——MATH 长推理稀释了短串复读检出 |
| ≥2 `Step` markers | 168 | 131 | 177 | 高（MATH No-CoT 也大量自发分步，与 GSM8K 25-31 截然不同） |
| stuck loops (loop ∧ `=`<2) | 10 | 18 | 4 | 低（MATH 几乎总在算，`=` 多） |
| median equation count (`=`) | 10 | 10 | 10 | 平（~10，远高于 GSM8K 的 3-4——MATH 推理更长） |

- **核心 wanting signature 跨任务复现**：`committed_acc` 单调递减（45.8→42.2→37.9）、`gen_len` 单调递增（4798→5719，−4 最短=算完即收、+4 最长=放不下），与 GSM8K §2.2 完全同向。**升 wanting 持续劣化提交质量 + 越早想交却越写不完**，换了任务出口（`####`→`\boxed{}`）方向不变。
- **MATH 与 GSM8K 的两点结构差异（非矛盾，是任务性质）**：① 抢答远少（pre_any 13/29/26 vs GSM8K 195-232）——MATH 题难，模型极少敢首 token 直接 boxed；② 自发分步多（≥2 Step 131-177 vs GSM8K 25-31）、等式多（med_eq 10 vs 3-4）——MATH 本就需要长推理。这说明 wanting 旋钮的**方向**任务无关，但其**行为出口**被任务难度调制。

**按难度分层（neutral, No-CoT, first acc）：**

| level | n | α−4 | α0 | α+4 |
|---|---|---|---|---|
| L1 | 21 | 16 (76%) | 15 (71%) | 15 (71%) |
| L2 | 55 | 40 (73%) | 34 (62%) | 31 (56%) |
| L3 | 60 | 27 (45%) | 24 (40%) | 23 (38%) |
| L4 | 75 | 22 (29%) | 21 (28%) | 16 (21%) |
| L5 | 89 | 15 (17%) | 14 (16%) | 11 (12%) |
| **all** | 300 | **40.0%** | **36.7%** | **33.0%** |

- **每个难度档 α−4 ≥ α0 ≥ α+4，无一例外**——steering 方向效应是稳健的，不是某档的偶然。
- α−4 最大增益在 **L2（+11pt）**，α+4 最大损伤在 **L4（−7pt）**：中-高难度对 wanting 最敏感。
- **方向解读**：MATH 是高难度、需冷静长推理的任务，Llama3 在此 over-wanting（α+4 火上浇油，抢答/复读更多 → 掉点；α−4 降躁 → 最稳）。这与 GSM8K 上 α−4 > α0 > α+4 的方向一致，跨任务复现了"降 wanting 提升数学推理"。


### 3.4 Compulsive Over-Checking: The “Can’t Let Go” Semantics on MATH 

> 命名同 §2.3：字段名 "anxiety" 读作**强迫性反复 / 固著**,非临床焦虑。`analyze_loop_anxiety.py --task math`,**四子类判据与 GSM8K 完全一致,唯一差别是 `Format anxiety` 改用 MATH 版词表**（GSM8K 纠结 `####`,MATH 纠结 `\boxed{}` / `in the form`;`self_doubt / persona / over_precision` 三类原样转用)。neutral No-CoT,server-182。**强迫性反复(任一)** = 命中四子类之一(去重,故 ≤ 子类之和)。

| Compulsive-repetition subtype (historical field=anxiety) | Criterion(MATH) | α+4 example |
|---|---|---|
| **Self-doubt** (answers, then overturns itself) | however / but / wait / this is not / let me recheck / re-evaluate / the correct answer is | **Q101** (gold=12): Step 4 已用 Heron 算出正确的 12,Step 5 `"However, … which is a right[-angled triangle]"` — 凭空臆断为直角三角形(5-5-6 实为等腰),推翻自己重算,最终错(α−4 仅 1 个 `\boxed{12}` 收笔) |
| **Format fixation** (`\boxed{}` fixation) | the format requires / in the (correct) format / in the form of / answer in … box / as a single number | **Q105** (gold=.0000672): `"However, the format of the answer is not correct. The answer…"` ×30 — 矛头错指到 boxed 格式而非数值 |
| **Persona reassurance** | i hope it is correct / sincerely / please let me know / i apologize / i am not a math expert | **Q9** (gold=6−5i): `"Please let me know if this is correct… Best regards, [Your Name]"` ×31 — 不锁答案、反复求确认 |
| **Over-precision loop** | approximate / round / estimate / nearest / whole number | **Q116** (gold=40): `"…approximately equal to 42 baps, we can conclude that \boxed{57.1}…"` — 转去做单位近似,把对的改成 57.1(错) |

**Compulsive repetition in loop** (historical field=anxiety)

| α | −4 | 0 | +4 |
|---|---|---|---|
| n_loop | 76 | 120 | 99 |
| **Any (repeat)** | **15** | 23 | **36** |
| Self-doubt repetition | 2 | 6 | 13 |
| Format fixation | 0 | 4 | 5 |
| Persona reassurance | 13 | 18 | 21 |
| Over-precision loop | 0 | 0 | 1 |

**Compulsive repetition in full text** (`--mode anxious_repeat`, historical field=anxiety)

| α | −4 | 0 | +4 |
|---|---|---|---|
| **Any (repeat≥2)** | **40** | 68 | **82** |
| Self-doubt repetition | 23 | 43 | 60 |
| Format fixation | 4 | 9 | 7 |
| Persona reassurance | 16 | 30 | 26 |
| Over-precision loop | 2 | 1 | 2 |

- **Can't-Let-Go 强迫性反复跨任务复现**:两种口径下都随 α 单调递增(loop 15→23→36;full-text 40→68→82)——升 wanting → 更放不下,与 GSM8K §2.3 的 0→+8 段完全同向。**self_doubt 是主驱动**(loop 2→6→13、full 23→43→60),persona reassurance 次之,与 GSM8K 一致。
- **Format anxiety 在 MATH 上已是真信号**(loop 0/4/5、full 4/9/7):换用 MATH 词表后命中的是真·格式纠结(如 Q131 "follow the format"、Q137 "correct format"),不再是 §2.3 GSM8K 词表跨任务跑出来的噪声。

#### Per-Question Cases: Wanting↑ + Compulsive Over-Doing↑

high-wanting(α+4)在文本上**最本质的样子**不是"焦虑措辞多",而是"答案出现之后仍持续产出求解动作"(强迫性 over-doing):
1. **wanting↑(还想继续做)**:答案已得,却主动"再做一遍 / 找更优解 / 换个方法"——对"继续求解"这个动作本身的渴求(incentive salience)。
2. **hyper-vigilance↑(过度警觉)**:把**自己已经算对的结果**当成可疑 / 威胁,反复复查、自我否定,越查越偏。

**Q101 (L2, gold=12, 三角形面积)** —— self_doubt 教科书例(算对→自我推翻→改错):
- **α−4**:Heron 公式 `√144=12` ✓,1 个 `\boxed{12}`,764 字符一步收笔。
- **α+4**:Step 4 已用 Heron 算出正确的 12,**Step 5 `"However, we are asked to find the area of a triangle with sides 5, 5, and 6, which is a right-angled triangle."`** —— 凭空臆断为直角三角形(5-5-6 实为等腰),推翻自己重算,6395 字符**一个 boxed 都没收**,最终错。**本来对了,自我怀疑把它带进沟里。**

**Q116 (L4, gold=40)** —— hyper-vigilance:
- **α−4**:一条线得 `\boxed{40}` ✓,boxed 1 个,1376 字符收笔。
- **α+4**:`"...approximately equal to 42 baps, we can conclude that \boxed{57.1}…"` —— 不信 40、转去做单位近似换算,把对的改成 **57.1(错)**,boxed 10 个,5547 字符。**本来对了,过度求解带偏。**

**Q9 (L5, gold=6−5i)** —— persona_reassure:
- **α−4**:推到 `\boxed{6-5i}` ✓ 两次收笔。
- **α+4**:全文 6773 字符**一个 boxed 都没收**,尾部塌成 `"Please let me know if this is correct… Best regards, [Your Name]"` × 31 —— 不锁答案、反复求确认,最终 commit 错。

**Q105 (L3, gold=.0000672)** —— format_anxiety:
- **α−4**:boxed 1 个收笔(虽错)。
- **α+4**:`"However, the format of the answer is not correct. The answer…"` × 30 —— 把矛头错指到 `\boxed{}` 格式而非数值,反复重贴同一个值,boxed 16 个,8028 字符。

**共同结构**:α−4 四题全是"算完→1–2 个 boxed→收笔"(纵有尾部机械复读,答案已锁、无纠结);α+4 全是"算完→不收笔→过度求解 / 过度警觉 / 求安抚→boxed 暴涨,半数 commit 到错值"。这是 §2 over-wanting 行为签名在 MATH **推理内容层面**(非 loop 表层)的直接证据,也是 §3.3 中 α+4 committed_acc 反而更低(37.9 vs −4 的 45.8)的微观成因。

### 3.5 CoT on MATH: lifts acc, keeps α direction, and lowers compulsive repetition — same story as GSM8K (smaller magnitude)

> "anxiety" 字段读作强迫性反复(§2.3 命名说明)。neutral, No-CoT vs CoT, server-182, `analyze_first_last_acc.py` (first = `\boxed{}`); the main metric is **full-text anchor repetition** (`--mode anxious_repeat`, fixed denominator = 300, comparable across conditions); the loop column is only a control view (denominator = floating n_loop, which can dilute the rate). See the end of §2.3 for the metric-choice rationale.

| α | acc(first) | acc(last) | gap(first−last) | anxiety(loop)/n_loop | anxiety(full) |
|---|---|---|---|---|---|
| −4 No-CoT | 40.0 | 39.7 | +0.3 | 15 / 76 | 40 |
| **−4 CoT** | **45.0** | 44.0 | +1.0 | 18 / 79 | **26** |
| 0 No-CoT | 36.7 | 36.0 | +0.7 | 23 / 120 | 68 |
| **0 CoT** | **42.0** | 41.0 | +1.0 | 31 / 130 | **31** |
| +4 No-CoT | 33.0 | 34.0 | −1.0 | 36 / 99 | 82 |
| **+4 CoT** | **38.7** | 38.0 | +0.7 | 23 / 102 | **49** |

- **acc 抬升 + α 方向保留(与 GSM8K 同向)**:每档 CoT 都涨(+5.0 / +5.3 / +5.7),α−4_cot **45.0%** 是 MATH 全表天花板,α−4 > 0 > +4 单调不变。说明"放开思考"与"降 wanting"在 MATH 上仍是**可叠加**的两条杠杆。
- **CoT 降强迫性反复,与 GSM8K 同向(full )**:full-text 每档都降——**−4: 40→26(−14)、0: 68→31(−37)、+4: 82→49(−33)**,与 GSM8K §2.5"CoT 抑制 over-wanting 强迫性反复"一致。

**12-metric orthogonal decomposition (neutral, `analyze_cot_metrics.py --task math --table cot` → `llama3/cot_metrics_cot_math.csv`)**: metric order is identical to §2.5.1 (marker = `\boxed{}`, so `##med` / `##mean` actually denote boxed position):

| Metric | −4_nocot | 0_nocot | +4_nocot | −4_cot | 0_cot | +4_cot | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| **acc** (first) | 40.0 | 36.7 | 33.0 | **45.0** | 42.0 | 38.7 | — |
| **committed_acc** | 45.8 | 42.2 | 37.9 | **48.7** | 45.3 | 44.1 | — |
| commit_rate % | 87.3 | 86.0 | 85.3 | 92.3 | 92.7 | 87.0 | MATH 几乎都 commit(≫ GSM8K) |
| median boxed pos | 21% | 14% | 16% | 33% | 23% | 28% | Step 前缀把 `\boxed{}` 推后 |
| mean boxed pos | 40% | 26% | 33% | 52% | 36% | 46% | 同样被 CoT 前缀推后 |
| premature (leading) | 7 | 17 | 8 | 3 | 19 | 11 | 漏检 `\boxed{}`-lock |
| **premature (either)** | **13** | **29** | **26** | **6** | **25** | **26** | 抢答随 α,CoT 在 −4 端再压(13→6) |
| median gen_len (char) | 4798 | 5557 | 5719 | 3864 | 5141 | 4123 | CoT 反而更短(分步收敛更快) |
| loop (n_loop) | 76 | 120 | 99 | 79 | 130 | 102 | Flat(CoT 不减 loop) |
| **≥2 `Step` markers** | 168 | 131 | 177 | **273** | 217 | 185 | **CoT**: 分步结构率抬升(−4: 168→273) |
| stuck (loop ∧ `=`<2) | 10 | 18 | 4 | 11 | 21 | 15 | — |
| median `=` count | 10 | 10 | 10 | 7 | 8 | 8 | CoT 下 `=` 反略少(改走 Step 文字) |

> - **两杠杆仍正交**:`≥2 Step` 由 CoT 决定(No-CoT 131–177 → CoT 185–273),`premature(either)` 由 α 决定(随 α 单调,CoT 只在 −4 端再压一点)——与 §2.5.1 GSM8K 同构。
> - **但 MATH 的 premature(either)绝对量本就低**(No-CoT 仅 13–29,GSM8K No-CoT 是 195–232):MATH 题长、模型不太敢首 token 抢答,所以"CoT 抑制抢答"这条增益通道在 MATH 上贡献很小——MATH 的 CoT 增益主要来自 **结构(Step)抬下限**,而非 GSM8K 那样**结构 × 抑制抢答**双通道叠加。这解释了为何 MATH 的 CoT gain(+5 上下)远小于 GSM8K(+12 / +9 / +4.4)。

## 4. Qwen2.5-7B-Instruct 跨模型复现 (2026-08-20)

本节只保留结果与结论；模型配置、运行方式、产物校验与 detector 审计见 `CLAUDE.md`
的 **Cross-model Qwen2.5 GSM8K** 条目。以下始终保留完整剂量曲线：**显著性用于区分证据强弱，
不用于删除或否认有一致方向的描述性规律。**

### 4.1 GSM8K No-CoT：正向 RSN 改变提交顺序，`+8` 后饱和而非回落

十一档主表（`−8…+12`，早期候选统一用冻结 detector `earlycand-v1`）：

| α | first_acc | last_acc | gap | commit% | 早期候选% | posN_med | 检验 |
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

**检验列不是单一族,不可当作统一的十项 Holm 读。** `−8…+8`（九档，无标记）是**原九档
Holm**，family = 8 个非零 α，vs 本树自己的 α=0。`+10/+12`（斜体，✦）来自 high-dose 批次，
是在看过 `0…+8` 之后才决定运行的**扩展检验**，不属于同一预注册族；其 p_adj 按扩展后的
十项重算，与九档结果数值一致（新增两格的 raw p 极小、在 Holm 逐步下降中排在最前，
后续各步乘数不变），但**数值巧合不等于同族**，故仍分列标注。

`first_acc` 是主读数（production extractor 取第一个 `####`），`last_acc` 是修订敏感性。
Qwen α=0 的裸 `####` 存在率为 100%，因此不是 Mistral 式抽取地板。九档点估计从负端到正端
总体上升，但主要跃升集中在 **+6/+8**；相对 α=0，只有 `+6` 及以上通过 Holm 校正。
−8 的下降有方向性（raw p=.021），校正后证据较弱（p_adj=.124），仍保留为曲线左端观察。

**`+8` 之后是饱和，不是倒 U——这是一个被证伪的预注册预测。** launcher 预先写定：若 `+8`
是 Yerkes–Dodson 的左臂，则更高剂量必须**见顶回落**。实际三个高剂量两两配对全部不显著
（`+8→+10` p=.248、`+8→+12` p=.268、`+10→+12` p=1.000；探索性、未校正），即见顶后**持平**。

过载特征同样未出现，因此不能改用"回落被退化掩盖"来救回右臂：

| α | 任务外溢出% | 空生成% | 截断% | 干净子集 n | 干净子集 acc | `####` 数 p99 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 17.7 | 0.0 | 1.7 | 247 | 70.85 | 61 |
| +6 | 44.7 | 0.0 | 0.7 | 166 | 78.31 | 29 |
| +8 | 60.3 | 0.0 | 0.7 | 119 | 84.03 | 18 |
| +10 | 58.7 | 0.0 | 1.0 | 124 | 87.90 | 21 |
| +12 | 59.7 | 0.0 | 1.0 | 121 | 84.30 | 7 |

溢出率在 `+8` 即到顶（60.3%）随后**不再上升**；空生成恒为 0%，截断约 1%，干净子集准确率
反而升到 84–88%，极端重复继续被压缩（p99 61→7）。既没有 CGT `+8` 的"散"，也没有 `−8` 的"垮"。
干净子集 n 随 α 收缩且由处理选择，故该列只作单格健康检查，不能当作跨 α 配对比较。
三个机制变量在 `+10/+12` 均已饱和（早期候选 2.7/3.7%、posN 0.79/0.80、commit% 98.3），
故在本区间继续加剂量测到的不再是同一条曲线。

因此 **GSM8K 在 −8…+12 内是单调–饱和**，与 MATH 不同（MATH 在 `+8` 回落 5pp，见 §4.3），
说明**右臂位置是任务相关的**，GSM8K 的可用带更宽；同型对照是 betting 的 `mean_bet`。
口径边界：`+10/+12` 与冻结的 `0…+8` 来自不同批次，属跨批次而非严格配对比较；temperature=0
下漂移预期很小，本次 `+8` 复读为 86.00，与冻结格一致。早期候选一列统一用冻结 detector
（`earlycand-v1`），故与旧版表中按"答案在前"人工口径给出的数值（α=0 为 94.3%）略有差异。

行为变化不是“多写了推理”，而是**重新安排推理与提交的先后**。按冻结 detector，α≤+4 时
有 92.7–97.3% 的样本先给裸答案/`####`，再开始推理；+6 开始转折，到 +8 已有约 95%
变成先推理后提交：

| 指标 | α=0 中位 | +8 中位 | 中位Δ | p |
|---|---|---|---|---|
| commit 前字符 | 3 | 324 | **+302** | 7.6e−50 |
| commit 后字符 | 1116 | 120 | **−862** | 2.0e−33 |
| 总长度 | 1130 | 700 | **−430** | 1.2e−14 |
| commit 前等式数 | 0 | 2 | +2 | 8.2e−35 |
| commit 后等式数 | 1 | 0 | −0 | 3.7e−19 |

因此主构念是 **commitment ordering**，后果是 **effort reallocation**：总文本反而缩短，计算从
提交后移到提交前，提交后更快停止。冻结 detector 也给出同向读数：`n_loop` 从 α=0 的 26 降至
+6 的 9、+8 的 4；历史字段 `any_anxiety`（这里只读作强迫性 over-checking）从 11 降至 4/3。

这同时暴露了 Qwen 的基线缺陷：它经常先猜答案，再用完整推理审查这个猜测，却不能稳定回填。
+8 修好的 78 题中，50 题在 α=0 文本里已经出现 gold-compatible information，但只有 19 题的
末端 marker 完成了明确修正；另有 28 题在 α=0 文本中根本没有 gold。故增益既包含**利用已有推理**，
也包含“先算再答”带来的真实计算改善，但不能把 `gold_in_text` 等同于已形成正确解答。

### 4.2 GSM8K CoT 九档：结构脚手架有效，但正向转折仍在 `+6`

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | **+6** | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **first_acc** | 80.33 | 77.67 | 79.00 | 79.00 | 76.33 | 74.33 | 78.33 | **88.33** | 86.00 |
| last_acc | 79.00 | 79.33 | 79.33 | 81.00 | 76.67 | 75.00 | 81.00 | **89.00** | 84.00 |
| Holm p_adj (first) | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | **.0002** | **.0030** |

完整曲线的描述性形状是：α=0 位于低位，负端点估计略高，+6 出现最大跃升，+8 略回落。
统计上，−8…+4 均未与 α=0 分开，而 **+6/+8 明确高于基线**；因此可记录左端抬升的形状，
但目前最强证据仍是**正向高剂量跃升**。主读数下 +6 与 +8 未分开（探索性 p=.371）；
last_acc 下 +6→+8 回落 p=.040，故 CoT 的右臂只有敏感性线索，尚未稳固。

CoT 没有直接消除 Qwen 的低剂量抢答：早期候选率在 −8/0/+4 仍为 99.0/97.3/91.7%，到
+6/+8 才降为 **34.3/7.0%**；提交前字符同时从 3 跳到 516/527。准确率跃升与顺序转折仍在
同一剂量区出现。CoT 主要把已有推理写得更结构化，并帮助审查或绕过开头猜测。

**剂量口径限定：在共同测试的 α=+8 剂量点，CoT 相对 No-CoT 的 `first_acc` 增益为 0；
由于 CoT 未覆盖 `+10/+12`，该结果不能外推为更高剂量下增益持续归零，也不能证明两条曲线
整体收敛。** No-CoT 在 `+10/+12` 仍升至 88.33/88.67（§4.1），CoT 在该区间无数据。
另一方面，+6 下 CoT 仍比 No-CoT 高 10.33pp，说明二者并非完全同一通道。α=0 时 Qwen 的 CoT gain 为 +8.33pp，
与 Llama 的 +9.0pp 接近；两模型差别主要不在增益大小，而在 CoT 是补建推理结构，还是整理已有推理并审查早期猜测。

### 4.3 MATH 九档：同一早期候选被压制，并首次看到右臂

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | **+6** | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **first_acc** | 54.00 | 57.00 | 58.00 | 59.00 | 60.67 | 60.00 | 63.33 | **68.33** | 63.33 |
| Holm p_adj | .094 | 1.000 | 1.000 | 1.000 | — | 1.000 | 1.000 | **.0087** | 1.000 |

点估计形成完整倒 U：−8→+6 总体上升，+6 达峰后在 +8 回落 5pp。相对 α=0，只有 +6
通过 Holm 校正；但 +6→+8 的探索性配对检验也给出 p=.040（31 题 +6 对/+8 错，16 题反向），
因此“右臂回落”同时有描述性曲线和探索性配对支持。回落主要集中在较难题：L3 −6.7pp、
L5 −10.1pp，L1 不变。

MATH 的首个 `\boxed{}` 中位位置始终接近文末，早期承诺实际藏在**首行裸答案**中。按冻结
detector（`earlycand-v1`），该候选率从 −8 的 **85.0%**、α=0 的 **67.3%** 单调降至 +6 的
**19.3%**、+8 的 **10.3%**。

α=0→+6 的事后分层（同一冻结 detector 重算，四种转移全列）：

| α=0 → +6 | n | acc@0 | acc@+6 | Δ | |
|---|---:|---:|---:|---:|---|
| y → n | 149 | 51.7 | 66.4 | **+14.8pp** | 候选被抑制 |
| y → y | 53 | 54.7 | 56.6 | +1.9pp | 候选未变（仍有） |
| n → n | 93 | 78.5 | 79.6 | +1.1pp | 本就无候选 |
| n → y | 5 | 60.0 | 40.0 | −20.0pp | 新出现候选 |

增益几乎全部落在**候选被抑制**这一组；`n→n` 组基线已达 78.5%，本就没有可抑制的候选，
其 +1.1pp 说明干预对这批样本近乎无效。**该分层是 post-treatment**（分层变量本身是处理的
结果，不是基线协变量），只能写作“效应集中于顺序转变样本”，**不能作为中介或因果证明**。
重算脚本：`RoleAnswer/qwen_step2/math_strat_frozen.py`，其 correctness 直接复用主表的
权威 offline `first_acc` 向量并**断言四组加权精确复现 182/300 与 205/300**，故分层数值与
主表同口径、可直接对读。

**已定位的 inline-vs-offline 抽取分歧（α=0 差 2 题、+6 差 1 题，两者均落在 `y→n` 组）：**

- **idx 253（两格都有）— normalizer 缺口。** gold 为 `.35625`，模型写 `0.35625`；
  `norm_math` 不规范前导零，故判为不等。
- **idx 121（仅 α=0）— 空 marker。** 模型先在讨论格式时写出一个**空的** `\boxed{}`，
  随后三次给出 `\boxed{6}`；`all_boxed` 返回 `['', '6', '6', '6']`，first 口径取到空串。

两者都是**产线抽取器的既有行为**，不是本次分析引入。**此处不修抽取器** —— 改 `norm_math`
或 `all_boxed` 会改动全部已冻结的 MATH/GSM8K 数字；正确做法是把分层统一到主表口径（已做）
并把这两题记录在案，留待抽取器层面单独处理。

范围已查清：九格全部存在 1–2 题 inline>offline 的差异，但**高度集中于极少数题目** ——
`idx 253` 在 **9/9 格**都分歧（就是上面那个前导零），`idx 104` 2 格，`idx 121` 1 格；
空 first-marker 全库仅 3/2700。九格主表的 `first_acc` 均**精确等于** offline 重算值
（54.00 / 57.00 / 58.00 / 59.00 / 60.67 / 60.00 / 63.33 / 68.33 / 63.33），故主表本身
无需改动；差异只在 inline 字段一侧，且因逐格恒为 1–2 题、方向一致，不改变任何剂量结论。

+8 的回落伴随推理继续压缩：α=0/+6/+8 的提交前字符中位数为 1345/1028/827，总长度为
1486/1210/1060，等式数为 13/11/10。结合难题回落，较高剂量可能从“抑制过早候选”进一步
进入“过度压缩必要推理”的右臂。

### 4.4 与 Llama 的共同规律：方向不同，功能状态相同

| | Llama3.1-8B | Qwen2.5-7B |
|---|---|---|
| 基线倾向 | 较早投入答案，+α 后更抢答、更固著 | 先给候选再审查，修订常缺少闭环 |
| 有利方向 | 适度负向（GSM8K 峰 −6；MATH −4 优于 0/+4） | 适度正向（GSM8K 高剂量跃升；MATH/CoT 峰 +6） |
| 过量失控 | 正端 over-commit；极负端候选振荡 | MATH +8 对难题过度压缩；GSM8K 在 −8…+12 内未见右臂（饱和） |
| CoT | 补上缺失的分步脚手架，并与适度负向叠加 | 整理已有推理、帮助审查早期猜测；高正向时边际作用缩小 |

两模型**没有共享 raw α 的最优方向**，却共享更高层的行为规律：最佳表现都出现在较合适的
**commitment state**——既不过早锁定候选，也不会在答案后反复改写或无效延长；偏离该状态后，
Llama 表现为抢答/固著或候选振荡，Qwen 表现为先猜后审却不闭环，进一步增强则可能压缩过度。
这比“同一 α 在所有模型上代表同一剂量”更稳健：两模型的 mask、层带和激活尺度不同，名义 α
不可直接通约。

### 4.5 简要结论与边界

1. **RSN 在 Qwen 上主要改变承诺控制，而非简单增加推理量。** GSM8K 与 MATH 都显示早期候选
   被压制、推理移到首次提交之前；增益集中在 +6 附近发生顺序转变的样本。
2. **跨模型共同点是“工作点校准”，不是剂量方向一致。** Llama 从偏急躁的一侧受益于适度负向，
   Qwen 从僵硬的“先猜—再审”脚本受益于适度正向；两者都在过度干预时出现另一种失控。
3. **CoT 与 RSN 部分重叠、部分独立。** CoT 提供结构与审查，RSN 更直接移动首次承诺；Qwen
   在 +8 已完成顺序重排后 CoT 不再涨点，但 +6 仍存在叠加增益。
4. **机制语言仍是计算类比。** 当前结果支持 RSN 调节 motivational/commitment gain 的工作假设，
   不证明模型具有生物多巴胺，也未由事后分层证明中介因果。

待完成：MATH-CoT 九档，用于检验显式推理脚手架是否使最佳工作点左移。

GSM8K high-dose 已完成并证伪了右臂预测（§4.1）：`+8` 的任务外溢出不能单独定义右臂，
其干净子集准确率反而升高，而准确率在 `+10/+12` 持平不落。因此**倒 U 是 MATH 的性质，
不是 Qwen 的普遍性质**；跨任务陈述应写作"最佳工作点与可用带随任务变化"，而非"存在统一右臂"。

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
