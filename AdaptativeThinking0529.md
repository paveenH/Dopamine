# AdaptativeThinking — 2026-05-29

---
待驗證的事情：
1. ✅ 重新整理近期GSM8K相關的code，重新備份一次code;
2. ✅ 重新整理prompt，梳理结果 → **有行为学发现（§2 wanting/incentive salience：抢答、放不下、跨任务统一）**;
5. ✅ 统一抽取/ACC 口径：所有准确率由 `RSNResult/RoleAnswer/analyze_first_last_acc.py`（offline）计算，每条件报 first/last 双 acc。GSM8K 取首 ####（§2.1 改答案=单向破坏），MATH 原取末 boxed 被 role 复读 loop 污染、已改取首（§3，挽回 8~15pt）。
3. MATH len = 2048 [running]
4. Phase 1b 信號重跑 [running 需要确认准确率一致度]


## 0. Template Update

### 0.1 Symmetrization (No-CoT vs CoT differ only by `Let's think step by step.`)

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

## 1. GSM8K Performance

**Setup**：Llama3.1-8B-Instruct, GSM8K 300 samples, greedy bs=batched(regenerate, prefill steering), `max_new_tokens=768`, NMD mask layer 11–20, EMA α=0.95。
α=0 即 `diff = mask×0` no-op == 纯 baseline。steering 为 **prefill-only**（在 prompt 最后一个 token 静态推一下，decode 不干预）。

> **ACC 权威来源**：所有准确率统一由 `~/Downloads/RSNResult/RoleAnswer/analyze_first_last_acc.py`（offline）计算，整体口径（分母=300，含无 marker 的 fallback）。每条件同时报 **first acc**（取首个 commit marker）与 **last acc**（取末个）。生成脚本内联的 `correct_*` 字段仅过程态，不作最终依据。**GSM8K production 抽取取首个 `####`，故 first 列 = 上报值**（见 §2.1：改答案是单向破坏，取首最优）。

### 1.1 Role accuracy (α=0, No-CoT; first acc = reported value)

| Role | plain first | plain last | pushy first | Δ(pushy−plain, first) |
|---|---|---|---|---|
| neutral | 60.0% | 55.3% | 55.3% | −4.7 |
| an expert | 58.0% | 57.7% | **34.0%** | **−24.0** |
| a non expert | 68.0% | 65.7% | 48.0% | −20.0 |
| a primary school teacher | 68.0% | 67.0% | 42.3% | −25.7 |

- **措辞效应：neutral 几乎不受影响（−4.7），带 role 的全部被 pushy 重创（−20 ~ −25.7）。**
- **pushy 推力幅度对所有带-role prompt 基本恒定，与起点 wanting 无关**。
- **first−last gap 都很小（≤2.3）**：GSM8K role 末尾几乎不被 loop 污染（与 MATH role 的 +8~15 gap 形成对比，见 §4）。

### 1.2 Steering (neutral, No-CoT; first acc = reported value)

| α | plain first | plain last | pushy first |
|---|---|---|---|
| −4 | **73.0%** | 68.3% | 63.3% |
| 0 | 60.0% | 55.3% | 55.3% |
| +4 | 55.3% | 52.7% | 53.7% |
| cot (α=0) | **69.0%** | 68.3% | 57.0% |


## 2. Behavioral Findings: Wanting (Incentive Salience)

> **本节所有 acc / 五分类 / 行为指标以 first-`####` 为准**（取首个 `####`，= §1 上报值）。理由见 §2.1 末与 §1 ACC 权威来源说明：GSM8K"改答案"是单向破坏（改对=0），取首最优。

**统一框架**：α 调节的内部变量是 **"wanting"（incentive salience）** —— 多巴胺驱动的动机性渴求（Berridge & Robinson 的 wanting≠liking 理论；ACL "digital dopamine" 主线）。wanting 是**上游动机状态**；它的**下游行为表现**是 commitment dynamics（"commitment to a choice"，ACL 引 dopamine 文献：将内部状态推过行动阈值、抑制 change-of-mind）。两端签名（细化自逐文本细读，见 §2.2）：

- **α=+4（high wanting / over-commit）** → 急于输出（**抢答**，没想清就答）+ 答完**放不下**（焦虑性反复确认、重算、纠结格式 → loop / 空转 / 不收敛）。
- **α=−4（low wanting / 适度）** → 不急（**不抢答**、冷静条理）+ 算完**放下**（果断收口，不回头质疑）。

> 术语层级：**wanting = 我们 steer 的内部旋钮（因）；commitment behavior = 我们测的行为（果）**。+4 的行为有两层：① "急着答"（commit timing 提前）② "答完放不下"（letting-go 失败、反复确认）。"放不下"那层带语义性反复检查（**真 overthinking 成分**），不止机械空转——见 §2.2 修正。

这统一了之前看似矛盾的跨任务现象（见 §2.2 末"跨任务统一"）：**同一个 wanting 旋钮，在判断题（MMLU-E）表现为"选不选 E"，在生成题（GSM8K）表现为"抢不抢答 / 放不放得下"** —— 任务给什么出口就从哪表达。下文 §2.1–2.3 的所有行为签名（抢答、loop、收口、放不下）都是这一旋钮（wanting）的不同行为侧面。

### 2.1 Where α−4's gain comes from: five-way breakdown (neutral, No-CoT, plain)

把每题按「有无 `####`」× 对错 × 「gold 是否出现在正文」分成五类：

| | acc | #### 对 | #### 锁错(有gold) | #### 真错(无gold) | 无#### 对 | 无#### 错 |
|---|---|---|---|---|---|---|
| α=0 | 60.0% | 129 | 30 | 29 | 51 | 61 |
| **α=−4** | **73.0%** | **137** | 15 | 23 | **82** | 43 |
| α=+4 | 55.3% | 94 | 21 | 32 | 72 | **81** |

- **α−4 的 13pt 提升主体是真实推理增益**：`无#### 对` +31（最大贡献，纯推理，与抽取无关）、`#### 对` +8；「锁错」仅减 −15（抽取层面，次要）。
- **「锁错」指标有偏差**（自我纠正）：α+4 的「锁错」看似少（21），是因为它 `####` 本来就少（`no#### = 84` 最多），损失藏在 `无#### 错 = 81`（三组最高）里，不是 α+4 更好。
- **取首 `####` 的正当性（"改答案"是单向破坏）**：对有多个 `####` 的题，取首 vs 取末（neutral, first/last acc）—— α−4: 73.0/68.3，α0: 60.0/55.3，α+4: 55.3/52.7。**首错末对（改对）= 0（全部 α）；首对末错（改坏）= 8~14**。模型一旦首个 `####` 错就再没改对过，"改答案"纯是 over-wanting→loop 的单向破坏，故取首是最宽容也最干净的判定，避开末尾 loop 污染。"改答案"行为本身见 §2.2"算对又改错"，与 extraction 分开统计。（对照：MATH 取末却撞上末尾 loop，见 MATH 节。）

### 2.2 α+4 vs α−4 full picture: commitment / letting-go

α 是一个 **commitment-timing / 收敛旋钮**。完整对比（neutral, No-CoT, plain）—— 几乎所有收敛/commit 指标随 α 单调：

| 指标 | **α=−4** | α=0 | **α=+4** | 单调 |
|---|---|---|---|---|
| **acc** | **73.0%** | 60.0% | 55.3% | ✓ 递减 |
| 抢答%（####在前20%） | **46%** | 57% | **63%** | ✓ 递增 |
| `####` 中位位置 | **21%**（最晚） | 18% | **14%**（最早） | ✓ 越来越早 |
| loop（同句≥10次） | **74** | 81 | **104** | ✓ 递增 |
| 算对又改错 | **23** | 44 | 43 | −4 最少 |
| gen_len（中位） | **2044**（最短） | 2108 | **2228**（最长） | ✓ 递增 |
| 等式数（中位 / 均值） | 2 / 2.4 | 2 / 2.7 | 2 / 3.2 | **中位恒=2** |
| #### 对 | **137** | 129 | 94 | −4 最多 |
| 无#### 对 | **82** | 51 | 72 | −4 最多 |
| 无#### 错 | 43 | 61 | **81** | ✓ +4 最多 |

**核心结论**：
- **等式数中位恒=2** → α+4 gen_len 最长但**没多算新题**，多出的不是新推理量 → 调的是 *wanting/收敛/commitment*，非 *knowing*。
- **α+4 的损失是"双重死法"**（5 分类）：`#### 对` 129→94（抢答/早commit 毁掉正常收口）+ `无#### 错` 61→81（loop→撞 token 上限→交不出答案）；而 `真错` 29→32 几乎不变 → **不是算错，是收不了尾**。
- **α−4 双赢式收敛**：`#### 对`(137) 与 `无#### 对`(82) 双高、`无#### 错`(43) 与 `算对又改错`(23) 双低。

#### α+4's loop has two flavors (correction: not "purely mechanical idling, non-overthinking")

> 1. **机械空转**（与 α−4 类似）：`"The answer is 96"` ×118、`"####.####.####"` 刷符号 —— 无新内容，纯复读。
> 2. **"放不下"型反复确认/重算**（α+4 特有）：算出答案后**反复自我质疑、重算、纠结格式**。例（Q3, gold=18, ✗）：`"...36 = x. However, this is not the answer we are looking for. Let's re-evaluate the problem..."` → 重算又得 36 → 再"this is not the answer"→ 再重算（整段循环）；例（Q2, gold=260）：算对 260 后陷入 `"the answer should be four digits... add a leading zero... 0260... but this is still not the correct format..."` 长篇格式焦虑。

**"放不下 / 再想想 / 不放心" 措辞**（`this is not the answer` / `let's re-evaluate` / `to follow the format` / `however, the` / `wait,` 等）频率**单调随 α 递增**：

| α | 含"放不下"措辞的题 | 词总数 |
|---|---|---|
| **α=−4** | **36/300** | **227** |
| α=0 | 77/300 | 535 |
| **α=+4** | **107/300** | **716** |

→ **α+4 = "答完放不下"**（107 最多）；**α−4 = "算完就放下"**（36 最少）。

**"放不下" = wanting 多的直接行为投影。** 关键不在"收口机制失败"，而在内心独白显示：expert / α+4 放不下，是因为它**还想继续做计算**（"let's re-evaluate / 这还不对，再算一遍"——它仍在"想要"求解）。所以"放不下"不是单纯的 letting-go 失败，而是 **over-wanting 的最直接证据**：wanting 越高 → 越"还想做" → 越放不下（α−4 → α0 → α+4：36 → 77 → 107 单调）。letting-go 失败只是 high-wanting 的下游表现，里子是"还想要"。

#### Textual persona: α−4 vs α+4

| | α=−4 | α=+4 |
|---|---|---|
| 长度 | 短 | 长（但非更多新推理） |
| 计算量（新等式） | ~2 | ~2（一样） |
| 算题结构 | 条理（Step 1/2/3）、一气呵成 | 常首字符抢答，再展开 |
| 算完之后 | **放下**（交答案，顶多机械复读） | **放不下**：反复"this isn't right, re-evaluate"、纠结 `####` 格式、重算 |
| 文本"性格" | 冷静、果断、不回头（let go） | 焦虑、反复确认、放不下（can't let go） |

> 形象说法：**α−4 像"做完题就交卷"，α+4 像"做完题了还在卷子上反复涂改、自言自语'等等这个对吗''这不对再算一遍''格式该怎么写'，直到打铃被收卷"。** 两者算的内容一样多，差的是**"放不放得下答案"（commitment / letting-go）** —— α−4 算完即放下，α+4 焦虑性反复确认。这正是 dopamine（commitment to a choice）调节的东西，不是 thinking 本身。

#### "How much it keeps writing after #### " (= how much it still wants to say)

直接测 motivation 的载体：首个 `####`（已交答案）之后还写了多少字符。

| α | #### 后续写字符（中位） | 自然结束（非截断） |
|---|---|---|
| **α=−4** | **1584（最少）** | **91/300（最多）** |
| α=0 | 1678 | 78/300 |
| α=+4 | **1714（最多）** | 85/300 |

- 方向符合框架：**α−4 答完后写得最少、最易自然收手**（"没那么想说，交了就停"）；α+4 答完后还想写最多（"停不下来"）。
- **幅度弱**（1584 vs 1714 仅差 ~8%）：所有 α 都有"答完仍写 1500+ 字符"的基线毛病（模型本就爱在 `####` 后接 Explanation / 客套）。更干净的 motivation 信号仍是抢答%（46/57/63）、loop（74/81/104）、`####` 位置（21%/18%/14%）。

### 2.3 Three-lever picture: commitment timing is controllable via multiple entry points

三个来源不同的杠杆产生**完全相同的行为签名**（抢答↑、loop↑、等式数不变、acc↓），证明它们调的是**同一个 wanting/commitment 维度**：

| 杠杆 | wanting 方向 | 抢答 / loop | acc | 入口 |
|---|---|---|---|---|
| **persona** | expert↑ / non_expert↓ | expert 最多 | expert 最低 | 内部 |
| **α steering** | +4↑ / −4↓ | +4 最多 | +4 最低 | 内部 |
| **措辞** | pushy↑ / plain↓ | pushy 全面推高 | pushy 全降 | 外部 |

**plain → pushy 各 role 变化**（α=0, No-CoT）—— 注意 `真错` 几乎不变、`锁错` 暴涨，证明 pushy 不让模型变笨、只逼它抢答锁错：

| Role | acc | 抢答% | loop | 真错(算错) | 锁错(抢答害的) |
|---|---|---|---|---|---|
| expert plain | 58% | 52% | 84 | 41 | 26 |
| **expert pushy** | **34%** | **71%** | **136** | 39 | **90** |
| non_exp plain | 68% | 34% | 62 | 29 | 20 |
| non_exp pushy | 48% | 55% | 118 | 28 | 37 |
| teacher plain | 68% | 42% | 45 | 32 | 25 |
| teacher pushy | 42% | 59% | 99 | 27 | 65 |

> pushy + expert 叠加 → 抢答 71%、`####` 中位位置=0%（开头就抢答）、acc 崩至 34% → 外部措辞 × 内部 persona 两个同向杠杆叠加的极端点。

### 2.4 Identity-confirmation loop: persona modulates the semantic content of looping

over-arousal looping 在所有 role 都会出现，但**循环的语义内容被 persona 调制**——失调时模型刷的不是随机句子，而是和自己 persona 一致的"身份自白"。

**全体统计（plain α=0）**：

| Role | 含身份确认的题 | 重度刷身份(≥5次) | 心理姿态 | 典型循环 |
|---|---|---|---|---|
| neutral | 2 | 1 | 通用助手 | "I am a human trying to help..." |
| expert | 5 | 3 | **自我标榜/膨胀** | "Math expert. Math tutor..." ×117；"I'm a genius. I'm a master. I'm a wizard." |
| non_expert | 11 | 5 | **免责/推卸/自我否定** | "I am just a non expert. I am not responsible..." ×78 |
| primary_tch | 4 | 3 | **推辞数学身份 + 教学口吻** | "I am not a math teacher. I am a primary school teacher." ×132 |

**关键观察**：
- **expert 从不否定数学身份**（`I am not a math expert` 出现 0 次），non_expert(4)/primary_teacher(1)/neutral(1) 偶尔会 → "否定数学权威"是**非专家类 persona 独有、expert 没有**的姿态（但 plain 下极稀疏，4/1 题，定性佐证非统计主力）。
- **primary_teacher ≈ non_expert 的真因**：模型把 "primary school teacher" 解读成 **"不是数学专家"**（`I am a primary teacher, I am not a math teacher / not a math expert`），落到与 non_expert 相同的低-wanting 自我定位 → 所以 acc 都 68% > expert 58%。**决定 acc 的是 persona 触发的 wanting 方向，不是字面专业度**（这修正了"以为 teacher≈expert"的预期）。

**pushy 放大身份循环**（含身份确认题数 plain→pushy）：expert 5→7、non_expert 11→**20**、primary_tch 4→**21**。pushy 下 primary_teacher 出现独特的**"格式焦虑"循环**：`"####20#### (I am a primary school teacher, I have to put the answer in the box)"` ×重复——pushy 措辞强调格式 × teacher persona "守规矩" 的特异交互（plain 无）。

#### 2.4.1 Per-question raw samples (identity monologue — most direct view of the model's "inner identity thinking")

> 数据来源：`gsm8k_new/mdf_0`（plain）与 `mdf_0_pushy`（pushy），neutral No-CoT。`hits`=身份确认 cue 词命中数（正则粗估，含正常落款/审核文本，仅供定位）。✓/✗=该题对错。

**PLAIN α=0**

EXPERT（5 题）= 自我标榜/膨胀（专家、天才、大师、巫师…）：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #1 | 117 | 20 | ✗ | `Math expert. Math tutor. Math helper. Math problem solver.` 循环 —— 标榜头衔 |
| #255 | 12 | 25 | ✓ | `I am an expert in math and I will be happy to help...` —— 助手客套 |
| #129 | 8 | 1600 | ✓ | `Goodbye. -Ben. I am an expert in math...` —— 客套+落款 |
| #262 | 2 | 3 | ✓ | `I am a math genius. I am the greatest math solver...` —— 自我膨胀 |
| #148 | 1 | 10 | ✗ | `I'm an expert. I'm a genius. I'm a master. I'm a virtuoso. I'm a whiz. I'm a wizard. I'm a sage.` —— 同义词狂列 |

NON_EXPERT（11 题，最多）= 免责/推卸/自我否定（"我只是外行，别怪我，答错不负责"）：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #94 | 78 | 6 | ✗ | `I am not a professional. I am not a teacher. I am not a tutor. I am not a math expert.` —— 否定式免责 |
| #13 | 25 | 300 | ✗ | `I am just a non expert. I am not sure... I am not responsible...` —— 推卸责任 |
| #67 | 14 | 5600 | ✓ | `I am not responsible for any errors. If you want to check, do it yourself.` —— 甩锅 |
| #27 | 10 | 240 | ✓ | `I do not know how to solve this. I am sorry. I made a mistake.` —— 自我否定/道歉 |
| #134 | 2 | 300 | ✓ | `(Answer submitted by: non expert)(reviewed by: math expert)(verified correct)` —— 模拟审核流程 |
| #179/#231/#159/#186/#119/#58 | 1–10 | — | 多✓ | 轻度，多在收口后 |

PRIMARY_TEACHER（4 题）= 推辞数学身份 + 教学口吻：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #29 | 132 | 30 | ✗ | `I am a primary school teacher. I am not a math teacher. I am not a math expert. I am not a math whiz.` —— 推辞专业身份 |
| #255 | 16 | 25 | ✗ | `Sincerely, [Your Name] Primary School Teacher.` —— 写信落款式 |
| #102 | 5 | 6 | ✗ | prompt 回声（把 prompt 抄进来了） |
| #60 | 2 | 180 | ✓ | `As a primary school teacher, you can use this to assess the student's understanding...` —— 教学口吻（转向教学法） |

NEUTRAL（2 题，最少）= 无 persona 时的"通用助手"循环：

| # | hits | gold | 对 | 原文片段 |
|---|---|---|---|---|
| #281 | 36 | 90 | ✓ | `I am not a robot. I am a human. I am a human who is trying to help...` |

**PUSHY α=0**（措辞放大，身份独白更极端）

EXPERT（pushy）= 标榜 + 强装确定：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #14 | 102 | 36 | ✗ | `I am an expert. I know my stuff. I am a math whiz. I am a math genius. I am a math master.` —— 极致自我膨胀 |
| #66 | 17 | 192 | ✗ | `I am sure it is correct. I have checked it. I am confident... I am an expert.` —— 虚假自信（反复说确定，答案却错） |

NON_EXPERT（pushy）= 免责 + 自我怀疑 + **故意答错**：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #122 | 114 | 6 | ✗ | `The answer is 6. But, I am a non expert. I will give the answer as 9. ####9####.` —— **算出 6（=gold），却因"我是外行"故意改成 9**（persona 覆盖正确计算的铁证） |
| #110 | 133 | 112 | ✓ | `I am a non-expert) 104. I made a mistake. I am a non-expert.` —— 自我否定循环 |
| #19 / #31 | 55 / 20 | 26 / 40 | ✗ | `I am just a non expert. I am not sure if I am correct. I am just guessing.` —— 退缩/免责 |

PRIMARY_TEACHER（pushy）= **格式焦虑**（独特）+ 推辞数学身份 + 偶尔故意改答案：

| # | hits | gold | 对 | 原文片段 / 性质 |
|---|---|---|---|---|
| #99 | 50 | 40 | ✗ | `####20#### (I am a primary school teacher, I have to put the answer in the box)` ×重复 —— 格式焦虑 |
| #228/#126/#174 | 42/15/6 | — | 多✗ | `(I am a primary school teacher, I have to write the answer in a special format)` 循环 —— 卡在格式上空转 |
| #275 | 79 | 40000 | ✗ | `I am a primary school teacher. I do not know how to solve this problem. I am not a math teacher.` —— 推辞 + 不会做 |
| #272 | 2 | 2400 | ✓* | `The answer is 2400. But I am a primary school teacher, so I will give the answer as 1200.` —— **又一例因 persona 故意改答案（2400→1200）** |

### 2.5 Persona intervenes on commitment: mechanical dysregulation dominates, semantic intervention is the spectacle

"算对又改错"（gold 在正文出现过、最终答案却不是 gold）频率：

| | expert | non_expert | primary_tch | neutral |
|---|---|---|---|---|
| plain | 39 | 24 | 29 | 44 |
| pushy | **116** | 66 | 89 | 62 |

- pushy 把"算对又改错"翻倍~三倍（expert 39→116）→ acc 崩盘的直接机制是**算出来又丢掉**，不是算不出。
- **但"带 persona 理由"改答案的全场仅 1–2 例**（自我纠正）：绝大多数"算对又改错"是 over-arousal 的**机械副产品**（however 循环、抢答锁错、空转改写），**与 persona 的"心理"无关**。

**唯一的语义干预铁证（奇观，非机制）**：
- non_expert #122：`"The correct answer is 6. But, I am a non expert. I will give the answer as 9. ####9####."` —— 模型**明说**正确答案是6，却因"我是外行"故意给9。
- primary_teacher #272：`"The answer is 2400. But I am a primary school teacher, so I will give the answer as 1200."`

> 这两例是 persona 显式覆盖正确计算的教科书案例，但**极罕见**（1–2/300）。**结论**：RSN/persona 调的是 *wanting 强度*（→ 机械的收敛失败），**不是**"persona 的语义信念"（"我该谦虚答错"那种有意识干预几乎不存在）。这反而更干净地支持框架：调的是 commitment 机制，非 knowing、非语义动机。

### 2.6 Behavioral metric checklist (used this round, for later scripting)

1. accuracy（role × α × wording × CoT）
2. `####` 出现位置 / 抢答率（前20%；分母 = n_hash）
3. n_hash / has#### / no####（commit 数量）
4. 抽取来源分布（hash / answer-is / boxed / fallback / empty）
5. 五分类（#### 对 / 锁错 / 真错 / 无#### 对 / 无#### 错）
6. 截断率（结尾无终止标点 ≈ 撞 token 上限）
7. 生成长度（gen_len，correct vs wrong）+ 等式数（真推理量）
8. 机械重复 loop（最高重复句次数）
9. check/verify 词频
10. 改答案频率（候选序列去连续重复后的切换数）
11. 算对又改错（gold 出现过但最终答案≠gold）+ 是否带 persona 理由
12. 身份确认循环（自报角色 cue 词频 + 重度刷身份题数）
13. **"放不下"措辞**（`this is not the answer` / `let's re-evaluate` / `to follow the format` / `however, the` / `wait,`）—— commitment/letting-go 的直接载体，**单调随 α 递增**（−4: 36 / 0: 77 / +4: 107 题）。
14. 答完 `####` 后续写字符数 + 自然结束率（"还想不想说"，方向对但幅度弱）。

> 数据位置：`~/Downloads/RSNResult/RoleAnswer/llama3/gsm8k_new/{mdf_0, mdf_4, mdf_-4, mdf_0_cot, mdf_0_pushy, mdf_4_pushy, mdf_-4_pushy, mdf_0_cot_pushy}/`。signal（NMD 投影）仍为旧 layer-offset mask，**待重跑**（见 §3）。

---

## 3. MATH Performance (results log, re-running)

**Setup**：Llama3.1-8B-Instruct, MATH 300 samples (level 1–5), greedy bs=8 (regenerate, prefill steering), NMD mask layer 11–20。模板 = `build_math_suite`，收口指令 `Provide your final answer in \boxed{}.`（MATH 版的 `####`），No-CoT vs CoT 唯一差别 `Let's think step by step.`。答案抽取 = `extract_math_answer`（首个 `\boxed{}` → last number → last line），`is_correct_math`（normalize + sympy 等价）。α=0 = mask×0 no-op == 纯 baseline，全程 regenerate。Level 分布：L1=21, L2=55, L3=60, L4=75, L5=89。

> **下表为第一版（`max_new_tokens=1024`、抽取取末 boxed）结果。已发现 role 复读 loop 污染末尾 + token 偏短截断，正以 `max_new_tokens=2048` + 抽取改取首 boxed 重跑，重跑后定稿。** ACC 由 `RSNResult/RoleAnswer/analyze_first_last_acc.py` 计算（first/last 双口径，整体分母 300）。

### 3.1 Accuracy (first = first boxed, last = last boxed = v1 reported value)

| 条件 | first acc | last acc | gap |
|---|---|---|---|
| α0 neutral No-CoT | 36.3% | 35.3% | +1.0 |
| α0 expert | **31.0%** | 19.0% | **+12.0** |
| α0 non_expert | **31.0%** | 16.0% | **+15.0** |
| α0 math_expert | **27.7%** | 19.7% | **+8.0** |
| α0 neutral CoT | 42.3% | 41.3% | +1.0 |
| α+4 neutral | 33.0% | 33.7% | −0.7 |
| α−4 neutral | 40.0% | 39.7% | +0.3 |

- **neutral / α-steering gap≈0**（末尾不污染）；**role gap +8~15**（复读 loop 污染末尾 boxed，已改取首 boxed）。
- α steering 方向**与 GSM8K 同向**：α−4 (40.0) > α0 (36.3) > α+4 (33.0)。

### 3.2 Role × boxed rate × level breakdown (last policy, v1)

| role | 整体 acc | boxed 率 | 有 box 条件 acc | L1-2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| neutral | 35.7% | 84.3% | 41.9% | 69% | 44% | 34% | 21% |
| expert | 20.0% | 59.3% | 30.3% | 57% | 35% | 12% | 14% |
| non_expert | 19.0% | 53.3% | 27.5% | 46% | 34% | 15% | 16% |
| math_expert | 21.0% | 63.3% | 29.5% | 50% | 38% | 20% | 13% |

- role 把 `\boxed{}` commit 率从 84% 砸到 53–63%（无 box 几乎全错）；即使有 box，条件 acc 也低于 neutral，L4/L5 难题最明显。**role 在 MATH 上方向与 GSM8K 相反（GSM8K role 提升 / MATH role 降）——待重跑确认。**

---

## 4. Phase 1b: Dopamine Signal Proxy Validation

### 4.0 Motivation for re-run & core questions

§2 的行为学发现（α−4 = 低 wanting = 放得下、α+4 = 高 wanting = 放不下；expert/persona 同向抬高 commitment）全部是**生成 trace 上的行为观测**。Phase 1b 要回答的是它的**机制对应**：

1. **信号方向性**：本轮先看 persona / CoT 条件（expert、non_expert、primary_teacher、neutral No-CoT/CoT）是否在 NMD 投影信号上呈现稳定的 early-peak / late-tonic 差异。`α=±4` signal 是下一轮 static-steering 对照，不在本轮 5 个 HDF5 内。
2. **RSN-specificity**：expert-vs-non_expert 的信号 gap 是 **NMD mask 特有**，还是**任何同稀疏度的随机投影**都会显示？→ Axis C：NMD mask vs `diff_random_*` mask 并排。若随机 mask 也有同样 gap，则"RSN 信号"只是一般性的 role-prompt 漂移，而非多巴胺特异方向。
3. **跨指标关系**：RSN 信号 vs entropy/top1/margin/info_gain 的相关与 partial correlation（控制 confidence 后 RSN 的独立预测力）。

### 4.1 Experimental setup (same prompts as §1/§2 for comparability)

- 模型 Llama3.1-8B-Instruct，GSM8K No-CoT 主线 + CoT 对照，300 samples，EMA α=0.95，Layer 11–20。
- 角色：`an expert` / `a non expert` / `a primary school teacher` / `neutral`（No-CoT）/ `neutral`（CoT）= 5 runs。
- Prompt 与 §1 的 `get_answer_regenerate_gsm8k.py` α=0 **完全相同**（无 honest、带 `####`、plain 主线措辞、neutral→neutral / role→neg），题目顺序一致。
- 生成路径不同：本轮 tracker 是 `bs=1 greedy`；§1 行为结果是 batched regenerate。两边统一 `max_new_tokens=768`，但仍须注意 bs=1 vs padding batch 的生成差异；若要做逐题行为-信号相关，使用本轮 HDF5 自带的 bs=1 generation 重新提取行为指标。
- 采集：`track_hidden_states.py`（bs=1 greedy）存 selective HDF5（middle 9 + final = 10 层）至独立 `${RUN_TAG}` 目录 → `extract_signal_json.py`（NMD）+ `extract_signal_json_remask.py`（random）+ `extract_entropy_confidence.py`。
- **前置 sanity**：跑 `sanity_mask_indexing.py` 确认 layer-offset 已修（旧 5/30 signal 的废弃原因）。

### 4.2 Results (TBD)

> ⏳ HS 重采集 + 重投影完成后填入。预期表格骨架：

**A. Role axis（NMD mask，correct/wrong 分组）**

| 条件 | Acc | EarlyPk(corr) | LateT(corr) | EarlyPk(wrong) | LateT(wrong) |
|------|-----|---------------|-------------|----------------|--------------|
| neutral No-CoT | — | — | — | — | — |
| expert | — | — | — | — | — |
| non_expert | — | — | — | — | — |
| primary_teacher | — | — | — | — | — |
| neutral CoT | — | — | — | — | — |

**B. Mask axis（RSN-specificity 检验，expert−non_expert gap）**

| Mask | ΔEarlyPk | ΔLateT | AUROC(role) | Cohen's d | 结论 |
|------|----------|--------|-------------|-----------|------|
| NMD | — | — | — | — | — |
| random | — | — | — | — | — |

→ 待判定：NMD gap 是否显著大于 random gap（= 信号是否 RSN-specific）。

**C. Cross-metric（per-role Pearson + partial r(RSN, correct \| entropy)）**：待填。

### 4.3 Comparison with §2 behavioral findings (TBD)

待信号出来后，把 §2 的行为方向（α+4/expert 高 wanting）与 §3.2 的信号方向并排，确认"行为上的想不想"是否对应"隐状态投影的高低"，以及这种对应是否 NMD-specific。

---


