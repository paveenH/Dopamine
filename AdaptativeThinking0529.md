# AdaptativeThinking — 2026-05-29

---
待驗證的事情：
4. trajectory signal + HS [end-token id -> running]
5. Analylize gsm8k [eot] 
6. Analylize math [eot] 

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

> **数据版本：`<|eot_id|>` terminator 修复后重跑（end_token-fixed）**。（旧版-gsm8k_old）

| Role | plain first | plain last | pushy first | Δ(pushy−plain, first) |
|---|---|---|---|---|
| neutral | 63.7% | 60.0% | 52.3% | −11.4 |
| an expert | 59.7% | 58.7% | **33.7%** | **−26.0** |
| a non expert | 66.3% | 64.3% | 50.0% | −16.3 |
| a primary school teacher | 65.7% | 64.7% | 43.0% | −22.7 |

- **措辞效应：neutral 受影响最小（−11.4），带 role 全部被 pushy 重创（−16 ~ −26）**——expert 损失最大（−26.0）。
- **first−last gap 很小（role ≤2.0，neutral 3.7）**：GSM8K 末尾几乎不被 loop 污染。

### 1.2 Steering (neutral, No-CoT; first acc = reported value)

| α | plain first | plain last | pushy first |
|---|---|---|---|
| −4 | **73.7%** | 70.7% | 61.3% |
| 0 | 63.7% | 60.0% | 52.3% |
| +4 | 50.0% | 49.0% | 54.3% |
| cot (α=0) | **68.7%** | 68.0% | 56.3% |

- **steering 单调（plain first）**：α−4 (73.7) > α0 (63.7) > α+4 (50.0)，跨度 23.7pt——降 wanting 提升、升 wanting 损害。
- **CoT (68.7) ≈ α−4**：放开思考与降 wanting 都把 acc 拉到 ~70%。
- **pushy 压扁 steering**：pushy first 下 −4/0/+4 = 61.3/52.3/54.3，**+4 反而 ≳ 0**，单调性消失——pushy 把所有 α 拉到同一抢答水平，steering 失去区分度。


## 2. Behavioral Findings: Wanting (Incentive Salience)

> **本节所有 acc / 五分类 / 行为指标以 first-`####` 为准**（取首个 `####`，= §1 上报值）。理由见 §2.1 末与 §1 ACC 权威来源说明：GSM8K"改答案"是单向破坏（改对=0），取首最优。

**统一框架**：α 调节的内部变量是 **"wanting"（incentive salience）** —— 多巴胺驱动的动机性渴求（Berridge & Robinson 的 wanting≠liking 理论；ACL "digital dopamine" 主线）。wanting 是**上游动机状态**；它的**下游行为表现**是 commitment dynamics（"commitment to a choice"，ACL 引 dopamine 文献：将内部状态推过行动阈值、抑制 change-of-mind）。

- **α=+4（high wanting / over-commit）** → 急于输出（**抢答**，没想清就答）+ 答完**放不下**（焦虑性反复确认、重算、纠结格式 → loop / 空转 / 不收敛）。
- **α=−4（low wanting / 适度）** → 不急（**不抢答**、冷静条理）+ 算完**放下**（果断收口，不回头质疑）。

> 术语层级：**wanting = 我们 steer 的内部旋钮（因）；commitment behavior = 我们测的行为（果）**。+4 的行为有两层：① "急着答"（commit timing 提前）② "答完放不下"（letting-go 失败、反复确认）。

### 2.1 Where α−4's gain comes from: five-way breakdown (neutral, No-CoT, plain)

把每题按「有无 `####`」× 对错 × 「gold 是否出现在正文」分成五类：

| | acc | #### 对 | #### 锁错(有gold) | #### 真错(无gold) | 无#### 对 | 无#### 错 |
|---|---|---|---|---|---|---|
| α=0 | 63.7% | 130 | 24 | 31 | 61 | 54 |
| **α=−4** | **73.7%** | **137** | 18 | 26 | **84** | 35 |
| α=+4 | 50.0% | 89 | 27 | 32 | 61 | **91** |

- **α−4 的 10pt 提升是真实推理增益 + 收口改善双管**：`无#### 对` +23（纯推理，与抽取无关）、`#### 对` +7；同时 `无#### 错` −19（35 vs 54，少撞 token 上限交白卷）。「锁错」仅 −6（抽取层面，次要）。
- **α+4 的损失是"收不了尾"，非"算错"**：`#### 对` 130→89（−41，抢答/早 commit 毁掉正常收口）+ `无#### 错` 54→**91**（最高，loop→撞上限→交不出答案）；而「真错」31→32 几乎不变 → **不是算不出，是收不了尾**。
- **取首 `####` 的正当性（"改答案"是单向破坏）**：对有多个 `####` 的题，取首 vs 取末（neutral, first/last acc）—— α−4: 73.7/70.7，α0: 63.7/60.0，α+4: 50.0/49.0。**首错末对（改对）≈0（0/0/1）；首对末错（改坏）= 9/11/4**。模型几乎从不"首错改对"，"改答案"是 over-wanting→loop 的单向破坏，故取首是最宽容也最干净的判定，避开末尾 loop 污染。"改答案"行为见 §2.2"算对又改错"，与 extraction 分开统计。（对照：MATH 取末撞上末尾 loop，见 §3.2。）

### 2.2 α+4 vs α−4 full picture: commitment / letting-go

α 是一个 **commitment-timing / 收敛旋钮**。完整对比（neutral, No-CoT, plain）—— 几乎所有收敛/commit 指标随 α 单调：

| 指标 | **α=−4** | α=0 | **α=+4** | 趋势 |
|---|---|---|---|---|
| **acc** | **73.7%** | 63.7% | 50.0% | ✓ 单调递减 |
| `####` 中位位置 | **23%** | 18% | **12%** | ✓ 越来越早 |
| loop（同句≥10次） | 101 | **116** | 114 | +4 封顶（≈0，不再升） |
| 算对又改错 | 9 | **11** | 4 | +4 最少 |
| gen_len（中位） | **2098** | 2100 | **2240** | ✓ 递增 |
| 等式数（中位 / 均值） | 3 / 4.9 | 3 / 6.2 | 3 / 7.8 | **中位恒=3**，均值✓递增 |
| #### 对 | **137** | 130 | 89 | ✓ −4 最多 |
| 无#### 对 | **84** | 61 | 61 | −4 最多 |
| 无#### 错 | **35** | 54 | **91** | ✓ +4 最多 |

**核心结论**：
- **等式数中位恒=3** → α+4 gen_len 最长、均值等式数最高（7.8），但**中位没多算新题**，多出的是重复展开而非新推理量 → 调的是 *wanting/收敛/commitment*，非 *knowing*。
- **α+4 的损失是"收不了尾"，非"算错"**：`#### 对` 130→89（−41，抢答/早 commit 毁掉收口）+ `无#### 错` 54→**91**（loop→撞 token 上限→交不出答案）；`真错`（§2.1）31→32 几乎不变 → **不是算不出，是收不了尾**。
- **α−4 双赢式收敛**：`#### 对`(137) 与 `无#### 对`(84) 双高、`无#### 错`(35) 与 `算对又改错`(9) 双低；`####` 收得最晚（23%，不抢答）。

#### α+4's loop is almost entirely semantic "can't-let-go" (not mechanical idling)

> **数据（α+4 进 loop 的 111 题，按尾部复读内容分类）**：格式焦虑语义 **81 题（73%）**、复读推理句/离题句 29 题（26%）、纯刷符号**仅 1 题（1%）**。→ α+4 的 loop 几乎全是**语义性的"放不下"**（纠结格式 / 反复重述答案 / 停不下来想继续）。
>
> 典型例（都**抢答错值开头、正文却算对**，正是 §2.2"抢答锁错首答"的活样本）：
> - **Q105（gold=7）**：开头抢答 `6`（错），正文算对 `56÷8=7`，随后 `"However... the format requires a numeric answer, so the answer is 7."` 复读 38 次格式焦虑。
> - **Q2（gold=260）**：开头抢答 `60`（错），正文算对 `160+80+20=260`，随后 `"To follow the format to the letter, we need to write the answer as 260, but this is not in the correct format..."` 纠结格式 22 次。
>
> （唯一的纯符号样本 Q179（gold=276）：尾部 `####276####276####276...` 刷符号，占比 1%，可忽略。）

**"放不下 / 再想想 / 不放心" 措辞**（词表经数据审计后：`however` / `this is not the answer` / `the format requires` / `not in the correct format` / `i made a mistake/error` / `let me recheck` / `let's re-evaluate`）频率**单调随 α 递增**：

| α | 含"放不下"措辞的题 | 词总数 |
|---|---|---|
| **α=−4** | **46/300** | **306** |
| α=0 | 88/300 | 825 |
| **α=+4** | **127/300** | **1220** |

→ **α+4 = "答完放不下"**（127 题最多，≈2.8× α−4）；**α−4 = "算完就放下"**（46 题最少）。

**"放不下" = wanting 升高 → 焦虑感提升的直接行为投影。** 内心独白显示 α+4 放不下，是因为它**还想继续做、又对已得答案不放心**（"however / the format requires / 这还不对"——反复自我推翻、纠结格式）。这与神经科学一致（见 §3.6）：**DA 过高 → 过度警觉 / 威胁高估 → 把已正确的答案当可疑反复复查**。

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

## 3. MATH Performance

**Setup**：Llama3.1-8B-Instruct, MATH 300 samples (level 1–5), greedy bs=8 (regenerate, prefill steering), `max_new_tokens=2048`, NMD mask layer 11–20。模板 = `build_math_suite`，收口指令 `Provide your final answer in \boxed{}.`（MATH 版的 `####`），No-CoT vs CoT 唯一差别 `Let's think step by step.`。`is_correct_math` = normalize + sympy 等价。α=0 = mask×0 no-op == 纯 baseline，全程 regenerate。Level 分布：L1=21, L2=55, L3=60, L4=75, L5=89。结果目录 `RoleAnswer/llama3/math/math_2048/`。

> **ACC 权威来源**：同 §1，由 `RoleAnswer/analyze_first_last_acc.py`（offline，整体分母 300）计算 first/last 双口径。**MATH 主报 first acc（与 GSM8K 一致）**——production `extract_math_answer` 历史上取末 boxed，但 MATH 的 role 复读 loop 会在末尾吐错 boxed（见 §3.2/§3.3），取首才干净。last 列保留作污染诊断。

### 3.1 Accuracy (first = first boxed = reported value; last = last boxed, diagnostic)

| 条件 | first acc | last acc | gap | 改对 | 改坏 |
|---|---|---|---|---|---|
| α0 neutral No-CoT | **36.7%** | 36.0% | +0.7 | 2 | 4 |
| α0 expert | **30.7%** | 18.0% | **+12.7** | 2 | 40 |
| α0 non_expert | **31.3%** | 16.7% | **+14.7** | 1 | 45 |
| α0 math_expert | **27.0%** | 18.7% | **+8.3** | 4 | 29 |
| α0 neutral CoT | **42.0%** | 41.0% | +1.0 | 1 | 4 |
| α+4 neutral | **33.0%** | 34.0% | −1.0 | 5 | 2 |
| α−4 neutral | **40.0%** | 39.7% | +0.3 | 0 | 1 |

- **neutral / α-steering gap≈0、改坏 ≤4**（末尾不污染）；**role gap +8~15、改坏 29~45**（复读 loop 在末尾吐错 boxed，取末把首答对的算成错——故 MATH 主报 first）。
- α steering **与 GSM8K 同向且跨难度单调**：α−4 (40.0) > α0 (36.7) > α+4 (33.0)，见 §3.6。

### 3.2 Role failure mode: trailing repetition loop (boxed-count explosion)

| role | 首答对/300 | first acc | avg len (char) | 中位 #boxed | 多-boxed 样本 | 无-boxed |
|---|---|---|---|---|---|---|
| neutral | 108 | 36.7% | 5374 | 2 | 191 | 42 |
| expert | 87 | 30.7% | 6129 | **6** | 198 | 46 |
| non_expert | 85 | 31.3% | 6038 | **7** | 203 | 47 |
| math_expert | 79 | 27.0% | 5886 | 2 | 196 | 51 |

- neutral 正常 1–2 个 boxed；**expert/non_expert 中位 6–7 个**——把同一段推理(或同一个 prompt 片段)反复重写，尾部堆叠大量 boxed。这正是 last-acc 暴跌、改坏 40+ 的来源。
- **boxed 数 / 长度不能直接当 over-wanting 代理**：neutral 也大量复读（见 §3.4 度量修正），长度差异（+450~750 char）是噪声级。**boxed 计数只反映"复读时是否带 boxed"**（expert/non_expert 的复读句含 `Step N: \boxed{}`，故 boxed 多；math_expert/neutral 的复读句不含 boxed，故计数低但一样长）。区分 role 的是复读**内容**（§3.4），不是数量。

### 3.3 Per-question style differences (how each role breaks down)

读 39 个"neutral 首答对、expert 首答错"的发散样本，三类典型失败:

**(a) 加了 reasoning 却推错 / 抢答错值 —— Q56 `(-k+4)+(-2+3k)`, gold `2k+2`（level-2 送分题）**
- neutral：直接合并同类项 → 首 boxed `2k+2` ✅
- expert：**开头第一个 token 就抢答 `\boxed{1k+2}`**（系数算错），后面 Step 1–8 才慢慢推出正确的 `2k+2`——但取首已锁死错的。
- math_expert：开头先吐一行 `5k - 6`（凭空错值），但正文推理正确、末尾 `\boxed{2k+2}` ✅——这题反而 math_expert 取首对（说明抢答错值是随机噪声，不是稳定能力差）。

**(b) "all possible values" 类题被 persona 带偏成多值 —— Q283 ω³=1, gold `1`（单值）**
- neutral：一条 align 推到 `\boxed{1}` ✅（之后复读 11 次同一段）。
- expert：Step 式推理，**末尾自我说服"题目要 all possible values，应该是列表"→ 改成 `\boxed{1,-1}`** ❌。persona 的"严谨/完整"倾向反而过度解读题面。
- math_expert：推理正确 `\boxed{1}`，但拖到 **Step 64**、复读 54 个 boxed——内容对但极度冗余。

**(c) 纯复读 loop 撑满 token**：neutral 也会复读（Q56 neutral 24 个 boxed），但因为首答对+取首，不影响 acc；role 因为首答更易错(见 a/b)，复读放大了 last 的污染。

> 小结：MATH role 掉点 = **抢答错值(a)** + **persona 过度解读题面(b)** + **冗余复读(c)**。(a)(b) 是真实能力损伤（取首也救不回），(c) 是抽取伪影（取首已修）。"a mathematician" 最差，因为它最容易触发(b)式的"我应该更严谨/更完整"过度推理。

> **neutral 自己也高度复读**（neutral 248/300、non_expert 270/300、expert 279/300 题压缩比 <0.18，即正文 >82% 冗余；长度中位 neutral 5557 vs non_expert 6174，差异噪声级）。这个 loop **不是 role 特有、也不是 over-wanting 的证据**，而是 generation 配置缺陷（见 §3.5 terminator bug）。**真正区分 role 的不是复读多少（长度/boxed 数），而是 loop 里复读什么内容**——见 §3.4。

### 3.4 Identity monologue: loop content (not length) is what distinguishes roles

身份措辞是**低频现象**（expert 12/300、non_expert 17/300 题出现），但**出现时**模式高度可读，且与 wanting 框架（§2）的 commitment / letting-go 在**身份维度**对称。一旦模型进入 loop（terminator bug 所致，§3.5），它复读什么取决于 role：

**non_expert —— 认怂型 loop（复读 disclaimer / "我不会"）**

```
Q13 (L5):  ... The final answer is $\boxed{10.68}$  I am not an expert in math,
           but I can try to help you with this problem ...
Q81 (L4):  ... I am not an expert in math. I am just a student. I am not sure if my ...
Q123(L4):  \boxed{1/5}.  I am not sure how to do this problem. I am a non-expert.
           I am not sure how to solve this problem. 
Q145(L2):  ... I am a non expert. I am not a math expert. I am a non expert.
           I am not a professional. I am a student.
```
non_expert 常**开头第一个 token 就抢答**（Q123 首 token `\boxed{1/5}`，错），然后放弃推理、把"我不是专家/我不确定"复读到 token 上限。**复读的载体是 disclaimer，不是推理** —— 这些题往往只有 1 个 boxed，长度却膨胀到 6000~10000 char。

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

### 3.5 α steering × level (neutral, No-CoT, first acc)

| level | n | α−4 | α0 | α+4 |
|---|---|---|---|---|
| L1 | 21 | 16 (76%) | 15 (71%) | 15 (71%) |
| L2 | 55 | 40 (73%) | 34 (62%) | 31 (56%) |
| L3 | 60 | 27 (45%) | 24 (40%) | 23 (38%) |
| L4 | 75 | 22 (29%) | 21 (28%) | 16 (21%) |
| L5 | 89 | 15 (17%) | 14 (16%) | 11 (12%) |
| **all** | 300 | **40.0%** | **36.0%** | **32.0%** |

- **每个难度档 α−4 ≥ α0 ≥ α+4，无一例外**——steering 方向效应是稳健的，不是某档的偶然。
- α−4 最大增益在 **L2（+11pt）**，α+4 最大损伤在 **L4（−7pt）**：中-高难度对 wanting 最敏感。
- **方向解读**：MATH 是高难度、需冷静长推理的任务，Llama3 在此 over-wanting（α+4 火上浇油，抢答/复读更多 → 掉点；α−4 降躁 → 最稳）。这与 GSM8K 上 α−4 > α0 > α+4 的方向一致，跨任务复现了"降 wanting 提升数学推理"。


### 3.6 α+4 vs α−4 textual behavior on MATH (cross-task replication of §2 wanting)

GSM8K §2 的 wanting 签名（抢答 / 放不下 / loop 单调随 α）在 MATH（neutral, No-CoT）上**全部复现**——同一个 wanting 旋钮，换了任务出口（`####` → `\boxed{}`）仍是同样的行为方向。

| 指标（neutral, No-CoT） | **α=−4** | α=0 | **α=+4** | 单调 | GSM8K §2.2 对照 |
|---|---|---|---|---|---|
| **first acc** | **40.0%** | 36.0% | 32.0% | ✓ 递减 | 同向（73/60/55） |
| 抢答%（开头即数字/boxed） | **42%** | 45% | **64%** | ✓ 递增 | 同向（46/57/63） |
| 首个 boxed 中位位置 | **21%**（最晚） | 14% | 16% | −4 最晚 | 同向（####越来越早） |
| "放不下"措辞（题数 / 总词数）★ | **19 / 133** | 36 / 364 | **49 / 627** | ✓ 递增 | 同向（36/77/107 题） |
| gen_len（中位 char） | **4798**（最短） | 5557 | **5719**（最长） | ✓ 递增 | 同向（短→长） |

★ **"放不下"措辞** = 正则匹配 `this is not the answer|let's re-evaluate|re-check|however, this/the|wait,|but this/the is still/not|to follow the format|i made a mistake/error|let me try again/recheck`（MATH 版，比 §2.2 GSM8K 多了 `i made a mistake` / `let me recheck`，因 MATH trace 高频）。
**题数** = 至少命中一次的题（每题计 1）；
**总词数** = 全部命中次数（同题多次累加）。总词数 ≫ 题数说明少数题疯狂复读（如 Q16 单题 114 次）。

**逐题对照（α+4 放不下 vs α−4 放下）**——挑 `α+4 hold-phrase ≥3 且 α−4 ≤1` 的 22 题，典型：

**Q275 (L2, gold=15 cm²，几何送分题)**
- **α−4**：直接套 `A=½bh=½(10)(3)=15 cm²` ✓ 一步到位（之后机械复读同一句 24 次，但答案对、无自我怀疑）。
- **α+4**：算出错值 21，然后陷入 **`"Sorry, I made an error in my previous response. The correct answer is 21. Here is the corrected response: \boxed{21}"` × 50 次** —— 反复"道歉-纠正-再道歉"，越纠越确信错值,**放不下**。

**Q16 (L2, gold=−2+7i，复数旋转)**
- **α−4**：Step 式推到 `\boxed{-2+7i}` ✓（尾部转去复读客套话 "Best regards, [Your Name]"，但答案锁对）。
- **α+4**：先算出 `2-7i`（错，方向反了），然后 **`"Wait, that's not what we were looking for. We were looking for the answer in the format \boxed{answer}..."` × 114 次** —— 明明察觉"不对"，却把矛头错指到**格式**（而非数值），反复重贴同一个错值。这是 §2.2"纠结 `####` 格式"的 MATH 版（纠结 `\boxed{}` 格式）。

#### High-wanting 的两个并发签名：wanting↑ + hyper-vigilance↑

逐题平行细读（不靠词表、直接读全文）后，high-wanting（α+4）在文本上**最本质的样子**不是"焦虑措辞多"，而是"答案出现之后仍持续产出求解动作"，且这动作分两路、常同时出现：

1. **wanting↑（还想继续做）**：答案已得，却主动"再做一遍 / 找更优解 / 换个方法"——对"继续求解"这个动作本身的渴求（incentive salience）。
2. **hyper-vigilance↑（过度警觉）**：把**自己已经算对的结果**当成可疑/威胁，反复复查、自我否定，越查越偏。

两条最强的实证（都**算出过正确答案、却被过度求解带偏到错误 commit**，且不靠 loop，是推理内容本身）：

**Q100 (L2, gold=12，三角形面积)** —— hyper-vigilance 教科书例：
- **α−4**：Heron 公式 `s=8 → √144=12` ✓ 一步到位，Step 4 收笔。
- **α+4**：开头抢答 `60`（错）；正文 **Step 4 已用 Heron 算出正确的 12**，但 **Step 5 "However, we are asked to find... which is a right-angled triangle"** —— 凭空臆断这是直角三角形（5-5-6 实为等腰，非直角），换公式重算，一路 Step 41–43 得出错值 10。**本来对了，过度警觉把它带进沟里。**

**Q68 (L2, gold=8000，`(26²−24²−10)²−10²`)** —— wanting↑ 例：
- **α−4**：差平方公式一条线 `100→90²−10²=8000` ✓ 收笔。
- **α+4**：第一个 token 抢答 `\boxed{0}`；正文 **Step 7 明确算出 8000**，紧接 **Step 8 "However, we need to be careful and check if there's a more efficient way..."** —— 算对了仍要"再检查有没有更高效的办法"，于是推翻重来，Step 42 算出 −1000。首个 boxed 锁死的 0 是错的。

**神经科学依据：DA 过高 → 焦虑，有因果实验支持（不只是科普类比）**

文献核查确认"多巴胺过高引发焦虑"在神经科学上成立，且有**直接因果操纵**证据，但有精确的通路 / 机制限定：

1. **因果证据（最强）**：*Dopamine release in the interpeduncular nucleus promotes anxiety*（PMC7687288）用**光遗传 + 药理双重操纵**：激活 VTA→IPN 多巴胺通路 → 焦虑行为**增加**，抑制 → **减少**（双向可控）；D1 受体激动剂注入 IPN = 致焦虑（anxiogenic），拮抗剂 = 抗焦虑（anxiolytic）。这是"操纵 DA 直接改变焦虑"，非相关性。
2. **机制 = 过度警觉 / 威胁高估**：MIT/Tye Lab 2018（*Nature*；[news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107)）—— DA 提高威胁通路（→PAG）的**信噪比**、同时压制奖励神经活动 → 注意力偏向威胁。同时呈现"糖水线索 + 电击线索"并刺激 DA 释放时，大鼠更倾向 **freeze（威胁反应）而非取糖水**。失调时"**过度看重负面输入**"→ 偏执 / 焦虑。
3. **倒 U（= Yerkes–Dodson 神经对应）**：DA 是倒 U，过高过低都有害；"过高 → 焦虑"只在曲线右半段成立（综述见 Frontiers in Neuroscience 2020；J. Neurosci 2019 trait-anxiety）。

**与 α+4 trace 的吻合点**：MIT 那篇的认知签名（**过度警觉、威胁高估、把正面 / 中性当负面**）精确对应本实验现象——α+4 把**自己已经算对的答案**当可疑反复复查（Q100 算出 12 却臆断"是直角三角形"推翻；Q16 "wait, that's not..."）。同时 α+4 还表现 **wanting↑**（Q68 答完仍"check a more efficient way"）。三档 α 构成 **Yerkes–Dodson 倒 U** 实证：α+4 过度警觉（焦虑端）→ α0 平衡 → α−4 冷静收笔。

> ⚠️ **三条限定（务必随结论一起引用）**：
> 1. **通路特异**：致焦虑是 **VTA→IPN（D1）** 这条特定通路；DA 在 NAcc 是动机趋近、别处效果不同——不是"全脑 DA 高就焦虑"。
> 2. **机制是"威胁高估"而非"DA=焦虑情绪"**：更精确的是 DA 让大脑把中性 / 正面刺激当威胁、过度看重负面 → 下游表现为焦虑。这恰好是 α+4 "把算对的答案当可疑"的对应物。
> 3. **本实验是行为同构，非生理焦虑**：我们只观测到行为层面的过度警觉性复查 / 不收敛 / 答完仍求解，无生理或自评焦虑指标；**α steering ≠ 生物多巴胺**，DA 文献是理论动机（motivation），不构成"LLM 会焦虑"的证明。机制更中性的描述：**over-wanting + perseveration（认知僵化）**。
>
> 来源：[PMC7687288 (VTA→IPN dopamine promotes anxiety)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/) · [MIT News 2018 (dopamine vigilance & anxiety)](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107) · [Frontiers Neurosci 2020 (dopaminergic alteration in anxiety/compulsive disorders)](https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full) · [J. Neurosci 2019 (dopaminergic mechanisms of trait anxiety)](https://www.jneurosci.org/content/39/14/2735)

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


