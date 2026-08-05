
## TODO
0. 用Qwen2.5-7B复现
1. B1 主实验核查：prompt -> 两阶生成讨论 -> steering作用点 -> 回答情况 
1. B2 stress test：Hard-bare 跑 α=−4、+4，用于判断 α 是否改变 coverage/lock-in；不预设 rescue。
1. 得到三点结果后再决定是否做更宽 α sweep 或 C2 uncertainty scaffold。
2. HumanLLM
3. Behaviour: 测一下和人类的行为学对齐关系

---
我同意重新梳理，而且我认为不能只看你列的四段，还要补上两个关键环节：

```text
环境与状态构造
    ↓
Stage 1 prompt → rationale 生成
    ↓ 仅通过文本传递
Stage 2 prompt → candidate scores → argmax choice
    ↓
reward → 下一轮状态
    ↓
跨轮 discovery / persistence / outcome
```


### 1. Prompt 与状态是否表达了我们真正想测的任务

先展示同一个 seed 的 Round 1、首次 reward 后、Round 50、Round 100：

- Stage 1 完整 prompt；
- raw / sanitized rationale；
- Stage 2 完整 prompt；
- 四个 candidate suffix；
- 实际注入 token。

重点检查：

- successes、trials、empirical rate 是否完全正确；
- 未尝试臂是否明确作为 unknown，而不是被理解成 0；
- 剩余轮数有没有 off-by-one；
- arm identity 和 display position 是否保持 run 内固定；
- Stage 2 中重复出现的 `Do not state a final choice yet` 是否影响候选打分；
- rationale 是否被 sanitizer 整行删除或截断；
- Stage 1 最后一个注入 token 到底是什么。当前只严格审核了 action 的 `Button` 尾 token，rationale 注入点也应显式记录。

这是第一步，后面所有解释都建立在这里。

### 2. Stage 1 到底生成了什么“决策”

不能只统计 `exploration`、`uncertainty` 等关键词，应对 rationale 做结构化编码：

| 文本维度 | 要回答的问题 |
|---|---|
| Evidence reading | 是否正确读取次数、reward 和 empirical rate？ |
| Arithmetic faithfulness | 有没有引用错误数字或错误排序？ |
| Intended policy | 文本是在建议 explore、exploit、stay、switch，还是没有明确策略？ |
| Intended arm | 是否隐含或明确指向某条臂？ |
| Uncertainty | 不确定性来自未试臂、样本少，还是泛泛而谈？ |
| Horizon sensitivity | 前期和后期的策略语言是否改变？ |
| Action agreement | 最终选择是否执行了 rationale 中的判断？ |

最好先盲化 α 标签，对分层抽取的文本进行人工核对，再考虑自动分类，避免为现有结果定制关键词。

### 3. Rationale 是否真的影响 Stage 2

这是目前最大的未回答问题。

需要比较同一状态下：

- 正常 rationale；
- 空 rationale；
- 来自同状态、不同 α 的 rationale；
- 保持 action steering 不变，只替换 rationale；
- 保持 rationale 不变，只改变 action steering。

然后看 candidate score 和 argmax 是否变化。

如果替换 rationale 几乎不改变分数，那么 Stage 1 只是“看起来在思考”，实际选择主要由 Stage 2 prompt/state 决定。反之才能说文本推理是中介机制。

### 4. 固定状态下重新做阶段分解

当前完整 episode 很快分叉，后续差异混合了：

- 当轮 steering；
- 不同选择；
- 不同 reward；
- 不同下一轮统计；
- 不同 rationale。

因此需要像 GSM8K 的 matched-state 一样，建立 frozen-state replay。对完全相同的状态做：

```text
rationale α ∈ {−4, 0, +4}
action α    ∈ {−4, 0, +4}
```

不一定马上跑全部 2000 个状态，可以预先冻结：

- 每个 seed 的 rounds 1、5、10、25、50、75、100；
- 未发现最优臂；
- 刚发现最优臂；
- 经验最优发生变化；
- 连续失败后。

这会直接回答：

- α 改变了 rationale 什么内容；
- 相同 rationale 下，action logits 怎么变；
- 两阶段是否存在 interaction；
- 当前镜像失败是直接作用，还是轨迹反馈放大的结果。

这比继续盲目跑完整 episode 更重要。

### 5. Candidate score 不能只看 entropy 和 margin

建议把每条臂的 score 分解为几个可解释成分：

\[
Score(a,t)
\sim
\beta_v \cdot empirical\_rate
+\beta_u \cdot uncertainty
+\beta_n \cdot untried
+\beta_p \cdot previous\_choice
+\text{label FE}
\]

然后看 α 改变哪个系数：

- `β_untried`：是否真的改变 information seeking；
- `β_value`：是否改变 reward/value sensitivity；
- `β_previous_choice`：是否改变 persistence；
- label fixed effect：是否只是偏好 Button A/B/C/D；
- round interaction：是否改变 exploration stopping。

尤其要排除固定 label bias。`action +4` 的低覆盖有可能不是“更贪心”，而是某个 Button label 被统一推高，从而形成错误锁定。

### 6. 查看 reward 后的即时反应

Bandit 最适合检查与多巴胺相关的动态行为，但当前结果主要是累计指标。还应分析：

- win-stay；
- lose-shift；
- 首次成功后的坚持时间；
- 连续失败几次后换臂；
- 经验最优改变后多久跟随；
- 已发现最优臂后再次离开的概率；
- 前 20、21–50、51–100 rounds 的变化。

这能区分：

- reward sensitivity；
- policy persistence；
- indiscriminate switching；
- uncertainty-driven exploration；
- 对负反馈过度敏感。

如果 RSN 真有类多巴胺意义，这些轨迹指标可能比最终 OptFrac 更有理论价值。

### 7. Steering 本身还要做 manipulation check

`steering_fires` 证明“非零向量确实加进去了”，但没有回答：

- α 相对于原 hidden-state norm 有多大；
- Stage 1 的影响在第几个生成 token 后衰减；
- 第一处文本分叉通常发生在哪个 token；
- `+α/−α` 是否沿预期 RSN 方向移动投影；
- 不同层贡献是否一致；
- action score 改变是对所有候选统一平移，还是改变候选间相对排序。

这里可以像 GSM8K 一样画 token-level RSN projection，但我建议放在文本与固定状态检查之后。

### 8. 理论解释应预先列出竞争假设

不要只寻找一个能解释结果的“多巴胺理论”。至少同时保留：

1. 动机/信息寻求调节；
2. precision 或 gain 调节；
3. response persistence / inertia；
4. reward sensitivity；
5. 固定 label/logit bias；
6. 一般扰动或 task-interface interaction。

每个理论提前写出可区分预测。例如：

- information seeking：应提高 untried bonus，而不是 non-novel churn；
- persistence：主要改变 previous-choice coefficient；
- reward sensitivity：主要改变 win-stay/lose-shift；
- label bias：跨状态持续偏向同一 Button；
- 一般扰动：entropy 变化但与状态变量没有结构关系。

这样最后找到的理论才不是事后包装。

我建议我们下一步只做第 1 项：选定一个 α=0 Easy-bare seed，完整展示 4 个代表性 rounds 的 Stage 1 prompt、rationale、Stage 2 prompt、candidate score 和 choice，同时核对两个注入 token。确认 prompt 因果链没有隐藏问题后，再进入文本编码。

另外我在代码里看到一个无行为影响的小重复：`score_candidates()` 连续执行了两次相同的 `best = max(scores, key=scores.get)`。它不会改变结果，可以之后整理，不影响当前审计。
---
行文可以参考
nc2026.Hippocampo-neocortical interaction as compressive retrieval-augmented generation
---

### 5. Cross-Model and Post-Training Replication

**目标：** 判断该 latent gain mechanism 是否可泛化，以及 post-training 是创造还是 sharpen 它。
 
- 在同一模型家族的 Base → SFT → DPO/Instruction-tuned checkpoints 上分别 self-localize RSN。
- 比较 neuron/layer overlap、direction similarity、`G_prefill` gain、behavioral working point 和 steering sensitivity。
- 先用 Betting + GSM8K 两个代表任务；主结论稳定后再扩展 Qwen/Mistral，不立即复制全部 benchmark。
- 所有模型使用各自定位的方向与各自校准的 α/`ΔG_prefill`，避免直接搬用 Llama 的 mask 和 raw dose。

**完成标准：** 至少一个独立模型或同家族训练阶段复现 direction-specific、task-dependent working point；若只在 Llama3-IT 成立，则将结论限定为 model-specific mechanism。

人脑、fMRI/EEG、commit prediction 与动态 controller 暂时放到以上验证之后。当前执行顺序：

```text
analysis freeze
→ RSN specificity
→ slow-state behavioral validation
→ α × anxiety scale
→ direction-specific causal controls
→ cross-model/post-training replication
```

# Follow-up

0. Adaptive CoT router：只用 prefill 或 very early decode features 預測要不要 think。
   - RSN features: x_prefill, RSN projection mean / variance, middle-layer RSN activation, role-sensitive direction projection, first 5-10 decode token RSN slope
   - uncertainty features: MSP, entropy, constrained entropy, logit margin, E-option logit / abstention probability
   - frequency/dynamic features（參考 ICLR2026 Balanced Thinking）: step-level confidence variance, local fluctuation，用來區分 overthinking / underthinking
   - info-theoretic features（參考 NIPS2025 Think or Not）: InfoBias, InfoGain，先作 diagnostic / baseline，不急著當主控制器
   - baselines: entropy threshold, MSP threshold, answer logit margin, question length, random routing, always CoT, always No-CoT

1. 推理過程中 Dopamine curve 與 Thinking curve 的關係：在 reasoning model 的 `<think>` trace 裡對齊 backtrack / first-commit / hedging / verification marker。



# Reference: candidate anxiety / mental-health benchmarks

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

# Brain

## B1. 理論框架

**RSA（Representational Similarity Analysis）核心邏輯：**
1. 給模型和人腦看同樣刺激
2. 分別產生 N×N 相似矩陣（RDM）
3. 比較兩個 RDM 的 Spearman 相關

**我們的預測**：RSN Δh 方向上的 RDM 應與 **ventral striatum / vmPFC** 相關，而與語言區（Broca / Wernicke）不相關。這直接說明 RSN 操縱的是 reward 表徵，不是語言表徵。

## B2. 兩條執行路徑

### 路徑 A：公開 fMRI 數據 + RSA（1–2 個月）

**推薦數據集：**

| 數據集 | 來源 | 優點 | 適用性 |
|---|---|---|---|
| **NARPS Mixed Gambles（ds001734）** | OpenNeuro | 108人，vmPFC+striatum，已預處理，BIDS | 最直接：gambling 行為 ↔ 我們 Betting 的 RDM 比對 |
| **Tom et al. Mixed-Gambles（ds000005）** | OpenNeuro | Poldrack lab 2007 Science，OFC+striatum 乾淨 | Reviewer 熟悉，說服力強 |
| **MID Task（多個數據集）** monetary incentive delay | OpenNeuro 搜尋 | Wanting 最直接的 fMRI 範式（reward anticipation） | 對應 Betting 的 incentive salience | 

**執行步驟：**
1. 提取我們的 LLM 在不同 α 條件下，layers 11–20 的 hidden states → 構建 RDM
2. 從公開數據集提取 ventral striatum ROI 的 trial-level 激活向量 → 構建腦區 RDM
3. 計算兩個 RDM 的 Spearman 相關（需設計共享的刺激結構）
4. 對照組：同一套分析在語言區（IFG / STG）的相關應接近零

**關鍵挑戰**：LLM 刺激（MCQ 題目）和 fMRI 刺激（賭注任務）的對齊——需要設計一批「LLM 和人腦都能做」的共享刺激集。

### 路徑 B：行為層次對齊（1–2 週，不需要 fMRI）

找已發表的人類行為數據（帕金森 vs 健康人 vs DA-agonist 組的 Cambridge Gamble / Iowa Gambling Task），與我們的 α 劑量比對：

| 組別 | 人腦 DA 狀態 | 預測對應 α |
|---|---|---|
| 帕金森未服藥 | 低 tonic DA | α<0 |
| 健康控制 | 正常 DA | α≈0 |
| DA 激動劑 / L-Dopa ON | 高 DA | α>0 |

#### 文獻偵察結論（2026-07-23，已查）

**① 方向學高度一致（可寫，定性）。** 人類 DA 藥理學在 gambling task 上的方向與我們的 α 預測完全對齊：
- α+ (高DA) → 更衝動 / 早承諾 / delay aversion↑：Multiple Modes 2013（高 levodopa 劑量 → delay aversion↑）；Cools 2003（L-Dopa ON → 理性決策但衝動下注）；Riba/Pizzagalli 2008（pramipexole → boost 後保守傾向消失，更 reward-seeking）。**直接對上我們 CGT-Sequential 的 α+ → accept_step↓ / DAI 展寬。**
- α− (低DA / 未服藥帕金森) → risk-averse / 保守；DA 治療才轉向 risk-taking。

**② 嚴格「三組行為向量相關係數」做不了。** 查過的三篇關鍵文獻都沒有可對齊的 trial-level 或乾淨組均值表：Riba 2008 只報聚合百分比（placebo 47% vs pramipexole 49%，n.s.，核心效應在 boost 試次，無 mean bet/SD）；Multiple Modes 2013 只有 patients-vs-controls 總體、無 ON/OFF 分組、無 CGT 子量表 mean/SD；Cools 2003 結論為文字性。→ 缺三組可比數值向量，原設想的相關表無法計算。

**③ 落地形態改為「藥理學方向定性對齊」**：一段 correspondence-to-human-DA-pharmacology 敘述 + 一張定性對照表（未服藥帕金森↔α−；agonist/L-Dopa↔α+；我們的 α 在 CGT-seq / IGT 上再現人類 DA 方向）。夠撐 EMNLP/NeurIPS discussion 一節，但為定性方向複現，非定量 RSA。措辭停在行為層，勿跳神經層。嚴格定量須寫信向作者（Cools / Djamshidian 組）要 raw data（合作級，非 1–2 週自辦）。

#### ★ 可做的定量版本：用 Steingroever 617人常模校準 α 軸零點

反過來用「只有健康人常模」這個限制：掃 α∈{−8…+8}，看哪個 α 的 IGT 行為**分布**與 617 名健康人最接近。
- 若最佳對齊 α≈0 → **數據驗證**出 α=0 baseline = 人類正常 DA 水平（比「假設 α=0=健康人」強），順勢錨定 α−=帕金森方向、α+=agonist 方向，間接補上缺的兩組。
- 對齊在分布層做：主讀數 = 逐 block 學習曲線（net_block1–5，形狀 RMSE/相關）或 net_score 整體分布（KS / Wasserstein）。指標口徑與 `analyze_igt.py` 一致（net_score / net_block / deck preference = IGT 標準指標，與 Steingroever 可比）。
- **關鍵前提（待確認）**：須挑 Steingroever 中用**經典 Bechara 100-trial payoff scheme** 的子集（我們的 IGT 是這個），否則學習曲線不可比；並確認其提供 trial-level 選牌序列（才能算 block 曲線）。
- 零 GPU（IGT 各 α 數據已有）、零合作門檻（公開可下載）。措辭：「α axis 的行為零點與人類健康常模一致」，勿 over-claim 到神經層。

**可用公開數據 / 文獻**：
- [617人 Iowa Gambling Task 數據（Steingroever et al.）](https://openpsychologydata.metajnl.com/articles/jopd.ak)（純行為，完全公開；健康常模，用於零點校準）
- Riba, Krämer, Heldmann, Richter, Münte (2008) *Dopamine Agonist Increases Risk Taking but Blunts Reward-Related Brain Activity*, PLOS ONE — [PMC2423613](https://pmc.ncbi.nlm.nih.gov/articles/PMC2423613/)
- *Multiple Modes of Impulsivity in Parkinson's Disease* (2013), PLOS ONE — [10.1371/journal.pone.0085747](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0085747)
- Cools, Barker, Sahakian, Robbins (2003) — L-Dopa ON/OFF CGT（理性決策但衝動下注）
- 帕金森 CGT/IGT 文獻（Djamshidian et al. 2010/2011；DBS ON/OFF delay aversion, PMC3439437）— 多為論文表格均值，非 raw data


# PLAN

### Phase 1（2 週內，低成本或 zero-cost）

| # | 實驗 | 成本 | 目的 |
|---|---|---|---|
| 1 | Bandit 逐輪收斂曲線（現有數據） | 0 | 可視化證據 |
| 2 | Known-correct subset analysis（現有 log） | 0 | 純 wanting 證據 |
| 3 | 行為層次腦對齊（路徑 B，文獻比對） | 0 | 最快的腦連結 |
| 4 | Cambridge Gamble Task 設計 + 跑 Llama3 | ~1 GPU day | 排除 Betting 的信心 confound |
| 5 | Betting alpha sweep α∈{−8..+8} | ~1 GPU day | 畫出完整倒 U |

### Phase 2（1 個月內，核心 claim）

| # | 實驗 | 成本 | 目的 |
|---|---|---|---|
| 6 | **Tülu-3 SFT vs DPO on Betting** | ~2 GPU days | RLHF punchline |
| 7 | **Llama3-Base self-localized mask** | ~3 GPU days | 排除 mask 適配問題 |
| 8 | Random/PCA/Prompt baselines | ~1 GPU day | Reviewer 必問 |
| 9 | 公開 fMRI RSA 初版（NARPS ds001734） | 1–2 週 | 腦對應量化 |

### Phase 3（如有空間）

| # | 實驗 | 目的 |
|---|---|---|
| 10 | Qwen3 + Mistral Betting 跨模型 | 廣度 |
| 11 | Loss-aversion framing on Betting | Prospect theory 連結 |
| 12 | 共享刺激集設計 + neural encoding | 登頂級期刊的路 |
| 13 | **Pressure × Confidence Dissociation** | 區分 DA-like commitment vs confidence |
| 14 | **Task Difficulty × RSN Activation（現有數據）** | DA effort/uncertainty 對應 |

