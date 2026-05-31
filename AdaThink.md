# Role-Conditioned Thinking in Reasoning Models

<!-- 主線導覽（三份文檔共用，每份開頭都有）

整條研究主線（三段）：
  RSN
    → 行為學多巴胺（Behavioral Dopamine）← Ada_Dopamine.md
        → 腦科學多巴胺（Brain Dopamine）← Ada_Dopamine2.md §五
            → 多巴胺與思考曲線（Dopamine & Thinking Curve）← AdaptativeThinking.md

  本文檔是 Thinking Curve 的額外延伸驗證，不在主線框架內。

【本文檔定位】
這是交給學弟的執行文件。
核心問題：Roleplaying（persona prompt）是否會對 reasoning model 的思考過程造成影響？

與主線其他文檔的關係：
  - Ada_Dopamine.md 在非推理模型上已確認 RSN = wanting 控制（行為學層面）
  - AdaptativeThinking.md 在 Llama3-8B 上用閉環控制測試 EMA 波形 ↔ acc 的關係
  - 本文檔進一步問：如果 persona 能改變 non-reasoning model 的 wanting state，
    那在 reasoning model 裡，這個 state 是否會表現在 <think> trace 上？
    （thinking token 數、backtrack 頻率、first-commit 位置、calibration）

【為什麼這是「升華」】
Non-reasoning model 的 wanting 只能從 output logit 間接推斷（acc、E rate、bet size）。
Reasoning model 把思考過程外顯為 <think> trace，讓我們第一次能直接觀察
motivational state 如何調節推理行為的每一步——這是從行為黑箱到過程透明的躍升。

【執行分工】
學弟負責：Phase 1（行為層 pilot）→ Phase 2（完整 profiling）→ 視結果決定 Phase 3/4
主線對接：Phase 3/4 的 hidden state 分析復用 RSN pipeline（track_hidden_states.py），
          結果回饋到 AdaptativeThinking.md 的 dopamine-thinking curve 框架。

關聯文件：
  Ada_Dopamine.md — 行為學理論框架（wanting/knowing/Yerkes-Dodson）
  Ada_Dopamine2.md — 腦科學 RSA 方向
  AdaptativeThinking.md — EMA 波形控制實驗 + dopamine-thinking curve 理論
-->

## Adaptive Reasoning

- **01.nips2025.Reasoning Models Better Express Their Confidence**
  Reasoning models show better verbalized confidence calibration than non-reasoning counterparts, and calibration improves as CoT unfolds. The most relevant point for this project is that slow thinking seems to support uncertainty refinement, not just longer answer generation.

- **02.nips2025.Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL**
  AutoThink trains R1-style models to choose when to think and when to answer directly, using multi-stage RL to avoid collapse into always-think or always-no-think. This is the closest prior work to the "when-to-think" routing problem.

- **03.nips2025.Think or Not? Exploring Thinking Efficiency in Large Reasoning Models via an Information-Theoretic Lens**
  This paper frames long CoT as potentially inefficient: later reasoning steps often bring decreasing InfoGain and increasing semantic drift. Its entropy-based Adaptive Think method is a strong baseline for early stopping and token-cost reduction.

- **04.iclr2026.Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling**
  DECS separates necessary reasoning prefix tokens from redundant post-answer thinking, then penalizes only the redundant part. This supports the view that the target is not "short reasoning" in general, but removing unnecessary continuation after the answer is already recoverable.

- **05.iclr2026.Efficient Reasoning with Balanced Thinking**
  ReBalance uses step-level confidence and confidence variance to detect overthinking versus underthinking, then dynamically steers hidden states. This is especially relevant because it treats adaptive reasoning as a continuous control problem rather than a binary CoT switch.

- **06.iclr2026.Rethinking LLM Reasoning: From Explicit Trajectories to Latent Representations**
  Latent Reasoning Tuning replaces explicit CoT tokens with latent reasoning vectors produced by an auxiliary reasoning network. The implication is that reasoning support need not always appear as visible CoT; some adaptive thinking may happen through latent representations.

- **07.aaai2026.Promoting Efficient Reasoning with Verifiable Stepwise Reward**
  VSRM scores intermediate reasoning steps by whether they improve verifiable answer correctness from partial trajectories. This gives a process-level alternative to crude length penalties: reward useful steps, suppress ineffective steps.

- **08.nips2025.When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs**
  CoT can hurt instruction-following by shifting attention away from hard constraints such as format, length, keywords, and language restrictions. This is a cautionary result: reasoning should be selectively invoked, not treated as universally beneficial.

- **09.acl2026.Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens**
  This paper argues that CoT may often reflect learned in-distribution structured patterns rather than robust algorithmic reasoning. For our framing, it warns against equating CoT behavior with genuine reasoning capacity.

- **10.2025.Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective**
  RaML interprets reasoning trajectories as test-time pseudo-optimization: each reasoning token changes the hidden state and answer distribution like an inner-loop update. This provides a useful theoretical lens for why CoT can help on hard tasks but also why shorter effective trajectories may exist.

- **11.acl2024.An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs**
  This work links CoT prompting to stronger activation of arithmetic-reasoning-related FFN neurons. It is relevant to RSN because it provides precedent for studying CoT effects through neuron-level activation patterns rather than only surface prompts.

## Thinking-process analysis (trace-level)

這組是分析 reasoning model 的 thinking trace 本身（cue word、step structure、overthinking/underthinking 等）的方法學先例。

- **2025.Demystifying Long Chain-of-Thought Reasoning in LLMs (Yeo et al., arXiv 2502.03373)**
  系統分析 R1-style 模型的 long-CoT trace，定義 reflection / verification / backtracking 等行為單元並做 cue word 頻率統計。**本項目 Layer 1–2 metric 的最直接方法學模板**，可借用其 taxonomy 與 cue-word 列表。

- **2024.Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs (Chen et al., arXiv 2412.21187)**
  「Overthinking」概念的代表性源頭，提出 early-answer、redundant verification 等 trace-level 指標。可參考其 verification depth / first-commit position 的操作化方式。

- **2025.Thoughts Are All Over the Place: On the Underthinking of o1-like LLMs (Wang et al., arXiv 2501.18585)**
  反向研究 underthinking — 模型頻繁切換 reasoning path 卻不深入，提出 thought-switching frequency 指標。直接補進 Layer 1 metric 列表。

- **2025.s1: Simple Test-Time Scaling (Muennighoff et al., arXiv 2501.19393)**
  「budget forcing」一節分析強制延長 thinking 時 trace 如何變化，是 Phase 1 baseline / control 的有用對照。

- **2023.Let's Verify Step by Step (Lightman et al., OpenAI PRM800K)**
  最經典的 step-level annotation 資料集，定義 step 邊界與正確性標籤。Layer 2 標註方案的直接參考。

- **2024.ProcessBench: Identifying Process Errors in Mathematical Reasoning (Zheng et al., arXiv 2412.06559)**
  step-level error detection 的 benchmark，step 分類粒度比 PRM800K 更細，對「5 類 step」定義有直接參考。

- **2024.LLMs Cannot Find Reasoning Errors, but Can Correct Them! (Tyen et al., arXiv 2311.08516)**
  探討用 LLM 當 annotator 標 reasoning step 的可行性與限制，對應 Layer 2 §標註方法 B。

- **2024.Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters (Snell et al., arXiv 2408.03314)**
  分析 test-time compute 如何分配到 thinking trace 不同部分，sequential vs parallel revisions 的概念可借來分析 backtracking。

- **2024.Chain of Thoughtlessness? An Analysis of CoT in Planning (Stechly et al., arXiv 2402.08164)**
  批判 CoT 是否真為 algorithmic reasoning，對「什麼算 genuine reasoning step」提供批判性視角，寫 limitation 有用。

## Persona / Role × Reasoning (prior work)

**gap：以下工作幾乎全部在 non-reasoning model（GPT-3.5/4、Llama-2/3、Mistral）上做，且只看 acc / bias，不看 thinking trace。Reasoning model（R1 / o1 / QwQ）上的 persona × thinking process 研究目前空白 — 這正是本項目的 novelty。**

- **nips2023.In-Context Impersonation Reveals Large Language Models' Strengths and Biases (Salewski et al.)**
  最早系統做 persona × task 的代表作，跨多 task 跑 expert / non-expert / domain-specific persona。只報 acc，不看 trace；模型為 GPT-3.5 / Vicuna，皆非 reasoning model。本項目的主要 contrast paper。

- **iclr2024.Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs (Gupta et al.)**
  研究 persona 如何改變 reasoning bias，但停在 outcome-level，trace 內部未分析。模型為 GPT-3.5 / 4。

- **naacl2024.Better Zero-Shot Reasoning with Role-Play Prompting (Kong et al.)**
  role-play prompt 提升 reasoning acc 的實證，但僅 outcome-level；模型為 GPT-3.5 / 4 / Llama-2。

- **acl2024.Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization (Tseng et al., arXiv 2406.01171)**
  Persona / role-play 文獻 survey，可用來定位本項目在整個 persona 研究地圖上的位置。Survey 範圍幾乎不涉及 reasoning model。

- **2024.When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of LLMs (Zheng et al., arXiv 2311.10054)**
  Persona system prompt 大多無顯著 acc 提升，提示 persona 效應更可能在 thinking process 而非 outcome — 強化本項目「看 trace 而非看 acc」的 framing。

## Motivation

- Persona prompt 是否會系統性地調節 reasoning model 的 thinking process（depth / verification / backtracking / commitment / calibration）？
- 這種行為層的調節，是否能被 RSN signal / dopamine-like curve 解釋或預測？

最終目標：把 persona → thinking behavior → RSN signal → dopamine-thinking curve 串成一條完整證據鏈。

## Hypothesis

Persona prompt 不只是改語氣，而是改變模型的 motivational / commitment state。在 reasoning model 中，這個 state 表現為：

- 是否進入 long thinking、thinking token length
- backtracking / verification / hedging 頻率
- first-commit position（多早出現候選答案）
- final answer confidence / calibration
- overthinking / underthinking 

## Persona dimensions

精簡到三個（近似）正交的軸 — 太多軸會互相污染（confidence 與 emotion 行為重疊），且 prompt 工程負擔大。先把這三軸做穩，其它軸（emotion、social role）放 future work。

| Axis | Levels | Note | 預期影響的 thinking 維度 |
|---|---|---|---|
| Confidence | uncertain / confident | confidence | backtrack rate、hedge、first-commit、calibration |
| Cognitive style | analytical / intuitive | Secondary — Kahneman 雙系統對應 | exploration vs commitment 比例、step 數 |
| Expertise | expert / non-expert | Dopamine | CoT 長度、術語、跳步、acc |

組合 2 × 2 × 2 = 8 個 persona，可控；主分析在 Confidence 軸看效應強度，Cognitive style 看 dual-system 對應，Expertise 做 anchor / 3-way interaction。

## Models

- DeepSeek-R1-Distill-Qwen-1.5B — pipeline debug
- DeepSeek-R1-Distill-Qwen-7B — main result
- Qwen3 thinking / non-thinking — same-family cross-mode validation
- QwQ-32B 

## Datasets

- GSM8K — 簡單數學
- MATH500 — 中等難度
- GPQA — 科學常識
- optional: AIME / AMC — 困難數學

選擇邏輯：難度梯度 + math vs non-math，能看 persona × task 的 interaction。

## Observation framework

Thinking process 從 surface 到 deep 分四層觀測，前三層為 behavioral（black-box），第四層為 mechanistic（hidden state）。先把行為層做穩，再回頭分析 hidden state — 否則 hidden state 分析沒有 anchoring 的對象。

### Layer 1 — Surface statistics

Regex / token count，無需 parse：

| Metric | 計算 |
|---|---|
| thinking_tokens | `len(tokenize(<think> block))` |
| answer_tokens | `len(tokenize(after </think>))` |
| think_answer_ratio | thinking_tokens / answer_tokens |
| backtrack_count | regex on `{wait, actually, hmm, let me reconsider, on second thought}` |
| verify_count | regex on `{let me check, verify, double-check, make sure}` |
| hedge_count | regex on `{maybe, perhaps, I'm not sure, might}` |
| switch_count | regex on `{alternatively, another approach, or maybe}` |
| commit_count | regex on `{so the answer is, therefore, the final answer}` |
| accuracy | exact match / GSM8K parser |
| answer_confidence | logit margin / entropy on final answer token |

### Layer 2 — Segment structure

把 `<think>` 切成 reasoning steps（按段落 / 雙換行 / cue word），給每個 step 打標籤：

- **Exploration** — 提出新假設、嘗試新方法
- **Verification** — 檢查已有結果
- **Backtracking** — 放棄之前的路徑
- **Commitment** — 提出候選答案
- **Meta-reflection** — 自我評價

標註方法：
- A. rule-based on cue words（快、precision 高 / recall 低）
- B. GPT-4o or Claude 當 annotator（準、約 $20 / 1000 traces）
- C. A 跑全量 + B 標 200 條 validation，報告 agreement

Metrics：
- step type 分布（5 類佔比）
- first_commit_pos（第幾個 step 出現第一個候選答案）
- backtrack_rate = backtracking / n_steps
- verification_depth（連續 verification 的最長鏈）
- n_steps

### Layer 3 — Reasoning graph (optional extension)

把 thinking 解析成 DAG，節點是 candidate / sub-claim，邊是 derivation / refutation：

- branching factor — 探索路徑數
- path abandonment rate
- convergence speed — 從第一個 candidate 到 final answer 的距離

難度大，作為 paper 的 extension section。

### Layer 4 — Hidden state / RSN signal

對 Layer 1–3 找到的 behavioral effect，回到 hidden state 找對應的 mechanistic signal

<!-- - RSN activation at prefill / early decode / mid-thinking / pre-commit
- EMA 軌跡形狀（early peak height、plateau level、decay）
- 與 dopamine-thinking curve 框架（Yerkes–Dodson）對應

復用 `track_hidden_states.py`：把 `--role` 擴成 persona axis × level，per-sample HDF5 dump（prefill + decode + all layers）。 -->

## Phased roadmap

### Phase 1 — Behavioral pilot 

最小可行實驗，目的：判斷 persona 對 thinking trace 有沒有系統性效應。

- Model: R1-Distill-Qwen-7B
- Persona: Confidence × Cognitive style × Expertise = 2 × 2 × 2 = 8 personas + 1 neutral baseline
- Datasets: GSM8K 200 + GPQA 200
- Metrics: Layer 1 全部
- 產出：persona × metric heatmap，加 per-axis main effect + 三軸 interaction 分析

**Decision:**
- ≥3 個 metric 有顯著 persona 效應 → 進 Phase 2
- 只有 1–2 個顯著 → 換 framing，做「為何 persona 效應窄」
- 完全無效應 → stop

### Phase 2 — Behavioral profiling

確認 Phase 1 有 signal 後，擴展到完整 behavioral taxonomy：

- 擴 model：加 R1-Distill-Qwen-1.5B + Qwen3 thinking
- 擴 confidence levels（2 → 5：very uncertain / uncertain / neutral / confident / very confident），看是否仍 monotonic
- 擴 dataset：加 MATH500（看 task interaction）
- 加 Layer 2 metrics（LLM annotation）
- 跑 persona × task interaction 分析
- 產出：完整 behavioral findings（3–5 個 main results）

### Phase 3 — Mechanistic linkage 

對 Phase 2 中效應最強的 1–2 個 persona axis，跑 hidden state：

- 算 RSN signal（NMD mask projection）
- Correlation：behavioral metric vs RSN signal（per-sample, per-persona）
- 若 correlation 強 → 進 Phase 4 causal intervention

### Phase 4 — Causal validation (optional, week 11+)

把 Phase 3 的 correlation 升級成 causal：

- Activation steering 把 RSN signal 推向 confident persona 的形狀，看 thinking behavior 是否被誘導改變
- 反向：steer 成 uncertain 的形狀，看 backtrack rate 是否上升
- 連回 dopamine-thinking curve：哪段曲線 segment 對應哪個 behavior 維度

## Controls

- neutral prompt baseline
- random sparse neuron mask（vs NMD mask）
- shuffled-layer RSN mask
- entropy / MSP / logit-margin routing baseline
- thinking-on vs thinking-off mode 對比（Qwen3）
- 同題不同 persona 的 within-question paired comparison

## Key analysis questions

1. Persona 系統性改變了 thinking trace 的哪些維度？哪些軸效應最強？
2. Persona 效應在不同 task / difficulty 下是否穩定 / 反轉？
3. Persona 是改 acc，還是只改 token cost、calibration？trade-off 曲線是什麼？
4. RSN signal / EMA 軌跡形狀是否能預測 behavioral metric？
5. RSN signal 是否能 causally 誘導對應的 thinking behavior？
6. 同一 persona 在 thinking model vs non-thinking model 上的效應是否一致？

## Potential claim

Reasoning models inherit role-sensitive motivational circuits from their base models. Persona prompts modulate adaptive deliberation — thinking depth, verification, commitment — through these circuits, and the modulation is mechanistically traceable to RSN signal dynamics that follow the dopamine-thinking curve.

## Backup path: IT model anchoring

風險：Persona prompt 在 reasoning model（R1-Distill / o1-style）上經過大量 RL，對 system-prompt persona 的 sensitivity 可能遠低於 base IT model。Phase 1 pilot 跑出來可能發現 trace 幾乎沒差。

對策不是 stop，而是退一步到 base IT model 上 anchoring：

1. **在對應的 IT base model 上做完整 persona experiment**（Qwen2.5-7B-Instruct → R1-Distill-Qwen-7B；Llama3-8B-Instruct → 對應 distill 版本）。IT base model 上 persona 效應通常更明顯，behavioral signal 更強。
2. **在 IT model 上提取 RSN neurons / NMD mask / signal trajectory**。這部分復用現有 RSN paper pipeline，零成本。
3. **把 IT model 上找到的 circuit 投影到 reasoning model**。R1-Distill 是從 Qwen2.5 蒸餾來的，weight space 相關 — 看相同 neurons / mask 在 reasoning model 中是否仍承載類似 signal。
4. **直接 hidden-state intervention 繞過 prompt sensitivity**。即使 persona prompt 在 reasoning model 上失效，activation steering 沿 IT-derived diff vector 可直接介入。

這個 backup 路徑反而強化了 mechanism claim — 證明 "reasoning model 從 base IT model 繼承了 role-sensitive circuit"，這本身就是有發表價值的 finding，並與 RSN 主線的「circuit 跨模型可遷移」假設一致。

## Why this matters

不僅是 incremental analysis paper，**核心科學問題是**：

> Reasoning model 的 thinking 是 deterministic algorithmic process，還是 motivationally modulated behavior？

如果 persona 能系統性、可預測、單調地改變 thinking process — 那 thinking 就**不是純 algorithmic**，而是 motivational state 的函數。對當前 reasoning model 文獻是概念衝擊，因為主流敘事把 thinking 當成「模型在做 search / reasoning」，而本工作可能說明 thinking 更接近「模型在按某種內在 motivational state 演出 reasoning」。

這個 framing 同時解釋多個現有發現：

- **Zheng 2024 (#25) negative result**（persona system prompt 不改 acc）→ motivational state 只改過程不改能力上限
- **Chen 2024 (#13) overthinking / Wang 2025 (#14) underthinking** → 兩者皆為 motivational state 失調的不同極端，而非單純 algorithmic 缺陷
- **Yeo 2025 (#12) reflection / backtracking behavior unit** → 都是 motivational state 在 trace 上的 surface manifestation
- **本項目 RSN / dopamine curve 主線** → 提供 motivational state 的 neural correlate

### 三層 contribution

1. **Phenomenological** — 首個系統 reasoning-model × persona × thinking-trace 的 study；Persona / Role × Reasoning 文獻全在 non-reasoning model 且只看 acc，本項目填補空白
2. **Mechanistic** — Persona 不是黑箱 prompt，而是可分解到 hidden-state circuit 上的 motivational signal；RSN / dopamine-thinking curve 提供 neural correlate
3. **Conceptual** — 重新定義 "thinking"：reasoning model 的 deliberation 是 motivationally modulated 而非 purely algorithmic；為 adaptive reasoning / when-to-think / overthinking-underthinking 提供統一機制框架

