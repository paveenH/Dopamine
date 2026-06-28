<!-- 主線導覽（三份文檔共用，每份開頭都有）

整條研究主線（四段）：
  RSN
    → 行為學多巴胺（Behavioral Dopamine）← Ada_Dopamine.md
        → 腦科學多巴胺（Brain Dopamine）← Ada_Dopamine2.md §五
            → 多巴胺與思考曲線（Dopamine & Thinking Curve）← 本文檔

  附：AdaThink.md 是 Thinking Curve 的額外延伸驗證（學弟執行），不在主線框架內。

【本文檔定位】
最終升華階段，兩層目標：
  1. 閉環控制（Phase 2，§三–§四）：用 RSN 信號做 decode-time 閉環調控，
     測試能否透過操控 EMA 波形（early peak / tonic plateau）提升推理準確率。
     結論：形狀控制（Plans A-H3）不等於 acc 控制，轉向 Phase 1b 信號驗證。
  2. Thinking Curve（Phase 3，理想目標）：在 reasoning model（DeepSeek-R1、
     Qwen3-thinking）的 <think> trace 裡觀察 persona 如何調節思考深度、
     backtrack、first-commit 等行為，對應多巴胺動力學，並透過 LLM 實驗
     模擬人腦思考過程中的 motivation dynamics。
     → 詳見 AdaThink.md（trace-level 分析框架）

【前兩段的任務】
Ada_Dopamine.md：行為學基礎驗證（wanting/knowing 解離、Bandit、Pressure）。
Ada_Dopamine2.md：腦科學升華（RSA：RSN Δh 是否對應 ventral striatum / vmPFC）。

關聯文件：
  Ada_Dopamine.md — 行為學理論框架與實驗結果
  Ada_Dopamine2.md — 腦科學 RSA 方向 + 實驗 Roadmap
  AdaThink.md — Reasoning model trace-level 分析框架（Thinking Curve 執行細節）
-->

# RSN as Dopaminergic Adaptive Calibration

## 1. 相關文獻

### 1.1 Adaptive Reasoning

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

- **12.2026.Reasoning emerges from constrained inference manifolds in large language models**
  This paper argues that LLM reasoning trajectories collapse into low-dimensional, information-preserving hidden-state manifolds. It is relevant because it supports the view that reasoning should be diagnosed from internal dynamics, not only from CoT text or final-answer accuracy. For this project, intrinsic dimension / information-volume style features may serve as additional router features for adaptive thinking, complementary to RSN projection, entropy, confidence variance, and early decode slope.

### 1.2 Dopamine Neuroscience

- **Tonic dopamine and biases in value learning** (NC 2025)
  高水平多巴胺增加 D1 受體敏感性，促進積極預期學習（樂觀）；低水平則增加對消極結果的敏感性（悲觀）。

- **Dopamine dynamics are dispensable for movement but promote reward responses** (Nature 2024)
  快速多巴胺動態並非運動必需，基礎 tonic 水平足以維持運動；快速釋放對動機和獎勵反應至關重要。

- **Dopamine release plateau and outcome signals in dorsal striatum** (NC 2024)
  持續努力任務中，多巴胺從初始尖峰轉變為「調製高原反應」（Modulated Plateau Responses）。高原在表現最佳個體中最明顯，代表穩定多巴胺供應是維持高水準表現的核心機制。
  - 對 RSN signal 的啟發：自然 flow-like waveform 應是 task-onset phasic peak + sustained tonic plateau + small feedback-linked micro-spikes，而不是高頻振盪或外力停止後的 cliff。

- **Reward magnitude determines reinforcement learning efficiency** (Science 2026) — https://mp.weixin.qq.com/s/xMfCquiGTszFPC3TxZAIxg
  奬勵大小本身會改變強化學習效率：少數幾次大奬勵比大量小奬勵更能加速小鼠在導航、費力運動技能與感覺運動決策任務中的學習。大奬勵誘發更強、更持久的伏隔核多巴胺反應，同時改善 session 內 learning rate、跨 session retention，以及任務投入狀態（減少 disengagement）。
  - 對 RSN signal 的啟發：learning rate / engagement 不是固定參數，而會被 reward magnitude 與 dopamine duration 動態調節。這支持將 RSN 視為 adaptive gain / commitment-to-engagement signal，而不是單純的 correctness 或 knowledge signal。
  - 限制：多巴胺刺激只能部分複製大奬勵效果，主要改善 session 內學習與 engagement，不能完整複製跨天保留；對本專案意味著 waveform steering 可能只能調節 willingness-to-act / engagement，不能替代 CoT 內容、verifier feedback 或真正可驗證的推理步驟。

- **Dopamine in Motivational Control: Rewarding, Aversive, and Alerting** (Neuron 2010)
  多巴胺以兩種模式運作：Tonic（穩定背景水平，與努力程度正相關）和 Phasic（針對獎勵/驚喜的短暫尖峰）。心流中頻繁的微小 phasic 釋放疊加並維持 tonic 高原。

- **Individual differences in flow linked to dopamine D2-receptor availability** (NeuroImage 2013)
  PET + `[11C]raclopride` 測量紋狀體 D2 受體可用性，發現 flow proneness 與 striatal D2 receptor availability 正相關（r=.41），主要由 dorsal striatum / putamen 驅動。限制：測量 trait-level 受體密度，非心流中的動態釋放。

- **Individual differences in flow proneness are linked to a dopamine D2 receptor gene variant** (Consciousness and Cognition 2016)
  DRD2 C957T polymorphism 與 flow proneness 相關，尤其在 mandatory activities（工作/學習）中更明顯。這是 D2 系統與 flow trait 關聯的補充證據，但仍是個體差異層級，不是 online dopamine waveform。

- **Neural correlates of experimentally induced flow experiences** (NeuroImage 2014)
  以自適應難度 mental arithmetic 在 fMRI 中誘發 flow。Flow condition 相比 boredom / overload 顯示 left putamen 與 left anterior IFG 活化增加，mPFC 與 amygdala 活化降低。Putamen 結果支持 flow 涉及 basal-ganglia/reward-related circuitry；但 fMRI 不能直接證明 dopamine release。

- **Neural signatures of experimentally induced flow experiences identified in a typical fMRI block design** (SCAN 2016)
  30 秒 block design 也能誘發 flow；flow condition 下 anterior insula、inferior frontal gyri、basal ganglia、midbrain 活化增加，mPFC/PCC/medial temporal lobe/amygdala 活化降低。這提供 flow 與 basal ganglia / midbrain reward system 相關的第二組實驗證據。

- **Go with the flow: A neuroscientific view on being fully engaged** (European Journal of Neuroscience 2021)
  心流動機底層由多巴胺系統和 LC-NE 系統共同驅動。注意：tonic/phasic 討論主要集中在 NE，多巴胺波形為推論而非直測。
  - 可用於框架表述：dopamine reward system 支撐 intrinsic motivation / engagement / wanting，LC-NE system 支撐 arousal / attention。Flow 應是 dopamine 與 NE 都在 optimal zone，而非單一物質越高越好。

- **The brain in flow: A systematic review** (Cortex 2022)
  系統性回顧 25 項研究（471 名受試者）。心流涉及前額葉和獎勵系統的共同活化，但跨研究神經動態不一致，無研究直接測量心流期間的多巴胺釋放。

**小結 / 限制**：目前 flow-dopamine 文獻支持「dopaminergic reward system 與 flow proneness / flow-like engagement 相關」，但尚未直接提供人類 flow 狀態下 dopamine release 的完整時序波形。因此本文中的 early peak → plateau → slow decay 是基於 dopamine reward-task dynamics 與 flow fMRI/review 的功能性類比，不應表述為已被直接量測的 biological dopamine waveform。

### 1.3 Mechanistic Alignment / State vs Capacity

- **State and capacity in neural models of cognition and consciousness** (Trends Open 2026) — https://www.sciencedirect.com/science/article/pii/S311734702600012X
  這篇綜述提出 state × capacity 雙軸框架，用來區分神經網路模型到底只是行為/腦活動上像人，還是真的能提供機制解釋。State 指固定架構當下如何運行，例如 gain、noise、attention、decision threshold、normalization；capacity 指模型原則上能表徵什麼，例如 depth、recurrence、context length、multimodality、external memory。
  - 對本專案的定位：RSN / α steering 應被表述為 **state-level gain control**，主要調節 willingness-to-act、decisiveness、engagement、commitment threshold，而不是 capacity-level upgrade。

### 1.4 Other
- RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents (ACL2026)
- **Theory of Agent (ToA): Why "being correct" is not enough for Agents** (ICML 2026 Position Paper) — https://arxiv.org/abs/2506.00886
  這篇把 Agent 行為統一建模為 **internal reasoning vs external action/tool use** 之間的 epistemic effort allocation。核心不是只問 Agent 能不能答對，而是問它是否能根據自身 knowledge boundary `Q_int`、任務不確定性與成本比 `β`，合理決定什麼時候自己想、什麼時候調用工具、什麼時候停止。
  - 關鍵警示：如果 post-training 只獎勵 final correctness，Agent 會自然漂向過度外包（over-delegation / overacting），短期正確率高但內部能力不成長；因此 agentic RL/SFT 應加入 process-level cost / effort-aware reward，而不是只看 answer accuracy。

---

## 2. 核心理論框架

### 2.1 Core Idea

RSN（Role-Sensitive Neurons）的行為類比多巴胺的雙向校準機制（dynamic calibration），可以理解為一個 **state-level gain knob**：調節模型在當下狀態中有多願意啟動、投入、承諾、持續檢查或停止，而不是直接改寫模型能表徵的知識與推理能力。因此，RSN / α steering 的核心對象不是 `knowing`，而是 `wanting`。這與 incentive salience 的神經科學框架一致：多巴胺不等於「快樂」或「正確」，而是把某個目標、線索或行動賦予更高的動機顯著性。

這個 framing 也意味著：最佳 α 不是固定常數，而是任務相依的工作點。需要探索、下注、嘗試或持續追求回報的任務，可能需要較高 wanting；需要克制、驗算、延遲承諾或避免 premature commitment 的任務，可能需要較低 wanting。RSN 的理論角色是移動這個工作點，而不是保證 accuracy 單調上升。

### 2.2 神經科學類比

| 概念 | 神經科學 | RSN 對應 |
|------|---------|---------|
| Tonic Dopamine | 個體基礎多巴胺濃度，決定背景動機、投入程度與行動準備度 | 模型的 baseline wanting / engagement level |
| Phasic Dopamine | 對 cue、reward prediction error、任務節點的短暫反應 | decode 過程中的局部 spike / transition signal |
| Incentive Salience | 讓某個目標或行動變得「值得追求」 | 把某個 answer/action candidate 推向更高 commitment priority |
| Gain Control | 調節 action threshold、effort allocation、verification tendency，而非直接寫入知識 | α 作為 bidirectional state-gain knob，改變 initiation / commitment / persistence / stopping tendency |

### 2.3 雙向失調與任務相依最優點

RSN / α steering 應被理解為在 wanting 軸上雙向移動模型狀態。這條軸不是「越高越好」，而是存在任務相依的 optimal zone：

- **Wanting 過低**：可能表現為啟動不足、欠承諾、過早放棄、無法形成穩定答案候選，或在多個候選之間無法收束。
- **Wanting 適中**：模型既願意投入與探索，又能在足夠證據後形成穩定承諾。
- **Wanting 過高**：可能表現為過早承諾、過度追求 reward/action、反覆檢查、放不下已完成答案，或把錯誤 prior 包裝成更有力的生成。

因此，同一個 α 方向在不同任務中可以有不同效果。對需要 exploration / reward pursuit 的任務，較高 wanting 可能更接近最佳點；對需要 deliberation / verification / delayed commitment 的任務，較低 wanting 可能更接近最佳點。理論上，RSN 的作用不是提供一個通用的「更好」方向，而是揭示不同任務對 wanting level 的需求曲線。

### 2.4 RSN Trajectory 的定位

RSN trajectory（如 `x_t`、EMA、early peak、late level、decay rate）應首先被視為 **diagnostic readout**，用來描述模型在解碼過程中的 engagement / commitment / release dynamics。它不是一個可以直接最大化或固定成某種形狀的 accuracy objective。

在這個框架下：

- **Tonic component**：對應持續背景 drive，可用 EMA 近似。
- **Phasic component**：對應局部 decode 節點的 spike，不應一概視為噪聲。
- **Release dynamics**：模型何時從高 engagement 狀態退出，可能比單純的起點高度更有解釋力。
- **Shape ≠ capability**：trajectory shape 可以反映 state，但不能替代 reasoning content、verifier feedback 或外部工具提供的 capacity。

因此，trajectory 分析的目標不是尋找一條固定的「理想曲線」，而是回答三個問題：不同 prompt / role / α 是否移動 state；這些 state 如何影響 initiation、commitment、verification 與 stopping；以及這些變化是否與任務需求相匹配。

## 3. 信號定義

### 3.1 Projection Signal

RSN signal 是把 middle-layer hidden state 投影到 sparse NMD mask 上得到的 per-token scalar。對每個 decode step `t`：

```text
x_t = mean_l( h_t[l] · m_l )
ema_t = β · ema_{t-1} + (1 - β) · x_t
```

其中：
- `h_t[l]` 是第 `l` 個 middle decoder layer 在 token `t` 的 hidden state。
- `m_l` 是該 layer 對應的 sparse NMD mask / direction。
- `x_prefill` 是最後一個 prompt token 的 RSN projection，用作 sample-level starting state。
- `x_t` 是 decode-time raw projection，保留 token-level phasic fluctuation。
- `ema_t` 是 `x_t` 的 smoothed trajectory，作為 tonic-like diagnostic readout。

這裡的 `mean_l` 是逐層先獨立投影，再對 middle layers 取平均；不是把所有 layer 拼接後做一次投影。

### 3.2 Mask and Layer Alignment

NMD mask 按 layer 獨立計算，每層只保留 top sparse neurons。實際內積為：

```text
h_t[l] · m_l = Σ_i h_t[l][i] × m_l[i]
```

mask 的 layer index 對齊 decoder layer **output** hidden state，而不是 layer input。對 Llama-style HF hidden states：

```text
hidden_states[0]      = embedding output
hidden_states[i + 1]  = decoder_layers[i] output
saved mask row i      ↔ decoder_layers[i] output
```

因此，無論是 signal observation 還是 static steering，最乾淨的對齊方式都是在 decoder layer output space 上讀取 / 注入。這能避免把「第 L 層 output 上學到的方向」錯放到同一層 input space 裡。

### 3.3 Three Modes of Use

同一個 RSN mask 可以被用在三種不同模式；三者應明確分開：

| Mode | 目的 | α | Hook / timing | 用途 |
|---|---|---:|---|---|
| Observation-only tracking | 只讀取 `x_t` / `ema_t`，不干預模型 | 0 | forward output readout | signal validation、role / CoT / correct-vs-wrong trajectory analysis |
| Static steering | 固定 α 改變 state-level gain | fixed α | output-side addition `h_t[l] += α m_l` | GSM8K / MATH α scan、behavioral steering |
| Closed-loop control | 根據 trajectory 即時調 α | dynamic `α_t` | observation → next-step intervention | Phase 2 waveform-control experiments |

Observation-only 和 static steering 是目前最穩定、最可比的兩種用途。Closed-loop control 則是獨立的控制實驗，不應被視為 signal definition 本身。

### 3.4 EMA Interpretation

EMA 的角色是把 noisy token-level `x_t` 轉成慢變的 tonic-like trajectory。它有兩個用途：

1. **診斷用途**：描述模型的 engagement / commitment / release dynamics。
2. **控制用途（歷史 Phase 2）**：作為比 raw `x_t` 更慢、更穩定的 feedback variable。

但 EMA 不是生物多巴胺的直接量測，也不是 universal accuracy target。`β=0.95` 對應約 20 token 的時間常數，適合長生成中的 state smoothing；對很短的 action-output 任務則未必穩定。因此，EMA 應被解讀為 **tonic-state approximation**，而不是「越高越好」的 objective。

### 3.5 Closed-loop Caveat: 1-step Lag

只有 closed-loop control 模式存在嚴格的 1-step lag。其物理來源是控制器必須先完成當前 token 的 forward，才能觀測 `x_t` 並計算下一步的 `α_{t+1}`：

```text
step t:
  apply α_t
  forward
  observe x_t / ema_t
  compute α_{t+1}

step t+1:
  apply α_{t+1}
```

因此，feedback 永遠基於上一 token 的 observation，作用於下一 token。對慢變的 EMA，這個 lag 尚可接受；對 1–2 token 內自然消退的 raw spike，逐 token feedback 容易追尾並造成振盪。這是 Phase 2 closed-loop 設計需要單獨處理的控制問題，不是 observation-only signal analysis 的限制。

### 3.6 Multi-Metric Signal Suite

**Per-step raw trajectories**

- `rsn_ema` 來自 **middle-layer HS 投影到 sparse NMD mask** 上的 scalar（RSN 子空間活動 = wanting / drive）。
- `entropy / top1 / margin / info_gain` 來自 **final-layer whole hidden state**（非 mask）過 `RMSNorm + lm_head` 重建出的**真實 next-token logits**，再 softmax；即模型真實輸出分布。它們度量的是 **confidence / decisiveness**。

| Metric | Source | Computation | Interpretation |
|--------|--------|------|---------|
| `rsn_ema` | middle HS · **NMD mask** | `mean_l(h_t[middle]·mask[l])` → EMA(α=0.95) | wanting / drive (RSN subspace) |
| `entropy_decode` | **full final HS** → lm_head | `-Σ p log p` over vocab | confidence / decisiveness (inverse) |
| `top1_decode` | **full final HS** → lm_head | `max(softmax(logits))` = MSP | confidence / decisiveness |
| `margin_decode` | **full final HS** → lm_head | `top1 - top2` | confidence / decisiveness (≈collinear w/ top1; optional) |
| `info_gain_decode` | **full final HS** → lm_head | `H_{t-1} - H_t` | reasoning efficiency (paper 03) |

> 注意：`rsn_ema` 是 mask 子空間投影（wanting）；entropy/top1/margin/info_gain 是整條 HS 重建真實 logits（confidence），兩者**不共用 mask**。這個來源分離正是 wanting（internal drive）與其 confidence（output decisiveness）表現的對照基礎。

**Derived trajectories**

- `normalized_ema` = `ema_t / x_prefill`
- `cumulative_entropy_reduction` = `H_0 - H_t` (cumulative InfoGain; paper 03)
- `rolling_conf_variance` = `std(top1[t-W:t])`, W=10 (paper 05)

**Per-sample scalar summaries**

`late_tonic`, `early_peak`, `mean_entropy`, `mean_top1`, `mean_margin`, `info_gain_mean`, `entropy_prefill`, `top1_prefill`, `margin_prefill`

**Comparison axes**

| Axis | Measurement | Purpose |
|------|---------|------|
| **A. State** (Role: Neutral/Expert/Non-Expert/Teacher · CoT vs No-CoT · α-dose) | curve mean ± std band, KW/MWU + Cohen's d between states | **主要目的**：信號對 state 的敏感度（wanting/confidence 是否被 state 移動）|
| **B. Correct vs Wrong** | curve mean by group | 輔助：與對錯的關係。⚠ 差異多為難度副產品（會做的題 release 快），非 state/DA 結論 |
| **C. Mask** (NMD vs Random) | curve mean by mask | RSN-specificity control (僅對 `rsn_ema` 有意義) |

---

## 4. 觀察結果

<!-- 待填：state（role / CoT / α）是否移動 trajectory 與多指標信號的觀察結果。
     主軸 = state 對比（CoT vs No-CoT、persona），correct/wrong 為輔助對照。 -->


