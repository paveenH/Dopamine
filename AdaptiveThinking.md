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

## RSN as Dopaminergic Adaptive Calibration

### 1. 相關文獻

#### 1.1 Adaptive Reasoning

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

#### 1.2 Dopamine Neuroscience

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

#### 1.3 Mechanistic Alignment / State vs Capacity

- **State and capacity in neural models of cognition and consciousness** (Trends Open 2026) — https://www.sciencedirect.com/science/article/pii/S311734702600012X
  這篇綜述提出 state × capacity 雙軸框架，用來區分神經網路模型到底只是行為/腦活動上像人，還是真的能提供機制解釋。State 指固定架構當下如何運行，例如 gain、noise、attention、decision threshold、normalization；capacity 指模型原則上能表徵什麼，例如 depth、recurrence、context length、multimodality、external memory。
  - 對本專案的定位：RSN / α steering 應被表述為 **state-level gain control**，主要調節 willingness-to-act、decisiveness、engagement、commitment threshold，而不是 capacity-level upgrade。

#### 1.4 Other
- RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents (ACL2026)
- **Theory of Agent (ToA): Why "being correct" is not enough for Agents** (ICML 2026 Position Paper) — https://arxiv.org/abs/2506.00886
  這篇把 Agent 行為統一建模為 **internal reasoning vs external action/tool use** 之間的 epistemic effort allocation。核心不是只問 Agent 能不能答對，而是問它是否能根據自身 knowledge boundary `Q_int`、任務不確定性與成本比 `β`，合理決定什麼時候自己想、什麼時候調用工具、什麼時候停止。
  - 關鍵警示：如果 post-training 只獎勵 final correctness，Agent 會自然漂向過度外包（over-delegation / overacting），短期正確率高但內部能力不成長；因此 agentic RL/SFT 應加入 process-level cost / effort-aware reward，而不是只看 answer accuracy。

---

### 2. 核心理論框架

#### 2.1 Core Idea

RSN（Role-Sensitive Neurons）的行為類比多巴胺的雙向校準機制（dynamic calibration），可以理解為一個 **state-level gain knob**：調節模型在當下狀態中有多願意啟動、投入、承諾、持續檢查或停止，而不是直接改寫模型能表徵的知識與推理能力。因此，RSN / α steering 的核心對象不是 `knowing`，而是 `wanting`。這與 incentive salience 的神經科學框架一致：多巴胺不等於「快樂」或「正確」，而是把某個目標、線索或行動賦予更高的動機顯著性。

這個 framing 也意味著：最佳 α 不是固定常數，而是任務相依的工作點。需要探索、下注、嘗試或持續追求回報的任務，可能需要較高 wanting；需要克制、驗算、延遲承諾或避免 premature commitment 的任務，可能需要較低 wanting。RSN 的理論角色是移動這個工作點，而不是保證 accuracy 單調上升。

#### 2.2 神經科學類比

| 概念 | 神經科學 | RSN 對應 |
|------|---------|---------|
| Tonic Dopamine | 個體基礎多巴胺濃度，決定背景動機、投入程度與行動準備度 | 模型的 baseline wanting / engagement level |
| Phasic Dopamine | 對 cue、reward prediction error、任務節點的短暫反應 | decode 過程中的局部 spike / transition signal |
| Incentive Salience | 讓某個目標或行動變得「值得追求」 | 把某個 answer/action candidate 推向更高 commitment priority |
| Gain Control | 調節 action threshold、effort allocation、verification tendency，而非直接寫入知識 | α 作為 bidirectional state-gain knob，改變 initiation / commitment / persistence / stopping tendency |

#### 2.3 雙向失調與任務相依最優點

RSN / α steering 應被理解為在 wanting 軸上雙向移動模型狀態。這條軸不是「越高越好」，而是存在任務相依的 optimal zone：

- **Wanting 過低**：可能表現為啟動不足、欠承諾、過早放棄、無法形成穩定答案候選，或在多個候選之間無法收束。
- **Wanting 適中**：模型既願意投入與探索，又能在足夠證據後形成穩定承諾。
- **Wanting 過高**：可能表現為過早承諾、過度追求 reward/action、反覆檢查、放不下已完成答案，或把錯誤 prior 包裝成更有力的生成。

因此，同一個 α 方向在不同任務中可以有不同效果。對需要 exploration / reward pursuit 的任務，較高 wanting 可能更接近最佳點；對需要 deliberation / verification / delayed commitment 的任務，較低 wanting 可能更接近最佳點。理論上，RSN 的作用不是提供一個通用的「更好」方向，而是揭示不同任務對 wanting level 的需求曲線。

#### 2.4 RSN Trajectory 的定位

RSN trajectory（如 `x_t`、EMA、early peak、late level、decay rate）應首先被視為 **diagnostic readout**，用來描述模型在解碼過程中的 engagement / commitment / release dynamics。它不是一個可以直接最大化或固定成某種形狀的 accuracy objective。

在這個框架下：

- **Tonic component**：對應持續背景 drive，可用 EMA 近似。
- **Phasic component**：對應局部 decode 節點的 spike，不應一概視為噪聲。
- **Release dynamics**：模型何時從高 engagement 狀態退出，可能比單純的起點高度更有解釋力。
- **Shape ≠ capability**：trajectory shape 可以反映 state，但不能替代 reasoning content、verifier feedback 或外部工具提供的 capacity。

因此，trajectory 分析的目標不是尋找一條固定的「理想曲線」，而是回答三個問題：不同 prompt / role / α 是否移動 state；這些 state 如何影響 initiation、commitment、verification 與 stopping；以及這些變化是否與任務需求相匹配。

---

### 3. 方法設計

#### 3.1 RSN Signal Definition

##### 3.1.1 Projection Signal

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

##### 3.1.2 Mask and Layer Alignment

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

##### 3.1.3 Three Modes of Use

同一個 RSN mask 可以被用在三種不同模式；三者應明確分開：

| Mode | 目的 | α | Hook / timing | 用途 |
|---|---|---:|---|---|
| Observation-only tracking | 只讀取 `x_t` / `ema_t`，不干預模型 | 0 | forward output readout | signal validation、role / CoT / correct-vs-wrong trajectory analysis |
| Static steering | 固定 α 改變 state-level gain | fixed α | output-side addition `h_t[l] += α m_l` | GSM8K / MATH α scan、behavioral steering |
| Closed-loop control | 根據 trajectory 即時調 α | dynamic `α_t` | observation → next-step intervention | Phase 2 waveform-control experiments |

Observation-only 和 static steering 是目前最穩定、最可比的兩種用途。Closed-loop control 則是獨立的控制實驗，不應被視為 signal definition 本身。

##### 3.1.4 EMA Interpretation

EMA 的角色是把 noisy token-level `x_t` 轉成慢變的 tonic-like trajectory。它有兩個用途：

1. **診斷用途**：描述模型的 engagement / commitment / release dynamics。
2. **控制用途（歷史 Phase 2）**：作為比 raw `x_t` 更慢、更穩定的 feedback variable。

但 EMA 不是生物多巴胺的直接量測，也不是 universal accuracy target。`β=0.95` 對應約 20 token 的時間常數，適合長生成中的 state smoothing；對很短的 action-output 任務則未必穩定。因此，EMA 應被解讀為 **tonic-state approximation**，而不是「越高越好」的 objective。

##### 3.1.5 Closed-loop Caveat: 1-step Lag

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

#### 3.1.6 Multi-Metric Signal Suite 

**Per-step raw trajectories**

| Metric | Computation | Interpretation |
|--------|------|---------|
| `rsn_ema` | `mean_l(h_t[middle]·mask[l])` → EMA(α=0.95) | wanting / drive |
| `entropy_decode` |`-Σ p log p` over vocab | wanting (vocabulary spread) |
| `top1_decode` | `max(softmax(logits))` | output certainty |
| `margin_decode` | `top1 - top2` | output certainty (nearly collinear with top1; optional) |
| `info_gain_decode` | `H_{t-1} - H_t` | reasoning efficiency (paper 03) |

**Derived trajectories**

- `normalized_ema` = `ema_t / x_prefill`
- `cumulative_entropy_reduction` = `H_0 - H_t` (cumulative InfoGain; paper 03)
- `rolling_conf_variance` = `std(top1[t-W:t])`, W=10 (paper 05)

**Per-sample scalar summaries**

`late_tonic`, `early_peak`, `mean_entropy`, `mean_top1`, `mean_margin`, `info_gain_mean`, `entropy_prefill`, `top1_prefill`, `margin_prefill`

**Comparison axes**

| Axis | Measurement | Purpose |
|------|---------|------|
| **A. Role** (Neutral / Expert / Non-Expert / ...) | curve mean ± std band | Role-prompt sensitivity |
| **B. Correct vs Wrong** | curve mean by group | Predictive relation with correctness |
| **C. Mask** (NMD vs Random) | curve mean by mask | RSN-specificity control (mainly for `rsn_ema`) |

**Cross-metric relationships**

- **Pearson correlation matrix** (per-sample, per-role): shared variance across metrics
- **r(metric, correct)** ranked: predictive strength for correctness
- **Partial correlation** r(RSN, correct | entropy): RSN's independent predictive signal after controlling for entropy
- **r(metric, RSN)**: coupling between each metric and the RSN trajectory


#### 3.2 Phase 2 控制方案

控制方案分為四代設計，每一代都是對前一代失敗模式的回應：

| 代次 | 計畫 | 核心機制 | 結果 |
|------|------|---------|------|
| **v1** | A / B / C | 單向比例控制（只往上補 tonic） | 信號有效，acc 無提升 |
| **v2** | D / E / F | 雙向比例反饋控制 | 全部失敗（D 爆炸；E/F 穩定但 α 太小） |
| **v3** | G | 死區 + 固定脈衝（bang-bang） | 證明「α 太小」假設錯，但形狀可控 ≠ acc 可控 |
| **v4** | H1 / H2 | Timing-controlled（限定窗口） | H1 否證 commitment 假設；H2 測試「擬合任意軌跡」 |

---

##### v1：單向比例控制（A / B / C）

**Plan A：Tonic Floor（保守）**
```python
target_tonic = x_prefill * 0.65
deviation = target_tonic - ema_t
α_t = +k1 * deviation / x_prefill if deviation > 0 else 0
```

**Plan B：Plateau Imitation**
```python
target_t = x_prefill * (1.0 - 0.35 * t / T)   # 100% → 65%，對應 CoT early→late 結構
deviation = target_t - ema_t
α_t = +k1 * deviation / x_prefill if deviation > 0 else 0
```

**Plan C：Tonic + Spike Damping**
```python
# Tonic control（同 Plan B）
target_t = x_prefill * (1.0 - 0.35 * t / T)
α_t = +k1 * (target_t - ema_t) / x_prefill if ema_t < target_t else 0

# Spike damping
spike_ratio = x_t / (ema_t + 1e-6)
if spike_ratio > 1.5:
    α_t -= k2 * (spike_ratio - 1.5)
```

超參數：k1=2.0，k2=1.0，α_ema=0.95，Layer 11–20，floor_ratio=0.65，plateau_end_ratio=0.65。

---

##### v2：雙向比例反饋（D / E / F）

**Plan D：Waveform Smoothing**

**核心想法**：用 raw signal `x_t` 相對於 EMA 的瞬間偏差來驅動 α，讓實際激活貼近平滑的 EMA 曲線：

```python
# Plan D: Waveform Smoothing
deviation = x_t - ema_t           # spike: +, dip: -
α_t = -k1 * deviation / x_prefill
```

**失敗原因：1-step lag 振盪放大**

Plan D 的控制對象是瞬時信號 `x_t`，但注入有 1-step lag（step t 觀測，step t+1 才作用）。當注入幅度稍大，系統等效開環增益 > 1，負反饋變成正反饋：

```
x_t 偏高 → 負 α 注入 → x_{t+1} 反向偏低 → 正 α 注入 → x_{t+2} 更高 → ...
每步放大 ~1.43× → 在數十步內衝到 inf
```

- k1=2.0：257/300 樣本爆炸（86%），acc=7%
- k1=1.0：2/300 樣本爆炸，acc=58.7%（仍低於 baseline 61.7%）
- 原始信號（無注入）deviation/xp 最大僅 1.96，注入本身創造了不穩定的擾動

**結論**：對瞬時高頻信號做閉環控制在 1-step lag 下天然不穩定，需改用慢信號（EMA）作為控制輸入。

**Plan E：EMA Homeostasis（雙向 EMA 控制）**

**動機**：Plan A/B 是單向 floor control；Plan D 追瞬時 x_t（高頻，不穩定）。Plan E 改為以 EMA 本身作為控制輸入——EMA 每步變化僅約 0.05×x_t，遠小於 x_t，1-step lag 的相位影響可忽略，天然穩定。同時雙向，允許對 Llama3 over-wanting 做有效抑制。

**核心想法**：

```python
# Plan E: EMA Homeostasis
target = x_prefill * floor_ratio   # 預設 0.85，允許從 prefill 自然下降 15%
deviation = ema_t - target          # EMA 偏高（> target）→ 負α壓；偏低 → 正α補
α_t = -k1 * deviation / x_prefill
```

**穩定性保證**：EMA 每步最多變化 `(1−0.95)×x_t ≈ 0.03`，即使 k1=2.0，α 每步變化極小，不會振盪。

**超參數**：k1 sweep（0.5 / 1.0 / 2.0），floor_ratio=0.85，其餘同 v1。

**Plan F：Dual-EMA Filter（雙 EMA 濾波）**

**動機**：Plan D 使用瞬時 x_t 作為控制輸入（高頻，lag 致命）；Plan E 用慢 EMA 控制絕對水位（穩定，但沒有追蹤波形結構）。Plan F 介於兩者之間——用快慢 EMA 的差值（MACD 風格）作為控制信號，追蹤「短期趨勢相對於長期基線的偏離」，同時保留信號分析的思路。

**核心想法**：

```python
# Plan F: Dual-EMA Filter（MACD-style）
fast_ema_t = 0.3 * fast_ema_{t-1} + 0.7 * x_t   # 快 EMA，α=0.7
slow_ema_t = 0.95 * slow_ema_{t-1} + 0.05 * x_t  # 慢 EMA（同原有 EMA）

deviation = fast_ema_t - slow_ema_t   # 快 > 慢 → 短期趨勢偏高 → 負α壓
α_t = -k1 * deviation / x_prefill
```

**與 Plan D/E 的比較**：

| | 控制輸入 | lag 敏感度 | 控制目標 |
|--|---------|-----------|---------|
| Plan D | x_t（瞬時） | 極高，天然不穩定 | 追每個 spike |
| Plan F | fast_ema - slow_ema | 中（fast_ema 每步最多移動 70% of x_t） | 平滑短期趨勢偏離 |
| Plan E | slow_ema - target | 極低（每步移動 5% of x_t） | 控制長期水位 |

**穩定性**：fast_ema 每步最多變化 70% of x_t，開環增益 `c = 0.7 × k1 × β / xp`，穩定條件為 k1 ≲ 1.17（比 Plan D 的 ~0.82 寬鬆，但仍比 Plan E 嚴格）。

**超參數**：k1 sweep（0.5 / 1.0 / 2.0），無 floor_ratio，其餘同 v1。

---

##### v3：死區脈衝（G）

**Plan G：Bang-Bang Dead-Zone（最簡雙向脈衝）**

**動機**：Plan D/E/F 全部使用「α 隨 deviation 比例變化」的設計，導致系統陷入結構性矛盾——deviation 小時 α 太小（不足以影響推理，E/F 全部 α 落在 ±1.2 內），加大 k1 使 α 變大時又觸發 lag 振盪（D 爆炸、F k1=2.0 撞 clamp）。比例控制器在這個系統中找不到「足夠大但又穩定」的工作點。

**核心想法**：放棄比例控制，回到最簡的死區 + 固定脈衝。在 ema 偏離 prefill 起點超過閾值時，注入固定大小的 α，否則完全不動。

```python
# Plan G: Bang-Bang Dead-Zone
band = k2 * x_prefill              # 死區半寬（k2=0.1/0.2/0.3 × xp）
if ema_t > x_prefill + band:
    α_t = -k1                       # 太高：固定負脈衝
elif ema_t < x_prefill - band:
    α_t = +k1                       # 太低：固定正脈衝
else:
    α_t = 0                         # 死區內保留 phasic
```

**設計優勢**：

1. **完全無方向先驗**：α 符號由 ema 與 xp 的相對位置自動決定，不假設模型偏好 +α 或 -α
2. **直接破解「α 太小」死局**：α 是固定常數，與 deviation 大小解耦，可以強行使用 k1=4 達到 known-effective steering 量級
3. **保留 phasic 結構**：死區內完全不動作，自然波動（包括有用的 phasic spike）被允許
4. **抗 lag**：bang-bang 不依賴 deviation 大小，1-step lag 最多在邊界附近產生 chattering，不會放大振盪
5. **可解釋**：行為簡單到一眼能看懂哪一步發生了什麼，便於 debug

**超參數**：k1 ∈ {1, 2, 4}（α 強度，覆蓋從 E/F 上界到 known-effective 量級），k2 ∈ {0.1, 0.2, 0.3}（死區寬度比例），9 組合 sweep。

---

##### v4：Timing-controlled（H1 / H2）

**Plan H1：Early Peak Boost（早期峰值脈衝）**

**動機**：Plan A-G 的所有結果指向同一個結論——「持續控制信號形狀」改不了 acc，但 baseline 數據顯示 **correct vs wrong 的最大差異就在 early peak 高度**（CoT correct ~1.30 vs wrong ~1.20；No-CoT 同樣 +0.08）。神經科學上，dopamine 的 task-onset burst（Schultz 1997）負責「commit to engagement」，後續執行不需要持續高 dopamine。Plan G k1=4.0 全程注入 → 形狀過頭、acc 反降 3.4%，提示「持續干預」反而傷害推理。

**核心想法**：放棄全程控制，只在生成最開始的關鍵窗口（前 100 step）強力抬升 ema 至 peak target，模擬 task-onset burst；之後完全不干預，讓 KV cache 中的早期 commitment 通過自回歸自然傳播到後續所有 token。

```python
# Plan H1: Early Peak Boost
boost_window = 100              # 早期注入窗口（step）
peak_target = floor_ratio * xp  # 目標峰值（floor_ratio 重用為 peak_ratio，e.g. 1.5）

if self._step < boost_window:
    if ema < peak_target:
        α_t = +k1               # 固定大小注入
    else:
        α_t = 0                 # 已達標，停止
else:
    α_t = 0                     # 後期：完全不干預
```

**與 G 的關鍵區別**：
- G 全程注入 → 持續擾動推理過程 → acc 反降
- H1 只早期注入 → 設定起點後撒手 → 不擾動中後期推理流暢性
- 「設定起點」vs「持續控制」的 timing-controlled 對比

**超參數**：k1 ∈ {2.0, 4.0, 6.0}（覆蓋 G k1=4.0 的 known-stable 區間 + 略高），peak_target = 1.5 × xp（從 CoT correct early peak ~1.30 + 過量 buffer），boost_window = 100 step（對應 CoT early peak 持續長度約 50-75 step + 容差）。

**Plan H2：Trapezoid Tracking（梯形軌跡追蹤）**

**動機**：H1 用 binary 注入（α=k1 or 0）+ 固定 peak target，導致兩個問題——(1) ema 鋸齒震盪（過衝後停、衰減後再注入），無法精準追蹤指定水位；(2) boost window 結束後 EMA 在 ~20 步內斷崖式回到 baseline，KV cache 沒有「鎖定」到新狀態，commitment 假設被否證。H2 改用比例追蹤 + 平滑梯形目標軌跡，回答兩個新問題：「能否精準塑造一條指定的 EMA 軌跡？」「擬合 CoT-like 形狀（早升、平台、緩降）能否提升 acc？」

**核心想法**：定義一條三段式梯形目標軌跡 `target(t)`，雙向比例追蹤：

```python
# Plan H2: Trapezoid Tracking
rise_end = 50
plateau_end = 200
T = avg_gen_len                          # 預設 400
peak = floor_ratio                        # 平台高度，預設 1.25 (×xp)
end_lvl = plateau_end_ratio               # 結尾水位，預設 0.75 (×xp)

if t < rise_end:
    target_ratio = 1.0 + (peak - 1.0) * (t / rise_end)         # 線性上升 1.0 → 1.25
elif t < plateau_end:
    target_ratio = peak                                        # 平台 1.25
elif t < T:
    target_ratio = peak - (peak - end_lvl) * (t - plateau_end) / (T - plateau_end)  # 線性下降 1.25 → 0.75
else:
    target_ratio = end_lvl                                     # 鉗位 0.75

target = target_ratio * x_prefill
α_t = k1 * (target - ema_t) / x_prefill   # 雙向比例追蹤
```

**與 H1 的關鍵差異**：

| | H1 | H2 |
|--|----|----|
| α 性質 | binary（k1 or 0） | 連續比例（k1 × deviation/xp） |
| 目標 | 固定 peak（1.5·xp） | 時變梯形軌跡 |
| 方向 | 單向（只往上） | 雙向（既補也壓） |
| 干預時段 | 前 100 step 後完全停 | 全程跟蹤（前 50 升、50-200 平、200+ 降） |
| 行為 | 鋸齒震盪過衝 | 平滑追蹤 |

**設計選擇**：
- **梯形形狀來源**：模仿 CoT correct 樣本的「early peak → plateau → 緩降」結構，但用乾淨的線性段近似（測試「形狀像不像 CoT」對 acc 的影響）
- **T 用 avg_gen_len=400**：保證下降段斜率固定可重現（真實 T 在生成中未知，且因樣本而異）
- **沒有死區**：純比例控制，讓 deviation 自然決定 α 大小；穩定性由 EMA 慢動態保證（同 Plan E）
- **依賴全局 clamp(±8)**：k1=4 時 α 在 ema 距 target 2·xp 處才會撞 clamp，正常情況下不會飽和

**超參數**：k1 ∈ {1.0, 2.0, 4.0}（追蹤增益），peak=1.25·xp，end=0.75·xp，rise=50, plateau_end=200, T=400 step。

**判決性測試**：
- 如果 H2 能精準追蹤目標軌跡（EMA 與 target 重合）但 acc 不變 → 進一步證實「形狀無關 acc」，整個項目轉向「decode-time 形狀控制不可行」結論
- 如果 H2 acc 提升 → 「形狀因果」假設復活，但需要重新解釋為什麼 H1 不行（區別在「平滑追蹤」vs「binary 過衝」？）
- 如果 H2 acc 下降 → 雙向控制本身（特別是「壓制自然 peak」與「強行下拉 late tonic」）干擾推理

**Plan H3：Trapezoid v2 — Steeper Rise & Longer Plateau（CoT 形狀貼近版）**

**動機**：H2 k1=4.0 達到 acc=62.7%（首次明確超過 baseline +1.0%），且呈現「k1 越大 → fit_err 越小 → acc 越高」的單調關係。但與 CoT (76.0%) 仍差 14%，從 EMA 軌跡圖看，H2 與 CoT 有兩個結構性差異：
1. **CoT 早期更陡**：~5% progress 內衝到 peak，H2 用 50 step（~12.5%）才到 1.25
2. **CoT peak 更高且 plateau 更短**：CoT 早期峰 ~1.27（correct ~1.30），中段就開始緩降；H2 peak=1.25 還拖長 plateau 到 200 step

H3 修改 H2 的軌跡形狀，使其更接近 CoT 真實波形，同時把 k1 sweep 推到更高（{4, 6, 8}）測試「擬合度 → acc」的單調關係是否延續到更大 α 量級。

**核心想法**：保留 H2 的雙向比例追蹤控制律，僅修改 target 軌跡的三段邊界與 peak 高度：

```python
# Plan H3: Trapezoid v2
rise_end = 30                     # H2: 50 → H3: 30  (更接近 EMA 時間常數 ~20 step)
plateau_end = 120                 # H2: 200 → H3: 120 (縮短平台、加長下降段)
T = avg_gen_len                   # 400 (同 H2)
peak = floor_ratio                # H2: 1.25 → H3: 1.35 (對齊 CoT correct early peak ~1.30 + buffer)
end_lvl = plateau_end_ratio       # 0.75 (同 H2)

# target_ratio 與 H2 完全相同的分段邏輯，僅邊界值不同
α_t = k1 * (target - ema) / xp    # 雙向比例追蹤
```

**與 H2 的差異對比**：

| | H2 | H3 | 設計邏輯 |
|--|----|----|---------|
| rise_end | 50 step | **30 step** | 更接近 EMA 時間常數（τ≈20），讓「target 跑得多快 ema 就能跟多快」 |
| plateau_end | 200 step | **120 step** | 縮短平台、加長下降，更接近 CoT「短峰 + 長緩降」 |
| peak | 1.25·xp | **1.35·xp** | 對齊 CoT correct 早期峰 ~1.30，加 buffer 抗欠擬合 |
| k1 sweep | {1, 2, 4} | **{4, 6, 8}** | H2 k1=4 為已知最佳，測試 k1 繼續上升的單調性 |

**判決性測試**：
- 如果 H3 acc > H2 k1=4（62.7%）→ 「形狀越像 CoT 越好」假設成立，繼續往「H4: 完全模擬 CoT 真實軌跡」方向迭代
- 如果 H3 acc ≈ H2 k1=4 → H2 已接近 trapezoid 形狀的 acc 上限，差距源於 CoT 內容本身而非形狀
- 如果 H3 acc < H2 k1=4 → 過高 peak（1.35）或過快 rise（30 step）對推理有害，需要回退或重新尋找最優窗口

**風險**：k1=8 配合更陡的 rise 可能撞 ±8 clamp（H2 k1=4 max α=1.69，外推 H3 k1=8 + rise 加速約到 ±5-6，仍應安全）。若飽和率 >10% 需縮回。

---

### 4. 實驗結果

#### 4.1 Phase 1：Baseline Signal 觀察（Llama3-8B，300 samples/condition，raw NMD mask）

| 條件 | Acc | x_prefill correct | x_prefill wrong | p(c vs w) | early tonic ratio | late tonic ratio |
|------|-----|-------------------|-----------------|-----------|-------------------|-----------------|
| GSM8K No-CoT | 61.7% | 0.567 | 0.585 | 0.145 | 1.055 | 0.295 |
| GSM8K CoT | 76.0% | 0.556 | 0.534 | 0.101 | — | 0.776 |
| MATH No-CoT | 29.3% | 0.490 | **0.530** | **0.006** | — | 0.778 |
| MATH CoT | 40.0% | 0.521 | **0.554** | **0.008** | — | 0.729 |

**關鍵觀察：**
- GSM8K No-CoT EMA 崩塌最嚴重（late tonic 0.295）；CoT 顯著抑制崩塌（0.776）
- MATH 生成較長，EMA 相對穩定（0.73–0.78）

#### 4.2 Phase 2：閉環調控結果（GSM8K No-CoT，300 samples，v1 方案）

| 條件 | Acc | early tonic ratio | late tonic ratio | α 正向% | α 均值(正) |
|------|-----|-------------------|-----------------|---------|-----------|
| Baseline (α=0) | 61.7% | 1.055 | 0.295 | — | — |
| Plan A (tonic floor) | 62.3% | 1.068 | 0.527 | 60.4% | 0.352 |
| Plan B (plateau imitation) | 61.3% | 1.091 | 0.525 | 70.0% | 0.374 |
| Plan C (tonic + spike damping) | 60.3% | 1.085 | 0.512 | 68.4% | 0.394 |

**信號層面**：late tonic ratio 從 0.295 提升至 0.51–0.53，injection 確實有效（pre-hook 修復後確認）。

**準確率層面**：A/B/C 變化 +0.6% / −0.4% / −1.4%，均無顯著提升。

**CoT 條件補充**（Plan A/B/C，baseline=76.0%）：

| 條件 | Acc |
|------|-----|
| Plan A (CoT) | 75.67%（−0.3%） |
| Plan B (CoT) | 73.00%（−3.0%） |
| Plan C (CoT) | 75.33%（−0.7%） |

**診斷：**
- v1 幾乎只有正向注入（A/B 100% 正向，C 負向僅 2.2%）
- 注入觸發時 EMA/x_prefill ≈ 0.47–0.56（信號已崩塌後補救），效果滯後
- late tonic ratio：wrong 樣本（0.56）反而高於 correct（0.51）→ late EMA 不是因果關鍵

#### 4.3 Phase 2：Plan D 結果（GSM8K No-CoT，300 samples）

| 條件 | Acc (all) | Acc (valid only) | 爆炸樣本數 | alpha 最大值 |
|------|-----------|-----------------|-----------|------------|
| Baseline | 61.7% | 61.7% | 0/300 | — |
| Plan D k1=0.5 | 60.0% | 60.0% | 0/300 | ±0.83 |
| Plan D k1=1.0 | 58.7% | 59.9% | 11/300 | ±5.6 |
| Plan D k1=2.0 | 7.0% | 40.0% | 290/300 | ±120 |

注：「爆炸」定義為 ema_decode 或 x_decode 中出現 nan/inf 或超過 100×x_prefill 的樣本。

**三組結果的共同結論：**

- k1=0.5：系統穩定，alpha 從未超過 ±1，無樣本爆炸。但 acc=60.0% vs baseline 61.7%，幾乎沒有正向效果。平滑效果極弱（每步推力僅 ~1.5% of xp），波形由語言模型自然動態主導。
- k1=1.0：11 個樣本爆炸，alpha std 已達天文數字。臨界不穩定，acc 輕微下降。
- k1=2.0：290/300 樣本爆炸，acc=7%。valid only acc=40%（未爆炸的 10 個樣本答對率尚可），說明爆炸本身而非控制方向導致 acc 崩潰。

**根本矛盾：**
```
k1 小 → 穩定，但推力太小，平滑效果消失（k1=0.5）
k1 大 → 有推力，但系統爆炸（k1≥1.0）
中間沒有「夠用又穩定」的 k1
```

穩定邊界估算 `k1 < xp/β ≈ 0.574/0.7 ≈ 0.82`，與實驗完全吻合。

**失敗的更深原因：信號時間尺度不匹配**

Plan D 的失敗不只是「spike 太多」，而是 RSN 信號的自然時間尺度（1-2 token spike）比 1-step lag 還短：

```
如果 spike 持續 1 步（語言模型的實際情況）：
  step t:   觀測到 spike → 計算負 α
  step t+1: 注入負 α，但 spike 已自然消退
            → 壓制一個已經不存在的東西 → 製造 dip
  step t+2: 觀測到 dip → 計算正 α → 製造新 spike
```

控制器的反應延遲（1 step）≥ 被控信號的持續時間（1-2 steps），導致「追尾」而非「平滑」。即使完全沒有 spike，只要信號變化比 lag 快，Plan D 就會製造振盪。

**→ 改用 Plan E（EMA Homeostasis）**：EMA 每步只移動 ~5% of x_t，天然比 lag 慢，規避時間尺度不匹配問題。

#### 4.4 Phase 2：Plan E / F 結果（GSM8K No-CoT，300 samples，完整 k1 sweep）

| 條件 | acc | valid acc | exploded | α 範圍 | clamp 飽和率 |
|------|-----|-----------|----------|--------|-------------|
| Baseline | 61.7% | 61.7% | 0/300 | — | — |
| Plan E k1=0.5 | 61.0% | 61.0% | 0/300 | [-0.54, +0.39] | 0% |
| Plan E k1=1.0 | 60.7% | 60.7% | 0/300 | [-0.83, +0.58] | 0% |
| Plan E k1=2.0 | 61.3% | 61.3% | 0/300 | [-1.21, +0.91] | 0% |
| Plan F k1=0.5 | 60.3% | 60.3% | 0/300 | [-0.58, +2.29] | 0% |
| Plan F k1=1.0 | 60.7% | 60.7% | 0/300 | [-1.22, +1.22] | 0% |
| Plan F k1=2.0 | 56.0% | 56.0% | 0/300 | [-8.0, +8.0] | 5.0% |

**Plan E 的觀察：**

- **三個 k1 全部穩定**，α 自然停留在 ±1.2 內，clamp 從未觸發
- **acc 三個都接近 baseline**（±1% 內波動），EMA 形狀被有效拉向 0.85 target
- 形狀符合預設意圖（前期壓抑過高、後期托起過低），但**形狀的改變沒有轉化為 acc 提升**
- 證實了「homeostasis 抗 lag」的設計假設：用慢信號（slow_ema）作控制輸入確實天然穩定

**Plan F 的觀察：**

- k1=0.5/1.0 表現類似 E（α 小、acc 持平）
- k1=2.0 出現 phase transition：撞 clamp 5%、acc 掉 5.7%
- 證實了 F 的 lag 不穩定假設：fast_ema 時間常數僅 ~1.4 步，本質仍接近 raw x_t，開環增益在 k1=2.0 突破穩定邊界
- F 的目標（fast - slow）本身在劇烈波動，控制器追一個自己也在抖的目標 → 越控越亂

**整個比例反饋路線（D/E/F）的死局確認：**

```
反饋越激進（D） → 越不穩定，爆炸
反饋折衷（F）   → 部分發散，部分無效
反饋越保守（E） → 完全穩定，但無效
```

α 自然穩定於 ±1.2 內，遠低於 known-effective steering 量級（±4），驗證了「**閉環反饋控制天然找不到「足夠大但又穩定」的工作點**」這個架構性矛盾。

**→ 改用 Plan G（Bang-Bang Dead-Zone）**：放棄比例控制，用固定 α 脈衝直接繞開「α 大小由 deviation 決定」這個機制性瓶頸。

#### 4.5 Phase 2：Plan G 結果（GSM8K No-CoT，300 samples，k1 sweep，k2=0.2）

| 條件 | acc | late tonic | α 範圍 | 介入率 | 爆炸 |
|------|-----|------------|--------|--------|------|
| Baseline No-CoT | 61.7% | 0.295 | — | — | 0/300 |
| Baseline CoT    | 76.0% | 0.776 | — | — | 0/300 |
| Plan G k1=1.0 | **61.7%** | 0.796 | ±1.0 | 52.3% | 0/300 |
| Plan G k1=2.0 | 61.0% | 0.826 | ±2.0 | 27.6% | 0/300 |
| Plan G k1=4.0 | 58.3% | **0.857** | ±4.0 | 14.5% | 0/300 |

**核心觀察：**

- **「α 太小」假設被推翻**：G k1=4.0 達到了 last-token steering 同量級（±4），系統穩定 0 爆炸 —— bang-bang 確實突破了 D/E/F 的「α 大小由 deviation 決定」機制性瓶頸
- **「形狀因果」假設被強烈否證**：G k1=4.0 的 late tonic（0.857）已超過 CoT（0.776），但 acc 反而比 baseline 降 3.4% —— 形狀塑造成功，acc 反向移動

**Plan G 的 timing-direction 分解（關鍵發現）：**

統計 G 在「早期 step<50」和「後期 step≥50」的注入方向比例：

| 階段 | +k1（抬升）| -k1（壓制）| 主導行為 |
|------|-----------|-----------|---------|
| 早期 step<50 | ~25% | **~75%** | **以壓制為主** |
| 後期 step≥50 | **~96%** | ~4% | 以抬升為主 |

**物理解釋**：早期 ema 從 prefill 自然衝高到 ~1.18，**超過 G 的上界 xp+0.2·xp=1.2 邊界附近**，所以 G 主要在做 **-k1 壓制 peak**；後期 ema 自然衰減到 0.30，遠低於下界 0.8·xp，所以 G 主要在做 **+k1 拉升 tonic**。

**這解釋了為什麼 G k1 越大 acc 越降**：
- 不是「過度推高 late tonic 傷害推理」
- 而是「越激進地扼殺自然的 task-onset peak」（baseline early peak ~1.18 → G k1=4.0 早期被壓到 ~1.10）
- early peak 是 dopamine 的 commitment-to-engagement signal，被壓制 → 推理質量下降

**baseline 的早期峰值差異也支持這個解讀**（從 §4.1 可推算）：

| 條件 | correct early peak | wrong early peak | 差異 |
|------|-------------------|------------------|------|
| CoT | ~1.30 | ~1.20 | +0.10 |
| No-CoT | ~1.18 | ~1.10 | +0.08 |

correct vs wrong 的最大形狀差異就在 early peak 高度，late tonic 反而是次要差異。

**→ 改用 Plan H1（Early Peak Boost）**：反向操作 G 的早期壓制行為，在前 100 step 主動抬升 ema 至 peak target，後續零干預。判決性測試「early peak = task-onset commitment signal」假設。

#### 4.6 Phase 2：Plan H1 結果（GSM8K No-CoT，300 samples，k1 sweep，peak_target=1.5·xp，window=100）

| 條件 | Acc | EarlyPeak (mean/xp) | LateTonic (mean/xp) | 注入次數 | EP_correct | EP_wrong |
|------|-----|---------------------|---------------------|----------|-----------|---------|
| Baseline No-CoT | **61.7%** | 1.256 | 0.295 | 0 | 1.285 | 1.210 |
| H1 k1=2.0 | 58.3% | 1.597 | 0.331 | 13167 | 1.615 | 1.572 |
| H1 k1=4.0 | 56.0% | 1.637 | 0.332 | 7463 | 1.648 | 1.623 |
| H1 k1=6.0 | 34.0% | 1.689 | 0.319 | 5425 | 1.717 | 1.674 |

**關鍵數據**：
- 注入確實發生在前 100 step（early 99%+ vs late <1%），說明 H1 邏輯正確
- Early peak 成功抬升：baseline 1.256 → H1 k1=2.0 1.597（+27%），k1=6.0 達 1.689
- 但 acc 全面下降：k1=2.0 → 58.3%（-3.4%），k1=4.0 → 56.0%（-5.7%），k1=6.0 → 34.0%（-27.7%）
- EP_correct 與 EP_wrong 差距維持不變（baseline Δ=0.075 → H1 k1=2.0 Δ=0.043）甚至縮小

**EMA 軌跡的「平頂 + 斷崖」結構**（從 `H1_full_ema.png`）：

- 0–20% progress（前 100 step boost window 內）：被強行壓在 1.5·xp，幾乎平頂方波
- ~20% 處（step 100，window 結束）：斷崖式下降，幾乎垂直
- 25–100%：與 baseline No-CoT 軌跡幾乎完全重合（後段 ~0.3）

斷崖位置與 EMA 時間常數（α=0.95 → τ ≈ 20 步 ≈ progress 5%）吻合，外加能量在 ~20 步內衰減到 1/e。**後段曲線完全回到 baseline → 前 100 步的高 dopamine 沒有通過 KV cache「鎖定」後續推理狀態**。

**結論：「Early peak = task-onset commitment signal」假設被否證**

1. **形狀可操控，acc 不可操控**：H1 成功把所有樣本的早期峰值拉到 1.5·xp，但 acc 反而下降。早期峰值高度不是 acc 的因果驅動因素，而是已有知識強度的結果。
2. **注入量越大，破壞越大**：k1=6.0 acc=34.0% 接近隨機，超大早期注入直接干擾 LLM 的 prefill 狀態延伸（KV cache 被強烈改寫，後續推理流程崩潰）。
3. **EP_correct vs EP_wrong 差異不可放大**：baseline 中 correct 比 wrong 早期峰稍高（Δ=0.075）是模型自身推理品質的信號，外部注入把兩組都拉高但 Δ 縮小，進一步說明這個差異是「結果」非「原因」。
4. **commitment 沒有建立**：後段曲線完全回到 baseline 軌跡，外力一停立刻回歸自然動態 → KV cache 並沒有被早期注入「鎖定」到新的 attractor。

#### 4.7 Phase 2：Plan H2 結果（GSM8K No-CoT，300 samples，k1 sweep，peak=1.25·xp，end=0.75·xp）

| 條件 | Acc | EarlyPk | LateT | α 範圍 | clamp | fit_err |
|------|-----|---------|-------|--------|-------|---------|
| Baseline No-CoT | 61.7% | 1.256 | 0.295 | — | — | — |
| Baseline CoT | 76.0% | 1.354 | 0.776 | — | — | — |
| H1 k1=2.0（對照）| 58.3% | 1.597 | 0.331 | +2.0 固定 | 0% | — |
| H2 k1=1.0 | 59.7% | 1.220 | 0.501 | [-0.64, +0.87] | 0% | 0.308 |
| H2 k1=2.0 | 60.7% | 1.211 | 0.584 | [-1.00, +1.25] | 0% | 0.221 |
| **H2 k1=4.0** | **62.7%** | 1.210 | 0.643 | [-1.43, +1.69] | 0% | **0.141** |

**核心發現**：

1. **k1 ↑ → fit_err ↓ → acc ↑（單調）**：擬合誤差從 0.308 → 0.141 約 1/k1 衰減，對應 acc 從 59.7% → 62.7% 單調上升。**首次在 A–H1 系列中觀察到「形狀擬合度」與「acc」的單調正相關**。

2. **H2 vs H1 戲劇性反轉**：
   - H1 binary 過衝 → 早期 EMA 推到 1.6（超過目標 1.5），後段斷崖回 0.33 → acc -3.4%
   - H2 比例追蹤 → 早期 EMA 維持 1.21（接近 baseline 1.256，**沒抑制 task-onset peak**），全程平滑跟蹤 → acc +1.0%

**未解問題與謹慎解讀**：

- **+1.0%（300 樣本）≈ 3 題差**，統計顯著性需更大樣本或 bootstrap 驗證
- 距離 CoT (76.0%) 仍有 14% gap，**H2 不等於「重現 CoT 效果」**
- k1=4.0 平均 α=0.5，可能等價於「弱化版 static steering」——需要對照 static α=+0.5 排除這個解釋
- H2 trapezoid 與 CoT 真實波形仍有結構性差異（CoT 早期更陡、peak 更高、plateau 更短）

**→ 改用 Plan H3（Trapezoid v2）**：把 trapezoid 邊界調得更貼近 CoT 形狀（rise 50→30, plateau 200→120, peak 1.25→1.35），並把 k1 推到 {4, 6, 8} 測試「擬合度→acc」單調關係的上限。

#### 4.8 Phase 2：Plan H3 結果

設置：k1 ∈ {4.0, 6.0, 8.0}（3 runs），peak=1.35·xp，end=0.75·xp，rise=30, plateau_end=120, T=400。

| 條件 | Acc | EarlyPk | LateTonic | ⟨α⟩ | α_max | α_min | fit_H3 | fit_H2 | fit_CoT |
|---|---|---|---|---|---|---|---|---|---|
| Baseline No-CoT | 61.7% | 1.256 | 0.295 | 0.000 | 0.00 | 0.00 | 0.529 | 0.525 | 0.494 |
| Baseline CoT | 76.0% | 1.354 | 0.776 | 0.000 | 0.00 | 0.00 | 0.333 | 0.332 | 0.336 |
| **H2 k1=4.0** | **62.7%** | 1.210 | 0.643 | 0.503 | 1.69 | −1.43 | 0.152 | 0.141 | 0.121 |
| H3 k1=4.0 | 61.0% | 1.317 | 0.646 | 0.519 | 1.78 | −1.35 | 0.143 | 0.148 | 0.122 |
| H3 k1=6.0 | 60.3% | 1.334 | 0.679 | 0.561 | 2.01 | −1.37 | 0.103 | 0.113 | 0.089 |
| H3 k1=8.0 | 59.7% | 1.342 | 0.699 | 0.576 | 2.14 | −1.86 | 0.081 | 0.094 | 0.073 |

**判讀**：落入第三種情況——**H3 acc < 62.7%**，且 acc 隨 k1 上升單調下降（61.0 → 60.3 → 59.7%），全部低於 No-CoT baseline。

**關鍵反證**：H3 的形狀**更**貼近 CoT（EarlyPk 1.32–1.34 vs CoT 1.354；LateTonic 0.65–0.70 vs CoT 0.776；fit_CoT 0.073–0.122 比 H2 的 0.121 更低），但 acc 反而比 H2 更差。這直接否定「trapezoid 形狀貼近 CoT → 高 acc」的因果讀法。

**對 H2 +1% 的重新理解**：H2（rise=50, peak=1.25）的小幅領先**不是**來自形狀拟合度本身。可能機制：
- 較緩 rise 落在 controller 穩定區間，沒有觸發 1-step lag 的振盪放大
- 較低 peak 沒有過度抑制 early-token 的探索性 token 選擇
- H3 的過陡 rise（30 步達 peak）+ 過高 peak（1.35）→ early decode 階段強行偏移 RSN 表徵，反而干擾解題路徑

**結論**：Phase 2 形狀模擬路線基本結案。H2 的 +1% 在現有 controller 下是天花板；要進一步逼近 CoT 的 +14%，需要 CoT 的**內容**（中間 token 攜帶的推理步驟），不是 RSN 投影的形狀。專案轉向 Phase 1b 訊號代理驗證（§4.9）。

---

#### 4.9 Phase 1b 訊號代理驗證 — 已搬遷至 `AdaptativeThinking0529.md`

> **此處原 §4.9 / §4.9.1（Neutral / Expert / Non-Expert 信號對比、multi-metric 補充）已刪除。**
> 那批數字用的是舊 prompt（角色 `a mathematician expert` / `a non-mathematician expert`、兩輪 GSM8K 結果對不上、extraction 不穩），與 2026-05-30 對稱化模板 + `####` directive 修正後的 pipeline **不可比**，故不保留。
>
> Phase 1b 信號重跑（paper-aligned roles：`an expert` / `a non expert` / `a primary school teacher` + neutral No-CoT/CoT）的所有新結果寫入 **`AdaptativeThinking0529.md`**（current-state 文檔）。本歷史文檔僅保留 §3.1b 的 Multi-Metric Signal Suite SOP（流程規範，不綁定具體數字）作為 pipeline 參考。

---

### 5. 分析腳本

| 腳本 | 對應分析 | 輸出圖 |
|------|---------|--------|
| `analyze_dopamine_signal.py` | Phase 1 EMA / x_prefill 分析 | `prefill_distribution`, `ema_curves_absolute`, `ema_curves_normalized`, `prefill_by_difficulty`, `prefill_by_condition`, `early_ema_by_condition` |
| `analyze_dopamine_spikes.py` | Phase 1 spike 結構分析 | `spike_gallery_*` ×4, `spike_density_normalized`, `raw_signal_normalized`, `spike_stats_summary` |
| `analyze_flow_shapes.py` | EMA 軌跡形狀分類（flow hypothesis） | `flow_shape_gallery`, `flow_shape_distribution`, `flow_target_hypotheses`, `flow_individual_traces_*` ×4 |
| `phase1_summary.py` | Phase 1 四條件總覽 | `phase1_summary_all_conditions` |
| `analyze_closed_loop.py` | Phase 2 Plan A/B/C 閉環結果分析 | `loop_accuracy_bar`, `loop_ema_normalized`, `loop_late_tonic_ratio`, `loop_alpha_profile`, `loop_intervention_stats`, `loop_ema_gallery` |
| `analyze_planD.py` | Phase 2 Plan D 三組 k1 分析（含爆炸診斷） | `planD_accuracy_bar`, `planD_ema_normalized`, `planD_alpha_profile`, `planD_divergence`, `planD_ema_gallery`, `planD_raw_signal_k05` |
| `analyze_planEF.py` | Phase 2 Plan E / F 完整 k1 sweep 對比 | `EF_accuracy_bar`, `EF_ema_normalized_E`, `EF_ema_normalized_F`, `EF_alpha_profile_E`, `EF_alpha_profile_F`, `EF_late_tonic`, `EF_alpha_hist`, `EF_ema_gallery` |
| `analyze_planG.py` | Phase 2 Plan G k1 sweep + 全方案總覽 | `G_accuracy_bar`, `G_ema_vs_CoT`, `G_alpha_profile`, `G_alpha_hist`, `G_ema_gallery`, `ALL_plans_with_G` |
| `analyze_planH1.py` | Phase 2 Plan H1 k1 sweep + 全方案總覽（含 H1） | `H1_accuracy_bar`, `H1_early_phase_ema`, `H1_full_ema`, `H1_alpha_profile`, `H1_early_peak_hist`, `H1_ema_gallery`, `ALL_plans_with_H1` |
| `analyze_planH2.py` | Phase 2 Plan H2 k1 sweep + 軌跡擬合度分析 | `H2_accuracy_bar`, `H2_full_ema`（含 target 疊圖）, `H2_tracking_error`, `H2_alpha_profile`, `H2_alpha_hist`, `H2_ema_gallery`, `ALL_plans_with_H2` |
| `analyze_planH3.py` | Phase 2 Plan H3 k1 sweep + 與 H2/CoT 形狀貼近度對比 | `H3_accuracy_bar`, `H3_full_ema`, `H3_tracking_error`, `H3_alpha_profile`, `H3_alpha_hist`, `H3_vs_H2_compare`, `ALL_plans_with_H3` |
| `extract_signal_json.py` *(server-side)* | Phase 1b：從 HDF5 raw HS 重投影 mask（NMD / random），輸出與 `signal/` schema 一致的 JSON | 無圖；產出 `dopamine_signal_gsm8k_8B_nocot_<role>[_<mask>]_ema0.95_L11-20.json` |
| `extract_entropy_confidence.py` *(server-side)* | Phase 1b：從 HDF5 final-layer HS 過 lm_head 算 entropy / top1 / margin / info_gain per step | 無圖；產出 `metrics_gsm8k_8B_nocot_<role>_ema0.95_L11-20.json` |
| `analyze_expert_vs_non_expert.py` | Phase 1b：Neutral / Expert / Non-Expert NMD-投影 EMA 軌跡對比 | `expert_vs_non_expert_correct_wrong` |
| `analyze_multi_metric.py` | Phase 1b：dopamine + entropy + top1 + margin + info_gain 多指標 × role × correct/wrong 對比，含 cross-metric Pearson 矩陣與 partial correlation | `multi_metric_curves`, `multi_metric_prefill`, `multi_metric_summary`, `multi_metric_correlation` |

資料目錄：`llama3/dopamine/signal/`（Phase 1）、`llama3/dopamine/loop/`（Phase 2）、`llama3/dopamine/plots/`（所有圖）、`/data1/paveen/Dopamine/components/hidden_states/<task>/`（Phase 1b raw HS HDF5，僅 server）

## TODO

排序原則：先驗證 RSN/dopamine signal 本身，再看 α=-4 為什麼有效，最後才做 router、reasoning model 和大 benchmark。

1. expert vs non-expert vs neutral (non-cot)：看RSN curve是不是有差異 ✔
2. expert vs non-expert vs neutral (non-cot)： Other metrics & Random mask ✔ 
3. 更新Expert的設定 + 多指標分析 neutral (cot & non-cot)

1. Validate dopamine signal proxy: selected RSN vs random projection, CoT vs No-CoT.

2. Validate dopamine signal proxy: selected RSN vs random projection, expert vs non-expert vs neutral.

3. 功能神經元 baseline：用 Language-sensitive / Emotion-sensitive neurons 做對比，檢驗 role-sensitive neurons 的獨特性和必要性。

4. Check Llama3-8B curves under static α=-4 / α=+4，對比 α=0、CoT、No-CoT。

5. 提前干預 / prefill intervention：比較 last prefill token、decode step 0、decode-time 全程注入；看曲線和 acc 是否不同。

6. Multi-metric tracking：除了 RSN activation，也同步收集 MSP / confidence logit、entropy、constrained entropy、logit margin、E-option logit / abstention probability。

7. Calibration on RSN-steered outputs：算 ECE / Brier / AUROC，檢查 α steering 是否造成 unwarranted certainty。

8. Probe validation：分開做 knowledge probe 和 commitment / decisiveness probe，確認 RSN 主要改的是 knowing 還是 willingness-to-act。

9. Adaptive CoT router：只用 prefill 或 very early decode features 預測要不要 think。
   - RSN features: x_prefill, RSN projection mean / variance, middle-layer RSN activation, role-sensitive direction projection, first 5-10 decode token RSN slope
   - uncertainty features: MSP, entropy, constrained entropy, logit margin, E-option logit / abstention probability
   - baselines: entropy threshold, MSP threshold, answer logit margin, question length, random routing, always CoT, always No-CoT

10. 加入 frequency feature：參考 ICLR2026 Balanced Thinking，用 step-level confidence variance / local fluctuation 區分 overthinking 和 underthinking。

11. 加入 InfoBias & InfoGain：參考 NIPS2025 Think or Not，先作為 diagnostic / baseline，不急著變成主控制器。

12. Base model & reasoning model：先做 Llama3-Base vs Llama3-IT；reasoning model 等前面 signal / intervention 站穩後再做。

13. 推理過程中 Dopamine curve 與 Thinking curve 的關係：在 reasoning model 的 `<think>` trace 裡對齊 backtrack / first-commit / hedging / verification marker。

14. RLHF 和 dopamine 出現的關係：整理 Notion `Model Analysis; Hallucination & Origin -> 15. Origin Analysis`，看 post-training 是否 sharpen 了 decisiveness axis。

15. Benchmark scale-up：數學推理先做 AIME24、AIME25、AMC23、MATH-500、Minerva、OlympiadBench；再考慮 GPQA-D、LiveCodeBench。

MATH-500GSM8KMinerva-MathAIME24AMC23OlympiadBench

与正确答案之间的互信息
---
