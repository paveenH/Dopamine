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

## 1. Related Literature

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

- **Prolonged dopamine signalling in striatum signals proximity and value of distant rewards** (Howe et al., Nature 2013) — https://www.nature.com/articles/nature12475
  大鼠朝遠處獎勵移動時，紋狀體 DA 呈現**持續 ramping**，並隨與目標的**距離**及**獎勵大小** scale。這是 dopamine ramping（tonic/phasic 之外第三時間尺度）最直接的行為學量測，本專案 §2.2 ramping / vigor 的主要神經科學依據。

- **Dopamine Ramps Are a Consequence of Reward Prediction Errors** (Gershman, Neural Computation 2014, MIT Press) — https://direct.mit.edu/neco/article/26/3/467/7956/
  TD-learning 框架下推導 ramping：對 goal proximity 做**凸 / 二次變換**（Weber's-law 式空間壓縮）即近似線性 ramp。即 §2.2 假說 A——ramping = 密集微小正向 phasic 脈衝的長程積分。（作者 Gershman 在 Harvard。）相關的 value-decay 機制見 Current Biology 2022「Dopamine ramps with fuzzy value estimates」。局部環路假說（B：紋狀體軸突獨立於 VTA 胞體放電形成 ramp，Mohebi et al.）見 Frontiers 2014「Local control of striatal dopamine release」。

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

## 2. Core Theoretical Framework

### 2.1 Project Positioning

RSN（Role-Sensitive Neurons）被視為一個 **state-level gain control**：它調節模型在當下有多願意啟動、投入、承諾、繼續檢查或停止，而不是直接增加模型的知識或推理 capacity。本文以 `wanting` 描述這類 action-readiness / commitment tendency，並將 dopamine 作為**功能類比與可檢驗假說**。

### 2.2 Temporal Components

本研究提出一個三成分假說，將 generation dynamics 分為 **task-entry tonic、ramping 與 decode-time phasic**：

| Component | Functional interpretation | Operational signal | Main prediction |
|---|---|---|---|
| **Task-entry tonic** | 進入任務時設定初始增益與 commitment threshold | `G_prefill` | α 改變起始狀態，並影響後續解題與提交策略 |
| **Ramping / Vigor** | 解碼期間朝目標推進的速度與 effort intensity | `s_t = EMA(Z_t)` 的斜率 | 斜率越陡，推進與 commitment 越快 |
| **Phasic** | decode 中相對慢基線的快速 pulse / dip | `p_t = Z_t - s_{t-1}` | 與慢變的 `s_t` 分離，呈現 token-level transient |

在此模型中，`G_prefill` 設定 generation 的初始條件，不作為 decode trajectory 的持續加數；decode 期間的信號表示為：

```text
Z_t = s_t + p_t
```

其中 `s_t` 表示慢變的 ramping / vigor component，`p_t` 表示相對慢基線的 phasic component。

### 2.3 Working Hypotheses

**H1 — Prefill steering acts through initial-condition / boundary-gating.**

α 只在最後一個 prompt token 注入一次，直接改變 `G_prefill`，但 decode 不再持續注入。現有三項觀察為：

1. `G_prefill` 隨 α 近線性移動；
2. 到 `decode[0]` 時約 95% 回彈；
3. 後續各 α 的 `G_decode` trajectory 基本重合。

**Prefill-only 已顯著改變行為，而 `G_decode` 重合 → 支持 initial-condition / boundary-gating。** α 在 generation boundary 設定 task-level gain 與 commitment regime，後續效應經由 KV cache、早期 token 選擇與 autoregressive path dependence 保留。

**H2 — Slow decode dynamics encode ramping / vigor.**

`s_t` 的斜率表示模型朝答案推進的 vigor。預測較陡的斜率對應較短 generation length、較早 commitment 與較高推進強度。分析時同時控制 output length 與 response format，以區分 vigor 和單純的提前停止。

**H3 — Fast decode residuals encode phasic dynamics.**

`p_t` 表示相對 slow baseline 的快速 phasic change，用來分離 decode 中的 pulse / dip 與 `s_t` 的慢變趨勢。第一步先驗證 `p_t` 是否具有穩定的快時間尺度結構；其後再探索它與中間推理步、答案形成、commitment 或其他 generation events 的關係，目前不預設特定 event anchor。


## 3. Signal Definition

### 3.1 Signal Architecture

信號分成三個層級：

1. **RSN state signal**：middle-layer hidden states 在 NMD direction 上的活動，用來描述 task-entry gain 與 decode dynamics。
2. **Output-distribution signal**：由 final-layer logits 計算 entropy、top1、margin 與 information change，用來描述 confidence / decisiveness。
3. **Behavioral readout**：accuracy、generation length、commitment timing 與 stopping failure，用來檢驗內部信號是否對應可觀察行為。

RSN signal 是主軸；logit metrics 用於判斷 wanting 是否只是 confidence 的另一種表示；behavioral metrics 則提供外部效度。

### 3.2 RSN Projection and Gain Coordinates

對 token `t`、middle layer `l`，先計算原始投影：

```text
r_{t,l} = h_{t,l} · m_l
```

其中 `h_{t,l}` 是 decoder layer output hidden state，`m_l` 是同一 output space 中的 sparse NMD direction。每層先獨立投影，再進行跨層聚合。

為了固定零點、保留 α 的干預單位並避免少數 layer 因尺度較大而主導聚合，使用 neutral、α=0、No-CoT 的 prefill distribution 作為 reference：

```text
μ_l^ref = E[r_{prefill,l} | neutral, α=0, No-CoT]
g_{t,l} = (r_{t,l} - μ_l^ref) / ||m_l||²
σ_l^ref = Std[g_{prefill,l} | neutral, α=0, No-CoT]
z_{t,l} = g_{t,l} / σ_l^ref
```

由此得到兩種跨層 readout：

```text
G_t = mean_l(g_{t,l})
Z_t = mean_l(z_{t,l})
```

| Signal | Role |
|---|---|
| `G_t` | 保留 α-equivalent unit；用於 steering calibration、dose linearity 與 intervention sanity check |
| `Z_t` | 各層先標準化後聚合；作為主要 observational trajectory，避免 layer-scale dominance |

Reference `μ_l^ref` 與 `σ_l^ref` 在所有 role、α、token 與 event 中固定，不隨條件重新估計，才能保留條件間與 generation 階段間的真實差異。

**坐標分工**：涉及 α 單位的定量陳述（dose linearity、`boundary_jump` 回彈比例、steering calibration，即 H1）一律在 `G` 坐標；涉及 trajectory 形狀的分析（`s_t` slope、`p_t` residual，即 H2/H3）在 `Z` 坐標，以確保 layer-fair、不被個別大尺度 layer 主導。

**Layer alignment**：mask row `l` 對齊 `decoder_layers[l]` 的 **output**。Signal observation 與 steering injection 必須使用同一 output space，避免一層 offset。

### 3.3 Temporal Signal Decomposition

#### 3.3.1 Task-Entry Tonic

最後一個 prompt token 的 gain 定義為：

```text
T = G_prefill
```

`T` 是 task-entry tonic 的主 readout，表示 generation boundary 上的初始 gain / commitment set point。`Z_prefill` 可作為 layer-fair 的 condition comparison；`G_prefill` 則保留 α 單位，作為主要 calibration signal。

prefill 到第一個 decode token 的回彈另記為：

```text
boundary_jump = G_0 - G_prefill
```

`boundary_jump` 用來描述 task-entry pulse 如何進入自然 decode dynamics；在 `G` 坐標計算，以與 H1 引用的 α-單位回彈比例（約 95%）保持一致。

#### 3.3.2 Ramping / Vigor

在 decode 內，對 `Z_t` 建立 slow component：

```text
s_0 = Z_0
s_t = β · s_{t-1} + (1 - β) · Z_t
```

`s_t` 只由 decode token 初始化與更新，不以 prefill 作 EMA seed。主要 readout 是 `s_t` 的 trajectory slope，而不是其絕對高度：

```text
vigor_slope = slope(s_t)
```

後續可分別估計 early、middle、late slope；β 與 window 的精確設定留待 sensitivity analysis。

#### 3.3.3 Fast Residual（candidate phasic-like）

decode-time fast component 定義為相對上一時刻 slow baseline 的 residual：

```text
p_t = Z_t - s_{t-1},  t ≥ 1
```

`p_t` 是一個 **EMA high-pass 殘差**（當前 `Z_t` 減上一步慢 EMA），不是 event-locked phasic 信號。`p_t > 0` 表示瞬時高於 slow baseline 的 pulse，`p_t < 0` 表示瞬時 dip。現階段只把它當作 **fast residual / candidate phasic-like component**，優先檢驗其 amplitude、variability 與時間結構；之後再分析它是否與特定 reasoning / commitment event 對齊（不預先指定 event anchor）——通過 event alignment 等驗證後方可升級為正式的 phasic 信號。

第一階段使用以下 summaries：

- `phasic_pos_peak = max(p_t)`
- `phasic_neg_peak = min(p_t)`
- `phasic_abs_mean = mean(|p_t|)`
- `phasic_std = std(p_t)`

### 3.4 Multi-Metric Signal Suite

單一 RSN trajectory 不能區分 wanting、confidence、task performance 與 response failure，因此使用下列 multi-metric suite：

| Family | Metric | Source / computation | Main interpretation |
|---|---|---|---|
| **Task-entry state** | `G_prefill` | α-unit RSN gain at last prompt token | task-entry tonic / intervention strength |
| **Task-entry state** | `Z_prefill` | layer-standardized gain at last prompt token | layer-fair boundary state |
| **Boundary transition** | `boundary_jump` | `G_0 - G_prefill`（G 坐標） | prefill pulse 的回彈 / carry-over |
| **Slow decode** | `s_t` | decode-only EMA of `Z_t`，seed `s_0 = Z_0` | post-launch slow generation dynamics |
| **Relaxation slope** | `vigor_slope` | slope of `s_t` | slow decode component 的下降速度（見下方 naming 說明） |
| **Phasic** | `p_t` | `Z_t - s_{t-1}` | fast pulse / dip relative to slow baseline |
| **Uncertainty** | `entropy_decode` | `-Σ_v q_t(v) log q_t(v)` | next-token uncertainty；越低通常越 decisive |
| **Confidence** | `top1_decode` | `max_v q_t(v)` | maximum next-token probability |
| **Confidence** | `margin_decode` | `top1 - top2` | local choice separation；與 top1 高度相關，作輔助 |
| **Distributional change** | `info_gain_decode` | `H_{t-1} - H_t` | token-to-token uncertainty reduction，不直接等同 reasoning quality |
| **Cumulative change** | `cumulative_entropy_reduction` | `H_0 - H_t` | 相對 generation 起點的累積 certainty change |
| **Confidence stability** | `rolling_conf_variance` | `Std(top1[t-W:t])` | local confidence volatility |
| **Task performance** | `accuracy` | answer correctness | knowing / task success |
| **Behavioral vigor** | `generation_length`, `commit_position` | output tokens；首次答案標記位置 | 推進與 commitment timing |
| **Stopping control** | `loop_rate`, `post_commit_tokens`, `eos_failure` | output pattern diagnostics | stopping failure；不併入 vigor 或 phasic |

Entropy、top1、margin 與 information-change metrics 均由 final-layer hidden state 經 final norm + `lm_head` 得到的真實 next-token logits 計算；RSN metrics 則來自 middle-layer sparse subspace。兩組信號來源不同，必須分開解讀。

### 3.5 Analysis Protocol

每個 sample 同時保存三種資料：

1. **Boundary snapshot**：`G_prefill`、`Z_prefill`、`boundary_jump` 與 prefill logit metrics。
2. **Full token trajectory**：`Z_t`、`s_t`、`p_t`、entropy、top1、margin 與 info gain。
3. **Scalar outcome**：accuracy、generation length、commit position、format validity 與 stopping diagnostics。

不同長度的回答以各自實際 decode length 映射到 `0–100%` progress 後再做 group trajectory comparison；完整 token-step 曲線另行保留，用於 early-step 與局部 pulse 分析，不以 `max_new_tokens` padding。

主要比較軸為：

| Axis | Purpose |
|---|---|
| **α dose** | 檢驗 intervention strength、boundary state 與行為的 dose-response |
| **CoT vs No-CoT** | 檢驗 process scaffold 如何改變 task-entry 與 decode dynamics |
| **Expert vs Non-Expert** | 檢驗 persona state 對 RSN signal 的自然調制 |
| **Correct vs Wrong** | 檢驗 signal 與 performance 的關係；作 outcome analysis，不單獨推論 dopamine |
| **NMD vs Random mask** | 檢驗 trajectory 是否具有 RSN-direction specificity |

舊版 `x_t`、prefill-seeded `ema_t`、`ema_t / x_prefill` 與 `late_tonic` 可保留作 reproduction / ablation，但不再作為新框架的主要 tonic 或 ramping 定義。


## 4. Observed Results Phase1

### 4.0 Acc

**signal vs gsm8k original —— 跨機/batch 偏差存證**：

| cond | signal | original | Δ |
|---|---:|---:|---:|
| α−8 | 40.7 | 40.3 | +0.4 |
| α−6 | 79.7 | 78.0 | +1.7 |
| α−4 | 74.3 | 73.0 | +1.3 |
| **α0** | **60.0** | **60.0** | **0.0** |
| α+4 | 51.3 | 55.3 | −4.0 |
| α+6 | 51.7 | 55.0 | −3.3 |
| α+8 | 49.7 | 53.7 | −4.0 |
| α0·expert | 57.0 | 58.0 | −1.0 |
| α0·non_exp | 66.0 | 68.0 | −2.0 |
| α0·teacher | 68.7 | 68.0 | +0.7 |
| α0_cot | 67.7 | 69.0 | −1.3 |
| α−4_cot | 82.7 | 85.0 | −2.3 |

### 4.1 Correct vs Incorrect Responses

Correctness 不是主要 intervention axis，而是**事後分組**（post-hoc grouping），非受控 intervention condition；本節將其作為 **outcome analysis**，描述 RSN dynamics 與 task performance 的相關結構，而非因果證據。分析使用 neutral No-CoT（180 correct / 120 incorrect）與 CoT（203 / 97）；commit 定義為首個 `####`，若不存在則使用首個 answer candidate。具可辨識 commit marker 的樣本較少（wrong 組尤然）：No-CoT 180 correct / 117 wrong，CoT 203 correct / 94 wrong。

#### Commit Timing

在具有可辨識 commit marker 的 responses 中，correct responses 並未更早提交答案。其 median commit step 顯著晚於 incorrect responses：

| Condition | n (correct / wrong) | Correct median | Incorrect median | MWU p |
|---|---:|---:|---:|---:|
| No-CoT | 180 / 117 | 101 | 56 | 0.006 |
| CoT | 203 / 94 | 203 | 96 | 0.00014 |

因此，length-normalized trajectory 中的 correct / incorrect 差異不能簡單歸因於「correct 更早完成」。相反地，在可辨識 commit 的 samples 中，incorrect responses 較早形成 answer candidate。

#### Commit-Aligned Dynamics

將各 sample 對齊至自己的 commit step（`t=0`）後，可觀察到三個主要模式：

1. **Pre-commit slow RSN state**：correct 組在 commit 前維持較高、較持久的 `s_t`；commit 後則下降得更快，約在 10–15 tokens 後低於 incorrect 組。前者表示組間差異不只是 commit timing 錯位，後者則提示 correct responses 在答案形成後具有更快的 state release / termination dynamics。此屬**描述性差異**，不宜單向解讀為「高 `s_t` 導致答對」——同樣相容的路徑是：模型對可解的題較早形成可行思路（perceived solvability / viable reasoning path）→ sustained engagement → 較晚 commit → 較高正確率，即 engagement 與 correctness 可能同受題目可解性驅動。
2. **Commit-centered transition**：`Z_t`、`s_t`、`p_t` 與 entropy / top1 / margin 均在 commit 附近快速變化。這支持 commit marker 對應一個明顯的 generation-state transition；但 `####` 本身即改變 token distribution，故 entropy / top1 / margin 的變化可能部分來自**格式轉換**，不能全數歸因於 confidence 的實質改變。
3. **Post-commit separation**：correct 組呈現較強的 `s_t` decline 與較完整的 confidence recovery；incorrect 組下降較慢、恢復較弱，可能反映不同的 termination dynamics（confidence recovery 無 CI，屬描述性觀察）。

| Signal family | Main observation | Interpretation |
|---|---|---|
| `s_t` / `Z_t` | correct 在 commit 前較高，commit 後下降較深 | slow RSN dynamics 與 response outcome 描述性相關 |
| `p_t` | commit window 有明顯 pulse / dip，但兩組在 commit 前無穩定分離 | 通用 transition signal，非穩定 correctness predictor（不作 RPE 解讀） |
| entropy / top1 / margin | commit 附近共同出現 uncertainty increase / confidence decrease | commit 是 output-distribution transition，部分來自 `####` 格式轉換 |
| info gain | commit 附近波動，但組間結構不穩定 | 僅作 distributional-change diagnostic |

**Overall:** correct / incorrect responses 在 commit 前後的 **slow RSN dynamics** 上存在描述性差異——correct 組 pre-commit `s_t` 較高、較持久，post-commit release 較快；task-entry gain、單一 phasic amplitude 與 confidence metrics 均不是穩定的 correctness predictor。這些圖來自同一批資料的兩種對齊方式，應視為互補分析，而非獨立證據。

> This pattern is **consistent with** adaptive engagement and termination dynamics, but may reflect perceived solvability or the availability of a coherent reasoning path rather than a direct causal effect of wanting on accuracy. 三種因果方向無法在此區分：(1) engagement↑ 促成答對；(2) 題目可解 → engagement↑；(3) 難度/熟悉度同時影響 engagement 與正確率。本節支持「RSN tracks engagement during viable reasoning」，但**不能單獨證明**「更高 dopamine → 更高正確率」——Dopamine 的主證據仍來自 α intervention、dose-response 與行為學實驗。

**Boundary:** commit-centered logit change 可能部分來自 `####` / answer-format transition；slow RSN difference 仍需 difficulty-matched analysis 驗證。

### 4.2 CoT vs No-CoT

#### Scope and Analysis Framework

只分析 **neutral、α=0、相同 300 道 GSM8K**。兩組除 CoT 增加 `Let's think step by step.` 外，其餘生成條件一致。統計統一採 paired mean difference、bootstrap 95% CI、Cohen's `d_z` 與 Wilcoxon signed-rank；軌跡圖的 bootstrap band 則使用各條件可用樣本作描述性展示。

本節依序觀察三類訊號：task-entry gain（`G_prefill` / `Z_prefill`）、slow RSN state（`s_t`）與 fast residual（`p_t`）。`s_t` / `p_t` 建於中層 sparse NMD projection；entropy / top1 / margin 則來自最終輸出分布，只作獨立 confidence-proxy 對照。

由於 **97% 樣本撞 767 max-token cap**，且大量 commit 後文字退化為 `#### N ...` 重複，decode 以第 1、2 個 `####` 切成三個互斥階段：

- **pre-commit** = decode start → 第 1 個 `####`（**答案形成**，正文主分析）
- **post-commit** = 第 1 個 `####` → 第 2 個 `####`（**提交後延續生成 / state release**）
- **loop tail** = 第 2 個 `####` → generation end（**stopping failure**，僅作診斷）
- *full = 三段之和*（僅用於展示 loop contamination，不作機制結論）

`s_t = EMA(Z_t)` 在整條 decode 上只計算一次（`s_0=Z_0`），`p_t=Z_t-s_{t-1}` 再由同一條軌跡取得；三階段不重新 seed。C1/C2-centered 圖只是放大兩個階段邊界，不構成另一套切分。C2-based 分析只涵蓋具有第二個 `####` 的 loop-prone subset，因此屬診斷性結果。

#### Integrated RSN Dynamics

**Task-entry gain.** CoT 在尚未 decode 時已提高 generation-boundary RSN state：

| Readout | No-CoT | CoT | ΔCoT−No | 95% CI | `d_z` | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| `G_prefill` | 0.000 | 0.071 | **+0.071** | [0.058, 0.085] | **0.59** | 1.9e-19 |
| `Z_prefill` | 0.000 | 0.158 | **+0.158** | [0.138, 0.178] | **0.89** | 1.1e-32 |
| `boundary_jump_G` | 0.167 | 0.094 | −0.073 | [−0.138, −0.006] | −0.13 | 2.4e-02 |

No-CoT 的 `G_prefill=0` / `Z_prefill=0` 來自 reference 定義。CoT effect 在 α-unit 與 layer-standardized 坐標中均成立，且兩組 `G_decode[0]` 幾乎相同（約 0.17）；因此主要差異已在 boundary 出現，`boundary_jump_G` 只呈現弱小的補償性縮減。`G_prefill` 是最後一個 prompt token 的 task-entry **tonic-like readout**，不能單獨證明後續 decode state 完全由該點決定。

**Slow and fast RSN results by stage.** 下表把原本分散的 `s_t` 與 `p_t` 結果放在同一時間軸；fast-residual 主證據限於 `abs_mean` / `std`，極值因 length bias 與 EMA lag 不作主讀。

| Stage | Slow RSN `s_t` | Fast residual `p_t` | Functional reading |
|---|---|---|---|
| **pre-commit** (n=281) | mean Δ=**+0.471**, `d_z`=**0.65**; `end−start` Δ=+0.228, `d_z`=0.27 | `abs_mean` Δ=+0.056, `d_z`=0.42; `std` Δ=+0.092, `d_z`=**0.63** | CoT 維持較高、較少 relax 的 answer-formation state，且 fast residual dispersion 較強 |
| **post-commit** (n=144) | `end−start` Δ=**−0.166**, `d_z`=−0.31 | `abs_mean` / `std` 小幅反轉（`d_z`=−0.19 / −0.18） | CoT 在首次提交後釋放較快，fast variability 差異同步減弱 |
| **loop tail** (n=141) | mean Δ=+0.185, `d_z`=0.32；tail 短 **83.6 tokens**, `d_z`=−0.51 | `std` Δ=−0.083, `d_z`=−0.33 | 低 RSN state 下的 stopping failure；CoT tail 較短、churn 較低，只作診斷 |

CoT 的主效應是 **level-dominant but not a pure shift**：task entry 已升高（`Z_prefill d_z`=0.89），pre-commit slow level 仍維持明顯差距（`d_z`=0.65），而 No-CoT 在答案形成期間 relax 得更多。strict-`####` subset（n=233）確認 pre-commit level effect 不依賴 fallback；absolute-token control 亦重現較弱的 CoT relaxation，因此不是單由長度歸一化製造。由於 CoT instruction 持續存在於 KV cache，這些觀察不能證明後續差異完全由 `G_prefill` 單點造成。

**Commit-centered boundaries.** C1 前 CoT `s_t` 維持高位，C1 後兩組共同快速下降，但 CoT 保留較高 offset；`p_t` 同時出現明顯負向 transition。C2-centered 圖重現近似結構，顯示第二個 `####` 也伴隨 marker-locked transition，但尚不能證明它是獨立的 loop-onset mechanism。C1/C2 都涉及 `####` / answer-format 改變，因此 fast transition 仍需 token-class-matched pseudo-event control。

**Full-decode diagnostic.** 若把整條 767-token decode 當作一段，slow slope 與 relaxation 會被 loop tail 反轉；full trajectory 因此只展示污染，不代表正常 reasoning 的 late stage。

`p_t` 是 EMA high-pass residual，不是已識別的 biological phasic dopamine。其 pre-commit `abs_mean` / `std` 效應在 FDR 後穩定，但 post-commit 小效應不宜強調；`pos/neg_peak` 同時受 segment length 與 EMA lag 影響，不能作獨立證據，也不能由較深負尾推斷 downward skew。

Event alignment 現已顯示 C1/C2 附近存在 marker-locked residual transition，但它同時伴隨 entropy spike / top1 dip，且 C2 幾乎複製 C1，因此可能包含 answer-format transition。現階段最穩健的結論仍是 **stage-dependent fast RSN dynamics**。若要升級為 commitment-related phasic-like component，仍需 pseudo-event / token-class control、length-matched quantile、`β∈{0.90,0.95,0.98}` sensitivity、其他 baseline estimator，以及 NMD-mask vs random-mask specificity。GSM8K 沒有 reward feedback，因此此處不能作 RPE 解讀。

#### Output-Distribution Confidence Controls

**基座警示（load-bearing）：RSN wanting 與 confidence proxy 不可視為同一訊號或疊在同一軸。** 前者是中層 hidden states 對 sparse NMD mask 的投影；後者是最終層經 RMSNorm、full lm_head 與全詞彙 softmax 後的 next-token distribution。entropy / top1 / margin 衡量 output decisiveness，不是最終答案的 epistemic confidence。

Confidence 現以**與 wanting 相同的三段切分**（pre-commit / post-commit / loop-tail，用同一組 1st/2nd `####` boundary）分析，使兩軸可在同一 stage boundary 下比較 measurement pattern。主分析限於兩組都有有效 commit 的 paired subset（pre-commit n=203；post/loop 因需有效 2nd `####` 降至 n=144/141，屬 C2-valid subset）。每格為 ΔCoT−No（括號內 `d_z`）。

**表 A：三段 stage means（與 wanting 同 boundary，paired）。**

| Metric | pre_commit (live) | post_commit（飽和·診斷） | loop_tail（飽和·診斷） |
|---|---:|---:|---:|
| entropy | **−0.176** (−0.83)*** | −0.052 (−0.06) ns | +0.074 (0.10)*** |
| top1 | **+0.039** (0.74)*** | +0.019 (0.09) ns | −0.014 (−0.10)*** |
| margin | **+0.048** (0.66)*** | +0.024 (0.08) ns | −0.018 (−0.11)*** |
| info_gain | −0.005 (−0.07)*** | −0.240 (−0.37)*** | −0.032 (−0.11)*** |
| roll_std | **−0.023** (−0.63)*** | −0.058 (−0.51)*** | +0.002 (0.04)*** |

**Stage 結論（load-bearing）。** entropy/top1/margin 的 CoT−No-CoT 效應**只在 pre_commit 可穩定判讀**（d_z=0.66–0.83）；進入 post_commit 後掉到 ns，loop_tail 只剩 |d_z|≈0.10 的殘量。主要原因是**測量飽和**：degenerate `#### N #### N` loop 內 top1≈0.98、entropy≈0.12，已無足夠 dynamic range。post/loop 的星號多半只反映 saturation 附近的細微變動，**不是可解讀的 confidence 效應**（info_gain post_commit 的負值來自 loop 內近零 surprise；roll_std 則反映趨近飽和天花板時的方差收縮）。因此本節確認的是 **pre-commit output-decisiveness effect**。同一 stage boundary 下，wanting 在 commit 後仍可測，而 confidence proxy 已失去分辨力；這是兩套讀數的 **measurement contrast**，但不能單獨作為 construct dissociation 的證據。


#### Integrated Interpretation

1. **Task entry:** CoT 在 generation boundary 前已提高 RSN gain（`G_prefill d_z`=0.59；`Z_prefill d_z`=0.89）。
2. **Answer formation:** CoT 在 pre-commit 維持較高 slow RSN level，且 fast residual dispersion 較強。這支持 sustained process engagement；effect 以 level 為主，但仍有較弱的 shape difference。
3. **Commitment and release:** C1 後兩組共同進入 slow-state decline；CoT 的 post-commit release 較快。C1/C2 的 fast transition 尚不能排除 `####` / answer-format effect，因此只稱 commitment-centered residual transition。此階段的 confidence proxy 已接近格式循環造成的飽和，不能用來判斷 release 是否伴隨 confidence change。
4. **Stopping failure:** loop tail 會反轉 full-decode trajectory；CoT tail 較短且 residual variability 較低，但這是 stopping diagnostic，不作正常 reasoning 或 dopamine level 的主結論。
5. **Wanting vs confidence:** CoT 在 pre-commit 同時提高 RSN engagement 與 output decisiveness，證明兩軸會被同一 prompt manipulation 共同調制；但兩者使用不同表徵基座，且 post-commit confidence proxy 飽和，因此本節既不證明兩者是同一構念，也不構成 dissociation 或直接因果證據。較乾淨的 dissociation 來自 §4.3 persona comparison；causal dissociation 仍需 α intervention 下的同軸檢驗。

整體而言，CoT 的最強 RSN 證據是 **task-entry gain + sustained pre-commit engagement + post-commit release**；confidence control 則只支持 **pre-commit output decisiveness 提高**，commit 後不作實質解讀。Fast residual 與 process salience / cognitive updating / commitment-related phasic-like dynamics 相容，但仍缺 event specificity 與 random-mask controls；Dopamine-specific 的主要因果證據仍需來自 α intervention、dose-response 與行為學實驗。

#### Figures and Analysis Files

| Figure | Role |
|---|---|
| `fig42_5panel_s_t.png` | C1/C2-centered 與三階段 lifecycle 的 slow RSN 主圖 |
| `fig42_5panel_p_t.png` | 同一時間框架下的 fast residual；主看 dispersion 與 marker transition |
| `fig42_5panel_entropy.png` / `fig42_5panel_top1.png` | 獨立 confidence-proxy controls |
| `fig42_B_lifecycle.png` | C2 / full loop contamination 診斷 |
| `fig42_step2_shape_test.png` | pre-commit level-vs-shape robustness |

分析腳本：`analyze_cot_step2_tonic.py`、`analyze_cot_step3_slow.py`、`analyze_cot_step4_confidence.py`、`analyze_cot_5panel.py`、`analyze_cot_figB_lifecycle.py`。

### 4.3 Persona

只比較 **Expert vs Non-Expert**（α=0、No-CoT、同 300 GSM8K questions、index-paired），contrast = Expert − Non-Expert（正值表示 Expert 較高）。沿用 §4.2 的 task-entry、三階段 RSN、confidence control 與 commit-centered 流程。Persona 是 **prompt manipulation**，不是 α intervention；本節描述自然 state modulation，不提供 steering 的因果證據。分析腳本為 `analyze_persona.py`，主圖為 `fig43_persona_main.png`，C2-centered 與 full lifecycle 僅作 Supplement。

**核心結果：Persona 不是單一的高低位移，而是 temporal redistribution。** Expert 在 task entry 具有極高 RSN gain；進入 decode 後，此排序在 commitment formation 附近反轉為 Non-Expert `s_t` 較高，並在 commit 後表現為更快的 state release。這與 CoT 的 sustained pre-commit elevation 是不同的調制模式。

#### Task-Entry Gain and Same-Time Confidence

Paired Δ(Expert−Non-Expert)：

| Readout @ prefill | Δ | `d_z` | Interpretation |
|---|---:|---:|---|
| `G_prefill` | +0.161 | **+2.81** | Expert task-entry gain 大幅升高 |
| `Z_prefill` | +0.284 | **+3.17** | layer-standardized 結果一致 |
| `boundary_jump_Z` | −0.265 | −0.54 | decode 起點差距縮小；與 `Z_prefill` 代數耦合，不作獨立機制證據 |
| entropy | +0.052 | +0.18 | Expert 略高 entropy |
| top1 | −0.010 | −0.15 | 弱反向 |
| margin | −0.010 | −0.12 | 弱反向 |

同一 prefill token 上，RSN gain 的 effect size 約 `d_z=3`，output-distribution confidence proxy 則僅有 `|d_z|≤0.18` 的弱反向，支持 **task-entry RSN gain 不可化約為 output decisiveness**。但 NMD mask 本身來自 MMLU Expert−Non-Expert contrast，因此此結果主要是**跨任務 manipulation check**，不是完全獨立的發現。

#### Commit-Centered Paired Analysis

現有 C1 locator（首個 `####`；缺失時使用 answer-candidate fallback）下，Expert 有 219/300、Non-Expert 有 236/300 個 C1-analyzable responses；共同有效 194 題，Expert-only 25、Non-Expert-only 42、neither 39。Non-Expert 的 analyzable rate 高 5.7 percentage points，paired exact `p=0.0498`，屬邊界顯著的行為診斷。

為排除原始 C1 圖兩組 valid-sample composition 不同，以下只在 common-valid questions 上，以每組自己的 C1 為中心計算完整 paired windows：

| Window | Readout | Expert | Non-Expert | ΔExp−Non | `d_z` | p | Reading |
|---|---|---:|---:|---:|---:|---:|---|
| `[−50,−10]` | `s_t mean` | 0.114 | 0.241 | −0.126 | −0.247 | .023 | 較早的 pre-commit reversal，探索性 |
| `[−20,0]` | `s_t mean` | 0.137 | 0.278 | **−0.141** | **−0.278** | **.0013** | Non-Expert commitment-formation state 較高 |
| `[0,+20]` | `s_t slope` | −0.046 | −0.056 | **+0.009** | **+0.291** | **.0004** | Non-Expert post-commit release 更快 |
| `[0,+20]` | `p_t abs_mean` | 1.039 | 1.130 | **−0.091** | **−0.248** | **.0016** | Non-Expert release transient 較強 |
| `[−20,0]` | entropy | 0.482 | 0.439 | +0.043 | +0.167 | .010 | Non-Expert 較 decisive，但效應較弱 |

在 3 windows × 7 readouts 的 21 項 exploratory comparisons 中，經 BH-FDR 後最穩定的是三項：`[−20,0] s_t mean`、`[0,+20] s_t slope` 與 `[0,+20] p_t abs_mean`；較早的 `[−50,−10] s_t` 與 entropy effect 只作輔助。這證明 reversal 不是由 Expert n=219 / Non-Expert n=236 的樣本組成差異製造。

全 pre-commit 平均仍接近 null（`s_t mean d_z=−0.10, ns`），並不與上述結果矛盾：Persona effect **不是貫穿整段的固定 offset**，而是集中在 commitment formation 與 release，整段平均會將局部差異稀釋。Pre-commit fast-residual dispersion 只有小效應（`abs_mean d_z=−0.15`；`std d_z=−0.17`）；post-commit / loop-tail 的全階段均值大致為 null。

#### Event and Confidence Controls

全 pre-commit confidence control 顯示 Expert entropy 較高（`d_z=+0.35`）、top1 / margin 較低（`d_z=−0.27 / −0.22`），方向上表示 Non-Expert 較 decisive；post-commit 與 loop-tail 受 `#### N #### N` saturation 污染，只作診斷。

Random interior pseudo-event 不出現 C1 的 sharp `s_t` transition，說明變化是 event-localized；但 C2-centered 圖幾乎複製 C1，且兩者都伴隨 `####` / answer-format transition，因此目前只能稱為 **answer-marker-centered dynamics**，不能直接命名為 biological commitment signal。

#### Integrated Interpretation

Persona 呈現三段式結構：

1. **Task entry:** Expert 具有極高 RSN gain，但沒有相應的 output decisiveness 增益。
2. **Commitment formation:** 在 common-valid paired subset 中，Non-Expert 的 slow RSN state 反超。
3. **State release:** Non-Expert 在 C1 後下降更快，fast-residual amplitude 也更大。

這提供了 Non-Expert accuracy 高於 Expert（68% vs 58%）的一個候選機制：**較低的入口 gain、較高的 commitment-formation engagement，以及較快的 post-commit release**。但本節仍未證明這三個 readouts 中介 accuracy；確認中介路徑需要分析 Expert-Wrong → Non-Expert-Correct 的 discordant items，或進行 gain-matched `role × α` rescue/cancellation experiment。

因此 §4.3 的主要貢獻是揭示 **Persona 與 CoT 具有不同的 temporal and representational modulation profiles**。CoT 在 pre-commit 同時提高 slow RSN engagement 與 output-distribution decisiveness，屬於較明顯的 **joint wanting–confidence modulation**；Persona 的主要效應則集中於 RSN/wanting 軸的時間重分配——從 task-entry gain、commitment-formation reversal 到 post-commit release。Persona 並非完全不影響 confidence：Non-Expert 在 pre-commit 略為更 decisive，但其效應明顯弱於 RSN gain，且較為局部。因此較準確的結論是：**CoT 同時調制 wanting 與 confidence，而 Persona 主要重組 wanting dynamics，並伴隨較弱的 confidence change。** 這支持 RSN 作為不同於 output confidence 的 dynamic state/gain readout，但 dopamine-specific 的因果證據仍主要來自 §4.4 α dose-response 與行為學結果。

### 4.4 α-Steering: Linear Task-Entry Gain and a Nonlinear Behavioral Working Point

本節分析 Llama3-8B 在 GSM8K、neutral No-CoT 條件下的 9 個 α doses（−8 至 +8）。α 是施加在既定 NMD/RSN 方向上的 intervention；本節首先將其視為 **task-entry RSN gain manipulation**，再檢驗其是否形成與 wanting / commitment 相容的下游動力學。

分析沿用 §4.2/§4.3 的 gain coordinates 與 commit locator：reference μ/σ 固定為 neutral α=0 No-CoT prefill；dose calibration 使用各 α 全量 300 題；slow/fast/confidence 則以每個 α 與 α=0 的 common-valid questions 進行 paired comparison，避免 −8 壓縮全 9 檔共同交集。Commit 定義為首個 `####`，缺失時使用 first answer-candidate fallback。

本節 signal–behavior alignment 使用同批 server-184 inline `correct`；production accuracy 仍以 [AdaDopamine_gsm8k.md](AdaDopamine_gsm8k.md) 的 server-182 offline first-`####` 口徑為準。兩套數值不可混算，但 dose 形狀一致，離散最佳點均在 α=−6。

#### Task-Entry Intervention Validity

| α | n | acc(184) | G_prefill | Z_prefill | boundary_jump_G |
|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 40.7 | −13.8454 | −26.4543 | +13.0577 |
| −6 | 300 | **79.7** | −10.2980 | −19.7427 | +10.5488 |
| −4 | 300 | 74.3 | −6.8022 | −13.1072 | +6.9866 |
| −2 | 300 | 68.3 | −3.3519 | −6.4887 | +3.4912 |
| 0 | 300 | 60.0 | 0.0000 | 0.0000 | +0.1675 |
| +2 | 300 | 55.3 | +3.2393 | +6.3403 | −3.0544 |
| +4 | 300 | 51.3 | +6.3835 | +12.5583 | −6.2388 |
| +6 | 300 | 51.7 | +9.4779 | +18.7204 | −9.4322 |
| +8 | 300 | 49.7 | +12.5354 | +24.8320 | −12.4419 |

`G_prefill` 隨 α 近乎完美線性（slope=1.648、intercept=−0.296、`R²=0.9992`），`Z_prefill` 結果一致（`R²=0.9997`）。正負方向近似對稱，但負側幅度在高 dose 略大。這建立了清楚的 manipulation check：**α 線性控制 generation boundary 上的 RSN gain**。

`boundary_jump_G` 與 `G_prefill` 反向且量級接近，表示一次性 prefill 注入到 `decode[0]` 時大幅弛豫。這與 **initial-condition / boundary-gating** 假說一致：α 改變 generation 的起始條件，而不是以固定 additive offset 留在整條 decode trajectory。此結果定位了非線性轉換發生在 generation boundary 之後，但尚不能證明 boundary rebound 本身造成後續行為差異。

#### Commitment-Formation RSN Dynamics

α 對 slow state 的主要影響集中在 pre-commit。下表為每個 α 相對 α=0 的 paired comparison：

| α | paired n | mean `s_t` | Δ vs 0 | `d_z` | exploratory p |
|---:|---:|---:|---:|---:|---|
| −8 | 229 | −0.573 | −0.325 | −0.418 | *** (崩) |
| **−6** | 291 | +0.260 | **+0.449** | **+0.584** | *** |
| −4 | 298 | −0.033 | +0.154 | +0.228 | *** |
| −2 | 298 | −0.122 | +0.065 | +0.143 | ns |
| 0 | 298 | −0.186 | 0 | 0 | — |
| +2 | 297 | −0.248 | −0.062 | −0.123 | ns |
| +4 | 298 | −0.308 | −0.121 | −0.191 | ** |
| +6 | 298 | −0.415 | −0.229 | −0.323 | *** |
| +8 | 298 | −0.347 | −0.161 | −0.221 | *** |

這不是平滑、對稱的標準 inverted-U，而是 **asymmetric peaked working-point response**：−6 形成明顯峰值，−4 至正 α 大致隨 α 增加而下降，−8 則進入另一種崩潰區。C1-centered trajectories 顯示此分離主要發生在答案形成期；越過 C1 後，各 α 共同下降並逐步收斂。Post-commit release 除 −8 外均接近 null，因此 α 並未普遍改寫全程 RSN 水位，而是選擇性地重組 commitment formation。

Pre-commit `end_minus_start` 顯示正 α 通常有較大的向下 relaxation（例如 +4 `d_z=−0.18`），但該 readout 同時受起點水平影響，只作 slow-state shape 的輔助證據。

#### Fast Residual Dynamics

**pre p_t std**:

| α | n | Δ vs 0 | d_z | p |
|---:|---:|---:|---:|---|
| −8 | 229 | −0.010 | −0.059 | ns |
| **−6** | 291 | **+0.066** | **+0.452** | *** |
| −4 | 298 | +0.026 | +0.176 | ** |
| −2…+8 | | ~0 | \|d_z\|<0.09 | ns |

Fast residual 提供一致但較窄的輔助證據：−6 的 pre-commit `p_t std` 明顯升高，−4 只有小效應，其餘 doses 接近 null。最佳工作點因而同時伴隨較高 slow-state level 與較強 fast-residual dispersion。

與 §4.2 相同，fast component 的主讀數限於 `abs_mean/std`，不使用易受長度與 EMA lag 影響的極值。`p_t` 仍是 EMA high-pass residual，C1 附近的共同轉折也可能包含 `####`/answer-marker effect，不能直接命名為 phasic dopamine。

#### Output-Distribution Confidence Controls

| Metric | α=−6 Δ vs 0 | `d_z` | α=+4 Δ vs 0 | `d_z` | Pattern |
|---|---:|---:|---:|---:|---|
| entropy | −0.174 | **−0.718** | +0.063 | +0.304 | −6 最確定 |
| top1 | +0.039 | **+0.643** | −0.012 | −0.248 | −6 達峰 |
| margin | +0.048 | **+0.593** | −0.014 | −0.214 | −6 達峰 |

α 的下游效應並非 selective wanting modulation。Pre-commit confidence 在 −6 同樣達到最佳工作點，且 effect size 與 slow RSN state 相當（−6：`s_t d_z=0.58`、top1 `d_z=0.64`、entropy `d_z=−0.72`）。正 α 則表現為 entropy 較高、top1/margin 較低。

因此 α 雖然直接施加在 RSN/NMD 方向上，進入 decode 後卻共同重組 **RSN engagement 與 output decisiveness**。這與 §4.2 CoT 的 joint wanting–confidence modulation 相似，不支持「α 只改 wanting、不動 confidence」的強版本。§4.3 Persona 的 task-entry RSN effect 遠大於同時點 confidence effect，仍是目前較清楚的 representational separation；但本研究尚未取得 wanting–confidence 的 causal dissociation。Post-commit confidence 受 answer-loop saturation 影響，仍不作實質解讀。

#### Signal–Behavior Alignment

| α | acc(184) | G_prefill | early_s_t |
|---:|---:|---:|---:|
| −8 | 40.7 | −13.85 | −0.405 |
| **−6** | **79.7** | −10.30 | +0.055 |
| −4 | 74.3 | −6.80 | −0.329 |
| −2 | 68.3 | −3.35 | −0.404 |
| 0 | 60.0 | 0.00 | −0.421 |
| +2 | 55.3 | +3.24 | −0.433 |
| +4 | 51.3 | +6.38 | −0.460 |
| +6 | 51.7 | +9.48 | −0.550 |
| +8 | 49.7 | +12.54 | −0.558 |

Accuracy 的 quadratic fit 優於 linear fit（`R²=0.352` vs `0.147`），但擬合峰值約 α=−1.9，與離散最高點 −6 不一致，說明資料不適合被描述成平滑、對稱的標準 inverted-U。更準確的形狀是：**−6 尖銳最佳點、−8 低端崩潰、正 α 端逐步下降後趨平**。

在 9 個 dose-level aggregates 上，inline accuracy 與 `G_prefill` 的相關較弱（`r=−0.37`），與 `early_s_t` 的相關較高（`r=+0.74`）。因此行為表現更接近 decode 期間形成的 commitment state，而不是 task-entry gain 的絕對大小。不過 `r=0.74` 只有 9 個 aggregate points，只能稱為 dose-level covariation，不能作為 `s_t` 中介 accuracy 的證據。

#### Integrated Interpretation

§4.4 最重要的結果是找到了**線性干預如何轉化為非線性內部與行為狀態**：

```text
α
→ linear task-entry RSN gain (G_prefill, R²=0.999)
→ nonlinear commitment-formation state (pre-commit s_t)
→ joint change in output decisiveness
→ asymmetric behavioral working point
```

三個工作區可暫時區分為：

1. **Adaptive range（−4 至 +2）：** α 對 commitment-formation state 進行較平滑的校準。
2. **Extreme negative（−8）：** 進入 commitment-formation collapse；event-centered 分析只保留仍可定位 C1 的條件子樣本，必須與 C1-analyzable rate 及文本中的 answer-candidate oscillation 共同解讀。
3. **High positive（+6/+8）：** pre-commit slow state 與 confidence 均較差；其 premature-commitment 解讀來自獨立的 commit-position / behavioral metrics，不能只由本節 trajectory 推斷。

整體而言，α 是可靠的 **task-entry RSN gain intervention**，但其有效作用不是「gain 越高越好」，而是將模型推入不同的 commitment-formation working point。這與 task-dependent dopamine/wanting calibration 及 Yerkes–Dodson framing 相容，但目前建立的是 computational and behavioral analogy，而非生物多巴胺機制的直接證明。

#### Figures and Analysis Files

| Figure | Role |
|---|---|
| `fig44_dose_main.png` | C1-centered `s_t` / `p_t` / entropy 與 dose-level working point 主圖 |
| `fig44_validity.png` | α 對 `G_prefill` / `Z_prefill` 的線性 manipulation check |
| `fig44_slow.png` | pre/post-commit slow-state paired comparisons |
| `fig44_fast.png` | fast-residual dispersion controls |
| `fig44_confidence.png` | entropy / top1 / margin confidence controls |
| `fig44_integrated.png` | inline accuracy、task-entry gain 與 early `s_t` 的 dose-level 對齊 |

分析腳本：`analyze_alpha_dose.py`（`--part validity/slow/fast/confidence/integrated/mainfig`）；raw stdout：`fig44_results.txt`。目前 slow/fast/confidence dose figures 顯示 absolute means，統計採 paired α-vs-0 comparisons；confirmatory reporting 前需統一改為 paired Δ/CI，並對 dose × metric comparisons 做 FDR correction。`fig44_dose_main.png` 的 trajectory panels 只顯示部分代表 doses；圖註應標示「selected α cells shown; all 9 doses used in dose-response panels」。

### 4.5 CoT × α=−4: Signal Interaction Analysis

**目標：** 檢驗 `α=−4` 如何調制 CoT 的內部信號動力學。這是一個 **2×2 單劑量 factorial**（`{No-CoT, CoT} × {α=0, α=−4}`），只能判斷 approximately additive / attenuation / interaction，**不能**推斷 CoT 條件下的最佳 α 或 dose-response 曲線（例如 CoT 是否移動 §4.4 的倒 U 峰位——需補採 CoT × dose signal 才能回答）。行為結果直接引用 `AdaDopamine_gsm8k.md` §2.5/§2.5.1，本節不重複展開。

**Scope 與口徑。** 四組同 300 道 GSM8K、index-paired（已驗證 common=300）；reference μ/σ 固定 = neutral α=0 No-CoT prefill（同 §4.2–4.4）。因四組全 paired，交互效應（difference-in-differences）以 **per-question** 計算再 Wilcoxon 檢驗，而非僅比四個 cell 平均：

```text
DiD_q = (cot_-4 − cot_0) − (nocot_-4 − nocot_0)   [每題 q]
```

inline acc（184）：nocot_0=60.0 / nocot_−4=74.3 / cot_0=67.7 / cot_−4=82.7，與 §2.5.1 行為（182：60/73/69/85）同向、量級一致。分析腳本 `analyze_cot_alpha.py`，主圖 `fig45_slow_centered.png` / `fig45_fast_centered.png`（四 cell C1-centered 疊圖）。

**核心結論（時間重心結構）。** 兩個 manipulation 的影響落在**不同的時間重心**上，而非簡單「正交可疊加」：

```text
Task entry   :  α effect  ≫  CoT effect
Decode / commitment formation : CoT effect ≫ α effect
Confidence output :  CoT 與 α=−4 大體呈 approximately additive
```

即 **α 直接控制 generation-boundary gain；CoT 主要重塑展開後的 reasoning dynamics；兩者最終共同提高 output decisiveness。** 統計上：**decode-stage 的 DiD 均不顯著；task-entry `G_prefill/Z_prefill` 存在統計顯著但量級極小的偏離（約佔 α 主效應 0.4%），實質意義上仍接近 additive。** 因此本節的可靠結論建立在**顯著的主效應（主導權翻轉）**之上，而非任何 DiD 判定；「approximately additive」不等於證明兩種機制獨立，單一 α=−4 亦不能證明整個干預機制皆可獨立疊加。

#### Step 1 — Task-Entry Gain：approximately additive，α 主導

| readout | nocot_0 | nocot_−4 | cot_0 | cot_−4 | α效應\|No-CoT | α效應\|CoT | DiD | DiD 顯著性 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `G_prefill` | 0.000 | −6.802 | 0.071 | −6.703 | −6.80 | −6.78 | +0.028 | *** but ≈0.4% of α |
| `Z_prefill` | 0.000 | −13.107 | 0.158 | −12.903 | −13.11 | −13.06 | +0.046 | *** but ≈0.4% of α |
| `boundary_jump_G` | 0.167 | 6.987 | 0.094 | 6.988 | +6.82 | +6.89 | +0.074 | ns |

α 效應幾乎不受 CoT 影響（G_prefill −6.80 vs −6.78），與 §4.4 co-design identity（`G_prefill(α) ≈ G_prefill(0) + α·‖mask‖²`，α 在 prefill 加一個與 CoT 無關的固定量）一致。**入口是 α 主導（|Δ|≈6.8）、CoT 次要（|Δ|≈0.07–0.16）。** G/Z 的 DiD 帶星號是 **statistical interaction**（n=300 + 極小配對方差可偵測到微小系統偏離），但其 **practical** 量級僅約 α 主效應的 0.4%——因此判為 **approximately additive**，不寫成 ns（effect 小 ≠ 不顯著），也不寫成「純 additive」。

#### Step 2 — Commitment-Centered Slow State：CoT 主導，early window α 效應減弱（attenuation trend）

以 C1 為中心，三段窗口的 `s_t mean`：

| Window | nocot_0 | nocot_−4 | cot_0 | cot_−4 | CoT效應(主) | α效應\|No-CoT | α效應\|CoT | DiD |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[-50,-20]` early (n=142) | −0.063 | 0.097 | 0.465 | 0.506 | +0.53*** | +0.159** | +0.040 ns | attenuation trend (DiD p=.14, ns) |
| `[-20,0]` commit (n=170) | −0.033 | 0.120 | 0.478 | 0.591 | +0.51*** | +0.153** | +0.113* | approx. additive |
| `[0,+20]` release (n=170) | −0.321 | −0.207 | −0.041 | 0.083 | +0.28*** | +0.114*** | +0.124** | approx. additive |

- **decode 內 CoT 絕對主導**（d_z 0.55–0.86），α=−4 次要（d_z 0.11–0.30），量級差 3–5 倍——與 Step 1 入口（α 主導）**正好相反**，構成上述時間重心翻轉。
- **early window 的 α 效應在 CoT 下由顯著（+0.159\*\*）降為 ns（+0.040）**，descriptive 上像 CoT 已把 early `s_t` 拉到 0.47、α 無額外空間的 saturation；但 **early 的 DiD 本身不顯著（p=.14）**，而「一顯著 + 一 ns」不等於兩者顯著不同——因此只記為 **attenuation trend**，redundancy/saturation 保留為**候選解釋**，不正式判定。commit / release 兩窗 α 效應維持（approximately additive）。
- release `s_t slope` 上 α 無穩定效應（No-CoT / CoT 皆 ns），CoT 則使 release 稍陡（d_z≈−0.5\*\*\*）。
- **cot_−4 在三窗 `s_t` 皆為四組最高**（0.506 / 0.591 / +0.083），且伴隨四組最高 acc（82.7）。**此處只作 co-occurrence 陳述**——§4.4 已證 signal 高不必然行為好（存在 nonlinear working point），故不將 cot_−4 的高 `s_t` 解讀為「解釋」或「複現」其高 acc，亦不暗示中介關係。

#### Step 3 — Fast Residual Dynamics：CoT 主導，DiD 全 ns

`p_t abs_mean` / `std`：主效應仍是 CoT（early abs_mean +0.092\*\*\*、std +0.143\*\*\*；release abs_mean +0.178\*\*\*），α=−4 邊際弱，early window 呈與 Step 2 同向的 attenuation trend（α 效應在 CoT 下轉 ns），commit / release **DiD 全部 ns**。`p_t` 僅作 fast residual dynamics 判讀，不命名為 phasic dopamine。

#### Step 4 — Confidence Controls：near-additive

pre-commit `[-20,0]`（與 wanting 分表；post/loop 飽和不讀）：

| Metric | nocot_0 | nocot_−4 | cot_0 | cot_−4 | α效應\|No-CoT | α效應\|CoT | DiD |
|---|---:|---:|---:|---:|---:|---:|---:|
| entropy | 0.553 | 0.466 | 0.339 | 0.253 | −0.087*** | −0.086*** | +0.001 (ns) |
| top1 | 0.840 | 0.863 | 0.894 | 0.916 | +0.024*** | +0.022** | −0.002 (ns) |
| margin | 0.753 | 0.785 | 0.830 | 0.859 | +0.032** | +0.029** | −0.003 (ns) |

α 效應在 CoT / No-CoT 下幾乎相同（entropy −0.087 vs −0.086），DiD Δ≈0 全 ns —— **near-additive**。兩者都朝更 decisive 方向疊加，cot_−4 在這四格中最確定（entropy 0.253 最低、top1 0.916 最高）。即 α=−4 **同時**改變 CoT 的 RSN state 與 output decisiveness，兩軸皆 near-additive；與 §4.4 結論一致（α 非 selective wanting intervention）。

#### Step 5 — Integrated Signal Interpretation

```text
CoT process  ×  α=−4 task-entry intervention
  → task-entry gain :  approximately additive（入口 α 主導；co-design identity）
  → slow s_t        :  CoT 主導；early α-attenuation trend（候選 redundancy）→ commit/release approx. additive
  → fast p_t        :  CoT 主導；α 弱，DiD 全 ns
  → output confidence: near-additive（α 效應不隨 CoT 變），共朝更 decisive
```

> **CoT and α=−4 are approximately additive at task entry and confidence output, while decode-stage dynamics are dominated by CoT. No reliable decode-stage interaction is detected; the early slow-state attenuation under CoT remains a descriptive redundancy candidate.**

本節真正有價值的不是「正交雙槓桿」，而是**兩個 manipulation 的時間重心不同**：α 直接控制 generation-boundary gain（入口 α≫CoT）；CoT 主要重塑展開後的 commitment-formation dynamics（decode CoT≫α）；兩者最終在 output decisiveness 上共同抬升（approximately additive）。cot_−4 在 `s_t` 與 confidence 上均為四格最高，且與最高 acc co-occur。

**限定。** 單 α 劑量，**不能**推斷 CoT 下的 dose-response 或最佳 α。所有 DiD 顯著性受 n=300 + 極小配對方差影響（入口 \*\*\* 但量級可忽略）：statistical 與 practical interaction 須分開讀，判定以量級比為準。「無顯著交互」不等於證明兩機制獨立。這是 computational/behavioral analogy，非生物多巴胺機制的直接證明。

#### Figures and Analysis Files

| Figure | Role |
|---|---|
| `fig45_slow_centered.png` | 四 cell C1-centered `s_t` 疊圖（主圖：CoT 抬高整條、α 次要調制） |
| `fig45_fast_centered.png` | 四 cell C1-centered `p_t` 疊圖 |

分析腳本：`analyze_cot_alpha.py`（`--part entry/slow/fast/confidence/all`）；raw stdout：`fig45_results.txt`；結果記錄：`fig45_SUMMARY.md`。DiD 的 verdict 標籤（additive/redundant/…）僅為量級比啟發式，非顯著性判定，不可單獨當結論。
