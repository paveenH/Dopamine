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

#### 3.3.3 Phasic

decode-time fast component 定義為相對上一時刻 slow baseline 的 residual：

```text
p_t = Z_t - s_{t-1},  t ≥ 1
```

`p_t > 0` 表示瞬時高於 slow baseline 的 pulse，`p_t < 0` 表示瞬時 dip。現階段先把它視為 decode 中的 phasic signal，優先檢驗其 amplitude、variability 與時間結構；之後再分析它是否與特定 reasoning / commitment event 對齊，不預先指定 event anchor。

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

Correctness 不是主要 intervention axis；本節將其作為 **outcome analysis**，檢驗 RSN dynamics 與 task performance 的關係。分析使用 neutral No-CoT（180 correct / 120 incorrect）與 CoT（203 / 97）；commit 定義為首個 `####`，若不存在則使用首個 answer candidate。

#### Commit Timing

在具有可辨識 commit marker 的 responses 中，correct responses 並未更早提交答案。其 median commit step 顯著晚於 incorrect responses（No-CoT: 180 / 117；CoT: 203 / 94）：

| Condition | Correct | Incorrect | MWU p |
|---|---:|---:|---:|
| No-CoT | 101 | 56 | 0.006 |
| CoT | 203 | 96 | 0.00014 |

因此，length-normalized trajectory 中的 correct / incorrect 差異不能簡單歸因於「correct 更早完成」。相反地，在可辨識 commit 的 samples 中，incorrect responses 較早形成 answer candidate。

#### Commit-Aligned Dynamics

將各 sample 對齊至自己的 commit step（`t=0`）後，可觀察到三個主要模式：

1. **Pre-commit slow RSN state**：Commit-aligned slow RSN dynamics：correct 組在 commit 前維持較高`s_t`；commit 後則下降得更快，約在 10–15 tokens 後低於 incorrect 組。前者表示組間差異不只是 commit timing 錯位，後者則提示 correct responses 在答案形成後具有更快的 state release / termination dynamics。
2. **Commit-centered transition**：`Z_t`、`s_t`、`p_t` 與 entropy / top1 / margin 均在 commit 附近快速變化。這支持 commit marker 對應一個明顯的 generation-state transition。
3. **Post-commit separation**：correct 組呈現較強的 `s_t` decline 與較完整的 confidence recovery；incorrect 組下降較慢、恢復較弱，可能反映不同的 termination dynamics。

| Signal family | Main observation | Interpretation |
|---|---|---|
| `s_t` / `Z_t` | correct 在 commit 前較高，commit 後下降較深 | slow RSN dynamics 與 response outcome 相關 |
| `p_t` | commit window 有明顯 pulse / dip，但兩組在 commit 前無穩定分離 | 對局部 state transition 敏感，不是穩定 correctness predictor |
| entropy / top1 / margin | commit 附近共同出現 uncertainty increase / confidence decrease | commit 是 output-distribution transition |
| info gain | commit 附近波動，但組間結構不穩定 | 僅作 distributional-change diagnostic |

#### Figures

| Figure | Alignment / signals | Main finding |
|---|---|---|
| `fig41_suite_rsn.png` | Length-normalized `Z_t / s_t / p_t` | correct 組 early slow RSN state 較高，之後下降更快；`p_t` 無穩定組間分離 |
| `fig41_suite_logit_nocot.png` | Length-normalized No-CoT logit metrics | incorrect 組 entropy / confidence volatility 略高、top1 / margin 略低；info gain 無穩定差異 |
| `fig41_suite_logit_cot.png` | Length-normalized CoT logit metrics | 大致重現 No-CoT pattern；confidence 差異存在但弱於 slow RSN 差異 |
| `fig41_commit_aligned_suite_nocot.png` | Commit-aligned No-CoT RSN + logit suite | commit 附近出現共同 state transition；`p_t` 有 pulse / dip，但 confidence 對 correctness 的分離較弱 |
| `fig41_commit_aligned_suite_cot.png` | Commit-aligned CoT RSN + logit suite | 重現 commit-centered transition 與 correct 組較快的 post-commit release |

**Overall:** correctness 主要與 commit 前後的 **slow RSN trajectory** 相關；task-entry gain、單一 phasic amplitude 與 confidence metrics 均不是穩定的 correctness predictor。六張圖來自同一批資料的兩種對齊方式，應視為互補分析，而非六份獨立證據。

**Boundary:** commit-centered logit change 可能部分來自 `####` / answer-format transition；slow RSN difference 仍需 difficulty-matched analysis 驗證。

### 4.2 CoT vs No-CoT

#### Scope

只分析 **neutral、α=0、相同 300 道 GSM8K**。两组除 CoT 增加 `Let's think step by step.` 外，其余生成条件一致。
> 本节重点是：CoT 如何改变 task-entry state 与 generation dynamics？

#### Step 1: Task-Entry Tonic-like Gain

| Readout | Question |
|---|---|
| `G_prefill` | CoT 是否改变 task-entry gain？ |
| `Z_prefill` | layer-fair 坐标下是否复现？ |
| `boundary_jump_G` | CoT 是否改变 prefill 到 decode[0] 的状态转换？ |

统计采用 paired mean difference、bootstrap 95% CI、Cohen’s `d_z` 和 Wilcoxon signed-rank（Step 1 已確認兩組為同一批 300 題、索引對齊，故用 paired）。

> 注意：`G_prefill` 讀的是 generation boundary 的**最後一個 prompt token**，不是整個任務期間的恆定 tonic state；故此小節測的是 task-entry tonic-**like** gain，而非嚴格意義的 tonic baseline。

**結果：CoT 在 generation boundary 就抬高了 RSN state**

| Readout | No-CoT | CoT | ΔCoT−No | 95% CI | `d_z` | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| `G_prefill` | 0.000 | 0.071 | **+0.071** | [0.058, 0.085] | **0.59** | 1.9e-19 |
| `Z_prefill` | 0.000 | 0.158 | **+0.158** | [0.138, 0.178] | **0.89** | 1.1e-32 |
| `boundary_jump_G` | 0.167 | 0.094 | **−0.073** | [−0.138, −0.006] | −0.13 | 2.4e-02 |

1. **`G_prefill`：CoT 顯著抬高 task-entry gain。** +0.071 α 單位，`d_z`=0.59（中等偏強），CI 不含 0。No-CoT 的 0.000 是因為它本身就是參考基準（`μ_l^ref` 定義在 neutral-α0-No-CoT prefill），故此列讀的是「CoT 相對 No-CoT baseline 的偏移」。意義：光是 prompt 多一句 `Let's think step by step.`，在**尚未 decode**、last token 同為 `Answer:` 的情況下，就已抬高 wanting——純 context 效應（前文 CoT instruction 改變了 last-token 的 residual 狀態）。

2. **`Z_prefill`：layer-fair 坐標下複現，且對逐層標準化穩健。** +0.158，`d_z`=**0.89**（強）。`d_z` 比 `G_prefill` 大，說明這個抬升**經 layer-wise standardization 後依然穩健，不是單靠少數 large-scale 層撐起來的**（Z-score 會放大低方差層的貢獻，故 `d_z` 增大只能推斷「非由大尺度層獨佔」，**不能**直接讀成「各層更均勻」——後者須另查 per-layer effect）。

3. **`boundary_jump_G`：CoT 反而縮小 prefill→decode[0] 的跳變。** No-CoT 跳 +0.167，CoT 只跳 +0.094，Δ=−0.073，但 `d_z`=−0.13（弱）。數據上兩組 decode[0] 的絕對高度高度接近（No-CoT `G_decode[0]`≈0.167；CoT ≈0.071+0.094=0.165），CoT 的差異**提前出現在 prefill**，故起點抬高後 decode 開場「還要往上跳的空間」變小。注意此處比較的是 α=0 的 CoT/No-CoT，**沒有注入 α**，因此這只是觀察到的邊界轉換差異，不能歸因於 §4.1 的 co-design steering identity。effect 弱，屬記錄性。

> **小結：CoT-related RSN elevation 在 generation boundary 就已可偵測。** CoT 抬高 `G_prefill` / `Z_prefill`，而兩組 `G_decode[0]` 到達相近水平，於是 CoT 下的 boundary jump 較小。這支持一個 task-entry gain difference，但**不足以斷定 `G_prefill` 單獨決定後續 decode dynamics**。

分析腳本：`analyze_cot_step2_tonic.py`。

#### Step 2: Slow Decode Dynamics

主信号：

```text
s_0 = Z_0
s_t = βs_{t-1} + (1-β)Z_t
```

统计同 Step 1：paired（同 300 題、索引對齊）、bootstrap 95% CI、Cohen's `d_z`、Wilcoxon signed-rank。

**方法決定：三個互斥功能階段。** Step 1 audit 發現 **97% 樣本撞 767 max_new_tokens 截斷，commit 後 70–85% 是 `#### N …` 退化 loop**。與其對「累計區間」再切 early/middle/late（重疊、重複計同一批 token），改把整條 decode 用兩個 marker（第 1、第 2 個 `####`）切成**互不重疊的三個功能階段**，每段有明確行為意義：

- **pre-commit** = decode start → 第 1 個 `####`（**答案形成**，正文主分析）
- **post-commit** = 第 1 個 `####` → 第 2 個 `####`（**提交後延續生成 / state release**）
- **loop tail** = 第 2 個 `####` → generation end（**stopping failure**，僅作診斷）
- *full = 三段之和*（僅示範「不切段會產生什麼污染」，不作機制結論）

> **口徑：`s_t` 只算一次。** slow EMA `s_t = βs_{t-1}+(1-β)Z_t`（`s_0=Z_0`）在**整條 decode 上算一次**，三段從同一條 `s_t` 切片取值——各段 level 因此共用同一基線、可直接互比（不對每段重新 seed EMA）。
>
> **端點來源兩組不對稱。** 用真 `####` 者：No-CoT 243/300、CoT 285/300；退回 answer-candidate 者 No-CoT 57、CoT 15。故 pre-commit 端點定義兩組不完全同構——下附 **strict-`####` paired subset**（僅取兩組都有真 `####` 的題）確認結論不因 fallback 差異而變。統計同 Step 1：paired、bootstrap 95% CI、`d_z`、Wilcoxon（每段只配對兩組都有效的題，故各段 n 不同）。

**主結果：三個互斥階段（ΔCoT−No，`d_z`；`***` p<.001 / `**` p<.01 / `*` p<.05 / ns）**

| Stage | Readout | No-CoT | CoT | ΔCoT−No | `d_z` | 显著 | 行為含義 |
|---|---|---:|---:|---:|---:|:--|---|
| **pre-commit** (n=281) | `mean` | −0.204 | +0.267 | **+0.471** | **0.65** | *** | CoT 顯著抬高推理段整體 wanting level（最穩健效應） |
| | `slope` | −0.001 | +0.000 | +0.002 | 0.14 | ** | 段內斜率差異弱 |
| | `end−start` | −0.258 | −0.030 | **+0.228** | 0.27 | *** | **No-CoT 在推理段內明顯 relax，CoT 幾乎不 relax → 殘餘 shape** |
| | `length` | 250.6 | 279.5 | +28.8 | 0.10 | *** | CoT 推理段稍長 |
| **post-commit** (n=144) | `mean` | −0.187 | −0.014 | +0.173 | 0.17 | * | level 差異縮小 |
| | `end−start` | −0.138 | −0.304 | **−0.166** | −0.31 | *** | **CoT 提交後掉得更多 → 較快 state release** |
| | `length` | 76.2 | 124.2 | +48.0 | 0.17 | ns | |
| **loop tail** (n=141) | `mean` | −1.421 | −1.236 | +0.185 | 0.32 | *** | loop 段 level 低（診斷用，非 wanting 結論） |
| | `length` | 619.0 | 535.4 | **−83.6** | −0.51 | *** | **CoT loop 尾更短 → 較早停止（stopping failure 較輕）** |

**pre-commit shape：兩條獨立證據。** 主表 `end−start`（+0.228 d_z=0.27 ***，末10%−首10%均值）已把「CoT relax 更少」量成一個標量。另有兩個 length-normalization-independent 控制佐證同一結論：

- **strict-`####` subset（n=233）**：pre-commit `mean` +0.255 d_z=0.35 ***、`slope` 弱、`relax`(end−start 同類) 仍近 ns（strict 子集 `relax_mag`=+0.019 ns）——結論不因 fallback 差異而變。
- **absolute-token-step**（不做長度歸一化）：前 50 token，centered `s_50−s_0` No-CoT −0.129 vs CoT +0.054（Δ+0.183 d_z=0.21 p=5e-4）；前 100 token，−0.299 vs −0.018（Δ+0.281 d_z=0.29 p=5e-4）。**「CoT relax 更弱」在真實 token-time 中重現，不只是歸一化坐標的假象。**

**full（不切段）只用來示範污染。** 若把整條 767 當一段，`relax_mag`=−0.49 d_z=−0.59 ***、`slope` 反號（d_z=−0.38 ***）——這全來自 loop tail 嚴重下拉整段軌跡，一旦切到 pre-commit 即消失。**注意**：此污染診斷屬 CoT/No-CoT 軸，不能直接推斷 §4.1 correct/incorrect 的下降也是 loop 假象（另一條比較軸，需各自檢驗）。

> **小結（level-dominant，非纯平移）**：CoT raises the RSN state at the generation boundary and maintains a higher slow decode level throughout answer formation (pre-commit `mean` d_z=0.65). The effect is **level-dominant but not a pure level shift**: within the pre-commit stage, No-CoT relaxes while CoT barely does (`end−start` d_z=0.27), a residual shape that survives both a strict-`####` subset and an absolute-token-step control. After commitment CoT releases *faster* (post-commit `end−start` d_z=−0.31) and its stopping failure is milder (loop tail 短 83 tokens). CoT-related elevation is already detectable at task entry (Step 1 `Z_prefill` d_z=0.89) and **remains present during pre-commit decoding**；由於 CoT instruction 一路存在於 KV cache，不能據此斷言後續差異完全由 `G_prefill` 決定或「非 decode 中動態生成」。

**圖**：`fig42_step3_slow_st.png`（三個互斥階段並排 s_t）、`fig42_step2_shape_test.png`（pre-commit level-vs-shape 裁決：raw + paired diff / baseline-centered）、`fig42_step3_commit_st.png`（commit-aligned level + slope）。分析腳本：`analyze_cot_step3_slow.py`（`report_stages` 出主表，`--plots` 出图；strict / abs-step 控制與 full 診斷隨主程序打印）。

> **圖例 n 與主表 n 不同（口徑差異，非錯誤）。** 主表每格是 **paired** 統計（僅取兩組同一題都在該階段有效的交集），故 pre-commit n=281 / post-commit n=144 / loop tail n=141。`fig42_step3_slow_st.png` 的均值帶用**各組獨立可用的全部軌跡（unpaired）**畫，故圖例 n 較大且兩組不等（pre-commit 298/281、post-commit 184/231、loop tail 219/189）。形態展示用 unpaired（軌跡越多帶越穩），顯著性檢驗用 paired（消除題間變異）——兩者都正確，只是不同用途。

#### Step 3：Fast Phasic Component

比较：

- `phasic_pos_peak`
- `phasic_neg_peak`
- `phasic_abs_mean`
- `phasic_std`
- early / middle / late phasic amplitude

本节暂时不做 commit alignment，也不预设特定 event。目标只是回答：

> CoT 是否改变 fast residual 的强度和时间分布？

如果 `p_t` 无稳定差异，应明确写成 CoT 主要改变 slow component，而非 fast transient。

---

#### Step 4：Wanting–Confidence Relationship

同步分析 final-logit metrics：

- entropy
- top1
- margin
- information gain
- rolling confidence variance

按以下窗口报告：

| Window | Meaning |
|---|---|
| Prefill | task-entry confidence |
| Early 0–25% | reasoning launch |
| Middle 25–75% | reasoning process |
| Late 75–100% | answer convergence |

主要判定：

- RSN 与 confidence 同时改变：CoT couples wanting and confidence。
- RSN 改变而 confidence 不变：支持 wanting–confidence dissociation。
- confidence 只在 early 改变：CoT 的 decisiveness effect 是短暂启动效应。

---

#### Step 5：Two Time Axes

必须同时画：

1. **Length-normalized trajectory**：每个回答自身映射至 `0–100%`。
2. **Absolute decode-step trajectory**：至少前 100 tokens。

第一张比较完整生成阶段，第二张检查 early difference 是否由 CoT 回答更长、归一化压缩造成。

Prefill 单独画在 decode 之前；不要重新使用 prefill-seeded EMA。

---

#### Step 6：Behavioral Anchoring

不重新做完整行为学分析，只引用已有结果：

- α=0 No-CoT accuracy：60.0%
- α=0 CoT accuracy：69.0%
- CoT 明显增加 stepwise structure；
- generation length、抢答和 commit marker 的口径差异沿用 `AdaDopamine_gsm8k.md` 的限制说明。

这些结果只作为 external behavioral anchor，不从 signal JSON 重算 accuracy。

---

#### Outputs

**表 1：Three-component summary**

| Metric | No-CoT | CoT | Paired Δ | `d_z` | p/FDR |
|---|---:|---:|---:|---:|---:|
| `G_prefill` | | | | | |
| `boundary_jump_G` | | | | | |
| slow slope | | | | | |
| `Z_late` | | | | | |
| phasic peaks/std | | | | | |

**表 2：Multi-metric temporal summary**

每行一个 RSN/logit metric，每列为 prefill、early、middle、late 的 CoT−No-CoT effect。

**图：**

1. Task-entry `G_prefill` 与 `boundary_jump_G`
2. `Z_t / s_t / p_t` length-normalized curves
3. CoT−No-CoT paired difference curves
4. 前 100 absolute decode steps
5. Entropy/top1/margin/info-gain suite

### 最终要回答的核心问题

1. CoT 是否在进入 generation 前就改变 `G_prefill`？
2. CoT 的主要影响落在 slow dynamics 还是 fast phasic residual？
3. 差异集中在 early decode，还是贯穿整个回答？
4. RSN wanting 与 confidence 是耦合还是可分离？
5. 这些内部变化能否与已有的 CoT accuracy gain 和 stepwise structure 对应？

这版不把 correctness subgroup 当主轴，能够保持 CoT 与 No-CoT 的完整、对称比较。

### 4.3 Persona

**Expert vs Non-Expert**

Cohen's d + MWU significance, **expert − non_expert** (+ = expert higher; `***` p<.001, `**` p<.01, `*` p<.05, ns; n=300). Decode split into four length-normalised quartiles Q1–Q4.

| Metric | prefill | Q1 0–25% | Q2 25–50% | Q3 50–75% | Q4 75–100% |
|---|---:|---:|---:|---:|---:|
| wanting | **+0.27\*\*** | **−0.24\*** | −0.01 ns | +0.01 ns | +0.01 ns |
| entropy | +0.05 ns | +0.12 ns | +0.06 ns | −0.06 ns | −0.03 ns |
| top1 | −0.06 ns | +0.00 ns | −0.03 ns | +0.08 ns | +0.06 ns |
| margin | −0.06 ns | +0.05 ns | −0.01 ns | +0.08 ns | +0.06 ns |
| info_gain | — | +0.04 ns | −0.03 ns | +0.06 ns | −0.07 ns |

**expert vs non_expert 是一个纯 wanting 的、极短暂的时间差异,confidence 四轴 × 五口径全程 ns。** wanting 是唯一显著的指标,而且**只活在 prefill 和 Q1 两个口径,且符号翻转**:prefill 处 expert 更高（+0.27\*\*,对齐 mask 方向 expert−non),Q1 处 non_expert 反超（−0.24\*),**Q2 起完全消散**（−0.01/+0.01/+0.01 全 ns）。所有 confidence 指标(entropy/top1/margin/info_gain)在五个口径下**没有一格显著** —— 两个 role 的输出笃定度完全无法区分。这就是 dissociation 最干净的形态:persona 只在 wanting 轴的 **prefill→Q1 边界**留下一个"起点高、随即反超"的瞬态,Q2 之后连 wanting 都归零,confidence 轴则自始至终什么都没有。

### 4.4 α-Steering: A Linear Wanting Knob Driving Inverted-U Behavior
