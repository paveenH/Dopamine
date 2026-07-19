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

### 2.1 Core Idea

RSN（Role-Sensitive Neurons）的行為類比多巴胺的雙向校準機制，可以理解為一個 **state-level gain knob**：調節模型在當下狀態中有多願意啟動、投入、承諾、持續檢查或停止，而不是直接改寫模型能表徵的知識與推理能力。因此，RSN / α steering 的核心對象是 `wanting`，不是 `knowing`。

### 2.2 Human Side: Tonic and Phasic Dopamine

**Tonic（張力性 / 背景多巴胺）**
- 多巴胺神經元自發放電維持的、弥散在突觸**外**的背景濃度；作用於高親和力 D2 受體。
- 時間尺度 **分鐘~小時**——是一條緩慢起伏的水位線，不是尖峰。
- 決定**背景動機、精力、行動準備度**。低 tonic → anhedonia / 拖延 / bradykinesia（under-wanting）。
- **它不編碼內容，只設定背景增益 / 提交閾值。** 在一次任務執行的時間尺度內基本恆定，**不實時更新**。

**Phasic（相位性 / 爆發多巴胺）**
- 疊加在 tonic 水位**之上**的快速 burst / dip；burst 放電 → 突觸**內**濃度毫秒飆升 → DAT 秒清；作用於低親和力受體。
- 時間尺度 **毫秒~秒**——一個尖峰後迅速消失。
- 編碼**事件**，且不只一種：**RPE**（獎勵預測誤差，需獎勵）、**salience / novelty**（顯著、新奇、關鍵線索，**無需獎勵**）、**決策節點**（下決心那一下）。

**Ramping** 是 tonic / phasic 之外常被忽略的第三個時間尺度，也是近年最活躍的前沿概念——朝目標逼近時多巴胺濃度的**持續單調爬升**（Howe et al. 2013 Nature：大鼠接近遠處獎勵時 DA 隨**距離與獎勵大小 scale** 地 ramp）。它不是恆定 tonic、也不是瞬時 spike，而是「朝完成推進的速度 / 勁頭」= **vigor**；其本質常被理解為**價值函數的梯度 / goal proximity**（「離算完還差多少步」）。Ramping 屬 tonic 還是 phasic，學界有兩個並行的前沿假說：
- **假說 A — TD-error 長程積分（Gershman 2014, Neural Computation, Harvard）**：ramping 是密集微小正向 phasic 脈衝在時間上的積分；對 goal proximity 做**凸 / 二次變換**即近似得到觀測到的**線性 ramp**（Weber's-law 式的空間壓縮）。映射：解釋了「單點 phasic 沒變，但 α 改變了脈衝在長程上的累積係數」。
- **假說 B — 局部環路流體動力學（Mohebi et al.）**：即使切斷 VTA 胞體放電（soma spiking），紋狀體軸突末梢仍能經局部（膽鹼能 / 谷氨酸能）控制漏出 DA 形成 ramp——DA ramp 可**獨立於 VTA 單元放電**。映射：prefill 最後一刻注入的 α 正像這種「不改微觀內核、卻改變整條後續序列釋放速率」的環境參數。

### 2.3 From Human to Signal

| 成分 | 人體 | 時間性質 | 是否 decode 中更新 | 信號量 |
|------|------|---------|------------------|--------|
| **Tonic** | 背景水位 / 閾值設定| 一道題內恆定 | **否**，進題時設好 | **`G_prefill`**（per-sample 常數） |
| **Ramping / Vigor** | 朝目標推進的速度 / 勁頭 | 緩慢漂移 | **是**| **`s_t` 的斜率**（decode EMA 爬升快慢） |
| **Phasic** | 關鍵節點 | 尖峰 | **是** | **`p_t = Z_t − s_{t-1}`** 在關鍵 token 處 |

**讀這張表的三個要點：**
- **Tonic ≠ decode EMA。** tonic 是 prefill 常數，一道題內恆定、不實時更新；decode 裡緩慢更新的成分是 **ramping**，不是 tonic。α 是干預、`G_prefill` 是讀數，`G_prefill(α)≈G_prefill(0)+α` 線性。
- **Vigor 承載在 `s_t` 的斜率，不是絕對高度**（高度混了 tonic）。「+α 搶答 / gen_len↓」= 高 vigor；「−α 磨蹭 / 反覆驗算」= 低 vigor。注意：**loop 重複（`####N####N`）不是 vigor，是 commitment 關不掉的 stopping failure**，另計，不入這三成分。ramping 的**第二種讀法**（承假說 A 的 value-gradient 觀點）：每步取 hidden state、投影到「起點（prompt 結束）↔ 終點（首個 `####` token 嵌入）」軸，看**到終點的幾何距離 / 餘弦隨 step 的曲線**——預言不同 α 調制的是這條幾何斜坡的到達斜率（−α 更穩更緩、+α 更陡）。此讀法即 value 定義「選擇 (A) commit 方向投影」的具體實現，與 `s_t` 斜率互為印證。
- **Phasic 必須減基線，不能用瞬時值 `Z_t`。** `Z_t = tonic + ramping + phasic` 三成分疊加；`p_t = Z_t − s_{t-1}` 等同高通濾波，減掉 `s_{t-1}` 估計的 tonic+ramping 慢基線後，只剩繞關鍵 token 的 phasic 尖峰。**關鍵 token** 分兩類：首個 `####`/答案候選（commitment）、中間關鍵步（process salience）——均為 phasic 的**待驗**錨點（path A）。

**Phasic 亞型與驗證分工**：phasic 按編碼內容分 salience / commitment / RPE 三亞型。salience 與 commitment **無需 reward，GSM8K 即可驗**（心流是其神經科學背書）；只有 **RPE 亞型需要 reward feedback → Bandit，不在 GSM8K**。

**value 定義的未鎖自由度**：
- **(A) 投影方向**：wanting NMD mask（現在）vs 朝 `####`/commit 方向——若 commit 處 wanting mask 平而 commit 方向有峰，phasic 是「提交方向」而非「wanting 方向」。
- **(B) 基線時間尺度 β**：`s_t` 現用 β=0.95（承自舊碼，未針對分離 commit transient 調過）。β 劃定 tonic/ramping 與 phasic 的分界；event-align 後 `p_t` 若平，可能是 transient 落進基線 → 掃 β∈{0.9,0.95,0.98}。


## 3. Signal Definition

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


### 3.3 EMA Interpretation

EMA 的角色是把 noisy token-level `x_t` 轉成慢變的 tonic-like trajectory。它有兩個用途：

1. **診斷用途**：描述模型的 engagement / commitment / release dynamics。
2. **控制用途**：作為比 raw `x_t` 更慢、更穩定的 feedback variable。

`β=0.95` 對應約 20 token 的時間常數，適合長生成中的 state smoothing；對很短的 action-output 任務則未必穩定。因此，EMA 應被解讀為 **tonic-state approximation**。

### 3.4 Closed-loop Caveat: 1-step Lag

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

因此，feedback 永遠基於上一 token 的 observation，作用於下一 token。對慢變的 EMA，這個 lag 尚可接受；對 1–2 token 內自然消退的 raw spike，逐 token feedback 容易追尾並造成振盪。

### 3.5 Multi-Metric Signal Suite

**Per-step raw trajectories**

- `rsn_ema` 來自 **middle-layer HS 投影到 sparse NMD mask** 上的 scalar（RSN 子空間活動 = wanting / drive）。
- `entropy / top1 / margin / info_gain` 來自 **final-layer whole hidden state**，`RMSNorm + lm_head` 重建出的**真實 next-token logits**，再 softmax；即模型真實輸出分布。它們度量的是 **confidence / decisiveness**。

| Metric | Source | Computation | Interpretation |
|--------|--------|------|---------|
| `rsn_ema` | middle HS · **NMD mask** | `mean_l(h_t[middle]·mask[l])` → EMA(α=0.95) | wanting / drive (RSN subspace) |
| `entropy_decode` | **last layer HS** | `-Σ p log p` over vocab | confidence / decisiveness (inverse) |
| `top1_decode` | **last layer HS** | `max(softmax(logits))` = MSP | confidence / decisiveness |
| `margin_decode` | **last layer HS** | `top1 - top2` | confidence / decisiveness (≈collinear w/ top1; optional) |
| `info_gain_decode` | **last layer HS** | `H_{t-1} - H_t` | reasoning efficiency (paper 03) |

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


## 4. Observed Results Phase1

### 4.1 Data Scope and Reading Convention

Plots: `RoleAnswer/plot_phase1_state.py` (overlay + diff, prefill drawn as point 0; EMA-smoothed for trend, `--raw` for unsmoothed). Statistics: per-port Cohen's d recomputed inline in `RoleAnswer/`.

Readout convention:
- `prefill` = the last prompt token before generation (point 0 in the trajectory plots).
- `early` = the first 20% of decode.
- `μ` = the full decode mean.
- Effect sizes are Cohen's d against neutral No-CoT unless stated otherwise.

**Three signal groups** (5 metrics → 3 axes; see §3.5 for sources):
- **wanting** = `rsn_ema` / `x_decode`: middle-layer HS projected on the sparse NMD mask = the RSN / drive axis.
- **head decisiveness** = `top1` ≈ `margin`: reads only the top 1–2 tokens of the output distribution.
- **global uncertainty** = `entropy` ≈ `info_gain`: reads the whole distribution including the tail. `info_gain = −ΔH` (per-step entropy change); its tok0 value ≈ 2–3 is just the prefill→tok0 entropy collapse, not a steady-state level.


### 4.2 CoT vs No-CoT: Process-Level Early-Window Amplification

Cohen's d + MWU significance, **CoT − No-CoT** (+ = CoT higher; `***` p<.001, `**` p<.01, `*` p<.05, ns; n=300). Decode split into four length-normalised quartiles Q1–Q4; prefill column shows the raw starting means `No-CoT→CoT` alongside its d.

| Metric | prefill (No-CoT→CoT) | Q1 0–25% | Q2 25–50% | Q3 50–75% | Q4 75–100% |
|---|---|---:|---:|---:|---:|
| wanting | 0.566→0.569 (+0.03 ns) | **+0.91\*\*\*** | **+0.55\*\*\*** | +0.14\*\* | +0.08 ns |
| entropy (raw, ↑ = more uncertain) | 3.92→4.26 (**+0.36\*\*\***) | **−0.70\*\*\*** | +0.14\*\*\* | −0.08\*\*\* | −0.03\*\* |
| top1 | 0.196→0.171 (−0.17 ns) | **+0.32\*\*\*** | **−0.21\*\*\*** | +0.03\*\*\* | +0.01\*\* |
| margin | 0.101→0.081 (−0.14 ns) | +0.16\* | **−0.24\*\*\*** | +0.03\*\*\* | +0.01\*\* |
| info_gain (−ΔH) | — | +0.11\*\*\* | +0.07 ns | +0.07 ns | −0.05 ns |

>**CoT = process-level reshaping**。四分位口径把 launch-phase 结构完整展开:
>- **wanting = 一条干净的单调衰减梯度**:prefill 不动（+0.03 ns）→ Q1 峰值 +0.91\*\*\*（Phase 1 全表最大效应）→ Q2 +0.55 → Q3 +0.14 → Q4 +0.08（收敛到共享 plateau）。
>- **confidence = Q1 正 / Q2 负 的瞬态翻转**:top1/margin 在 Q1 显著为正（CoT 更笃定,+0.32 / +0.16),Q2 翻成显著为负（top1 −0.21、margin −0.24,此时 No-CoT 反而更笃定),Q3/Q4 收敛。

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
