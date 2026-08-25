<!-- 主線導覽（三份文檔共用，每份開頭都有）

整條研究主線（四段）：
  RSN
    → 行為學多巴胺（Behavioral Dopamine）← AdaDopamine.md
        → 腦科學多巴胺（Brain Dopamine）← 規劃中,尚未成獨立文檔（AdaDopamine.md §5 登記人類/動物範式血統）
            → 多巴胺與思考曲線（Dopamine & Thinking Curve）← 本文檔

  附：AdaThink.md 是 Thinking Curve 的額外延伸驗證（學弟執行），不在主線框架內。

【本文檔定位】
本文檔目前的主體是 **Phase 1b signal validation**（§4：α=0 觀察 + α-dose signal 的 gain-coordinate 分析,含 CoT/persona/dose/direction-specificity）。歷史上還包含兩層目標:
  1. 閉環控制（Phase 2，§三–§四）：用 RSN 信號做 decode-time 閉環調控，
     測試能否透過操控 EMA 波形（early peak / tonic plateau）提升推理準確率。
     結論：形狀控制（Plans A-H3）不等於 acc 控制，轉向 Phase 1b 信號驗證（即當前 §4）。
  2. Thinking Curve（Phase 3，理想目標）：在 reasoning model（DeepSeek-R1、
     Qwen3-thinking）的 <think> trace 裡觀察 persona 如何調節思考深度、
     backtrack、first-commit 等行為，對應多巴胺動力學，並透過 LLM 實驗
     模擬人腦思考過程中的 motivation dynamics。
     → 詳見 AdaThink.md（trace-level 分析框架）

【前兩段的任務】
AdaDopamine.md：行為學基礎驗證（wanting/knowing 解離、Bandit、Pressure）。
腦科學升華（RSA：RSN Δh 是否對應 ventral striatum / vmPFC）：規劃中,尚未成獨立文檔。

關聯文件：
  AdaDopamine.md — 行為學理論框架與實驗結果（§5 登記人類/動物範式血統）
  AdaDopamine_gsm8k.md — GSM8K/MATH 當前 production accuracy 與 commitment-dynamics 權威來源
  AdaThink.md — Reasoning model trace-level 分析框架（Thinking Curve 執行細節）
-->

# RSN as Dopaminergic Adaptive Calibration

## 1. Related Literature

### 1.1 Adaptive Reasoning

- **Reasoning Models Better Express Their Confidence** (NeurIPS 2025)  
  Reasoning models 的 verbalized confidence calibration 优于非推理模型，并会随 CoT 展开而改善。本文据此将 confidence、entropy 与 commitment dynamics 作为推理过程读数，而不只观察最终正确率。

- **Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL** (NeurIPS 2025)  
  AutoThink 训练模型根据任务状态决定是否展开推理，避免退化为 always-think 或 always-no-think。它代表显式 reasoning routing；本文则研究 RSN 是否提供一种连续的内部状态调节轴。

- **Think or Not? Exploring Thinking Efficiency in Large Reasoning Models via an Information-Theoretic Lens** (NeurIPS 2025)  
  长 CoT 后期可能出现 information gain 递减和 semantic drift。该工作支持使用 entropy、information gain 与 cumulative entropy reduction 检查继续推理是否仍带来有效更新。

- **Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling** (ICLR 2026)  
  DECS 区分形成答案所需的推理与答案已经可恢复后的冗余延续。它与本文的 commitment position、post-commit continuation 和 stopping failure 分析直接相关。

- **Efficient Reasoning with Balanced Thinking** (ICLR 2026)  
  ReBalance 使用 step-level confidence 与 confidence variance 区分 underthinking 和 overthinking，并通过 hidden-state steering 调节推理。它说明 adaptive reasoning 可以被视为连续的内部状态控制问题，而不仅是二元 CoT 开关。

- **An Investigation of Neuron Activation as a Unified Lens to Explain Chain-of-Thought Eliciting Arithmetic Reasoning of LLMs** (ACL 2024)  
  该工作发现 CoT prompting 会改变 arithmetic-related FFN neuron activation，为本文通过稀疏神经元方向研究 CoT、reasoning state 与行为变化提供直接的方法先例。

- **Reasoning Emerges from Constrained Inference Manifolds in Large Language Models** (2026)  
  该工作将推理描述为 hidden states 在低维、信息保持区域中的动态轨迹。它为后续检验 RSN 是自然推理流形上的功能轴，还是使轨迹偏离该流形的干预方向提供几何分析基础。

### 1.2 Dopamine Neuroscience

- **Dopamine Dynamics Are Dispensable for Movement but Promote Reward Responses** (Nature 2024)  
  快速 dopamine dynamics 并非维持基本运动所必需，但会增强 reward-related responses。该结果支持区分持续的背景状态与事件相关的快速变化，而不把所有行为调节都归因于 phasic signal。

- **Dopamine Release Plateau and Outcome Signals in Dorsal Striatum** (Nature Communications 2024)  
  持续努力过程中，dopamine activity 可由初始反应转为较稳定的 plateau。该结果启发本文区分 task-entry signal 与 decode 期间的 sustained slow state，但不意味着 LLM 应复制相同的生物波形。

- **Dopamine in Motivational Control: Rewarding, Aversive, and Alerting** (Neuron 2010)  
  该综述区分 dopamine 的持续背景调节与快速事件响应，并强调其与 motivation、action readiness 和 salience 的关系。本文据此将 RSN 主要解释为 wanting / engagement / commitment-related signal，而不是 knowledge signal。

- **Prolonged Dopamine Signalling in Striatum Signals Proximity and Value of Distant Rewards** (Howe et al., Nature 2013)  
  大鼠接近远距离奖励时，striatal dopamine 呈持续 ramping，并随目标距离和奖励价值变化。这是本文将 `s_t` slope operationalize 为 ramping/vigor 候选读数的主要神经科学依据。  

- **Dopamine Ramps Are a Consequence of Reward Prediction Errors** (Gershman, Neural Computation 2014)  
  该工作说明 dopamine ramp 可能由一系列局部 reward-prediction-error updates 累积形成，而不一定需要独立的持续 ramp generator。这为慢状态与快速 residual 之间的关系提供计算解释。  

- **The Brain in Flow: A Systematic Review** (Cortex 2022)  
  Flow 与前额叶、basal ganglia 和 reward-related circuitry 有关，但现有研究尚未直接测量人类 flow 状态下完整的 dopamine release dynamics。因此，本文的 tonic、ramping 和 phasic-like 解释只能作为功能类比，不能表述为生物同构。

### 1.3 Mechanistic Alignment: State vs Capacity

- **State and Capacity in Neural Models of Cognition and Consciousness** (Trends 2026)  
  该框架区分模型当前的运行状态与模型原则上具有的能力。State 包括 gain、attention、noise 与 decision threshold；capacity 则包括知识、表征深度和计算资源。本文据此将 RSN / α steering 定位为 **state-level gain calibration**：它调节 engagement、action readiness 与 commitment behavior，而不是直接增加模型的知识或 reasoning capacity。  

### 1.4 Literature Boundary

以上文献分别提供 adaptive reasoning、内部状态分析与 dopamine multi-timescale dynamics 的理论背景，但不直接证明 RSN 等同于生物 dopamine system。本文中的 tonic、ramping/vigor 与 phasic-like 均为 operational definitions；对应现象是否在特定任务中得到支持，需要由 `G_prefill`、`s_t`、`vigor_slope`、`p_t` 及其行为关联分别检验。

## 2. Core Theoretical Framework

### 2.1 Project Positioning

RSN（Role-Sensitive Neurons）被視為一個 **state-level gain control**：它調節模型在當下有多願意啟動、投入、承諾、繼續檢查或停止，而不是直接增加模型的知識或推理 capacity。本文以 `wanting` 描述這類 action-readiness / commitment tendency，並將 dopamine 作為**功能類比與可檢驗假說**。

### 2.2 Temporal Components

本研究提出一個三成分假說，將 generation dynamics 分為 **task-entry tonic、ramping 與 decode-time phasic**：

| Component | Functional interpretation | Operational signal | Main prediction |
|---|---|---|---|
| **Task-entry tonic** | 進入任務時設定初始增益與 commitment threshold | `G_prefill` | α 改變起始狀態，並影響後續解題與提交策略 |
| **Ramping / Vigor** | 解碼期間朝目標推進的速度與 effort intensity | $s_t = \mathrm{EMA}(Z_t)$ 的斜率 | 斜率越陡，推進與 commitment 越快 |
| **Phasic** | decode 中相對慢基線的快速 pulse / dip | $p_t = Z_t - s_{t-1}$ | 與慢變的 `s_t` 分離，呈現 token-level transient |

在此模型中，`G_prefill` 設定 generation 的初始條件，不作為 decode trajectory 的持續加數；decode 期間的慢/快分解由 EMA 定義（$s_t = \beta s_{t-1} + (1-\beta) Z_t$，$p_t = Z_t - s_{t-1}$），因此瞬時信號與慢基線的精確關係為：

$$Z_t = s_{t-1} + p_t$$

$$s_t = s_{t-1} + (1-\beta)\,p_t$$
其中 `s_t` 是慢變的 ramping / vigor component、`p_t` 是相對**上一時刻**慢基線 `s_{t-1}` 的 phasic residual（注意 `Z_t` 分解到的是 `s_{t-1}` 而非 `s_t`；`s_t` 再以 `(1-β)` 比例吸收該 residual）。


### 2.3 Working Hypotheses

**H1 — Prefill steering acts through initial-condition / boundary-gating.**

α 只在 prompt 的最后一个 token 注入一次，直接改变生成开始前的 `G_prefill`；进入 decode 后不再继续注入。目前观察到：

1. `G_prefill` 随 α 近似线性变化；
2. 到生成第一个 token（`decode[0]`）时，约 95% 的直接注入效应已经回弹；
3. 此后不同 α 条件下的 `G_decode` trajectories 大致重合。

尽管注入效应没有持续停留在 `G_decode` 上，prefill-only steering 仍显著改变了模型的生成行为。这说明 α 可能主要在 generation boundary 改变模型的初始状态与 commitment regime；随后，这一差异通过 KV cache、早期 token 选择和自回归路径依赖延续到整段生成。

因此，现有结果支持 **initial-condition / boundary-gating** 解释，而不是“α 在 decode 全程持续抬高 RSN signal”的解释。

**H2 — Slow decode dynamics encode ramping / vigor.**

本模型使用 `s_t` 的斜率来衡量模型在生成过程中朝答案推进的速度与 effort intensity，并将其作为 ramping/vigor hypothesis 的 operational measure。该假说预测：`s_t` 上升越快，模型的推进强度越高，因而可能表现为更短的 generation length 和更早的 commitment。分析中需要控制 output length 与 response format，避免把单纯的提前停止误判为 vigor。

这里需要区分指标定义与任务结果：`s_t` slope 作为 ramping/vigor 的 operational measure 保持不变，但特定任务是否呈现预期的 slope–behavior relationship，需要通过实验检验。GSM8K 的结果见 §4.8：该任务没有检出符合预期方向的 slope–vigor evidence；相比之下，`s_t` level 与 commitment timing 呈现稳定关联。

**H3 — Fast decode residuals encode phasic dynamics.**

`p_t` 表示当前 decode signal 相对于 slow baseline 的快速残差，用于分离被 `s_t` 平滑掉的瞬时 pulse 与 dip。我们首先检验 `p_t` 是否具有稳定的快时间尺度结构，再分析这些变化是否与中间推理步骤、答案形成、commitment 或其他 generation events 对齐。

目前不预设唯一的 event anchor，而是根据不同任务的生成结构检验可能的 transient–event association。若观察到稳定且可重复的事件对齐，则支持 `p_t` 捕捉了具有功能意义的 phasic dynamics；若未观察到，则说明该任务中尚未检出相应的快速事件结构。


## 3. Signal Definition

### 3.1 Signal Architecture

1. **RSN state signal**：middle-layer hidden states 在 NMD direction 上的活動，用來描述 task-entry gain 與 decode dynamics。
2. **Output-distribution signal**：由 final-layer logits 計算 entropy、top1、margin 與 information change，用來描述 confidence / decisiveness。
3. **Behavioral readout**：accuracy、generation length、commitment timing 與 stopping failure，用來檢驗內部信號是否對應可觀察行為。

RSN signal 是主軸；logit metrics 用於判斷 wanting 是否只是 confidence 的另一種表示；behavioral metrics 則提供外部效度。

### 3.2 RSN Projection and Gain Coordinates

對 token `t`、middle layer `l`，先計算原始投影：

$$r_{t,l} = h_{t,l} \cdot m_l$$

其中 `h_{t,l}` 是 decoder layer output hidden state，`m_l` 是同一 output space 中的 sparse NMD direction。每層先獨立投影，再進行跨層聚合。

為了固定零點、保留 α 的干預單位並避免少數 layer 因尺度較大而主導聚合，使用 neutral、α=0、No-CoT 的 prefill distribution 作為 reference：

$$\mu_l^{ref} = \mathbb{E}\big[r_{prefill,l} \mid \text{neutral},\ \alpha=0,\ \text{No-CoT}\big]$$

$$g_{t,l} = \frac{r_{t,l} - \mu_l^{ref}}{\lVert m_l \rVert^{2}}$$

$$\sigma_l^{ref} = \operatorname{Std}\big[g_{prefill,l} \mid \text{neutral},\ \alpha=0,\ \text{No-CoT}\big]$$

$$z_{t,l} = \frac{g_{t,l}}{\sigma_l^{ref}}$$

由此得到兩種跨層 readout：

$$G_t = \operatorname*{mean}_{l}\, g_{t,l} \qquad\qquad Z_t = \operatorname*{mean}_{l}\, z_{t,l}$$

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

$$T = G_{prefill}$$

`T` 是 task-entry tonic 的主 readout，表示 generation boundary 上的初始 gain / commitment set point。`Z_prefill` 可作為 layer-fair 的 condition comparison；`G_prefill` 則保留 α 單位，作為主要 calibration signal。

prefill 到第一個 decode token 的回彈另記為：

$$\text{boundary\_jump} = G_0 - G_{prefill}$$

`boundary_jump` 用來描述 task-entry pulse 如何進入自然 decode dynamics；在 `G` 坐標計算，以與 H1 引用的 α-單位回彈比例（約 95%）保持一致。

#### 3.3.2 Ramping / Vigor

在 decode 內，對 `Z_t` 建立 slow component：

$$s_0 = Z_0 \qquad\qquad s_t = \beta\, s_{t-1} + (1-\beta)\, Z_t$$

`s_t` 只由 decode token 初始化與更新，不以 prefill 作 EMA seed。主要 readout 是 `s_t` 的 trajectory slope，而不是其絕對高度：

$$\text{vigor\_slope} = \operatorname{slope}(s_t)$$

後續可分別估計 early、middle、late slope；β 與 window 的精確設定留待 sensitivity analysis。

#### 3.3.3 Fast Residual（candidate phasic-like）

decode-time fast component 定義為相對上一時刻 slow baseline 的 residual：

$$p_t = Z_t - s_{t-1}, \qquad t \geq 1$$

`p_t` 是一個 **EMA high-pass 殘差**（當前 `Z_t` 減上一步慢 EMA），不是 event-locked phasic 信號。`p_t > 0` 表示瞬時高於 slow baseline 的 pulse，`p_t < 0` 表示瞬時 dip。現階段把它當作 **fast residual / candidate phasic-like component**（此 operational 命名保留），優先檢驗其 amplitude、variability 與時間結構；之後再分析它是否與特定 reasoning / commitment event 對齊（不預先指定 event anchor）。**event alignment 等驗證決定的是「當前 task 是否提供 phasic-like empirical evidence」，而非決定 `p_t` 是否獲得該名稱。**

第一階段使用以下 summaries：

$$\text{phasic\_pos\_peak} = \max_t p_t \qquad\qquad \text{phasic\_neg\_peak} = \min_t p_t$$

$$\text{phasic\_abs\_mean} = \operatorname{mean}\big(\lvert p_t \rvert\big) \qquad\qquad \text{phasic\_std} = \operatorname{std}(p_t)$$

### 3.4 Multi-Metric Signal Suite

單一 RSN trajectory 不能區分 wanting、confidence、task performance 與 response failure，因此使用下列 multi-metric suite：

| Family | Metric | Source / computation | Main interpretation |
|---|---|---|---|
| **Task-entry state** | `G_prefill` | α-unit RSN gain at last prompt token | task-entry tonic / intervention strength |
| **Task-entry state** | `Z_prefill` | layer-standardized gain at last prompt token | layer-fair boundary state |
| **Boundary transition** | `boundary_jump` | $G_0 - G_{prefill}$（G 坐標） | prefill pulse 的回彈 / carry-over |
| **Slow decode** | `s_t` | decode-only EMA of $Z_t$，seed $s_0 = Z_0$ | post-launch slow generation dynamics |
| **Relaxation slope** | `vigor_slope` | slope of `s_t` | slow decode component 的斜率；欄名 `vigor_slope` 是 **H2 ramping/vigor 假說的 operational 名稱（保留）**，實測 decode 期間多呈鬆弛（下降），故功能上讀作 relaxation slope——命名指假說，數值方向指觀測 |
| **Phasic** | `p_t` | $Z_t - s_{t-1}$ | fast pulse / dip relative to slow baseline |
| **Uncertainty** | `entropy_decode` | $-\sum_v q_t(v)\log q_t(v)$ | next-token uncertainty；越低通常越 decisive |
| **Confidence** | `top1_decode` | $\max_v q_t(v)$ | maximum next-token probability |
| **Confidence** | `margin_decode` | $\text{top1} - \text{top2}$ | local choice separation；與 top1 高度相關，作輔助 |
| **Distributional change** | `info_gain_decode` | $H_{t-1} - H_t$ | token-to-token uncertainty reduction，不直接等同 reasoning quality |
| **Cumulative change** | `cumulative_entropy_reduction` | $H_0 - H_t$ | 相對 generation 起點的累積 certainty change |
| **Confidence stability** | `rolling_conf_variance` | $\operatorname{Std}\big(\text{top1}[t-W:t]\big)$ | local confidence volatility |
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

**口徑說明:** 本表及 §4 全節的 `signal`(server-184，bs=1 inline `correct`)只用於**同批 signal–behavior alignment**；**production accuracy 一律以 [AdaDopamine_gsm8k.md](AdaDopamine_gsm8k.md) 的 server-182 offline first-`####` 口徑為準**。兩套數值不可混算(跨機 bf16 + bs 差異),但 dose 形狀一致、離散最佳點均在 α=−6。

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
- **post-commit** = 第 1 個 `####` → 第 2 個 `####`（**提交後延續生成**；此處第 2 個 `####` 是 **second-answer-marker proxy**,非經獨立演算法驗證的 loop onset）
- **post-second-marker tail** = 第 2 個 `####` → generation end（與 stopping failure / loop 相容,但只作診斷,不等同真實 loop onset）
- *full = 三段之和*（僅用於展示 loop contamination，不作機制結論）

$s_t = \mathrm{EMA}(Z_t)$ 在整條 decode 上只計算一次（$s_0 = Z_0$），$p_t = Z_t - s_{t-1}$ 再由同一條軌跡取得；三階段不重新 seed。C1/C2-centered 圖只是放大兩個階段邊界，不構成另一套切分。**C2 是 second-answer-marker proxy**,其後段為 post-second-marker tail;C2-based 分析只涵蓋具有第二個 `####` 的 loop-prone subset，屬診斷性結果,不能把 C2 直接等同真實 loop onset。

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
| **post-second-marker tail** (n=141) | mean Δ=+0.185, `d_z`=0.32；tail 短 **83.6 tokens**, `d_z`=−0.51 | `std` Δ=−0.083, `d_z`=−0.33 | 第二個 `####` marker 之後的尾段（與 stopping failure / loop 相容，但只作診斷，不等同經獨立驗證的 loop onset）；CoT tail 較短、churn 較低 |

CoT 的主效應是 **level-dominant but not a pure shift**：task entry 已升高（`Z_prefill d_z`=0.89），pre-commit slow level 仍維持明顯差距（`d_z`=0.65），而 No-CoT 在答案形成期間 relax 得更多。strict-`####` subset（n=233）確認 pre-commit level effect 不依賴 fallback；absolute-token control 亦重現較弱的 CoT relaxation，因此不是單由長度歸一化製造。由於 CoT instruction 持續存在於 KV cache，這些觀察不能證明後續差異完全由 `G_prefill` 單點造成。

**Commit-centered boundaries.** C1 前 CoT `s_t` 維持高位，C1 後兩組共同快速下降，但 CoT 保留較高 offset；`p_t` 同時出現明顯負向 transition。C2-centered 圖重現近似結構，顯示第二個 `####` 也伴隨 marker-locked transition，但尚不能證明它是獨立的 loop-onset mechanism。C1/C2 都涉及 `####` / answer-format 改變，因此 fast transition 仍需 token-class-matched pseudo-event control。

**Full-decode diagnostic.** 若把整條 767-token decode 當作一段，slow slope 與 relaxation 會被 loop tail 反轉；full trajectory 因此只展示污染，不代表正常 reasoning 的 late stage。

`p_t` 是 EMA high-pass residual，不是已識別的 biological phasic dopamine。其 pre-commit `abs_mean` / `std` 效應在 FDR 後穩定，但 post-commit 小效應不宜強調；`pos/neg_peak` 同時受 segment length 與 EMA lag 影響，不能作獨立證據，也不能由較深負尾推斷 downward skew。

Event alignment 現已顯示 C1/C2 附近存在 marker-locked residual transition，但它同時伴隨 entropy spike / top1 dip，且 C2 幾乎複製 C1，因此可能包含 answer-format transition。現階段最穩健的結論仍是 **stage-dependent fast RSN dynamics**。若要**聲稱在 GSM8K 中檢測到 commitment-related phasic-like response**，仍需 pseudo-event / token-class control、length-matched quantile、`β∈{0.90,0.95,0.98}` sensitivity、其他 baseline estimator，以及 NMD-mask vs random-mask specificity。GSM8K 沒有 reward feedback，因此此處不能作 RPE 解讀。

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

整體而言，CoT 的最強 RSN 證據是 **task-entry gain + sustained pre-commit engagement + post-commit release**；confidence control 則只支持 **pre-commit output decisiveness 提高**，commit 後不作實質解讀。Fast residual 與 process salience / cognitive updating / commitment-related phasic-like dynamics 相容，但仍缺 event specificity 與 random-mask controls；**causal evidence for RSN gain modulation that supports the dopamine analogy** 仍需來自 α intervention、dose-response 與行為學實驗（α 能因果操縱 RSN,但不直接證明生物 dopamine mechanism）。

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

因此 §4.3 的主要貢獻是揭示 **Persona 與 CoT 具有不同的 temporal and representational modulation profiles**。CoT 在 pre-commit 同時提高 slow RSN engagement 與 output-distribution decisiveness，屬於較明顯的 **joint wanting–confidence modulation**；Persona 的主要效應則集中於 RSN/wanting 軸的時間重分配——從 task-entry gain、commitment-formation reversal 到 post-commit release。Persona 並非完全不影響 confidence：Non-Expert 在 pre-commit 略為更 decisive，但其效應明顯弱於 RSN gain，且較為局部。因此較準確的結論是：**CoT 同時調制 wanting 與 confidence，而 Persona 主要重組 wanting dynamics，並伴隨較弱的 confidence change。** 這支持 RSN 作為不同於 output confidence 的 dynamic state/gain readout，但 **causal evidence for RSN gain modulation that supports the dopamine analogy** 仍主要來自 §4.4 α dose-response 與行為學結果（α 因果操縱的是 RSN gain,非直接的生物 dopamine mechanism）。

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

與 §4.2 相同，fast component 的主讀數限於 `abs_mean/std`，不使用易受長度與 EMA lag 影響的極值。`p_t` 保留 phasic-like operational definition；本節僅觀察到 amplitude/dispersion evidence，C1 附近的共同轉折也可能包含 `####`/answer-marker effect，尚未建立 biological dopamine 或 event-specific phasic correspondence。

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

$$\mathrm{DiD}_q = \big(\text{cot}_{-4} - \text{cot}_0\big) - \big(\text{nocot}_{-4} - \text{nocot}_0\big) \qquad \text{[每題 } q\text{]}$$

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

α 效應幾乎不受 CoT 影響（G_prefill −6.80 vs −6.78），與 §4.4 co-design identity（$G_{prefill}(\alpha) \approx G_{prefill}(0) + \alpha\lVert m\rVert^{2}$，α 在 prefill 加一個與 CoT 無關的固定量）一致。**入口是 α 主導（|Δ|≈6.8）、CoT 次要（|Δ|≈0.07–0.16）。** G/Z 的 DiD 帶星號是 **statistical interaction**（n=300 + 極小配對方差可偵測到微小系統偏離），但其 **practical** 量級僅約 α 主效應的 0.4%——因此判為 **approximately additive**，不寫成 ns（effect 小 ≠ 不顯著），也不寫成「純 additive」。

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

`p_t abs_mean` / `std`：主效應仍是 CoT（early abs_mean +0.092\*\*\*、std +0.143\*\*\*；release abs_mean +0.178\*\*\*），α=−4 邊際弱，early window 呈與 Step 2 同向的 attenuation trend（α 效應在 CoT 下轉 ns），commit / release **DiD 全部 ns**。`p_t` 保留 phasic-like operational definition；本節僅得 fast residual dynamics evidence，尚未建立 biological dopamine 或 event-specific phasic correspondence。

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

### 4.6 RSN Direction Specificity: support-selection 與 generic-direction null

前面 §4.1–4.5 的所有 state 效應都是在 **NMD/RSN 方向**上投影得到的。一個必須回答的對照問題是：**這些效應是 NMD 方向特有的，還是任意一個稀疏方向都會出現同樣的 state 差異？**

本節用**兩級 null** 逐步收緊這個問題：

- **① support-selection null（`diff_random`，N=11）**：只隨機化「選哪些 neuron」（support），取值仍**複用 dense role-diff 的 coordinate-wise 數值與符號結構**。它回答「**top-|diff| 這組支撐是否特殊**」，但**不能**回答「role-diff 方向 vs 任意無關方向是否特殊」——因為它保留了 role-diff 的權重。
- **② generic-direction null（orthogonal Gaussian，各 N=10）**：把權重換成**逐層與 dense role-diff Δ_l 精確正交、且 norm-match 到 NMD 行**的 Gaussian，徹底去掉 role-diff 的權重方向。兩個子家族：`ortho_gauss_same`（用 NMD **自己**的 20 個 neuron 位置）與 `ortho_gauss_off`（用**與 NMD 不相交**的隨機位置）。前者測「同一批 support 上、非 role-diff 方向」，後者測「完全避開 NMD support 的任意稀疏正交方向」。

三個 null family 一致落入同一結論，故本節結論由前一版的 preliminary 上調為 **跨三類 null 一致的 exploratory direction-specificity evidence**（仍限 N=10–11 exploratory ordering、Llama3-8B、offline，見文末限定）。

**方法（offline re-projection null）。** 因果雙向 steering 的有效性已在 RSN 母論文中驗證，本節只做離線重投影：把**同一批已存的 HDF5 hidden states** 對不同 mask 重投影：

- **① support-selection null（`detection/nmd.py:get_diff_random_mask`）**：**隨機位置**，取值來自真實 role-diff。其 per-layer norm 天然約為 NMD 的 ¼（NMD 挑 top-|diff|），此 norm gap 是 NMD「取 top-k」操作的正確對照，**不做 norm-match**；與 ±1 的 `random` mask 不同。N=**11**（seed 1–10 + seed 42），p 地板 = 1/12 ≈ 0.083。
- **② generic-direction null（`detection/nmd.py:get_ortho_gauss_mask`，`ortho_gauss_{same,off}`）**：逐層 float64 建構，$m_l = g - \dfrac{g \cdot d_{sub}}{\lVert d_{sub}\rVert^{2}} d_{sub}$（`d_sub` 取自 **dense role-diff** 而非稀疏 mask），使 `m_l ⊥ Δ_l`，再 norm-match 到 NMD 該層。建構期硬性 assert 逐層 `|cos(m_l,Δ_l)|<1e-5`、norm-match、恰好 top_k 個非零、support 關係正確——一次乾淨跑完即通過驗證。各 N=**10**（seed 1–10），p 地板 = 1/11 ≈ 0.091。
- **每個 mask 用自己的 reference**：μ/σ 來自它自己的 neutral-No-CoT prefill，`‖m_l‖²` 來自它自己，使「raw projection 尺度較大」不能為 NMD 買到假優勢。（注意：在 decode 的 Z 座標下 `‖m‖²` 於標準化中相消，故 norm gap 不影響 decode 讀數；它只影響 `G_prefill`。）ortho null 已 norm-match，`G_prefill` 也可比。
- 三個 family 皆屬 exploratory（N=10–11），只讀 **effect 與 ordering**，不作正式顯著性宣稱。

**帶符號 temporal specificity。** 指標為**帶符號、commit-aligned** 的軌跡量（與 §4.1–4.5 同口徑）：commit-aligned、Z 單位，窗口 pre=`[−40,0)`、post=`[0,+10]`；每個 mask 用自己 reference，共用同一 K-gate（`K=max(30,⌈0.15·n⌉)`）。兩個 primary 讀數 `s_pre_mean`（commit 前 slow-state）與 `p_post_mean`（commit 後 fast-residual）：

| Contrast | `s_pre_mean` (NMD / null-med / pctile) | `p_post_mean` (NMD / null-med / pctile) |
|---|---|---|
| CoT vs No-CoT | +0.500 / −0.034 / **100%** | −0.361 / −0.054 / **0%** |
| Expert vs Non-Expert | −0.133 / +0.032 / **0%** | +0.157 / −0.007 / **100%** |
| α=−4 vs 0 | +0.129 / −0.003 / **100%** | −0.065 / −0.012 / **0%** |
| α=−6 vs 0 | +0.437 / −0.015 / **100%** | −0.264 / −0.062 / **0%** |
| α=+4 vs 0 | −0.167 / −0.014 / **0%** | +0.185 / +0.004 / **100%** |

（每個 primary cell 在全部 5 個 contrast 都落在 null 分布的極端端 pctile 0% 或 100%，p_emp 全為 N=11 地板 0.083。）

三點觀察：

1. **兩個 primary 在全部 5 個 contrast 都落 null 極端**，方向隨 contrast **系統性翻轉**：α−/CoT 是「commit 前 `s_t` 高、commit 後 `p_t` 低」，α+/Expert 完全鏡像。
2. **自然 state（CoT、Persona）與注入 α 表現一致。** 若特異性只出現在 α-dose，可能是 α-steering 的 injection–projection identity 造成的時間版假象；但 CoT vs No-CoT、Expert vs Non-Expert（皆非注入）與 α 條件同樣落極端——故此時間特異性**不能僅由 α-steering 的 injection–projection identity 解釋**。（其中 CoT 是最獨立的自然狀態證據；Expert–Non-Expert 雖非注入，但 NMD mask 本身即抽自該 contrast，故關聯性較弱。）
3. **整條軌跡距離（leave-one-out centroid RMS，`[−40,+20]`，`s_t`/`p_t` 分開）**：全部 10 個 (contrast × signal) cell 的 NMD signed 軌跡距離都**超過全部已採樣 null draws**（D_NMD 為 null median 的 3–7 倍，pctile 100%，p_emp 於 N=11 地板 0.083）。但 RMS 同時含水平/幅度/形狀，**不能單獨解讀為形狀特異**：`p_t` 的 shape-corr（NMD vs null 0.39–0.55）相近，顯示偏離**主要來自幅度/水平差異**（「同形不同幅」）；`s_t` 的 null 質心近平坦（centroid std < 0.02），shape-corr 不可解釋（已標註）。**是否存在獨立於幅度的形狀差異，仍需額外指標。**

commit-centered 圖（`fig46_commit_specificity_{contrast}.png`）直觀呈現：每個 contrast 的 NMD 曲線在 commit（step 0）附近急轉——commit 前一個平台、commit 後單調鬆弛；11 條 null 在 commit 處**未形成與 NMD 同等強度的 signed transition**（`s_t` null 近平坦，`p_t` null 有結構但幅度較弱）。CoT 的 `p_t` 更在 commit 出現一個 −0.7 的單步 transition（呼應 §4.2：commit 附近出現顯著 `p_t` transient，與 phasic-like hypothesis 相容，但仍有 `####`/marker-format confound，不宣稱為獨立的 phasic 事件節點）。

**升級：generic-direction null（權重去掉 role-diff 後，NMD 仍相對 null 保持極端）。** 上表 ① 的 null 保留了 role-diff 權重。把權重換成逐層 ⊥ Δ_l 的 norm-matched Gaussian（② `ortho_gauss_same`/`off`）後，兩個 primary 的 NMD-vs-null 極端 pctile 維持不變——下表列 NMD signed 與各 null 的 median（後者中心趨勢皆 ≈0，且採樣到的方向均未複現 NMD 的窗口平均 signed effect）：

| Contrast | `s_pre_mean` NMD | same-med / off-med | pctile (same / off) | `p_post_mean` NMD | same-med / off-med | pctile (same / off) |
|---|---|---|---|---|---|---|
| CoT vs No-CoT | +0.500 | +0.016 / +0.010 | **100% / 100%** | −0.361 | +0.027 / −0.004 | **0% / 0%** |
| Expert vs Non-Expert | −0.133 | −0.009 / −0.004 | **0% / 0%** | +0.157 | +0.005 / −0.013 | **100% / 100%** |
| α=−4 vs 0 | +0.129 | +0.007 / +0.006 | **100% / 100%** | −0.065 | −0.004 / +0.004 | **0% / 0%** |
| α=−6 vs 0 | +0.437 | +0.040 / −0.006 | **100% / 100%** | −0.264 | −0.005 / +0.031 | **0% / 0%** |
| α=+4 vs 0 | −0.167 | −0.040 / −0.007 | **0% / 0%** | +0.185 | +0.007 / −0.011 | **100% / 100%** |

三個 family（`diff_random` / `ortho_gauss_same` / `ortho_gauss_off`）合計 **30 個 primary cell 中，NMD 皆相對各自 null 保持極端**（pctile 0% 或 100%），且各 null 的**中心趨勢接近 0，採樣到的方向均未複現 NMD 的窗口平均 signed effect**（null median≈0；注意這是 null 的中心，不代表 null 曲線無結構——部分 null 尤其 `p_t` 仍有明顯 commit 附近波動，只是其窗口平均帶符號效應接近 0 且方向不定）。LOO-centroid RMS 亦全 90–100% pctile。把三個 family 排進 support × weight 的 **control matrix**（非嚴格 factorial——① 的 support 是隨機而非窮舉一格；每格記「NMD 是否相對該 null 保持極端」）：

| | role-diff 權重（①） | ⊥ role-diff 權重（②） |
|---|---|---|
| **NMD support**（same） | NMD remains extreme | NMD remains extreme |
| **NMD-disjoint support**（off） | NMD remains extreme（① random support 落此格） | NMD remains extreme |

**off-support 這格最吃重**（把 20 個位置**完全移出 NMD** 且權重正交於 role-diff，NMD 仍獨佔極端，null median≈0——off 方向自己並未命中某個 global mode）。三格合起來指向：特異性來自 **top-|diff| 支撐與 role-diff-aligned 權重的特定組合（兩者的匹配關係）**，而非任一單獨成分——`ortho_gauss_same` 失敗說明 NMD 支撐單獨不充分，`diff_random` 失敗說明 random 支撐上的 role-diff 權重不充分。並非 support 或 weight 不重要，恰恰是兩者的對應最重要。

**Leave-one-layer-out（不由任一單層驅動）。** 對 NMD mask 逐一踢掉 9 個 middle layer（L11–19）之一、在剩 8 層上重算兩個 primary 的 signed window mean：全部 10 個 (contrast × primary) cell 在**任一單層被踢掉後皆保持符號不變**（no sign flip），且 LOO 的 min/max 與 full 同側不跨 0（例：α=−6 `s_pre_mean` full=+0.437，LOO 範圍 [+0.334, +0.621]；`p_post_mean` full=−0.264，範圍 [−0.374, −0.211]）。最具影響力的單層為 L11，但踢除後效應仍穩健——故此時間效應**不由任一單層驅動**。**但這是 single-layer LOO,尚未排除兩三個 layer 共同貢獻**（未做 leave-two/three-out 或 null 側逐層對照）,故只滿足「不由任一單層驅動」,不宣稱「不由少數 layer 驅動」。

**兩層結論（分開讀）。**

1. **task-entry raw gain（`G_prefill`）**：NMD 遠強於 null（α-dose `d_z` ±72–80 vs null ±3–7），但這是 **co-design identity**（$x_{prefill}(\alpha) \approx x_{prefill}(0) + \alpha\lVert m\rVert^{2}$，NMD 因取 top-|diff| 而 norm 最大）+ mask 本身抽自該方向，屬 **manipulation check**，非獨立證據。
2. **commitment-locked temporal organization（`s_t`/`p_t` 軌跡）**：**這是目前最強的 NMD direction-specificity evidence**——commit 前後帶符號的結構化走向（幅度/水平）穩定超出**三個** null family（support-selection ①、same/off generic-direction ②），且在注入與自然 state 上一致。off-support generic-direction null 也保持 NMD 極端（權重去掉 role-diff、位置移出 NMD 後 NMD 仍獨佔極端），與 ① 對照後，證據指向 **top-|diff| 支撐與 role-diff-aligned 權重的特定組合**是特異性來源——排除了「僅是 top-|diff| 支撐」與「僅是複用 role-diff 逐坐標權重」兩種單成分解釋；是否另有獨立於幅度的形狀差異仍待定。

**限定。** (i) direction-specificity null **已補齊**：support-selection（①）+ same/off generic-direction（② orthogonal Gaussian）三個 family 一致 hold，已排除「僅 top-|diff| 支撐」與「僅複用 role-diff 逐坐標權重」。**尚未做**的僅剩 same-support sign-shuffle（檢驗「符號—位置對應」——註：NMD 支撐的 role-diff 符號**數量**近乎平衡（180 neuron 中 +86/−94，imbalance 0.044），但這只說明正負個數對稱，不代表該對應不重要，故列為次優先而非已知無判別力）；orthogonal null 已 norm-match 並徹底去除 role-diff 權重方向，sign-shuffle 屬更細的分解，非必要補強。(ii) **各 family N=10–11 draws 為 exploratory ordering，不作正式顯著性宣稱**（p 地板 0.083–0.091；且 draws 已參與指標選擇）。三個 family 與 30 cell 並非獨立重複——它們共享同一批 hidden states、conditions、baseline 與指標，故不可用「多 family 一致」推得低偶然機率。下一步優先**跨任務、跨模型與 causal-direction control**（見 (iii)），而非繼續擴增同類 seeds。(iii) 僅 Llama3-8B、僅 offline re-projection（不含 random-direction 因果 steering 對照）。(iv) leave-one-layer-out 已做且通過（10/10 no-flip，見上），僅滿足「**不由任一單層驅動**」；**尚未排除兩三個 layer 共同貢獻**（未做 leave-two/three-out），故不宣稱「不只由少數 layer 驅動」。per-layer 對照僅在 NMD 側檢驗軌跡穩健性，尚未與 null 做逐層對照。

分析腳本：`analyze_rsn_specificity.py`（`python3.10`；帶符號 commit-aligned temporal 指標 + LOO-RMS + leave-one-layer-out；`--null_family {diff_random,ortho_gauss_same,ortho_gauss_off}` + `--null_root` 切換 null family，凍結指標不變；`--plot` 出全部 5 張 commit-centered 圖，檔名帶 family tag）。讀 `llama3/dopamine/signal/` NMD + `llama3/dopamine/{random,ortho_same,ortho_off}/seed{1..10}/`；① 由 `run_random_null.sh`、② 由 `run_generic_null.sh` 生成（皆 server-side，zero-GPU offline re-projection）。server 步驟見 `GENERIC_NULL_RUNBOOK.md`。

### 4.7 Case Study: sample-level RSN trajectories and generated text

為核對聚合曲線的功能解讀，本節逐題對照 9 個 `sample_traj3_` case（Q10、Q80、Q92、Q140、Q152、Q189、Q225、Q251、Q284），每題疊加 neutral No-CoT 的 α=−6、0、+6 `s_t` / `p_t` 軌跡與實際生成文本。這是**定性 sanity check**，樣本不是預註冊或代表性抽樣，不提供新的 effect size、顯著性或因果證據；其用途是檢查聚合指標是否對應可辨識的生成階段，以及暴露事件定位與輸出格式的混淆。

#### Case-level observations

**Slow state `s_t`: sustained reasoning 與 state release。** 多個 case 中，模型仍在展開推理、修正候選答案或尚未正式提交時，`s_t` 維持較高或較持續；首次明確作答後則常快速下降。Q140 的 α=+6 在較長推理後答對，期間 `s_t` 長時間維持；相對地，較早提交並進入重複的條件更快下降。Q189 的 α=+6 長時間未形成正式提交，`s_t` / `p_t` 也持續活躍。這與 §4.2–4.5 的 **pre-commit engagement → post-commit release** 聚合結構一致，但不表示高 `s_t` 必然帶來正確答案：Q251 顯示持續生成也可能沿錯誤路徑推進。case study 支持 **`s_t` level 與 ongoing / unresolved processing 相關**，而不是 correctness 或 reasoning quality 的直接讀數；但這是 level 觀察，**不直接檢驗以 slope 定義的 ramping/vigor hypothesis**（該假說由 §4.8 Slow-State Behavioral Validation 專門檢驗，結果為 GSM8K 未檢出 slope-vigor evidence）——不能由此 case-level level 觀察推斷 slope 的 vigor 讀數成立或不成立。

**Fast residual `p_t`: generation-mode sensitivity。** Q80、Q92、Q140、Q189 等 case 顯示，開放式自然語言推理時的 `p_t` 往往較高幅且不規則；進入 `####`、數字或固定句式反覆輸出後，則常轉為較低幅、較規則的振盪。這個視覺觀察促成下列全樣本 follow-up；結果顯示,能穩定區分 reasoning 與 post-answer/loop 階段的是 `p_t` 的 **centered RMS(residual amplitude)**,而非頻率指標——其主要穩定變化是 **residual amplitude collapse**,不是獨立的 frequency reorganization。

#### Formal amplitude/frequency validation

**Formal `p_t` frequency test：幅度效應保留，未發現穩健的頻率組織。** `analyze_pt_frequency.py` 在 neutral No-CoT 的 α=−6/0/+6 上逐題計算兩套 paired comparison：(i) commit-centered `[−40,0)` vs `[0,+40)`（n=195–248/α）；(ii) reasoning `[0,C1)` vs 由 **strict repeated-ngram tail proxy**（全文最早重複 ≥3 次的 12-character n-gram 起點）定位的複讀尾段（n=24/42/41）——此 proxy 僅為重複性 tail 的近似,不等同經獨立驗證的 loop onset。每段先去均值，再報 centered RMS（residual variability）、zero-crossing rate、Welch dominant frequency、spectral centroid 與 normalized spectral entropy；因此此處的 RMS 是 fast residual 的段內變異幅度，不包含 level shift。

| post−pre | α=−6 | α=0 | α=+6 |
|---|---:|---:|---:|
| centered RMS | **−0.285**\*\*\* | **−0.186**\*\*\* | **−0.143**\*\*\* |
| spectral entropy | **−0.094**\*\*\* | **−0.074**\*\*\* | **−0.056**\*\*\* |
| zero-crossing rate | **−0.061**\*\*\* | **−0.038**\*\* | **−0.036**\*\* |
| spectral centroid | −0.007 ns | **−0.019**\*\*\* | −0.013\* |

（paired Wilcoxon：\* `p<.05`，\*\* `p<.01`，\*\*\* `p<.001`。）

Commit-centered 結果表面上呈現「提交後振幅降低、頻譜更集中且 crossing 變少」，與 case 圖的規則振盪印象一致；但 spectral-entropy change 與 `####` 比例變化高度共變（Pearson `r=−0.53/−0.58/−0.55`），與 repeated-12gram rate 亦中度共變（`r=−0.40/−0.46/−0.43`）。這不是控制後的因果分解，不能單由相關係數估計「多少效應由格式造成」；不過它清楚顯示 commit-centered frequency/regularity readout 與 answer-format、重複內容不可分離。

stage-based comparison（reasoning vs repeated-ngram tail proxy）給出不同結果：tail 段的 centered RMS 在三個 α 仍大幅下降（約 −0.31/−0.35/−0.36，皆顯著），但 zero-crossing rate 全部接近 null，dominant frequency / centroid 反而小幅上升，spectral entropy 只弱下降。換言之，跨兩種切分唯一穩定的量是**幅度下降**；「更低頻／更規則」並未跨口徑成立，也沒有 α-monotonic dose structure。此 stage subset 的 repetition/hash 與 frequency-change 相關雖接近 0，但**不應據此宣稱頻率效應已扣除 repetition confound**：detector 本身按重複性選樣，n 僅 24–42，同時存在 selection 與 range restriction，只能說「頻率差在兩種切分下不穩定」，不是一個 confound-free 的 negative test。

**Between-α：α 的資訊在 pre-commit residual amplitude,不在頻率。** 前述兩套皆為 within-α 的 pre→post/stage 對比;要分離 **α 主效應**,再逐題配對(同 300 題 index 對齊)比較同一段內 α=−6/α=+6 相對 α=0 的差異。**pre-commit(提交前的開放推理段)是唯一有干淨 α 信號之處**:α=−6 的 centered RMS 顯著高於 α=0（Δ`+0.046`，`p<.001`，paired n=200），α=+6 相對 α=0 接近 null（Δ`−0.006`，n.s.），而**所有 frequency metrics（zcr / dominant frequency / centroid / spectral entropy）在兩個方向皆為 null**。此段在提交前、不含 `####` 複讀,因此該幅度效應是**扣除 loop 後仍與 α 相關**的乾淨結果,與 §4.4 的 slow-state dose 結論同向（−α 抬高 pre-commit engagement）。post-commit 段 α=−6 的 RMS 更低、α=+6 更高,但此窗口混合答案收尾與複讀,只作 **post-marker diagnostic**,不作機制結論。reason / tail 段的 between-α 因要求「兩 α 在同題皆有 tail」而僅 n=10/14,不報告任何機制結論。這補上本節最重要的正結果:**`p_t` 攜帶的 α-related 訊息位於 pre-commit residual amplitude,而非頻率。**

**First-answer accuracy 與 stable completion 分離。** 個案也顯示「首次答對」不等於「穩定完成」。例如 Q251 的 α=0 首次提交正確答案 60，因此 first-answer protocol 判為 correct，但之後仍繼續除以 2 並產生錯誤候選。這不否定 GSM8K 以 first `####` 作 production accuracy 的口徑；它說明 accuracy 與 termination quality / post-answer degeneration 是兩個不同的行為維度。後者應由 answer switching、重複強度、自然 EOS、hit-cap 或 stable-final-answer 等獨立指標描述，不能由 first accuracy 代替。

**Declaration marker 的技術邊界。** 單一句子如 `The final answer is: \boxed{75}####` 會同時命中 `final answer`、`\boxed{}` 與 `####`；若不合併，圖中的第二條線可能只是同一次提交的另一個 marker，而非第二次作答。`plot_sample_traj.py` 現已將相距 25 characters 內的 marker 合併，dotted line 表示 **second distinct answer declaration**。但即使是第二次獨立 declaration，也只能視為 repetition / revision proxy，不能自動等同真正的 loop onset。§4.2 聚合分析使用 literal 第 1 / 第 2 個 `####`，因此不受同一句多類 marker 重複命中的繪圖問題影響；不過其中 C2 仍應解讀為 **second-answer-marker boundary**，其後段是 post-second-marker tail，而不是經獨立演算法驗證的 loop onset。

**Case-study conclusion：amplitude / frequency dissociation。** 這 9 題與全樣本 follow-up 共同把本節從「case study + frequency negative」升級為 **case-level validation + amplitude/frequency dissociation**:(i) `s_t` 的主結構與持續推理—提交後釋放相容;(ii) full-decode 會被 post-answer stopping failure 污染;(iii) `p_t` 的可靠訊息集中於 **signed change 與 residual amplitude / dispersion**——α=−6 在乾淨的 pre-commit 段提高 centered RMS,而 frequency metrics 不隨 α 穩定變化。Commit-centered 的頻譜變化對 answer-format / repetition 敏感,因此**頻率只保留為 negative control,不作 RSN 主讀數**;`p_t` 繼續作為 phasic-like fast-residual measure(operational 命名保留)——當前 task 支持 amplitude change,但未檢測到穩定的 frequency organization,亦未建立 biological phasic dopamine correspondence。這些結果也不支持「`s_t` 越高越正確」、不建立 α 的單調個案規律,且不能把第二個 marker 或 repeated-ngram tail proxy 當作經獨立驗證的真實 loop onset。

### 4.8 Slow-State Behavioral Validation

§4.2–4.4 主要以 `s_t` 的**水平（level）**描述模型是否仍處於持續推理與未完成提交的狀態。本節進一步檢驗另一項預測：如果 `s_t` 的**斜率（slope）**代表 ramping / vigor，較陡的上升是否應對應更快的答案提交。分析使用固定 early window 與尚未提交的 at-risk 樣本，以避免斜率和提交位置產生機械耦合；完整方法與統計規格見 `CLAUDE.md`。

結果清楚區分了 **state level** 與 **state slope**。`s_t` level 穩定關聯 commitment timing：水平越高，模型通常維持推理越久、提交越晚（ρ=**+0.379**），而且這項關係在回歸與 held-out questions 上均可重現。相反，slope 與提交時間的直接關係接近零（ρ=**−0.020**）；控制 level 後出現的 slope 效應方向反而是「斜率越正，提交越晚」，不符合「斜率越陡、推進越快」的 vigor 預測。premature commitment 的 slope 分析因多數樣本已在測量窗內提交，只保留為診斷，不作主要證據。

因此，GSM8K 支持把 `s_t` **level** 解讀為 ongoing engagement / commitment state 的 readout，但**未檢出 slope-based vigor evidence**。這不否定 ramping / vigor 的建模假說；GSM8K 缺乏逐步逼近獎勵的任務結構，更適合在 effort、betting 或 agentic progression 等任務中繼續檢驗。換言之，當前已驗證的是 **slow-state level 的行為意義**，而 slope 是否能表徵 vigor 仍是 open question。

## 5. Qwen2.5 Cross-Model Analysis

本節記錄 Qwen2.5-7B-Instruct 在 GSM8K、neutral 條件下，以 §4 的一維 state 分析鏈所得的結果。**所有結論限於一維投影層次；manifold 分析尚未完成**，於 §5.7 列為 open。

> **一句話結論：Qwen 的入口增益持續隨 α 線性增加，但進入 decode 後，回應沿一個相對固定的 RSN layer profile 被顯著壓縮；現有證據更支持「標量增益壓縮」，不支持「軌跡發生幾何重分配」。**

（標題用 *Analysis* 而非 *Replication*：分析鏈完整移植且入口線性複製成功，但**行為曲線並未複製**——見 §5.7。）

**跨模型比較的口徑限制（貫穿本節，不可放寬）。** Llama 與 Qwen 使用**不同的 mask、不同的 band（L=9 vs L=6）、不同的 activation scale**，因此相同數值的 α **不是相同強度的 intervention**。本節與 §4 的表格**分開呈現，不合併**；不得直接比較 raw α、raw projection 或 p 值。可比較的是**行為狀態的形狀**（commit position、response profile 的共線性、dose 曲線的飽和位置），不是絕對量。

### 5.1 協議差異

| 項目 | Llama3-8B（§4） | Qwen2.5-7B |
|---|---|---|
| band | `[11,20)`，L=9 | `[16,22)`，L=6 |
| mask | `nmd_0.5_11_20_8B.npy` | `nmd_0.5_16_22_7B.npy` |
| max_new_tokens | 512 | 768 |
| reference μ/σ | 該模型自身 α=0 No-CoT prefill | 同左（**模型內 reference**，不共用） |
| α 覆蓋 | −8…+8（9 檔） | −8…+12（11 檔，No-CoT）+ CoT {0,+6} |

band 位置是 per-model 的 mask 事實（Qwen 取 layer-wise Expert/Non-Expert Pearson 下降起點），不是可調參數。reference 固定在各自模型的 α=0 prefill，這是兩套 Z 座標不可互換的直接原因。

### 5.2 Task-entry gain 保持線性

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 | +12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `G_prefill` | −13.14 | −9.82 | −6.53 | −3.26 | 0.00 | +3.27 | +6.57 | +9.88 | +13.25 | +16.74 | +20.26 |
| `Z_prefill` | −32.44 | −24.23 | −16.11 | −8.03 | 0.00 | +8.05 | +16.19 | +24.35 | +32.68 | +41.33 | +50.06 |

`G_prefill ~ α` 線性 **R²=0.99987**（slope 1.661），`Z_prefill` **R²=0.99985**。與 §4.4 相同的 manipulation check 在 Qwen 上成立，且**線性一路維持到 +12**，即入口增益在行為與 commitment 都已飽和之後仍未彎折。

> **slope 1.661 與 Llama 的 1.648 接近純屬巧合，不得引用為跨模型一致性。** 兩者的 slope 由各自 mask 的 ‖m_l‖² 與層數決定，是不同量綱下的數字。

> 圖：`fig51_entry_gain.png`（左 §5.2 入口線性、中 accuracy 飽和、右 commit timing 與 pre-span 覆蓋率）

### 5.3 Accuracy 在 +8 後趨平，而 commitment 繼續延後

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 | +12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc(300) | 62.00 | 65.33 | 68.00 | 68.00 | 67.67 | 68.33 | 73.67 | 77.67 | 86.00 | 88.00 | 87.67 |
| commit c_med（絕對） | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 47 | 110 | 134 | 163 |
| `posN_med`（歸一化） | .010 | .010 | .010 | .010 | .010 | .010 | .010 | .190 | .828 | .839 | .854 |
| decode 長度 | 386.5 | 377.4 | 388.4 | 392.0 | 397.4 | 381.9 | 377.6 | 304.5 | 258.9 | 269.5 | 279.4 |
| pre-span ≥20 佔比 | 8.3% | 6.0% | 4.3% | 4.3% | 5.7% | 6.0% | 9.3% | 58.0% | 96.0% | 97.0% | 96.7% |

> **「趨平」只適用於 accuracy 與歸一化 commit 位置，不適用於絕對 commit step。** `posN_med` 在 +8 後停在 .828/.839/.854，而絕對 `c_med` 仍持續右移 110→134→163——因為生成長度同時從 258.9 拉長到 279.4。兩者不是矛盾而是不同量：**commitment 在生成中的相對位置飽和，但絕對延後仍在繼續**。任何「commit timing 趨平」的說法都必須指明用的是哪一個。

兩點與 Llama 明顯不同。**其一，Qwen 沒有右臂**：accuracy 在 −8…+12 內單調上升後**飽和**（+8→+10→+12 兩兩 n.s.），而 Llama 的最佳點在 α=−6 且兩側下降。**其二，Qwen 在低 α 是「先答後推」**：c_med=3 表示答案幾乎在生成開頭就出現，因此 α≤+4 的 pre-commit window 對 92–96% 的樣本**未定義**。

> **這使 α≤+4 的 cell 成為 coverage row，不是可比對照。** 後續所有 pre-commit 分析的 cohort 都**選擇在 manipulation 的結果上**（pre-span ≥20），這是設計上無法迴避的限制，必須隨數字一起陳述。

### 5.4 延遲提交子集上的 decode-response compression（非 ceiling）

固定 cohort：在 +6/+8/+10/+12 **全部**四檔都具備 ≥20 步 pre-commit span 的題目，n=167。

| α | `x_prefill` | `s_pre` | `Z_prefill` | `Z_pre` | c_med |
|---:|---:|---:|---:|---:|---:|
| +6 | 134.98 | 2.342 | 24.41 | 1.417 | 93 |
| +8 | 183.26 | 2.973 | 32.74 | 1.641 | 125 |
| +10 | 235.37 | 3.167 | 41.39 | 1.673 | 163 |
| +12 | 288.01 | 3.250 | 50.12 | 1.752 | 187 |

paired（同題、固定 cohort）：`+6→+8` p=0.00114、`+8→+10` p=0.0147、`+10→+12` p=0.361。入口每 +2α 遞增 ~50 個 raw 單位而 decode 側只遞增 0.63 → 0.19 → 0.08，**RAW response ratio 單調下降 0.0131 / 0.0037 / 0.0016**。

> **口徑三則。**（a）headline 用 **RAW**：Z ratio 為 0.0269/0.0037/0.0091，**非單調**，只作 sensitivity。（b）`s_t` 一律由 `x_decode` **decode-seeded** 重算，禁用 stored `ema_decode`（其 prefill 汙染隨 α 增長，正好落在 pre-commit window）。（c）commit 位置以 **tokenizer offset mapping** 定位，非字元比例。
>
> **這是 compression，不是 ceiling。** 一個「已到頂」的解釋要求 decode 側停止回應；此處 `+6→+8`、`+8→+10` 仍顯著，只是每單位 α 的回應變小。

> 圖：`fig52_compression.png`（左 entry、中 decode、右 RAW response ratio）

### 5.5 高低劑量 response profile 近乎共線

以 `[c−20, c)` 的逐層 pre-commit `s_t`，取每 +2α 的逐層回應向量：

| step | L15 | L16 | L17 | L18 | L19 | L20 | mean | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +6→+8 | 0.5374 | 0.3713 | 0.1690 | 0.2367 | 0.5221 | **−0.1986** | 0.2730 | 0.92 |
| +8→+12 | 0.1829 | 0.0899 | 0.0344 | 0.0528 | 0.1688 | **−0.0814** | 0.0746 | 1.19 |

最小二乘 $v_{high} \approx k\,v_{low}$：**k = 0.309**（幅度每劑量步縮小 3.24×），最佳縮放解釋 **97.4% 的 response energy**，歸一化**平方**殘差 **2.6%**（殘差**範數**比例 16.1%），**cos = 0.987**。

> **單位必須寫明**：「殘差 2.6%」是能量（平方）比，不是幅度差。另外對最小二乘的 k，$\text{resid} = 1 - \cos^{2}$ **恆等**（實測差 1e−16），故 cos 與平方殘差是同一幾何事實，**只有 k 提供獨立資訊**（幅度壓縮）。
>
> **逐層異質與 L20 反號本身不是重分配證據**：固定的負 loading 正是「單一潛在標量經固定異質 profile 投影」會產生的形態。因此本節結論為：**排除層間同步的均勻 ceiling；未見明顯方向旋轉；仍與沿固定 layer profile 的標量增益壓縮相容。**

> 圖：`fig53_profile.png`（左逐層回應、中 k·v_low 疊合、右兩劑量步散點共線）

### 5.6 Random / orthogonal readout controls

將**同一批** hidden states 重投影到三個 null family（各 10 draw，共 30）：

| | RSN | `diff_random` | `ortho_gauss_same` | `ortho_gauss_off` |
|---|---:|---:|---:|---:|
| resid（能量比） | **0.026** | 0.341 | 0.102 | 0.227 |
| pctile / p | — | 0.0% / .182 | 0.0% / .182 | 20.0% / .545 |
| CV(+6→+8) | **0.917** | 2.373 | 3.943 | 2.936 |
| k | 0.309 | 0.250 | 0.358 | 0.240 |

**RSN 的殘差與 CV 都低於全部三個 family**：RSN 方向比一般稀疏方向**更像干淨的標量通道**。這是**反對**把 §5.5 讀成重分配的證據，不是支持。

> **兩則不得跳過的限定。**（a）**§5.5 的「CV 0.92 偏高」在有了參照分布後不成立**——一般方向的 CV 是 2.4–3.9，RSN 屬低端，逐層異質性不構成特殊性證據。（b）**低 SNR caveat 只適用於三個 family 中的兩個**。逐 draw 回應幅度 `‖v₆₈‖` 中位數：RSN 0.907、`diff_random` 0.270、`ortho_gauss_same` **0.752**、`ortho_gauss_off` 0.349。對 `diff_random` 與 `ortho_off`（RSN 的 30–38%）caveat 成立——對近噪聲擬合 k 天然留下大殘差。但 **`ortho_gauss_same` 具備 RSN 83% 的回應幅度,殘差仍是 RSN 的約 4 倍（0.102 vs 0.026）**,且它是**配對最嚴格的 null**（NMD 自身 support、權重 ⊥ role-diff、norm-matched），因此無法用低 SNR 解釋掉,結論主要由它承載。措辭仍維持 **argues against** 而非 excludes（N=10、p 下限 0.182、三 family 非獨立）；`pctile 0.0%` 亦**不可**轉寫為「RSN 特殊」——percentile 只說明 RSN 在極端，而此處極端指向非預期方向。
>
> **邊界（binding）**：這批 hidden states 是在 RSN steering 下生成的，重投影因此界定的是 **readout**，不是 intervention。**steering direction 的因果特異性主張仍需另行注入 random/orthogonal 方向並重新採集。** 各 family N=10 → 雙邊 p 下限 0.182；三個 family 共享同一批 hidden states、cohort 與 reference，**非獨立重複**，「三者一致」不構成低機率論證。

> 圖：`fig54_null.png`（左殘差、中 CV、右逐層回應與各 family 幅度）

### 5.7 目前解釋與 open item

**Qwen 複用了 §4 的完整分析鏈，但「複用鏈條」與「複製結果」必須分開陳述。** 五個環節（§5.2 入口 manipulation check、§5.3 行為曲線、§5.4 commit-aligned slow state、§5.5 逐層 response profile、§5.6 random/orthogonal readout control）都以相同口徑跑通。其中**真正複製的是入口線性與 entry–decode 解耦**：`G_prefill ~ α` 在兩個模型上都近乎完美線性，且都在 decode 側轉為非線性。**行為曲線則沒有複製**——Llama 是倒 U（最佳點 α=−6、兩側下降），Qwen 是高劑量平台（單調上升後於 +8 飽和，band 內無右臂）。因此本節是**分析框架的跨模型可移植性 + 部分機制複製**，不是行為結果的複製。

**當前最合適的一維層次結論：高劑量下 decode response 幅度明顯壓縮，逐層 loading 高度異質且含 L20 反號，但兩個劑量區間的 response profile 近乎共線（cos=0.987、k=0.309、97.4% energy），且此近共線性比一般方向更明顯——因此結果與「沿固定 RSN profile 的標量增益壓縮」相容，不支持層間同步的均勻 ceiling，亦未顯示幾何重分配。**

**Open（下一小節）：manifold 分析。** 一維投影無法區分「軌跡等距移動但轉離 mask 方向」與「軌跡本身壓縮」；§5.6 的結果使**標量增益成為 manifold 分析要擊敗的假設**，而非待排除的形式選項。待補指標：PCA 譜 / participation ratio、重建誤差、trajectory speed 與 curvature、tangent alignment。另兩項 open：**−8 與 CoT 條件**尚未納入本節逐層分析；與 Llama 的對齊須以 **working state** 而非 raw α 進行。

離線腳本（`RoleAnswer/qwen_signal/`，`python3.10`）：`commit_aligned.py`（§5.3–5.4）、`hs_layerwise.py`（§5.5）、`hs_null_specificity.py`（§5.6）、`plot_section5.py`（全部圖）；凍結記錄 `entry_gain_RESULT.txt`、`commit_aligned_v3_RESULT.txt`、`hs_layerwise_RESULT.txt`、`hs_null_specificity_RESULT.txt`。伺服器端 `check_hs_qwen25.py`（H5 驗收）、`run_null_remask_qwen25.sh`（null 重投影）。

| 圖 | 內容 |
|---|---|
| `fig51_entry_gain.png` | §5.2 入口 gain 線性 + §5.3 accuracy 飽和 / commit timing |
| `fig52_compression.png` | §5.4 entry vs decode 回應與 RAW response ratio |
| `fig53_profile.png` | §5.5 逐層 profile + scalar-compression 擬合 |
| `fig54_null.png` | §5.6 RSN vs 三個 null family |

**圖檔位於 `qwen2.5/dopamine/plots_gain/`，與 Llama 的 `llama3/dopamine/plots_gain/` 分開存放**——兩模型數值不可比，共用目錄是跨模型混用的起點。每個 panel 都由產生凍結文字記錄的同一段程式重新推導，不從表格硬編數字。

## 6. Conclusion

本研究辨識出一組可調節 LLM reasoning state 的 **Role-Sensitive Neurons (RSNs)**。它們主要反映 task engagement、action readiness 與 commitment dynamics，而不是直接儲存知識或提升推理 capacity。觀察結果顯示，CoT、Persona 與 answer commitment 會以不同的時間模式調制 RSN state；其中 CoT 主要提高 pre-commit engagement，Persona 主要重組 task-entry、commitment formation 與 post-commit release 的時間分配。

更重要的是，沿既定 RSN/NMD 方向施加 α steering，可近乎線性地控制 task-entry gain，並進一步產生非線性的 commitment state、output decisiveness 與 behavioral working point。極端負向 steering 造成 commitment-formation collapse，過高正向 steering 則伴隨較差的 commitment state 與行為表現，而中等負向範圍形成較佳工作點。CoT 與 α=−4 的單劑量分析進一步顯示，α 主要控制 generation boundary，CoT 則主要重塑後續 decode dynamics，兩者具有不同的時間重心並可大致疊加。

因此，目前最合適的結論是：**RSNs constitute a controllable latent gain mechanism that functions as a computational analogue of dopaminergic adaptive calibration in LLMs.** 這些 neurons 能以 task-dependent、dose-dependent 的方式調節模型的投入、推進、承諾與停止，呈現與 dopamine-related wanting、vigor 和 optimal-level calibration 相容的功能結構。但此結論屬於 **computational and behavioral analogy**：α 不等同生物多巴胺濃度，`G_prefill`、`s_t` 與 `p_t` 也尚不能直接等同 tonic、ramping 與 phasic dopamine。就 `p_t` 而言,tonic/ramping/phasic 的研究框架保留,但證據邊界須明確:其 **amplitude / dispersion 驗證得到支持**（§4.7:α=−6 在乾淨 pre-commit 段提高 centered RMS，pre-commit residual dispersion 隨 α 變化）,而 **frequency organization 暫未得到支持**（頻率指標不隨 α 穩定變化,commit-centered 的頻譜變化對 answer-format / repetition 敏感）。因此 `p_t` 目前是 **candidate phasic-like signal**,而非已識別的 biological phasic dopamine。就 `s_t` 而言，證據呈現清楚的**強／弱分佈**：**task-entry gain 強**（`G_prefill` 隨 α 近線性）、**slow-state level 與 release 強**（level 呈倒 U 追蹤 acc、穩定關聯 commitment timing，release 隨 commit 快速下降），**但 GSM8K 中 slope-based vigor evidence 未檢出**（§4.8：leak-free at-risk 下 slope↔timing 為 null，控制 level 後僅剩一個方向與 vigor 相反的 suppressor 殘差）。因此 **ramping/vigor 的建模定義保留**，但其斜率預測在本任務未兌現，留待 effort / betting / agentic progression 等能誘發漸進逼近結構的 task 檢驗——已驗證的 `s_t` 經驗內容是 **slow engagement / commitment-state readout（level）**，vigor（slope）為 open 的 task-level 問題。

方向特異性已由**三個 null family** 收緊（§4.6：support-selection `diff_random` N=11 + generic-direction `ortho_gauss_same`/`off` 各 N=10）。兩層結論：task-entry raw gain（`G_prefill`）遠強於 null 但屬 co-design / manipulation-check；而 **commitment-locked 的 `s_t` / `p_t` 時間軌跡提供目前最強的 NMD direction-specificity evidence**——其 commit 前後帶符號的結構化走向（幅度/水平）穩定超出全部三個 family（30 個 primary cell 中 NMD 皆相對各自 null 保持極端 pctile，null median 皆 ≈0），且在注入 α 與自然 state（CoT / Persona，其中 CoT 最獨立）上一致，故**不能僅由 α-steering 的 injection–projection identity 解釋**。**off-support generic-direction null 這格最吃重**：把權重去掉 role-diff、位置移出 NMD 後 NMD 仍獨佔極端——與 ① 對照後，證據指向 **top-|diff| 支撐與 role-diff-aligned 權重的特定組合**是特異性來源，排除了「僅 top-|diff| 支撐」與「僅複用 role-diff 逐坐標權重」兩種單成分解釋。限定：各 family N=10–11 為 exploratory ordering（三 family/30 cell 共享同一批 hidden states 與指標，非獨立重複，不作正式顯著性宣稱），僅 Llama3-8B、僅 offline re-projection，是否另有獨立於幅度的形狀差異仍待定。後續優先跨模型、跨任務與含 random-direction 因果 steering 的對照，確認這套機制的普遍性及其與其他 latent control directions 的區別。
