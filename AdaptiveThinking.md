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

这里需要区分指标定义与任务结果：`s_t` slope 作为 ramping/vigor 的 operational measure 保持不变，但特定任务是否呈现预期的 slope–behavior relationship，需要通过实验检验。GSM8K 的结果见 §4.7：该任务没有检出符合预期方向的 slope–vigor evidence；相比之下，`s_t` level 与 commitment timing 呈现稳定关联。

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

對 token `t`、middle layer `l`，先計算原始投影：$r_{t,l} = h_{t,l} \cdot m_l$，其中 $h_{t,l}$ 是 decoder layer output hidden state，$m_l$ 是同一 output space 中的 sparse NMD direction。每層先獨立投影，再進行跨層聚合。

為了固定零點、保留 α 的干預單位並避免少數 layer 因尺度較大而主導聚合，使用 neutral、α=0、No-CoT 的 prefill distribution 作為 reference：

$$\mu_l^{ref} = \mathbb{E}\big[r_{prefill,l} \mid \text{neutral},\ \alpha=0,\ \text{No-CoT}\big]$$

$$g_{t,l} = \frac{r_{t,l} - \mu_l^{ref}}{\lVert m_l \rVert^{2}}$$

$$\sigma_l^{ref} = \operatorname{Std}\big[g_{prefill,l} \mid \text{neutral},\ \alpha=0,\ \text{No-CoT}\big]$$

$$z_{t,l} = \frac{g_{t,l}}{\sigma_l^{ref}}$$

由此得到兩種跨層 readout：$G_t = \operatorname*{mean}_{l}\, g_{t,l} \qquad\qquad Z_t = \operatorname*{mean}_{l}\, z_{t,l}$

| Signal | Role |
|---|---|
| $G_t$ | 保留 α-equivalent unit；用於 steering calibration、dose linearity 與 intervention sanity check |
| $Z_t$ | 各層先標準化後聚合；作為主要 observational trajectory，避免 layer-scale dominance |

Reference $μ_l^{ref}$ 與 $σ_l^{ref}$ 在所有 role、α、token 與 event 中固定，不隨條件重新估計，才能保留條件間與 generation 階段間的真實差異。

**坐標分工**：涉及 α 單位的定量陳述（dose linearity、`boundary_jump` 回彈比例、steering calibration，即 H1）一律在 $G$ 坐標；涉及 trajectory 形狀的分析（$s_t$ slope、$p_t$ residual，即 H2/H3）在 $Z$ 坐標，以確保 layer-fair、不被個別大尺度 layer 主導。

### 3.3 Temporal Signal Decomposition

#### 3.3.1 Task-Entry Tonic

最後一個 prompt token 的 gain 定義為：$T = G_{prefill}$，`T` 是 task-entry tonic 的主 readout，表示 generation boundary 上的初始 gain / commitment set point。`Z_prefill` 可作為 layer-fair 的 condition comparison；`G_prefill` 則保留 α 單位，作為主要 calibration signal。

prefill 到第一個 decode token 的回彈另記為：$\text{boundary\_jump} = G_0 - G_{prefill}$，`boundary_jump` 用來描述 task-entry pulse 如何進入自然 decode dynamics；在 `G` 坐標計算，以與 H1 引用的 α-單位回彈比例（約 95%）保持一致。

#### 3.3.2 Ramping / Vigor

在 decode 內，對 `Z_t` 建立 slow component：
$$s_0 = Z_0 \qquad\qquad s_t = \beta\, s_{t-1} + (1-\beta)\, Z_t$$
`s_t` 只由 decode token 初始化與更新，不以 prefill 作 EMA seed。

主要 readout 是 `s_t` 的 trajectory slope，而不是其絕對高度：$\text{vigor\_slope} = \operatorname{slope}(s_t)$

後續可分別估計 early、middle、late slope；β 與 window 的精確設定留待 sensitivity analysis。

#### 3.3.3 Fast Residual（candidate phasic-like）

decode-time fast component 定義為相對上一時刻 slow baseline 的 residual：

$$p_t = Z_t - s_{t-1}, \qquad t \geq 1$$

`p_t` 是一個 **EMA high-pass 殘差**（當前 `Z_t` 減上一步慢 EMA），不是 event-locked phasic 信號。`p_t > 0` 表示瞬時高於 slow baseline 的 pulse，`p_t < 0` 表示瞬時 dip。現階段把它當作 **fast residual / candidate phasic-like component**，優先檢驗其 amplitude、variability 與時間結構；之後再分析它是否與特定 reasoning / commitment event 對齊。

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

**signal vs gsm8k original —— Bias Verification**：

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

**口徑:** 本表及 §4 全節的 `signal` 只用於**同批 signal–behavior alignment**；**production accuracy 以 [AdaDopamine_gsm8k.md](AdaDopamine_gsm8k.md) 為準**。兩套數值不可混算,但 **dose 形狀一致、離散最佳點均在 α=−6**——alignment 依靠的是形狀,不是絕對水平。（兩批的機器/batch-size 差異來源與診斷見 `CLAUDE.md`。）

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


将每个 sample 按自身的 commit step 对齐（`t=0`）后，可以观察到三个主要模式：

1. **Pre-commit slow state**：correct 组在 commit 前维持更高、更持久的 `s_t`，说明组间差异不只是 commit timing 错位。但这是描述性关联，不能解释为高 `s_t` 导致正确；题目可解性也可能同时影响 engagement、commit timing 与正确率。

2. **Commit-centered transition**：`Z_t`、`s_t`、`p_t` 及 entropy、top1、margin 均在 commit 附近快速变化，说明 commit 对应明显的 generation-state transition。不过，`####` 也会改变输出格式与 token distribution，因此 confidence proxies 的变化可能包含格式效应。

3. **Post-commit separation**：commit 后，correct 组的 `s_t` 下降更快，并在约 10–15 tokens 后低于 incorrect 组；其 confidence recovery 也较明显。这可能反映 correct responses 具有更快的 state release / termination dynamics，但目前仍属于描述性结果。

| Signal | Main observation | Interpretation |
|---|---|---|
| `s_t` / `Z_t` | Correct 组在 commit 前整体较高，commit 后下降更快、更深 | Slow RSN dynamics 与回答结果存在描述性关联，但不能据此推断因果关系 |
| `p_t` | Commit 附近出现明显 pulse / dip，但 commit 前不能稳定区分 correct 与 incorrect | 主要反映 generation-state transition，不能稳定预测正确率，也不作 RPE 解读 |
| Entropy / top1 / margin | Commit 附近共同表现出 uncertainty 上升、output decisiveness 降低 | Commit 对应明显的 output-distribution transition，但其中可能包含 `####` 引起的格式效应 |
| Information gain | Commit 附近出现波动，但 correct 与 incorrect 的差异不稳定 | 仅作为 output distribution 发生变化的辅助诊断，不作为主要结果 |

**Overall.** Correct 与 incorrect responses 的主要差异出现在 commit 前后的 slow RSN dynamics。Correct 组在 commit 前维持更高、更持久的 `s_t`，并在 commit 后表现出更快的 state release。相比之下，task-entry gain、局部 `p_t` amplitude 及 confidence metrics 均不能稳定区分回答是否正确。

这一结果说明，`s_t` trajectory 与 viable reasoning、ongoing engagement 和 termination dynamics 存在描述性关联，但不能据此判断因果方向。目前至少有三种可能：

1. 更高的 engagement 有助于模型答对；
2. 模型先识别出可行的解题路径，因而维持更高 engagement；
3. 题目难度或熟悉度同时影响 engagement 与正确率。

因此，本节支持 **RSN slow state tracks engagement during viable reasoning**，但不能单独证明更高的 RSN state 会提高正确率。相关的因果证据仍需来自 α intervention、dose-response，以及 difficulty-matched analysis。

还需要保留两项限制：

- Commit 附近的 logit与 confidence 变化可能部分由 `####` 和 answer-format transition 引起；
- 不同图是同一批数据的不同对齐视角，彼此互补，但不能视为相互独立的重复证据。

### 4.2 CoT vs No-CoT

#### Scope and Shared Analysis Framework

本节比较 **neutral、α=0、相同 300 道 GSM8K** 的 No-CoT 与 CoT。两组仅相差一句 `Let's think step by step.`，其余生成设置一致。统计采用 paired mean difference、bootstrap 95% CI、Cohen’s `d_z` 和 Wilcoxon signed-rank；轨迹图的 bootstrap bands 仅作描述性展示。

RSN 读数包括 task-entry gain（`G_prefill` / `Z_prefill`）、slow state（`s_t`）和 fast residual（`p_t`），均来自中层 sparse NMD projection。Entropy、top1、margin、info gain 和 rolling variability 来自最终层的完整输出分布，仅作为独立的 **output-decisiveness controls**，不与 RSN projection 视为同一信号，也不直接表示最终答案的 epistemic confidence。

由于 97% 的样本达到 767-token 上限，且 commit 后常退化为 `#### N ...` 重复，decode 统一按前两个 answer markers 划分：

| Stage | Definition | Role in analysis |
|---|---|---|
| **pre-commit** | decode start → 第一个 `####`（C1） | 答案形成阶段，正文主分析 |
| **post-commit** | C1 → 第二个 `####`（C2） | 首次提交后的延续生成 |
| **post-second-marker tail** | C2 → generation end | stopping failure / repetition 诊断 |
| **full decode** | 上述三段之和 | 仅展示 post-commit tail 对整体轨迹的污染 |

C2 只是 **second-answer-marker proxy**，不能直接等同真实 loop onset；C2-based 结果也只适用于存在第二个 marker 的 loop-prone subset。`s_t=EMA(Z_t)` 在完整 decode 上连续计算一次，`p_t=Z_t-s_{t-1}` 由同一轨迹得到，各阶段不重新初始化。C1/C2-centered plots 只是放大阶段边界，不构成另一套切分。

#### Result 1 — CoT Raises Task-Entry Gain

CoT 在 decode 开始前已经提高 generation-boundary RSN state：

| Readout | No-CoT reference | CoT | Paired ΔCoT−No | 95% CI | `d_z` | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| `G_prefill` | 0.000 | 0.071 | **+0.071** | [0.058, 0.085] | **0.59** | 1.9e-19 |
| `Z_prefill` | 0.000 | 0.158 | **+0.158** | [0.138, 0.178] | **0.89** | 1.1e-32 |
| `boundary_jump_G` | 0.167 | 0.094 | −0.073 | [−0.138, −0.006] | −0.13 | 2.4e-02 |

No-CoT 的零值来自 reference definition，并不表示没有 RSN activation。CoT 在 α-unit 和 layer-standardized 坐标中都提高了 task-entry gain；但到 `G_decode[0]` 时两组已迅速接近（均约 0.17），说明该差异没有作为固定 offset 持续存在。`G_prefill` 因此是 **task-entry tonic-like readout**：它表明 CoT 改变了进入生成阶段时的状态，但不能证明后续 dynamics 完全由这一单点造成，因为 CoT instruction 仍通过完整 prompt、KV cache 和早期 token selection 影响生成。

#### Result 2 — CoT Sustains Pre-Commit Slow State and Changes Fast Residual Dispersion

下表将 `s_t` 与 `p_t` 放在同一阶段框架中。`p_t` 的主读数为 `abs_mean` 和 `std`；peak metrics 受 segment length 与 EMA lag 影响，不作主证据。

| Stage | Slow RSN `s_t` | Fast residual `p_t` | Main interpretation |
|---|---|---|---|
| **pre-commit** (n=281) | mean Δ=**+0.471**, `d_z`=**0.65**；`end−start` Δ=+0.228, `d_z`=0.27 | `abs_mean` Δ=+0.056, `d_z`=0.42；`std` Δ=+0.092, `d_z`=**0.63** | CoT 维持更高、较少 relaxation 的 answer-formation state，并提高 fast residual dispersion |
| **post-commit** (n=144) | `end−start` Δ=**−0.166**, `d_z`=−0.31 | `abs_mean` / `std` 小幅反转（`d_z`=−0.19 / −0.18） | CoT 在首次提交后 release 更快，fast variability 差异减弱 |
| **post-second-marker tail** (n=141) | mean Δ=+0.185, `d_z`=0.32；tail 短 **83.6 tokens**, `d_z`=−0.51 | `std` Δ=−0.083, `d_z`=−0.33 | CoT 的诊断性尾段更短、churn 更低；不等同真实 loop onset |

CoT 的 slow-state effect 以 **level difference** 为主，但不是整条曲线的简单平移：No-CoT 在答案形成期间 relaxation 更明显。Strict-`####` subset（n=233）和 absolute-token control 均重现这一趋势，说明结果不依赖 fallback locator，也不是单由长度归一化制造。

在 C1 附近，两组的 `s_t` 都快速下降，`p_t` 同时出现负向 transition；C2 附近也出现相似结构。由于两处都伴随 `####` 和 answer-format change，目前只能将其解释为 **marker-locked、stage-dependent fast RSN dynamics**，不能确认是独立的 commitment-specific phasic response。GSM8K 没有 reward feedback，因此不作 RPE 解读。

Full-decode trajectory 会被重复尾段反转，只能用于显示 loop contamination，不能代表正常 reasoning 的 late stage。

#### Result 3 — CoT Improves Pre-Commit Output Decisiveness

Confidence controls 使用与 RSN 相同的阶段边界。主分析限于两组均有有效 commit 的 paired subset：pre-commit n=203；post-commit 和 tail 因要求有效 C2，分别为 n=144 / 141。表中数值为 ΔCoT−No-CoT，括号内为 `d_z`。

| Metric | pre-commit（可判读） | post-commit（饱和诊断） | post-second-marker tail（饱和诊断） |
|---|---:|---:|---:|
| entropy | **−0.176** (−0.83)*** | −0.052 (−0.06) ns | +0.074 (0.10)*** |
| top1 | **+0.039** (0.74)*** | +0.019 (0.09) ns | −0.014 (−0.10)*** |
| margin | **+0.048** (0.66)*** | +0.024 (0.08) ns | −0.018 (−0.11)*** |
| info_gain | −0.005 (−0.07)*** | −0.240 (−0.37)*** | −0.032 (−0.11)*** |
| roll_std | **−0.023** (−0.63)*** | −0.058 (−0.51)*** | +0.002 (0.04)*** |

可稳定解释的 confidence result 仅出现在 pre-commit：CoT 降低 entropy，并提高 top1 和 margin（`|d_z|=0.66–0.83`），说明答案形成期间的 output distribution 更 decisive。进入 post-commit 后，degenerate loop 使 top1≈0.98、entropy≈0.12，confidence proxies 已缺乏 dynamic range；此后的星号主要反映 saturation 附近的细微变化，不作实质 confidence 解读。Wanting 在该阶段仍可测，而 confidence proxy 已饱和，这是一项 measurement contrast，但不能单独证明两者构成 dissociation。

#### Integrated Interpretation

CoT 的内部效应可以概括为：

> **higher task-entry gain → sustained pre-commit engagement and output decisiveness → faster post-commit state release**

- **Task entry：** CoT 在 generation boundary 前提高 `G_prefill`（`d_z`=0.59）和 `Z_prefill`（`d_z`=0.89）。
- **Answer formation：** CoT 维持更高的 pre-commit `s_t`，提高 fast residual dispersion，并使输出分布更 decisive。
- **Commitment and release：** C1 后两组均进入 slow-state decline，CoT release 更快；C1/C2 的 `p_t` transition 仍可能包含 marker-format effect。
- **Stopping failure：** CoT 的 post-second-marker tail 更短、residual variability 更低，但这只是 stopping diagnostic，不代表正常 reasoning 或 dopamine level。
- **Evidence boundary：** CoT 同时调节 RSN engagement 与 output decisiveness，但这既不证明两者是同一构念，也不构成 causal dissociation。RSN gain 的直接因果证据仍来自 α intervention 和 dose-response；这些结果支持的是 computational analogy，而不是 biological dopamine mechanism。

#### Figures

| Figure | Role |
|---|---|
| `fig42_main_2x2.png` | C1-centered `s_t` / `p_t` / entropy 与 pre-commit effect-size summary |
| `fig42_5panel_s_t.png` | C1/C2-centered 与三阶段 lifecycle 的 slow RSN dynamics |
| `fig42_5panel_p_t.png` | 同一时间框架下的 fast residual dispersion 与 marker transition |
| `fig42_5panel_entropy.png` / `fig42_5panel_top1.png` | 独立 output-distribution controls |
| `fig42_B_lifecycle.png` | C2 / full-decode loop contamination diagnostic |
| `fig42_step2_shape_test.png` | Pre-commit level-versus-shape robustness check |

### 4.3 Persona

#### Scope and Shared Analysis Framework

本節比較 **Expert 與 Non-Expert persona**（α=0、No-CoT、相同 300 道 GSM8K、依題目配對），所有差值均定義為 Expert − Non-Expert。分析沿用 §4.2 的 task-entry、commit-centered RSN 與 output-confidence readouts；C1 定義為首個 `####`，缺失時使用首個 answer candidate 作 fallback。

Persona 是 prompt manipulation，而不是 α intervention。因此，本節回答的是 persona 是否自然改變 RSN state，不能單獨提供 steering 的因果證據。整體結果顯示，Persona 並非造成持續一致的高低位移，而是改變 RSN state 在 **task entry、答案形成與提交後釋放**三個階段的時間分布。

#### Result 1 — Expert Strongly Raises Task-Entry RSN Gain

| Prefill readout | Δ Expert−Non-Expert | `d_z` | Interpretation |
|---|---:|---:|---|
| `G_prefill` | +0.161 | **+2.81** | Expert 的 task-entry gain 大幅升高 |
| `Z_prefill` | +0.284 | **+3.17** | layer-standardized 結果一致 |
| `boundary_jump_Z` | −0.265 | −0.54 | 進入 decode 後差距迅速縮小；與 `Z_prefill` 代數耦合，不作獨立證據 |
| entropy | +0.052 | +0.18 | Expert 的不確定性略高 |
| top1 | −0.010 | −0.15 | Expert 的 output decisiveness 略低 |
| margin | −0.010 | −0.12 | 弱反向差異 |

Expert 在最後一個 prompt token 上呈現很大的 RSN gain effect（`d_z≈3`），但同一時點的 confidence-proxy 差異很弱且方向相反（`|d_z|≤0.18`）。因此，task-entry RSN gain 不能簡化為 output decisiveness。不過，NMD mask 本身源自 MMLU Expert−Non-Expert contrast，這項結果主要是跨任務 manipulation check，而非完全獨立的方向驗證。

#### Result 2 — Persona Reorganizes Commitment-Period RSN Dynamics

C1 可定位的樣本為 Expert 219/300、Non-Expert 236/300；其中共同有效 194 題、Expert-only 25 題、Non-Expert-only 42 題、兩者皆無 39 題。Non-Expert 的可分析率高 5.7 percentage points（paired exact `p=.0498`），只作行為診斷。為避免兩組有效樣本構成不同，以下比較僅使用 194 道 common-valid questions，並分別對齊各自的 C1：

| Window | Readout | Expert | Non-Expert | Δ Expert−Non-Expert | `d_z` | `p` | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| `[−50,−10]` | `s_t mean` | 0.114 | 0.241 | −0.126 | −0.247 | .023 | 較早的 pre-commit reversal；探索性 |
| `[−20,0]` | `s_t mean` | 0.137 | 0.278 | **−0.141** | **−0.278** | **.0013** | Non-Expert 在答案形成前維持較高 slow state |
| `[0,+20]` | `s_t slope` | −0.046 | −0.056 | **+0.009** | **+0.291** | **.0004** | Non-Expert 在提交後釋放得更快 |
| `[0,+20]` | `p_t abs_mean` | 1.039 | 1.130 | **−0.091** | **−0.248** | **.0016** | Non-Expert 的 release transient 較強 |
| `[−20,0]` | entropy | 0.482 | 0.439 | +0.043 | +0.167 | .010 | Non-Expert 略為更 decisive，效應較弱 |

在 3 windows × 7 readouts 的 21 項探索性比較中，BH-FDR 後最穩定的是 `[−20,0] s_t mean`、`[0,+20] s_t slope` 與 `[0,+20] p_t abs_mean`。較早的 `s_t` reversal 與 entropy effect 僅作輔助。

全 pre-commit 平均接近 null（`s_t mean d_z=−0.10, ns`），因為 Persona effect 集中在 C1 附近，而非形成貫穿整段的固定 offset。其他整段統計也較弱：pre-commit `p_t abs_mean d_z=−0.15`、`p_t std d_z=−0.17`；post-commit 與 loop-tail 的全階段均值大致為 null。

#### Result 3 — Confidence Changes Are Weaker and More Localized

全 pre-commit 區間中，Expert 的 entropy 較高（`d_z=+0.35`），top1 與 margin 較低（`d_z=−0.27 / −0.22`），表示 Non-Expert 整體略為更 decisive。這些效應小於 task-entry RSN gain，且主要集中在 pre-commit；post-commit 與 loop-tail 受到 `#### N #### N` saturation 污染，只作診斷。

Random interior pseudo-event 未重現 C1 附近的 sharp `s_t` transition，支持變化具有事件局部性。但 C2-centered 圖幾乎複製 C1，且兩者都伴隨 `####` 與 answer-format transition，因此目前只能稱為 **answer-marker-centered dynamics**，不能直接解讀為獨立的 biological commitment signal。

#### Integrated Interpretation

Persona 呈現三階段變化：

1. **Task entry：** Expert 的 RSN gain 大幅升高，但 output decisiveness 沒有同步提高。
2. **Commitment formation：** 到答案形成前，排序反轉為 Non-Expert 的 `s_t` 較高。
3. **State release：** 提交後，Non-Expert 的 slow state 下降更快，`p_t` transient 也更強。

這一模式與 Non-Expert 較高的 accuracy（68% vs 58%）相容，但尚未證明 RSN dynamics 中介了正確率差異。要檢驗這條路徑，仍需分析 Expert-Wrong → Non-Expert-Correct 的 discordant items，或進行 gain-matched `role × α` rescue/cancellation experiment。

與 §4.2 相比，CoT 主要同時提高 pre-commit engagement 與 output decisiveness；Persona 則主要重新分配 RSN/wanting dynamics，並伴隨較弱、較局部的 confidence change。這支持 RSN 作為不同於 output confidence 的 dynamic state/gain readout，但 RSN gain 的因果證據仍主要來自 §4.4 的 α dose-response 與行為實驗；這些結果因果操縱的是 RSN gain，而不是生物 dopamine 本身。

### 4.4 α-Steering: Linear Task-Entry Gain and a Nonlinear Behavioral Working Point

#### Scope and Shared Analysis Framework

本節分析 Llama3-8B 在 GSM8K、neutral No-CoT 條件下的 **9 個 α doses（−8 至 +8）**。α 是沿既定 NMD/RSN 方向施加的 **task-entry intervention**；本節先驗證它是否線性控制入口 gain，再檢驗其下游動力學與行為。

分析沿用 §4.2/§4.3 的 gain coordinates 與 commit locator：reference μ/σ 固定為 neutral α=0 No-CoT prefill；dose calibration 使用各 α 全量 300 題；slow/fast/confidence 則以**每個 α 與 α=0 的 common-valid questions 作 paired comparison**（不取全 9 檔共同交集，否則 −8 會壓縮所有 cell）。C1 定義為首個 `####`，缺失時使用 first answer-candidate fallback。

**兩項貫穿全節的限制，後續不再重複。**（i）signal–behavior alignment 使用同批 server-184 inline `correct`；production accuracy 以 [AdaDopamine_gsm8k.md](AdaDopamine_gsm8k.md) 的 server-182 offline first-`####` 為準。兩套不可混算,但 dose shape 與離散最佳點一致。（ii）**−8 的 C1-analyzable subset 明顯減少**（paired n=229 vs 其餘 291–298），故所有 −8 的 event-centered 結果都是**條件子樣本**，須與 C1-analyzable rate 一起解讀。

#### Result 1 — α Linearly Controls Task-Entry RSN Gain

| α | n | acc(184) | `G_prefill` | `Z_prefill` | `boundary_jump_G` |
|---:|---:|---:|---:|---:|---:|
| −8 | 300 | 40.7 | −13.8454 | −26.4543 | +13.0577 |
| **−6** | 300 | **79.7** | −10.2980 | −19.7427 | +10.5488 |
| −4 | 300 | 74.3 | −6.8022 | −13.1072 | +6.9866 |
| −2 | 300 | 68.3 | −3.3519 | −6.4887 | +3.4912 |
| 0 | 300 | 60.0 | 0.0000 | 0.0000 | +0.1675 |
| +2 | 300 | 55.3 | +3.2393 | +6.3403 | −3.0544 |
| +4 | 300 | 51.3 | +6.3835 | +12.5583 | −6.2388 |
| +6 | 300 | 51.7 | +9.4779 | +18.7204 | −9.4322 |
| +8 | 300 | 49.7 | +12.5354 | +24.8320 | −12.4419 |

`G_prefill` 隨 α 近乎完美線性（slope=1.648、intercept=−0.296、`R²=0.9992`），`Z_prefill` 一致（`R²=0.9997`）；正負方向近似對稱，負側在高 dose 幅度略大。這是清楚的 manipulation check：**α 線性控制 generation boundary 上的 RSN gain**。

`boundary_jump_G` 與 `G_prefill` **反向且量級接近**，表示一次性 prefill 注入在 `decode[0]` 已大幅回彈。這與 **initial-condition / boundary-gating** 一致——α 改變 generation 的起始條件，而不是在整條 decode trajectory 上留下固定 additive offset——並把非線性轉換定位在 boundary 之後。但這**不能證明回彈本身造成後續行為差異**。slope 由該 mask 的 `‖m_l‖²` 與層數決定，是特定計算設定下的量，**不可當作跨模型或生物層面的可比劑量**。

#### Result 2 — Decode Dynamics Form a Nonlinear Working Point

α 對 slow state 的影響集中在 pre-commit。下表為每個 α 相對 α=0 的 paired comparison：

| α | paired n | mean `s_t` | Δ vs 0 | `d_z` | exploratory p |
|---:|---:|---:|---:|---:|---|
| −8 | 229 | −0.573 | −0.325 | −0.418 | *** (崩潰區) |
| **−6** | 291 | +0.260 | **+0.449** | **+0.584** | *** |
| −4 | 298 | −0.033 | +0.154 | +0.228 | *** |
| −2 | 298 | −0.122 | +0.065 | +0.143 | ns |
| 0 | 298 | −0.186 | 0 | 0 | — |
| +2 | 297 | −0.248 | −0.062 | −0.123 | ns |
| +4 | 298 | −0.308 | −0.121 | −0.191 | ** |
| +6 | 298 | −0.415 | −0.229 | −0.323 | *** |
| +8 | 298 | −0.347 | −0.161 | −0.221 | *** |

**入口是線性的，pre-commit slow state 卻不是。** 形狀為 **asymmetric peaked working-point response**：−6 是明顯峰值，−4 至正 α 大致隨 α 遞減，−8 進入另一種崩潰區。C1-centered trajectories 顯示分離主要發生在 **commitment formation**；越過 C1 後各 dose 共同下降並逐步收斂，且 post-commit release 除 −8 外接近 null——α 並未普遍改寫全程 RSN 水位，而是選擇性重組答案形成期。

Pre-commit `end_minus_start` 顯示正 α 通常有較大的向下 relaxation（例如 +4 `d_z=−0.18`），但該 readout 同時受起點水平影響，**只作 shape 輔助，不單獨承擔結論**。−8 是當前計算干預下的 **extreme-negative failure regime**，不等同「低 dopamine」。

#### Result 3 — α Modulates Pre-Commit Residual Amplitude and Output Decisiveness

**Fast residual（pre-commit `p_t std`）：**

| α | n | Δ vs 0 | `d_z` | p |
|---:|---:|---:|---:|---|
| −8 | 229 | −0.010 | −0.059 | ns |
| **−6** | 291 | **+0.066** | **+0.452** | *** |
| −4 | 298 | +0.026 | +0.176 | ** |
| −2…+8 | | ~0 | \|d_z\|<0.09 | ns |

−6 的 pre-commit residual dispersion 明顯升高，−4 只有小效應，其餘 doses 接近 null——支持一致但**較窄**。

**Pre-commit residual amplitude 的 between-α 檢驗。** 上表為 dispersion；另以逐題配對（同 300 題 index 對齊）比較同一段內 α=−6 / α=+6 相對 α=0 的 **centered RMS**（先去均值，故為段內變異幅度，不含 level shift）：

| pre-commit centered RMS | Δ vs α=0 | 顯著性 |
|---|---:|---|
| α=−6 | **+0.046** | **`p<.001`**，paired **n=200** |
| α=+6 | −0.006 | n.s. |

**所有 frequency metrics（zcr / dominant frequency / centroid / spectral entropy）在兩個 α 方向皆為 null。**

> **`p_t` 中與 α 相關的資訊主要體現於 pre-commit residual amplitude，而非穩定的頻率變化。**

此段在提交前、不含 `####` 複讀，因此該幅度效應是**扣除 loop 後仍與 α 相關**的乾淨結果，與本節 slow-state dose 結論同向（−α 抬高 pre-commit engagement）。**三項限制：**（i）此 between-α 對比只覆蓋 **α=−6 / 0 / +6 三個代表 dose**，**不是完整 dose curve**，不可外推為所有劑量的連續效應；（ii）**α=−6 的 RMS 上升不是正負對稱效應**——+6 為 null，故只能寫成負向劑量的單側結果；（iii）post-commit 段的 α 差異**只作 post-marker diagnostic**（該窗口混合答案收尾、格式變化與重複生成），reason / tail 段的 between-α 因要求「兩 α 在同題皆有 tail」而僅 **n=10/14**，**不作任何機制結論**。

`p_t` 是**同一 RSN projection 相對 EMA baseline 的快殘差**，不是獨立通道；本節只支持 amplitude/dispersion change，**不支持 biological phasic dopamine 或 RPE 解讀**（C1 附近的共同轉折也可能含 `####`/answer-marker effect）。主讀數限於 `abs_mean/std` 與 centered RMS，不用易受長度與 EMA lag 影響的極值。

**Output-distribution confidence controls：**

| Metric | α=−6 Δ vs 0 | `d_z` | α=+4 Δ vs 0 | `d_z` | Pattern |
|---|---:|---:|---:|---:|---|
| entropy | −0.174 | **−0.718** | +0.063 | +0.304 | −6 最確定 |
| top1 | +0.039 | **+0.643** | −0.012 | −0.248 | −6 達峰 |
| margin | +0.048 | **+0.593** | −0.014 | −0.214 | −6 達峰 |

−6 **同時**具有較高 slow state、較高 pre-commit residual amplitude 與較強 output decisiveness，且 effect size 相當（`s_t d_z=0.58`、top1 `d_z=0.64`、entropy `d_z=−0.72`）；正 α 則 entropy 較高、top1/margin 較低。這三項共同構成 asymmetric working point 的內部組成。因此 **α 不是 selective wanting intervention**：它同時改變 RSN engagement 與 confidence-related output distribution。這與 §4.2 CoT 的 joint modulation 相似，不支持「α 只改 wanting、不動 confidence」的強版本；§4.3 Persona 的 task-entry separation 仍是目前較清楚的一例，但本研究尚未取得 wanting–confidence 的 causal dissociation。Post-commit confidence 受 answer-loop 與格式轉換污染，不作實質解讀。

#### Result 4 — Commitment Is Followed by Residual-Amplitude Release, while Frequency Effects Are Unstable

Result 3 檢驗 α 主效應；本節改為 within-α 的**階段對比**，在 α=−6 / 0 / +6 上做兩套 paired comparison：commit-centered，以及 reasoning 段 vs **repeated-ngram tail proxy**。同樣先去均值再取指標。

| post−pre | α=−6 | α=0 | α=+6 |
|---|---:|---:|---:|
| centered RMS | **−0.285**\*\*\* | **−0.186**\*\*\* | **−0.143**\*\*\* |
| spectral entropy | **−0.094**\*\*\* | **−0.074**\*\*\* | **−0.056**\*\*\* |
| zero-crossing rate | **−0.061**\*\*\* | **−0.038**\*\* | **−0.036**\*\* |
| spectral centroid | −0.007 ns | **−0.019**\*\*\* | −0.013\* |

（paired Wilcoxon：\* `p<.05`，\*\* `p<.01`，\*\*\* `p<.001`。）

- **commit 後最穩定的變化是 centered RMS 下降**，三個 α 條件一致；stage-based 切分複現同一方向，tail 段 centered RMS 約 **−0.31 / −0.35 / −0.36**（皆顯著）。
- commit-centered frequency metrics 也會變化，但**方向與窗口敏感**（stage 切分下 zcr 近 null，dominant frequency / centroid 反而小幅上升，spectral entropy 只弱下降），亦無 α-monotonic dose structure，**不能宣稱存在獨立的 frequency reorganization**。
- 這些 frequency 變化與 `####` 比例（Pearson `r=−0.53/−0.58/−0.55`）及 repeated-12gram rate（`r=−0.40/−0.46/−0.43`）**高度共變**：commit-centered frequency/regularity readout 與 answer-format、重複內容**不可分離**。**相關係數不是控制後的因果分解**，不能據以估計「多少效應由格式造成」，也不能寫成格式或重複**導致**了 residual change。
- stage-based subset 僅 **n=24–42**，且由 repetition detector 選樣，同時存在 **selection 與 range restriction**；即使其 repetition/hash 與 frequency-change 相關接近 0，也只能說明**頻率效應不穩定**，**不能稱作 confound-free negative test**。

> 因此當前穩健結果是 **commit 後的 `p_t` residual amplitude release**，頻率指標僅保留為 negative control。

**兩個 proxy 的邊界：** repeated-ngram tail 只是重複尾段的近似，**不是經驗證的 loop-onset detector**；第 2 個 answer marker 只是 repetition / revision proxy，**同樣不等同 loop onset**（§4.2 聚合分析用 literal 第 1 / 第 2 個 `####`，其 **C2 應讀作 second-answer-marker boundary**）。窗口長度、頻率指標計算、12-gram detector 與腳本參數見 `CLAUDE.md`。

#### Result 5 — Internal State and Behavioral Performance Align at an Asymmetric Working Point

| α | acc(184) | `G_prefill` | `early_s_t` |
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

離散行為最佳點是 **α=−6**。Accuracy 的 quadratic fit 雖優於 linear（`R²=0.352` vs `0.147`），但擬合峰值約 α=−1.9，**與離散峰值 −6 不一致**，故不能描述成平滑、對稱的標準 inverted-U。正確形狀是：**−6 尖銳最佳點、−8 低端崩潰、正 α 端逐步下降後趨平**。

在 9 個 dose-level aggregates 上，inline accuracy 與 `early_s_t` 的相關（`r=+0.74`）高於與 `G_prefill`（`r=−0.37`）。行為表現因此更接近 **decode 期間形成的 commitment state**，而非入口 gain 的絕對大小。但只有 **9 個 aggregate points**，這僅是 dose-level covariation，**不能作為 `s_t` 中介 accuracy 的證據，也不可寫成因果**。

#### Integrated Interpretation

§4.4 的核心結果是**線性干預如何轉化為非線性內部與行為狀態**：

$$\alpha \;\to\; \text{linear task-entry gain } (G_{prefill}) \;\to\; \text{nonlinear commitment-formation state } (s_t) \;\to\; \text{joint change in output decisiveness} \;\to\; \text{asymmetric behavioral working point}$$

其中 `s_t` 是主要載體，`p_t` 的支持較窄——α 相關資訊集中在 **pre-commit residual amplitude**（且僅在負向的 α=−6 顯著），而非頻率組織；commit 之後的殘差變化則為 amplitude release，其頻率讀數與 answer format / repetition 不可分離。三個工作區：

1. **Intermediate calibration range（−4 至 +2）：** α 對 commitment-formation state 作較平滑的校準。**這是描述 state 的變化平緩，不是說此區間表現較好**——區間內 accuracy 實為單調下降（74.3→68.3→60.0→55.3），且離散最佳點 −6 落在區間之外。
2. **Extreme-negative collapse（−8）：** commitment-formation collapse；event-centered 結果僅為條件子樣本，須與 C1-analyzable rate 及文本中的 answer-candidate oscillation 共同解讀。
3. **High-positive degradation / flattening（+6/+8）：** pre-commit slow state 與 confidence 均較差；其 **premature-commitment 解讀來自獨立的 commit-position / behavioral evidence，不能僅由本節 trajectory 推斷**。

整體而言，α 是可靠的 **task-entry RSN gain intervention**，但有效作用**不是「gain 越高越好」**，而是把模型推入不同的 commitment-formation working point。這與 task-dependent dopamine/wanting calibration 及 Yerkes–Dodson framing 相容，但目前建立的是 **computational and behavioral analogy**，而非生物多巴胺機制的直接對應。

#### Figures

| Figure | Role |
|---|---|
| `fig44_dose_main.png` | C1-centered `s_t` / `p_t` / entropy 與 dose-level working point 主圖 |
| `fig44_validity.png` | α 對 `G_prefill` / `Z_prefill` 的線性 manipulation check |
| `fig44_slow.png` | pre/post-commit slow-state paired comparisons |
| `fig44_fast.png` | fast-residual dispersion 與 pre/post-commit amplitude controls |
| `fig44_confidence.png` | entropy / top1 / margin confidence controls |
| `fig44_integrated.png` | inline accuracy、task-entry gain 與 early `s_t` 的 dose-level 對齊 |

**報告限制。** 目前 slow/fast/confidence dose figures 顯示 absolute means，而統計採 paired α-vs-0 comparisons；confirmatory reporting 前需統一為 paired Δ/CI 並對 dose × metric 作多重比較校正。Trajectory panels 只顯示部分代表 doses，dose-response panel 則使用全部 9 檔。（腳本、`--part` 選項與輸出檔位置見 `CLAUDE.md`。）Result 3–4 的 amplitude/frequency 分析為獨立腳本產出，**只覆蓋 α=−6 / 0 / +6 三個代表 dose**，與其餘 9-dose readout 的取樣密度不同，不可並列為同一條 dose curve。

### 4.5 CoT × α=−4: Signal Interaction Analysis

#### Scope and Shared Analysis Framework

本節檢驗 `α=−4` 如何調制 CoT 的內部信號動力學。這是一個**固定的 2×2 單劑量 factorial**（`{No-CoT, CoT} × {α=0, α=−4}`）：目標是判斷兩個 manipulation 是 approximately additive、attenuation 還是 interaction，**不是**尋找 CoT 條件下的最佳 α。四組同 300 道 GSM8K、index-paired（common=300 已驗證）；reference μ/σ 固定 = neutral α=0 No-CoT prefill（同 §4.2–4.4）。

因四組全 paired，交互效應以 **per-question difference-in-differences** 計算後作 Wilcoxon，而非比較四個 cell 平均：

$$\mathrm{DiD}_q = \big(\text{cot}_{-4} - \text{cot}_0\big) - \big(\text{nocot}_{-4} - \text{nocot}_0\big) \qquad \text{[每題 } q\text{]}$$

inline acc（184）：nocot_0=60.0 / nocot_−4=74.3 / cot_0=67.7 / cot_−4=82.7，與 §2.5.1 行為（182：60/73/69/85）同向、量級一致；行為結果直接引用 `AdaDopamine_gsm8k.md` §2.5/§2.5.1，本節不重複展開。

**三條貫穿全節的統計限制，後續不再重複。**（i）**statistical 與 practical interaction 須分開讀**：n=300 加上極小的配對方差，能偵測到量級可忽略的系統偏離（入口 DiD `***` 但僅約 α 主效應的 0.4%），判定以**量級比**為準，既不因量級小而降級為 ns，也不寫成「純 additive」。（ii）**「一顯著 + 一 ns」不等於兩者顯著不同**——early window 的 α 效應在 CoT 下轉 ns，但其 DiD p=.14，故只記為 **attenuation trend**，不作 redundancy/saturation 的正式判定。（iii）**單一 α 劑量**：不能推斷 CoT 下的 dose-response 或最佳 α（例如 CoT 是否移動 §4.4 的 asymmetric working point / discrete optimum——需補採 CoT × dose signal）；「無顯著交互」也不等於證明兩機制獨立。

#### Result 1 — Task Entry: α Dominates, and CoT Barely Changes It

| readout | nocot_0 | nocot_−4 | cot_0 | cot_−4 | α效應\|No-CoT | α效應\|CoT | DiD | DiD 顯著性 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `G_prefill` | 0.000 | −6.802 | 0.071 | −6.703 | −6.80 | −6.78 | +0.028 | *** but ≈0.4% of α |
| `Z_prefill` | 0.000 | −13.107 | 0.158 | −12.903 | −13.11 | −13.06 | +0.046 | *** but ≈0.4% of α |
| `boundary_jump_G` | 0.167 | 6.987 | 0.094 | 6.988 | +6.82 | +6.89 | +0.074 | ns |

**入口是 α 主導（|Δ|≈6.8）、CoT 次要（|Δ|≈0.07–0.16）**，且 α 效應幾乎不受 CoT 影響（`G_prefill` −6.80 vs −6.78）。這與 §4.4 的 co-design identity 一致（$G_{prefill}(\alpha) \approx G_{prefill}(0) + \alpha\lVert m\rVert^{2}$：α 在 prefill 加上一個與 CoT 無關的固定量）。G/Z 的 DiD 帶星號屬 statistical interaction，量級僅 0.4%，故判為 **approximately additive**。

#### Result 2 — Decode Dynamics Shift the Dominant Effect from α to CoT

以 C1 為中心，三段窗口的 `s_t mean`：

| Window | nocot_0 | nocot_−4 | cot_0 | cot_−4 | CoT效應(主) | α效應\|No-CoT | α效應\|CoT | DiD |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `[-50,-20]` early (n=142) | −0.063 | 0.097 | 0.465 | 0.506 | +0.53*** | +0.159** | +0.040 ns | attenuation trend (DiD p=.14, ns) |
| `[-20,0]` commit (n=170) | −0.033 | 0.120 | 0.478 | 0.591 | +0.51*** | +0.153** | +0.113* | approx. additive |
| `[0,+20]` release (n=170) | −0.321 | −0.207 | −0.041 | 0.083 | +0.28*** | +0.114*** | +0.124** | approx. additive |

**decode 內 CoT 絕對主導**（`d_z` 0.55–0.86），α=−4 次要（`d_z` 0.11–0.30），量級差 3–5 倍——**兩個 manipulation 的主次排序與 Result 1 的入口相反**（入口 α≫CoT，decode CoT≫α）。**這是主導權的翻轉，不是信號數值反號**：α 的效應在兩處同向為正（入口 −6.80 是 α=−4 的負向注入，decode 三窗皆 +0.11~+0.16），改變的只是誰的量級更大。這個翻轉就是本節的核心發現，它建立在**顯著的主效應**上，而非任何 DiD 判定。

Early window 的 α 效應在 CoT 下由 `+0.159**` 降為 ns（`+0.040`），descriptive 上像是 CoT 已把 early `s_t` 拉到 0.47、α 無額外空間；但依限制（ii），這只記為 **attenuation trend**，redundancy/saturation 保留為候選解釋。commit / release 兩窗的 α 效應維持（approximately additive）。Release `s_t slope` 上 α 無穩定效應（No-CoT / CoT 皆 ns），CoT 則使 release 稍陡（`d_z≈−0.5***`）。

**cot_−4 在三窗 `s_t` 皆為四組最高**（0.506 / 0.591 / +0.083），且伴隨四組最高 acc（82.7）。**此處只作 co-occurrence 陳述**——§4.4 已證 signal 高不必然行為好，故不將高 `s_t` 解讀為其高 acc 的解釋或中介。

#### Result 3 — Fast Residual: CoT Dominates, All Decode DiD Null

`p_t abs_mean` / `std` 的主效應仍是 CoT（early `abs_mean +0.092***`、`std +0.143***`；release `abs_mean +0.178***`），α=−4 邊際弱。Early window 呈與 Result 2 同向的 attenuation trend（α 效應在 CoT 下轉 ns），commit / release 的 **DiD 全部 ns**。`p_t` 保留 phasic-like operational definition；本節僅得 fast residual dynamics evidence，尚未建立 biological dopamine 或 event-specific phasic correspondence。

#### Result 4 — Confidence Output: Near-Additive

pre-commit `[-20,0]`（與 wanting 分表；post/loop 飽和不讀）：

| Metric | nocot_0 | nocot_−4 | cot_0 | cot_−4 | α效應\|No-CoT | α效應\|CoT | DiD |
|---|---:|---:|---:|---:|---:|---:|---:|
| entropy | 0.553 | 0.466 | 0.339 | 0.253 | −0.087*** | −0.086*** | +0.001 (ns) |
| top1 | 0.840 | 0.863 | 0.894 | 0.916 | +0.024*** | +0.022** | −0.002 (ns) |
| margin | 0.753 | 0.785 | 0.830 | 0.859 | +0.032** | +0.029** | −0.003 (ns) |

α 效應在 CoT / No-CoT 下幾乎相同（entropy −0.087 vs −0.086），DiD Δ≈0 且全 ns —— **near-additive**。兩者都朝更 decisive 疊加，cot_−4 在四格中最確定（entropy 0.253 最低、top1 0.916 最高）。即 α=−4 **同時**改變 CoT 的 RSN state 與 output decisiveness，兩軸皆 near-additive，與 §4.4「α 非 selective wanting intervention」一致。

#### Integrated Interpretation

本節的價值不在「正交雙槓桿」，而在**兩個 manipulation 的時間重心不同**：

| Stage | 主導 | 交互判定 |
|---|---|---|
| Task entry | **α ≫ CoT** | approximately additive（DiD *** 但僅 0.4%） |
| Decode / commitment formation | **CoT ≫ α**（3–5×） | 無可靠交互；early 為 attenuation trend |
| Confidence output | 兩者共同抬升 | near-additive（DiD 全 ns） |

即 **α 直接控制 generation-boundary gain；CoT 主要重塑展開後的 commitment-formation dynamics；兩者最終在 output decisiveness 上共同抬升。**

> **CoT and α=−4 are approximately additive at task entry and confidence output, while decode-stage dynamics are dominated by CoT. No reliable decode-stage interaction is detected; the early slow-state attenuation under CoT remains a descriptive redundancy candidate.**

cot_−4 在 `s_t` 與 confidence 上均為四格最高，並與最高 acc co-occur——僅為共現，不作中介宣稱。本節建立的是 computational / behavioral analogy，非生物多巴胺機制的直接證明。

#### Figures

| Figure | Role |
|---|---|
| `fig45_slow_centered.png` | 四 cell C1-centered `s_t` 疊圖（主圖：CoT 抬高整條、α 次要調制） |
| `fig45_fast_centered.png` | 四 cell C1-centered `p_t` 疊圖 |

**報告限制。** DiD 的 verdict 標籤（additive / redundant / …）僅為**量級比啟發式，非顯著性判定**，不可單獨當結論——應**結合 DiD 的效應量、實際量級與 Wilcoxon p 共同判斷**（見 Scope 限制 (i)：兩者須分開讀，任一單獨都不足以定案）。（腳本、`--part` 選項與輸出檔位置見 `CLAUDE.md`。）

### 4.6 RSN Direction Specificity: support-selection 與 generic-direction null

#### Scope and Null-Control Design

§4.1–4.5 的所有 state 效應都是沿 **NMD/RSN 方向**投影讀出的。本節檢驗的對照問題是：**這些 commitment-related temporal effects 是否對 NMD/RSN 方向具有特異性，還是任意稀疏方向都能讀出類似結構？**

本節是對**同一批已存 hidden states 的 offline re-projection control**。它檢驗的是 **RSN 是否是一條具有特異性的 state readout direction**；它**不能**證明「只有沿 RSN 方向做 causal steering 才會改變行為」——後者需要真正注入 random/orthogonal directions 並重新跑模型，是另一個實驗。

三個 null family 從不同方向收緊這個問題：

| Null family | Support | Weight direction | 主要回答的問題 |
|---|---|---|---|
| `diff_random`（N=11） | 隨機 support | 保留 role-diff coordinate weights | top-\|diff\| support 是否特殊 |
| `ortho_gauss_same`（N=10） | NMD 原 support | 與 dense role-diff 逐層精確正交、norm-matched | 相同 support 本身是否足夠 |
| `ortho_gauss_off`（N=10） | 與 NMD 不相交 | 與 dense role-diff 逐層精確正交、norm-matched | 完全無關的稀疏方向能否複現效果 |

- **每個 mask 用自己的 reference**（μ/σ 與 `‖m_l‖²` 皆取自它自己），使「raw projection 尺度較大」不能為 NMD 買到假優勢。`‖m‖²` 在 decode 的 Z 座標中相消，故 `diff_random` 的 ¼-norm gap 只影響 `G_prefill`，不影響 decode 讀數。
- **Primary metrics 為帶符號、commit-aligned**（與 §4.1–4.5 同口徑，Z 單位，共用 K-gate）：`s_pre_mean` = commit 前 `[−40,0)` 的 slow state；`p_post_mean` = commit 後 `[0,+10]` 的 fast residual。**重點是 signed temporal organization，不是 unsigned amplitude 比較**——早期的 unsigned 口徑會把 commit-locked 的符號結構平均掉，產生 false negative。
- **null draws 極少，只作 exploratory ordering**：p 地板約 0.083–0.091，percentile 0%/100% **不等於**正式顯著性。
- 三個 family **共享同一批 hidden states、conditions、baseline 與指標**，不是獨立重複。

#### Result 1 — RSN Shows Extreme Signed Temporal Effects Across All Contrasts

對 `diff_random`（support-selection null）：

| Contrast | `s_pre_mean` (NMD / null-med / pctile) | `p_post_mean` (NMD / null-med / pctile) |
|---|---|---|
| CoT vs No-CoT | +0.500 / −0.034 / **100%** | −0.361 / −0.054 / **0%** |
| Expert vs Non-Expert | −0.133 / +0.032 / **0%** | +0.157 / −0.007 / **100%** |
| α=−4 vs 0 | +0.129 / −0.003 / **100%** | −0.065 / −0.012 / **0%** |
| α=−6 vs 0 | +0.437 / −0.015 / **100%** | −0.264 / −0.062 / **0%** |
| α=+4 vs 0 | −0.167 / −0.014 / **0%** | +0.185 / +0.004 / **100%** |

1. **兩個 primary readouts 在全部 5 個 contrast 都落在 null 分布的極端端點**（pctile 0% 或 100%）。
2. **方向隨條件系統性翻轉**：CoT 與負 α 為「commit 前 `s_t` 較高、commit 後 `p_t` 較低」，Expert 與正 α 呈鏡像。
3. **null median ≈ 0 不代表每條 null trajectory 都平坦**——部分 null（尤其 `p_t`）在 commit 附近仍有明顯波動，只是其**帶符號窗口平均**方向不定、中心趨近 0。

**自然狀態與注入條件須分開解讀。** 若特異性只出現在 α-dose，可能是 α-steering 的 injection–projection identity 造成的時間版假象。但 **CoT 是最獨立的自然狀態證據**（無任何注入）；**Persona 雖也非注入，但 NMD mask 本身即抽自 Expert–Non-Expert contrast，獨立性較弱**。自然 CoT 與 α contrasts 呈現一致結構，說明結果不能完全歸因於 α injection–projection identity。

#### Result 2 — Specificity Survives Generic Orthogonal Directions

把權重換成逐層 ⊥ dense role-diff Δ_l 的 norm-matched Gaussian（徹底去掉 role-diff 的權重方向）後：

| Contrast | `s_pre_mean` NMD | same-med / off-med | pctile (same / off) | `p_post_mean` NMD | same-med / off-med | pctile (same / off) |
|---|---|---|---|---|---|---|
| CoT vs No-CoT | +0.500 | +0.016 / +0.010 | **100% / 100%** | −0.361 | +0.027 / −0.004 | **0% / 0%** |
| Expert vs Non-Expert | −0.133 | −0.009 / −0.004 | **0% / 0%** | +0.157 | +0.005 / −0.013 | **100% / 100%** |
| α=−4 vs 0 | +0.129 | +0.007 / +0.006 | **100% / 100%** | −0.065 | −0.004 / +0.004 | **0% / 0%** |
| α=−6 vs 0 | +0.437 | +0.040 / −0.006 | **100% / 100%** | −0.264 | −0.005 / +0.031 | **0% / 0%** |
| α=+4 vs 0 | −0.167 | −0.040 / −0.007 | **0% / 0%** | +0.185 | +0.007 / −0.011 | **100% / 100%** |

合計三個 family 的 **30 個 primary cell 中，NMD 皆相對各自 null 保持極端**，各 null 中心趨勢皆 ≈0。分成 support × weight 的 **control matrix**（每格記「NMD 是否相對該 null 保持極端」）：

| | role-diff 權重 | ⊥ role-diff 權重 |
|---|---|---|
| **NMD support** | —（即 NMD 本身） | `ortho_gauss_same`：NMD remains extreme |
| **random support** | `diff_random`：NMD remains extreme | — |
| **strictly NMD-disjoint support** | — | `ortho_gauss_off`：NMD remains extreme |

> 這是幫助解釋三個對照關係的 **control matrix，不是完整、嚴格的 factorial design**——`diff_random` 的 support 是**隨機抽樣**、與 NMD 可能有少量重疊，並非嚴格 disjoint；只有 `ortho_gauss_off` 的 support 才與 NMD 完全不相交。表中的「—」是未取樣的格，不是空結果。

- **`ortho_gauss_same`** 表明：僅使用 NMD support **不足以**複現 NMD 效應。
- **`diff_random`** 表明：僅保留 role-diff coordinate weights、但放到隨機 support 上，也**不足以**複現。
- **`ortho_gauss_off` 是最嚴格的對照**：support 完全移出 NMD、weights 與 role-diff 正交，NMD 仍保持極端，且 off 方向自身的窗口平均 signed effect ≈0（未命中某個 global mode）。

結論應寫為：**當前證據指向 top-\|diff\| support 與 role-diff-aligned weights 的特定匹配關係**，而不是 support 或 weight 任一單獨成分。不可寫成「support 不重要」「weights 不重要」或「已完成所有可能的方向特異性分解」。

#### Result 3 — The Effect Is Mainly Amplitude/Level Specificity, Not Proven Shape Specificity

整條軌跡距離以 **LOO-centroid RMS**（窗口 `[−40,+20]`，`s_t`/`p_t` 分開）度量：

- 全部 **10 個 (contrast × signal) cell** 的 NMD signed 軌跡距離都**超過已採樣的全部 null draws**（pctile 90–100%）。
- NMD 的 RMS 距離約為 null median 的 **3–7 倍**。
- **RMS 同時包含 level、amplitude 與 shape**，不能單獨解讀為軌跡形狀不同。
- `p_t` 的 NMD/null **shape correlation 約 0.39–0.55**：兩者存在**部分共同形狀**，但當前 RMS 差異主要體現的是 **level/amplitude**，**尚未建立獨立的 shape specificity**。
- `s_t` 的 null centroid 近乎平坦（centroid std 極小），其 shape correlation **不穩定亦不可解釋**。

> RSN 的特異性主要體現在 **commitment-locked signed level/amplitude**，而**尚未證明**存在獨立於幅度的 trajectory-shape specificity。

因此不可將此結果寫成 neural manifold reorganization、trajectory rotation，或獨立於幅度的形狀差異——這些仍需額外的幾何分析。

#### Result 4 — The Temporal Effect Is Not Driven by Any Single Layer

逐一移除 middle layer L11–L19 中任一層、在剩餘 8 層上重算兩個 primary 的 signed window mean：

- **全部 10 個 (contrast × primary) cell 均保持原符號**，LOO 的 min/max 與 full 同側且不跨 0。
- 例（α=−6）：`s_pre_mean` full=**+0.437**，LOO 範圍 **[+0.334, +0.621]**；`p_post_mean` full=**−0.264**，LOO 範圍 **[−0.374, −0.211]**。
- **L11 影響最大**，但移除後效應仍成立。

準確結論只能是：**效應不由任意單一層獨立驅動。** 不能升級為「效應不由少數層驅動」——尚未做 leave-two/three-out，也未完成 null 側的逐層對照。

#### Integrated Interpretation

分兩個層次讀，不可混為一談：

**1. Task-entry gain（manipulation check）。** `G_prefill` 在 NMD 上遠強於 null（α-dose `d_z` ±72–80 vs null ±3–7），但這主要來自 **co-design identity**（$x_{prefill}(\alpha) \approx x_{prefill}(0) + \alpha\lVert m\rVert^{2}$）、NMD 因取 top-\|diff\| 而 norm 最大，以及 mask 本身即抽自該方向。因此它是 **manipulation check，不是獨立的 direction-specificity evidence**。

**2. Commitment-locked temporal organization。** `s_pre_mean` 與 `p_post_mean` 在三類 null 中都保持極端；自然 CoT 與注入 α conditions 呈現相容結構；same-support、random-support 與 off-support controls 共同說明效應與 NMD support–weight 的特定匹配有關。**這是目前最強的 exploratory RSN readout-specificity evidence。**

> Across support-randomized and orthogonal generic-direction controls, the NMD/RSN projection consistently exhibits stronger signed commitment-locked temporal organization. The effect is primarily expressed through state level and amplitude and appears to depend on the specific matching between top-|diff| support and role-diff-aligned weights.

> 在隨機支撐與正交方向對照下，NMD/RSN 方向始終表現出更強的、與 commitment 對齊的帶符號時序結構。當前證據主要支持 **level/amplitude 層面的 readout specificity**，並指向 top-\|diff\| 支撐與 role-diff 權重之間的特定匹配關係。

#### Evidence Boundary

1. 三類 null 均為 **N=10–11**，只支持 exploratory ordering。
2. **0%/100% percentile 不等於正式顯著性**（p 地板 0.083–0.091，且 draws 已參與指標選擇）。
3. 三個 family 共享同一批 hidden states、conditions、baseline 與指標，**不能視為獨立重複**並據此推算很低的偶然機率。
4. 當前只分析 **Llama3-8B**。
5. 當前是 **offline re-projection**，只證明 **readout specificity**，不證明 **causal steering-direction specificity**。
6. **Trajectory distance 主要反映 level/amplitude**，獨立的 shape specificity 尚未建立。
7. Leave-one-layer-out 只排除**任一單層驅動**，未排除兩三層聯合驅動。
8. **Sign-shuffle 尚未完成**，但屬更細的分解，不是當前結論成立的必要條件。
9. 下一步優先級是**跨模型、跨任務與 causal random/orthogonal injection**，而非繼續增加同類型的 null seeds。

#### Figures

| 圖 | 說明 |
|---|---|
| `fig46_commit_specificity_{contrast}.png` | commit-centered 疊圖：NMD 與各 null 在 C1 前後的 signed trajectory。NMD 曲線在 commit（step 0）附近急轉——commit 前一個平台、commit 後單調鬆弛；**null 可能有局部波動，但未穩定複現 NMD 的窗口平均方向與幅度**（`s_t` null 近平坦，`p_t` null 有結構但幅度較弱）。CoT 的 `p_t` 在 commit 出現一個 −0.7 的單步 transition（呼應 §4.2；仍有 `####`/marker-format confound，不宣稱為獨立的 phasic 事件節點）。 |

（分析腳本、null family 切換方式、mask 建構式與正交/norm-match assert、seed 與目錄結構、server launcher 見 `CLAUDE.md`。）

### 4.7 Slow-State Behavioral Validation

#### Scope and Analysis Framework

§4.2–4.4 主要以 `s_t` 的**水平（level）**描述模型是否仍處於持續推理與未完成提交的狀態。本節檢驗 sample-level 的 `s_t` 是否關聯行為：在控制 α condition 後，**level** 與 **slope** 各自能否預測 commitment timing。這回答的是逐題的 state–behavior 關係，**不是 α 的 dose-response**（後者見 §4.4）。

分析 pool 了 **11 個 conditions ×300 題**（No-CoT dose −8…+8 含 ±2，CoT α=0/−4）；`s_t` 與行為讀數取自**同一份 signal JSON**、index 對齊。三項設計邊界是本節成立的前提：

1. **固定 early window `[0,20)`。** 以 `[0,c1)` 計算的 slope 與 c1 機械耦合，故主要 predictor 使用**不會看到 commit 的固定窗口**。
2. **at-risk subset（`commit_step ≥ 20`）。** 有 23.4% 的樣本在 token 20 前已提交，該窗口會跨越 commit 而混入 post-commit release；因此 commit-timing 分析僅在尚未提交的樣本上進行，target 為 `commit_excess`。
3. **α-condition fixed effects**，避免跨條件的 pooled association 被誤讀為 within-sample 關係；held-out 為 question-level 切分，且 **train scaler 凍結後套用於 test**。

（完整統計規格、腳本與 `--part` 選項見 `CLAUDE.md`。）

#### Result 1 — Slow-State Level Is Associated with Commitment Timing

at-risk subset 上，`s_t` level 與提交時間穩定正相關：水平越高，模型通常維持推理越久、提交越晚。

| Readout | 值 |
|---|---|
| level ↔ `commit_step`（Spearman） | ρ=**+0.379**，`p<1e−85` |
| 回歸加入 level 後 `R²` | 0.235 → **0.259**（β=**+19.6**，`p=7e−7`） |
| question-level held-out test `R²` | ≈**0.25** |

這項關係在 descriptive、cluster-robust 回歸與 held-out questions 上皆可重現。但它是 **descriptive association**：本節**不能**推斷 `s_t` level 造成較晚提交，也**不能**推斷高 `s_t` 帶來較高正確率。

#### Result 2 — Slow-State Slope Does Not Support the Predicted Vigor Relationship on GSM8K

若 slope 代表 ramping / vigor，較陡的上升應對應更快提交。leak-free 窗口下未檢出此關係：

| Readout | 值 |
|---|---|
| slope ↔ `commit_step` marginal（Spearman） | ρ=**−0.020**，n.s. |
| corr(level, slope) | **r=−0.48** |
| 控制 level 後的 slope 效應 | 顯著但為 **suppressor**，方向 **與 vigor 預測相反**（斜率越正 → 提交**越晚**） |

也就是說，slope 的邊際關聯為 null，控制 level 後殘留的效應方向與預設相反，因此**不構成 vigor evidence**。premature commitment 的 slope 分析另有時間順序混淆——**749/754** 個 premature 樣本在 token 20 前已提交，測量窗口與 post-commit release 重疊——故僅保留為診斷，不作主要證據。

#### Integrated Interpretation

GSM8K 支持把 `s_t` **level** 解讀為 ongoing engagement / commitment state 的 readout；**未檢出**符合預設方向的 **slope-based vigor evidence**。

依 §2.2 命名約定，這是 **task-level finding，不否定 ramping / vigor 的建模假說，也不改名或降級 `s_t` slope 的 operational definition**。GSM8K 缺乏逐步逼近獎勵的任務結構，slope 預測更適合在 effort、betting 或 agentic progression 等任務中繼續檢驗。當前已驗證的經驗內容是 **slow-state level 的行為意義**；vigor（slope）仍是 open 的 task-level 問題。

#### Evidence Boundary

1. level ↔ timing 為 **descriptive association**，非因果，且不等同「高 `s_t` → 正確」。
2. slope 的 null 是**未檢出穩定效應**，不是「沒有效應」的證明。
3. 控制 level 後的 slope 係數是 suppressor 殘差，**不可單獨引用為 vigor 的反向證據**。
4. premature 分析 time-confounded，僅作診斷。
5. 結果 pooled 自 11 個 α conditions，已加 fixed effects，但仍非 within-condition 的獨立複製。

### 4.8 Case Study: Sample-Level RSN Trajectories and Generated Text

#### Scope and Purpose

本節逐題對照 9 個 `sample_traj3_` case（Q10、Q80、Q92、Q140、Q152、Q189、Q225、Q251、Q284），每張圖同時疊加 neutral No-CoT 下 α=−6 / 0 / +6 的 `s_t`、`p_t` 軌跡與實際生成文本。目的是檢查聚合指標是否對應**可辨識的生成階段**，並暴露事件定位與輸出格式的混淆。

**本節只承擔定性展示功能，不含任何全樣本 Result。** 正式統計在別處：`s_t` 的 sample-level 行為驗證見 **§4.7**；`p_t` 的 amplitude / frequency 全樣本檢驗見 **§4.4 Result 3–4**。

證據邊界：

- 這些樣本**不是預註冊、不是隨機抽樣、不具代表性**，**不提供** effect size、顯著性或因果證據。
- **不能由個案軌跡證明高 `s_t` 導致正確。**
- **不得僅憑視覺印象宣稱頻率重組**——case plot 只能**提出**問題。
- 第 2 個 answer marker 只是 **second-answer-marker proxy，不是真實 loop onset**；repeated tail 亦僅與 stopping failure / loop-like generation 相容。
- `p_t` 是 EMA residual，**不作 RPE 或 biological dopamine 解讀**。

（marker 合併規則、圖檔生成腳本見 `CLAUDE.md`。）

#### Case Observation 1 — `s_t` Tracks Ongoing Processing and Post-Commit Release

- 推理仍在展開、答案尚未形成或仍在修正時，`s_t` 往往**維持較高水平**。
- 首次明確提交後，`s_t` 通常**下降**，表現為 state release。
- **Q140**（α=+6 較長推理後答對，期間 `s_t` 長時間維持）與 **Q189**（α=+6 長時間未形成正式提交，`s_t` / `p_t` 持續活躍）是 sustained / unresolved processing 的代表案例。
- **Q251 是必須保留的反例**：持續較高的 `s_t` 也可能沿**錯誤路徑**推進。

> 可寫的結論只有：**`s_t` level 與 ongoing / unresolved processing 及 post-commit release 相容。**

這與 §4.2–4.5 的 **pre-commit engagement → post-commit release** 聚合結構同向，但 case 圖只是**核對**該結構，不構成新的證據。不可寫成「高 `s_t` 導致正確」或「`s_t` 是 reasoning quality」。**本節觀察的是 level，且僅為定性**；slope-vigor 的正式檢驗在 §4.7（GSM8K 未檢出），不能由 level 的案例觀察推斷 slope 讀數成立或不成立。

#### Case Observation 2 — `p_t` Illustrates Stage-Dependent Residual Dynamics

- 開放式自然語言推理階段的 `p_t` 往往**幅度較大、變化較不規則**（Q80、Q92、Q140、Q189）。
- 進入 `####`、數字或固定句式的**重複尾段**後，殘差幅度通常**下降**。

這與 §4.4 Result 4 的全樣本結果方向一致（commit 後 centered RMS 下降），但**案例圖本身不能判定 amplitude 或 frequency 效應**——正式判定完全來自 §4.4 的 paired 全樣本分析，該處同時說明頻率指標與 answer format / repetition 不可分離。

#### Case Observation 3 — First-Answer Accuracy Does Not Guarantee Stable Completion

**Q251**：α=0 首次 `####` 給出正確答案 60，因此 first-answer protocol 判為 **correct**；但模型之後仍繼續除以 2 並產生其他候選答案。

> **first-answer accuracy 與 termination quality / stable completion 是不同的行為維度。**

這**不否定** GSM8K 以 first `####` 作 production accuracy 的口徑；它只說明該指標**不能替代** answer switching、重複強度、自然 EOS、hit-cap 或 stable-final-answer 等停止品質指標。作為單一案例，它**提出**而非**估計**這項區分。

#### Integrated Interpretation

1. **`s_t` level** 與持續處理及提交後的 state release 相容，但**不是** correctness 或 reasoning quality 的直接指標（Q251）。
2. **`p_t`** 在推理段與重複尾段呈現不同的視覺波動，方向與 §4.4 Result 4 的 amplitude 結果一致；頻率則不能由圖判讀。
3. **first-answer accuracy 與 stable completion 是不同維度**，需以獨立的停止品質指標度量。

三項觀察都只作為聚合結果的**直觀核對**，其統計地位由 §4.4 與 §4.7 承擔。

#### Evidence Boundary

1. 9 個案例**不是代表性抽樣**（非預註冊、非隨機）。
2. case plots **不提供** effect size、顯著性或因果證據。
3. **高 `s_t` 不等於**正確或高品質推理（Q251）。
4. 案例**不能**用於判定 frequency reorganization。
5. **repeated-ngram tail 與 second marker 都不是真實 loop onset**。
6. **`p_t` 是 EMA residual**，不是已識別的 biological phasic dopamine。

#### Figures

| Figure | Role |
|---|---|
| `plots_gain/sample_traj3_q*.png`（Q10、Q80、Q92、Q140、Q152、Q189、Q225、Q251、Q284） | 展示 α=−6/0/+6 下 `s_t`、`p_t` 與生成文本的樣本級對應；**僅作 qualitative sanity check** |

正文只引用真正承載不同解釋邊界的案例：**Q140 / Q189**（sustained / unresolved processing）、**Q251**（高 `s_t` 的反例，兼 first-accuracy vs stable-completion）、**Q80 / Q92**（`p_t` 的 generation-stage 敏感度）。其餘案例（Q10、Q152、Q225、Q284）保留於 Figures，不逐題展開。

## 5. Qwen2.5 Cross-Model Analysis

本節記錄 Qwen2.5-7B-Instruct 在 GSM8K、neutral 條件下，以 §4 的一維 state 分析鏈所得的結果。**本節結論限於一維投影層次**；manifold 分析已於 2026-08-28 完成並關閉（見 `AdaManifold.md`），跨模型行為差異的定位見 §5.8。

> **一句話結論：Qwen 的入口增益持續隨 α 線性增加，但進入 decode 後，回應沿一個相對固定的 RSN layer profile 被顯著壓縮；現有證據更支持「標量增益壓縮」，不支持「軌跡發生幾何重分配」。**

**閱讀順序即分析順序**：操控驗收（§5.2）→ 主鏈條（§5.3）→ slow state（§5.4）→ fast residual / confidence（§5.5）→ 高劑量壓縮與 null（§5.6）→ 跨模型綜合與證據邊界（§5.7）→ commitment transfer（§5.8）。正文重點放在**主鏈條、高劑量 compression 與 working-state 對齊的可行性判定**三處；其餘為支撐與控制。

### 5.1 Scope and Cross-Model Comparison Rules

**跨模型比較的口徑限制。** Llama 與 Qwen 使用**不同的 mask、不同的 band（L=9 vs L=6）、不同的 activation scale、不同的詞表**，因此相同數值的 α **不是相同強度的 intervention**。

| 項目 | Llama3-8B（§4） | Qwen2.5-7B |
|---|---|---|
| band | `[11,20)`，L=9 | `[16,22)`，L=6 |
| mask | `nmd_0.5_11_20_8B.npy` | `nmd_0.5_16_22_7B.npy` |
| decoder layers | 32 | 28 |
| max_new_tokens | 512 | 768 |
| reference μ/σ | 該模型自身 α=0 No-CoT prefill | 同左（**模型內 reference**，不共用） |
| α 覆蓋 | −8…+8（9 檔） | −8…+12（11 檔，No-CoT）+ CoT {0,+6} |

band 位置是 per-model 的 mask 事實（Qwen 取 layer-wise Expert/Non-Expert Pearson 下降起點），不是可調參數。reference 固定在各自模型的 α=0 prefill，這是兩套 Z 座標不可互換的直接原因。

**四條比較規則，本節逐條遵守：**

1. **一切統計先在模型內標準化**（Z 座標、或相對各自 α=0 的 paired Δ）後才跨模型並列。
2. **比較內部狀態，不比較 raw α。** 可比較的是**形狀**（dose 曲線的飽和位置、response profile 的共線性、commit position 的相對移動），不是絕對量。
3. **entropy 若進入比較，一律使用 `entropy/log(V)`**（Llama V=128k、Qwen V=152,064，未歸一化的 nats 不可比），並採用與 Llama 相同的 commit locator 與 `±20` 窗口。Qwen 的 logit family 已於 §5.5 解封，故本規則現已可套用；但 `entropy/log(V)` 只是**詞表大小歸一化，並非 model-free axis**——tokenizer 粒度與詞表結構仍不同。
4. **cohort 定義必須連同數字陳述。** Qwen 的 pre-commit cohort 選擇在 manipulation 自身的結果上（見 §5.3），這一點在 Llama 不成立，是兩者最重要的不對稱。

### 5.2 Manipulation Check and Effective Dose

本节首先确认 α steering 是否按设计作用于 Qwen 的 task-entry state，并检验入口响应在高剂量下是否仍保持线性。

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 | +12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `G_prefill` | −13.14 | −9.82 | −6.53 | −3.26 | 0.00 | +3.27 | +6.57 | +9.88 | +13.25 | +16.74 | +20.26 |
| `Z_prefill` | −32.44 | −24.23 | −16.11 | −8.03 | 0.00 | +8.05 | +16.19 | +24.35 | +32.68 | +41.33 | +50.06 |
| `boundary_jump`（Z） | +32.40 | +24.17 | +16.07 | +8.05 | +0.11 | −7.87 | −15.91 | −23.49 | −31.63 | −40.36 | −49.07 |
| `jump/Z_prefill` | −0.999 | −0.997 | −0.997 | −1.003 | — | −0.977 | −0.983 | −0.965 | −0.968 | −0.976 | −0.980 |

**入口操控保持线性。** `G_prefill ~ α` 的线性拟合为 `R²=0.99987`（slope=1.661），`Z_prefill ~ α` 为 `R²=0.99985`。该关系一直维持到 α=+12；因此，+8 之后出现的行为平台不能归因于入口注入失效。

**Qwen 模型内的有效剂量**为：

$$
\Delta G_{\text{prefill}}/\Delta\alpha = 1.661,
\qquad
\Delta Z_{\text{prefill}}/\Delta\alpha = 4.100.
$$

这些斜率只用于 Qwen 内部的剂量校准。其数值不能与 Llama 的斜率直接比较，因为两模型使用不同的 mask、layer band 与 reference。

**入口偏移在 decode 开始时近乎完全释放。** `boundary_jump = Z_decode[0] − Z_prefill` 与 `Z_prefill` 方向相反，二者比值在全部 11 个剂量上稳定为 `−0.965…−1.003`，CoT α=+6 时为 `−0.960`。这说明一次性的 prefill steering 没有在 decode 中留下持续的线性 offset，复制了 Llama 中观察到的 entry–decode decoupling / boundary-gating 形态。

因此，本节得到两个结论：

1. α 对 Qwen task-entry gain 的控制有效，并在完整剂量范围内保持线性；
2. 入口状态虽然在 decode 起点迅速释放，仍可伴随后续 commitment 与行为变化，支持 initial-condition / boundary-gating 的解释。

> 图：`fig51_entry_gain.png`。左图显示入口线性，中图显示 accuracy 平台，右图显示 commitment timing 与 pre-commit coverage。

### 5.3 Main Chain: `G_prefill` → pre-commit `s_t` → commitment timing → accuracy

| α | −8 | −6 | −4 | −2 | 0 | +2 | +4 | +6 | +8 | +10 | +12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acc(300) | 62.00 | 65.33 | 68.00 | 68.00 | 67.67 | 68.33 | 73.67 | 77.67 | 86.00 | 88.00 | 87.67 |
| commit `c_med` | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 47 | 110 | 134 | 163 |
| `posN_med` | .010 | .010 | .010 | .010 | .010 | .010 | .010 | .190 | .828 | .839 | .854 |
| decode length | 386.5 | 377.4 | 388.4 | 392.0 | 397.4 | 381.9 | 377.6 | 304.5 | 258.9 | 269.5 | 279.4 |
| post-commit step | 374.2 | 368.9 | 381.2 | 383.0 | 387.9 | 371.4 | 362.0 | 237.3 | 111.5 | 104.1 | 103.5 |
| pre-span ≥20 % | 8.3% | 6.0% | 4.3% | 4.3% | 5.7% | 6.0% | 9.3% | 58.0% | 96.0% | 97.0% | 96.7% |
| `s_t` early / mid / late | .66/.81/.96 | .65/.83/.91 | .68/.84/.96 | .72/.86/1.08 | .74/.86/1.07 | .73/.83/1.04 | .76/.88/1.07 | .89/.89/1.17 | .90/.91/1.28 | .89/.93/1.33 | .89/.92/1.35 |
| loop% | 13.7 | 10.7 | 11.7 | 14.3 | 13.0 | 8.7 | 10.3 | 5.3 | 2.7 | 2.7 | 3.3 |
| eos_fail% | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

表格中每一列代表一个 α 剂量，各行含义如下：
- **Accuracy**：300 道题的正确率。
- **Commit step (`c_med`)**：首次生成 `####` 的 token 位置中位数；越大表示越晚正式提交答案。
- **Normalized commit position (`posN_med`)**：commit step 除以总生成长度，范围约为 0–1；越接近 0 越像“先答后推理”，越接近 1 越接近在推理末尾提交。
- **Decode length**：平均生成长度。
- **Post-commit steps**：首次 `####` 之后仍继续生成的平均 token 数；越长表示提交答案后仍有大量输出。
- **Pre-span ≥20 coverage**：至少具有 20 个 commit 前 token 的样本比例；决定该剂量下的 pre-commit `s_t` 分析是否可靠。
- **`s_t early / mid / late`**：commit 前慢状态在早期、中期和后期窗口的平均水平，用于观察模型在正式提交前如何维持或释放内部状态。
- **Loop %**：输出出现明显重复或循环的样本比例。
- **EOS fail**：没有自然生成结束符、最终撞到长度上限的比例。

**主要結果。** Qwen 的核心鏈條在 `+4→+8` 間最明顯：隨 task-entry gain 線性上升（§5.2），pre-commit `s_t` 整體抬高（early：`.74→.90`），答案提交位置由生成開頭移至後段（`posN_med: .010→.828`），accuracy 同時由 `67.67%` 提升至 `86.00%`。loop rate 亦由 `13.0%` 降至 `2.7%`，說明表現提升並非來自更長的重複輸出。

`+8` 之後，accuracy 與相對 commit 位置進入平台：accuracy 為 `86.00%→88.00%→87.67%`，`posN_med` 為 `.828→.839→.854`。但絕對 commit step 仍由 `110→134→163` 持續延後，因為生成長度同時由 `258.9→269.5→279.4` 增加。因此，趨平的是 commitment 在完整生成中的**相對位置**，而非絕對提交時間。

**與 Llama 的差異。**

- Qwen 在觀察範圍內呈現「上升後平台」，未出現 Llama 以 `α=−6` 為最佳點、兩側下降的非線性工作點。
- Qwen 在 `α≤+4` 時通常於生成開頭提交答案（`c_med=3`），因此 `91–96%` 的樣本沒有足夠的 pre-commit window。這些低劑量 cell 只能用於呈現 coverage，不能作為後續 pre-commit state analysis 的直接對照。

**解讀邊界。** `+8` 之後的行為平台與 §5.6 的 decode-response compression 同期出現，但目前只能視為相互對應的現象。現有分析尚未證明 decode compression 導致 accuracy 飽和，也未建立 `s_t → commitment timing → accuracy` 的中介因果鏈。

**CoT（α ∈ {0,+6}，各 n=300）沿同一鏈條移動，且不改變上述形態：**

| Condition | acc | `Z_prefill` | `jump/Zpre` | `c_med` | `posN` | `s_pre`(n_risk) | loop% |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoT α=0 | 77.00 | 0.00 | — | 3 | .0095 | 1.697 (12) | 12.3 |
| CoT α=+6 | 87.67 | +24.00 | −0.960 | 163 | .857 | 1.672 (175) | 5.0 |

- **acc**：回答正确率。
- **`Z_prefill`**：生成开始前的标准化 task-entry RSN gain；α=0 按定义为 0。
- **`jump/Zpre`**：从 prefill 到第一个 decode token 时，入口增益回弹的比例；`−0.960` 表示约 96% 的入口偏移迅速消退。
- **`c_med`**：首次生成 `####` 的 token 位置中位数。
- **`posN`**：commit 在完整输出中的相对位置。
- **`s_pre`**：commit 前 slow state 的平均水平。
- **`n_risk`**：有足够 commit 前窗口、能够计算 `s_pre` 的样本数。
- **loop%**：出现重复或循环输出的比例。

整体上，`+6` 主要让 CoT 模型更晚提交答案、提高正确率并减少循环；`s_pre` 的水平几乎不变。CoT 把 α=0 的 accuracy 由 67.67 抬到 77.00，**但沒有改變「低 α 先答後推」**（`c_med` 仍為 3，coverage 5.0%）。CoT α=+6 已達到 No-CoT α=+8…+12 的 accuracy 平台（87.67）與 commit 位置（`posN` .857），即 **CoT 與 +α 在此任務上推動同一條鏈條，且 CoT 讓較低的 α 就達到平台**。兩個 cell 不足以構成 CoT 的 dose curve，不可外推。

#### 5.3.1 Post-commit slow-state release：same-sign but attenuated

相同 `±20` token 窗口（`[c−20,c)` vs `[c,c+20)`），同一份 `phase1_gain` 程式碼路徑：

| Condition | `s_pre` | `s_post` | release | `d_z` | coverage | n |
|---|---:|---:|---:|---:|---:|---:|
| **Llama α=0 No-CoT** | −0.001 | −0.280 | **−0.279** | −0.785 | 73.3% | 220 |
| **Llama α=0 CoT** | +0.466 | +0.029 | **−0.436** | −1.544 | 77.3% | 232 |
| **Qwen α=+8** | 1.448 | 1.300 | **−0.148** | −0.356 | 70.7% | 212 |
| **Qwen α=+10** | 1.432 | 1.261 | **−0.171** | −0.416 | 73.0% | 219 |
| **Qwen α=+12** | 1.492 | 1.259 | **−0.233** | −0.557 | 75.7% | 227 |
| *Qwen α=+6* | *1.172* | *1.156* | *−0.015* | *−0.042* | *41.3%* | *124* |
| *Qwen α=+4* | *1.114* | *1.172* | *+0.058* | *+0.146* | *7.0%* | *21* |
| *Qwen α=+2* | *0.900* | *0.948* | *+0.048* | *+0.103* | *4.3%* | *13* |
| *Qwen α=0* | *0.926* | *1.069* | *+0.144* | *+0.321* | *4.0%* | *12* |
| *Qwen α=−2* | *1.041* | *1.032* | *−0.008* | *−0.022* | *3.0%* | *9* |
| *Qwen α=−4* | *0.960* | *0.973* | *+0.013* | *+0.024* | *3.0%* | *9* |
| *Qwen α=−6* | *1.019* | *1.074* | *+0.055* | *+0.121* | *4.0%* | *12* |
| *Qwen α=−8* | *1.209* | *1.097* | *−0.112* | *−0.232* | *6.3%* | *19* |

斜體列 coverage < 50%，僅為 coverage diagnostic，不可用於機制判斷。

**結論**

> Qwen 与 Llama 在 commit 后都出现状态释放，但 Qwen 的释放更弱、更缓慢，属于 **方向一致但效应减弱（same-sign but attenuated）**，不能视为完全复制。由于 Qwen 只有高剂量条件具备足够的 pre-commit 数据，且缺少可用的 α=0 对照，该结果仅作为补充观察，不能进行严格的跨模型比较。

> 圖：`qwen2.5/dopamine/plots_gain/fig5_qwen_commit_centered.png`（`qwen_signal/plot_qwen_mainfig.py`）。

### 5.4 Slow-State Behavioral Validation

複現 §4.7 的兩問：`s_t` 的 **level** 是否關聯 commitment timing，`s_t` 的 **slope** 是否兌現 vigor 預測。**預測窗一律是固定的 `[0,20)` early window**（§4.7 口徑），at-risk = `commit_step ≥ 20`，即窗口不跨越 commit。

| α | n_risk | coverage | `vigor_slope` | 95% CI | `s_t` level |
|---:|---:|---:|---:|---|---:|
| −8 | 25 | 8.3% | — | *(coverage too low)* | — |
| −6 | 18 | 6.0% | — | *(coverage too low)* | — |
| −4 | 13 | 4.3% | — | *(coverage too low)* | — |
| −2 | 13 | 4.3% | — | *(coverage too low)* | — |
| 0 | 17 | 5.7% | — | *(coverage too low)* | — |
| +2 | 18 | 6.0% | — | *(coverage too low)* | — |
| +4 | 28 | 9.3% | — | *(coverage too low)* | — |
| +6 | 174 | 58.0% | **−0.0211** | [−0.0269, −0.0162] | 1.163 |
| +8 | 288 | 96.0% | −0.0063 | [−0.0102, −0.0023] | 1.063 |
| +10 | 291 | 97.0% | +0.0008 | [−0.0028, +0.0043] | 1.052 |
| +12 | 290 | 96.7% | +0.0018 | [−0.0017, +0.0052] | 1.082 |

（CoT：α=0 coverage 5.0% 拒絕；α=+6 coverage 73.0%，slope **−0.0117** [−0.0165, −0.0066]，level 1.158。）

**结论** Qwen 在 GSM8K 上同样没有出现 ramping/vigor 信号。可分析的 `+6～+12` 条件呈现的是逐渐趋近于零的负斜率，即状态缓慢松弛，而非随推理推进持续增强。`α≤+4` 时模型通常在 decode step 3 就提交答案，因此不存在足够的 pre-commit 区段；这是“过早提交”的行为结果，而不是待补的数据缺口，也不能通过扩大窗口解决。

另一方面，`s_t` 的整体水平随剂量升高，并与 commit position 后移同向，说明较高 slow-state level 与较晚提交相伴。但不同剂量的可分析样本覆盖率高度分离，无法复现 Llama 的同剂量、leak-free 回归。因此这里只能报告 **slow-state level 与 commitment timing 的描述性共变**，不能比较关联强度或建立因果关系。

> `n_risk` 必须按表解释：`vigor_slope` 只要求 commit 前 ≥20 步；`s_pre/s_post` 和 `p_t` 要求 commit 前后各 ≥20 步，因此数值不同是正确的。

### 5.5 Fast Residual and Output Decisiveness

`p_t = Z_t − s_{t-1}`，是**同一條一維投影相對 EMA baseline 的殘差**。

| α | `abs_mean` | `std` | `pre_abs` | **`at_commit`** | `post_abs` | n_risk |
|---:|---:|---:|---:|---:|---:|---:|
| −8 | 1.0518 | 1.3074 | 0.9329 | **−1.0375** | 1.0243 | 19 |
| −6 | 1.0494 | 1.3055 | 0.9106 | **−0.7589** | 1.1456 | 12 |
| −4 | 1.0497 | 1.3077 | 0.8829 | **−0.6232** | 1.0715 | 9 |
| −2 | 1.0599 | 1.3222 | 0.9753 | **−0.5499** | 1.0313 | 9 |
| 0 | 1.0566 | 1.3129 | 0.9157 | **−0.5777** | 1.0508 | 12 |
| +2 | 1.0540 | 1.3100 | 0.8886 | **−0.7355** | 1.1339 | 13 |
| +4 | 1.0460 | 1.3089 | 0.9227 | **−0.8534** | 1.0294 | 21 |
| +6 | 1.0077 | 1.2608 | 0.9006 | **−0.8124** | 0.9646 | 124 |
| +8 | 0.9997 | 1.2520 | 0.9180 | **−1.1815** | 1.0369 | 212 |
| +10 | 0.9986 | 1.2528 | 0.9303 | **−1.2080** | 1.0534 | 219 |
| +12 | 1.0072 | 1.2642 | 0.9416 | **−1.3219** | 1.0668 | 227 |

- **α**：RSN steering 的剂量。
- **`abs_mean`**：整条 decode 中 `|p_t|` 的平均值，表示 fast residual 的整体振幅。
- **`std`**：整条 decode 中 `p_t` 的标准差，表示 residual 的整体波动程度。
- **`pre_abs`**：commit 前窗口内 `|p_t|` 的平均值，表示提交答案前的 residual 振幅。
- **`at_c`**：首次生成 `####` 当步的带符号 `p_t`；负值表示 commit 时 RSN projection 相对 slow baseline 突然下降。
- **`post_abs`**：commit 后窗口内 `|p_t|` 的平均值，表示提交答案后的 residual 振幅。
- **`n_risk`**：commit 前后都有足够窗口、能够计算 `pre_abs` 和 `post_abs` 的样本数。

**结论**

> Qwen 的 α 干预不会整体放大生成过程中的快速波动，但会改变答案提交瞬间的状态转折。换言之，α 主要调节 **commit 时刻的快速切换**，而非全程提高 residual amplitude。该结果与 Llama 部分一致，但 `p_t` 是 slow state 的残差成分，只能作为同一机制链的补充读数，不能视为独立的因果证据。

需要注意：

- `p_t` 是从 slow state 中分解出的快残差，不是独立的第二条因果证据。
- output decisiveness 只有 7 个剂量条件，因此只能描述单调趋势，不能判断峰值或倒 U 曲线。
#### Result 2 — Task-Entry Output Decisiveness Increases with α

最後一個 prompt token 的輸出分布：

| α | `entropy` | `entropy/log(V)` | `top1` | `margin` |
|---:|---:|---:|---:|---:|
| −8 | 2.099 | 0.1759 | 0.2402 | 0.0561 |
| 0 | 1.031 | 0.0864 | 0.6385 | 0.4671 |
| +6 | 0.969 | 0.0812 | 0.6816 | 0.5170 |
| +8 | 0.572 | 0.0479 | 0.7979 | 0.6502 |
| +12 | 0.315 | 0.0264 | 0.8924 | 0.7961 |

| 擬合（n=5 cells） | slope | R² |
|---|---:|---:|
| `entropy` | −0.0846 | 0.935 |
| `top1` | +0.0311 | 0.941 |
| `margin` | +0.0348 | 0.953 |

**Task entry 隨 α 單調變得更 decisive。** 與 §5.2 的 `G_prefill` 入口線性同向：α 在 task entry 即已同時改變 RSN state 與 output distribution。**僅為單調趨勢**——5 個 cell 無法判斷是否存在峰值。

#### Result 3 — Commit Produces a Stage-Specific Confidence Transition

對稱 `±20` 窗口，Δ = post − pre：

| α | cov% | n | Δ`entropy` | `d_z` | Δ`top1` | `d_z` | Δ`margin` | `d_z` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +6 | 41.3 | 124 | **+0.2145** | +0.603 | **−0.0566** | −0.554 | **−0.0809** | −0.564 |
| +8 | 70.7 | 212 | **+0.1502** | +0.473 | **−0.0360** | −0.376 | **−0.0503** | −0.360 |
| +12 | 75.7 | 227 | **+0.1400** | +0.399 | **−0.0301** | −0.302 | **−0.0401** | −0.279 |

（三個可讀 cell 全部 `p<0.01`。）

`0` 與 `−8` 的 coverage 僅 **4.0%（n=12）** 與 **6.3%（n=19）**，是 **coverage 診斷列**，不承載機制解讀。

commit **當步**本身則隨劑量進一步 sharpen（`pre` 窗口平均 entropy 減去當步 entropy）：

| α | +6 | +8 | +12 |
|---|---:|---:|---:|
| at-commit sharpening | +0.044 | +0.058 | +0.092 |

**α 使答案提交瞬間更 decisive，但提交後的分布反而變鬆。** 此結構與 Result 1 的 `p_t` commit-locked dip、§5.3.1 的 `s_t` post-commit release 是**同一事件的不同讀數**。

三項限制隨數字同行：

- **格式效應無法排除。** `####` 恰在對齊點改變輸出格式，故 post-commit 的 entropy 上升不可全部解讀為真實 confidence 下降。
- **Cohort 選自 manipulation 自身結果。** `≥20` 步 pre-commit span 的涵蓋率由 α=0 的 4.0% 升至 +12 的 75.7%——可讀 cell **無配對的 α=0 對照**（同 §5.3.1）。
- **固定 decode 分位僅為描述性。** α 將 commit 中位數由 decode step ≈3 推遲至 ≈187，固定分位在各 cell 取樣的是不同生成階段。

#### Result 3b — CoT

CoT `+6`（cov 58.3%、n=175）：

| 指標 | Δ | p |
|---|---:|---:|
| `entropy` | +0.022 | 0.46 |
| `top1` | −0.0002 | 0.59 |
| `margin` | −0.0018 | 0.88 |

**A commit-locked confidence transition was not detected in the CoT +6 readable cohort.** 不寫「CoT abolishes the transition」——可讀 CoT cell 僅此一個，且 cohort 由 commitment timing 篩選產生（CoT `α=0` coverage 僅 4.0%、n=12，無法構成對照）。

#### §5.5 綜合結論

> 对 Qwen 做 α 干预，不仅改变了模型内部的 RSN 状态，也改变了模型输出 token 时的“确定程度”，而且这种变化随生成阶段不同。

具体分三段：

- **刚进入任务时**：α 越高，输出分布越集中，模型更快进入明确的生成状态。
- **正式提交答案时**：这种集中进一步增强，模型更明确地选择答案。
- **提交答案之后**：entropy 上升、top1 probability 和 margin 下降，说明答案一旦写出，模型的紧绷/确定状态开始释放。

所以它不是一种只改变抽象“推理意愿”、而完全不触碰输出决策的干预；它也会改变实际的 token 分布，因此不能称为非常纯粹的 `selective wanting intervention`。

结果是：commit 本身伴随阶段性的输出分布变化，但 **Llama 的过早 commit 并不是因为置信度特别高**。因此当前更准确的理解仍是“承诺时机发生失调”，而不是“模型过度自信”。

### 5.6 High-Dose Compression and Direction Specificity

#### 5.6.1 Decode-Response Compression on the Delayed-Commit Subset (Not a Ceiling)

固定 cohort：在 +6/+8/+10/+12 **全部**四檔都具備 ≥20 步 pre-commit span 的題目，n=167。

| α | `x_prefill` | `s_pre` | `Z_prefill` | `Z_pre` | c_med |
|---:|---:|---:|---:|---:|---:|
| +6 | 134.98 | 2.342 | 24.41 | 1.417 | 93 |
| +8 | 183.26 | 2.973 | 32.74 | 1.641 | 125 |
| +10 | 235.37 | 3.167 | 41.39 | 1.673 | 163 |
| +12 | 288.01 | 3.250 | 50.12 | 1.752 | 187 |

| 劑量步 | Δ decode | RAW response ratio | paired p |
|---|---:|---:|---:|
| +6→+8 | 0.63 | **0.0131** | 0.00114 |
| +8→+10 | 0.19 | **0.0037** | 0.0147 |
| +10→+12 | 0.08 | **0.0016** | 0.361 |

入口每 +2α 遞增約 50 個 raw 單位，decode 側的每單位回應則單調下降。

**結論** 這是 **compression 而非 ceiling**：`+6→+8`、`+8→+10` 仍顯著，decode 側並未停止回應，只是每單位 α 的回應變小。

- headline 用 **RAW** ratio；Z ratio（0.0269 / 0.0037 / 0.0091）非單調，僅作 sensitivity。
- cohort 選擇在 manipulation 的結果上（§5.3），故本表**不含可比的低劑量對照**，是高劑量區間內部的相對比較。

> 圖：`fig52_compression.png`（左 entry、中 decode、右 RAW response ratio）

#### 5.6.2 High- and Low-Dose Response Profiles Are Near-Collinear

以 `[c−20, c)` 的逐層 pre-commit `s_t`，取每 +2α 的逐層回應向量
（在固定的 167 道题上，α 增加后，该层在 commit 前 20 个 token 内的平均 slow-state s_t 改变量。）

| step | L15 | L16 | L17 | L18 | L19 | L20 | mean | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +6→+8 | 0.5374 | 0.3713 | 0.1690 | 0.2367 | 0.5221 | **−0.1986** | 0.2730 | 0.92 |
| +8→+12 | 0.1829 | 0.0899 | 0.0344 | 0.0528 | 0.1688 | **−0.0814** | 0.0746 | 1.19 |

最小二乘 $v_{high} \approx k\,v_{low}$：**k = 0.309**（幅度每劑量步縮小 3.24×），最佳縮放解釋 **97.4% 的 response energy**，歸一化**平方**殘差 **2.6%**（殘差**範數**比例 16.1%），**cos = 0.987**

这个表计算的是：**当 α 继续增加时，Qwen 各层的 pre-commit slow state `s_t` 分别改变了多少。**

- `+6→+8`：低剂量阶段每增加 `+2α`，L15–L20 各层的响应变化。
- `+8→+12`：高剂量阶段的变化，并换算为每 `+2α` 的响应。
- L15–L20 六个数字组成一个六维的 **layer-response profile**。
- `mean`：六层响应的平均幅度。
- `CV`：不同层之间响应是否均匀；越大表示层间差异越大。

然后比较两个 response profile：

- `cos=0.987`：两组向量几乎同方向，说明高低剂量主要保持相同的逐层响应形状。
- `k=0.309`：高剂量响应约为低剂量的 31%，即单位剂量带来的变化缩小了约 3.24 倍。
- `97.4% response energy`：只需把低剂量 profile 整体缩小，就可以解释高剂量 profile 的绝大部分。
- L20 在两行都为负，也说明这个反向 loading 是稳定结构，并非高剂量突然发生方向改变。

**结论**

> Qwen 在高剂量下仍保持相同的逐层 RSN response profile，但整体响应幅度明显减弱，符合**沿固定 layer profile 的标量增益压缩**。这不是所有层同时达到 ceiling，也没有证据显示响应方向发生旋转或层间几何重组。`k` 用于量化压缩幅度，其余指标仅作描述，不支持显著的层×剂量交互作用。
> 圖：`fig53_profile.png`

#### 5.6.3 Random / Orthogonal Readout Controls

將**同一批** hidden states 重投影到三個 null family（各 10 draw，共 30）：

| | RSN | `diff_random` | `ortho_gauss_same` | `ortho_gauss_off` |
|---|---:|---:|---:|---:|
| resid（能量比） | **0.026** | 0.341 | 0.102 | 0.227 |
| pctile / p | — | 0.0% / .182 | 0.0% / .182 | 20.0% / .545 |
| CV(+6→+8) | **0.917** | 2.373 | 3.943 | 2.936 |
| k | 0.309 | 0.250 | 0.358 | 0.240 |
| `‖v₆₈‖` 中位數 | 0.907 | 0.270 | 0.752 | 0.349 |

**结论**

> 与三类随机/正交方向相比，RSN 的 residual 和 CV 更低，说明其逐层响应更接近一个稳定、干净的标量通道，而非层间几何重分配。其中，证据主要来自幅度匹配最好的 `ortho_gauss_same` 对照。由于这些结果只是对 RSN steering 后的状态重新投影，因此仅支持 **readout specificity**，不能证明 RSN 干预方向具有因果特异性。受限于 null 数量，结论应写作 **argues against redistribution**，而非完全排除。

> 圖：`fig54_null.png`

### 5.7 Cross-Model Synthesis and Evidence Boundaries

本節整合 §5.2–§5.6 的跨模型結果，回答同一個問題：**兩模型哪些結果一致、哪些不同，以及目前能比較到什麼程度。** 分析鏈條的「複用」與結果的「複製」必須分開陳述——六個環節都以相同口徑跑通，但這只說明方法可移植。

#### 5.7.1 複製的結果（機制層次）

- **入口線性**：`G_prefill ~ α`，R²=0.9999，維持到 +12。
- **entry–decode 解耦**：`jump/Z_prefill` 穩定在 −0.965…−1.003，一次性 prefill 注入在 `decode[0]` 近乎完全釋放。
- **commit-locked 的帶符號 `p_t` dip**，隨劑量單調加深（−0.578 → −1.322）。
- **commit-locked 的 output-distribution transition**（§5.5 Result 3）：commit 後 `entropy` 上升、`top1`/`margin` 下降，方向與 Llama 一致。但其 **dose dependence 是階段特異且取樣稀疏的**——commit 後的 Δ 幅度隨劑量遞減（`d_z` −0.55 → −0.30），僅 7 cell，格式效應無法排除。

#### 5.7.2 同向但減弱（第三類，見 §5.3.1）

- **post-commit slow-state release**：相同 `±20` 窗口下方向一致但幅度較小（Qwen `+12` 為 `−0.233`，Llama α=0 No-CoT 為 `−0.279`，即 最大格約 **0.84 倍**（`−0.233` vs `−0.279`），較低兩格 0.53–0.61 倍）。Qwen 由高位緩慢回落而非明顯解除狀態。**不可歸入複製或未複製任一類**：歸為複製會抹去 Qwen 停在高位的事實，歸為未複製則與資料矛盾。

#### 5.7.3 未複製的結果

- **行為曲線**：**Llama 呈 asymmetric peaked response**（尖銳最佳點 α=−6、−8 崩潰、正 α 端逐步下降後趨平），**Qwen 呈高劑量平台**（單調上升後於 +8 飽和，band 內無右臂）。
- **`p_t` amplitude 的劑量效應**：Llama 在 α=−6 有乾淨的 pre-commit centered RMS 上升；Qwen 的 `abs_mean`/`std` 在整個 −8…+12 幾乎不動。

**兩項在此任務上同向為 null**：`vigor_slope`（§5.4，兩模型皆未在 GSM8K 誘發 ramping 斜率信號）與 frequency organization（§5.5，Llama 已測皆 null，Qwen 依此設計未計算）。

#### 5.7.4 比較邊界

**working state 對齊目前不可執行。** 下表逐量說明為何：

| 待比較量 | Llama 最佳點 `α=−6` | Qwen 高表現區 `α=+8…+12` | 當前判定 |
|---|---:|---:|---|
| `Z_prefill` | `−19.74` | `+32.68…+50.06` | 不可直接比較：兩者以各自模型的 α=0 reference 標準化 |
| pre-commit `s_t` | Δ vs α=0 = `+0.449`（`d_z=0.584`，paired `n=291`） | level = `1.448`（at-risk `n=212`） | 不可直接比較：一側是配對變化量，另一側是選定 cohort 的絕對水平 |
| commitment timing | 各劑量約 `77%` coverage，`posN` 隨 α 平緩變化 | `posN: .010→.828`，coverage `5.7%→96.0%` | 可比較變化形態，不可視為相同 commitment state |
| output decisiveness | 齊備（11 cell） | 已提取（7 cell，§5.5） | `PARTIALLY AVAILABLE / SPARSELY SAMPLED`：可比較 commit-locked transition 的**方向**，不可比較劑量曲線 |

四條邊界，任何跨模型陳述都必須同時聲明：

1. **raw α 與狀態絕對值不可比**——mask、L=9 vs L=6、activation scale 皆不同，等量 α 不是等量干預。
2. **Qwen 的 commit-aligned cohort 存在選擇偏差**——其 pre-commit window 之所以存在，正是因為 α 推遲了提交（α=0 僅 12/300 有 ≥20 個 commit 前 token，`+8/+10/+12` 則有 212–227 個）。這是 **post-treatment selection**，增加樣本無法解決。
3. **只能說共享部分調節結構，不能說到達相同 working state**——可確認的一致性是：兩模型都呈現 task-entry gain 與 decode dynamics 的解耦，且高表現區均伴隨 commitment timing、生成穩定性與退化尾巴的系統性變化。
4. **random/orthogonal remask 只是 readout control，不是因果注入**——§5.6.3 僅界定 readout specificity；steering direction 的因果特異性需另行注入並重新採集（→ P2）。

> Qwen 在高剂量时，RSN 的作用开始“变弱”，但作用方式没有改变。

具体来说：

- 不同层的响应强弱本来就不一样，L20 甚至方向相反，因此不是所有层一起碰到上限。
- 低剂量和高剂量的六层响应模式几乎保持相同，只是高剂量时整体缩小到约 **31%**（`k=0.309`）。
- 所以更像是把同一个 RSN 响应模式整体“调小音量”，而不是高剂量引发了新的层间结构或作用方向。

一句话结论：

> **Qwen 的高剂量平台更可能来自原有 RSN 通道的增益压缩，而不是神经表示发生了重新组织。**

### 5.8 Commitment Transfer Function: How Similar Entry Gains Produce a Peak and a Plateau

前述 manifold 分析顯示，entry geometry 不足以解釋兩模型的行為差異。本節利用兩模型完整劑量曲線與逐題配對資料，進一步定位下游的 commitment 環節。

兩模型的全部 20 個 cell 均包含相同的 300 道題且順序一致，因此可按題目進行配對分析；配對關係已逐 cell 驗證。分析實作與資料驗收細節見 `CLAUDE.md`。

#### 5.8.0 Commit State Is Categorical, Not Binary

Commit state 分为三类：

| State | 定义 | Llama α=0 |
|---|---|---:|
| `committed` | 存在可解析的 `#### <数字>` | 177 |
| `marker_unparsed` | 出现 `####`，但答案不可解析；其中 52 题为重复 commit loop | 66 |
| `no_marker` | 全文未出现 `####` | 57 |

三类互斥且穷尽（177+66+57 = 300），依据是 marker 是否**出现**。`marker_unparsed` 中的 loop 子集表示反复提交，而非从未提交，因此不能归入 `no_marker`；其余 14 题带 1–3 个 marker。此外，123 个不可解析样本（66+57）经 fallback 后仍有 63 题答对。

> **Commit state 衡量生成过程中的提交行为，不是 accuracy proxy；格式异常不等于答案错误。**

#### 5.8.1 Within-Model Dose–Response Curves (Item-Paired, n=300 per Cell)

每格對各自 α=0 逐題配對，附 bootstrap 95% CI。**raw α 不跨模型比較。**

| Llama α | acc | early-cand% | posN | `unparsed%` | (其中 `loop%`) | `nomk%` |
|---|---:|---:|---:|---:|---:|---:|
| −8 | 40.67 | **77.0** | **0.043** | 23.7 | 18.0 | 17.0 |
| **−6** | **79.67** | **19.3** | **0.298** | 37.3 | 32.0 | 4.0 |
| −4 | 74.33 | 30.3 | 0.247 | 32.3 | 27.3 | 10.3 |
| 0 | 60.00 | 47.7 | 0.219 | 22.0 | 17.3 | 19.0 |
| +8 | 49.67 | 69.3 | 0.193 | 22.0 | 18.7 | 23.7 |

| Qwen α | acc | early-cand% | posN | `unparsed%` | (其中 `loop%`) | `nomk%` |
|---|---:|---:|---:|---:|---:|---:|
| −8 | 62.00 | 95.0 | 0.487 | 17.3 | 4.3 | **0.0** |
| 0 | 67.67 | 96.3 | 0.553 | 20.3 | 5.0 | **0.0** |
| +6 | 77.67 | 47.7 | 0.587 | 9.0 | 1.3 | **0.0** |
| **+8** | **86.00** | **5.0** | 0.657 | 2.0 | 0.7 | **0.0** |
| +12 | 87.67 | 4.0 | 0.697 | 2.3 | 1.3 | **0.0** |

**两个主要读数的操作型定义（frozen 2026-08-21，两任务共用一套，不调参）：**

- `early-cand%`：**首个非空行**同时满足 (1) strip 后 ≤ **60 字符**、(2) 含至少一个数字 token、(3) 不是编号推理开头（`1. To find …`）、(4) 不是纯标题行（`Step 1:`／`Solution:`）——即模型在任何推导之前就写下一个答案形状的裸数字。长度上限 40/60/80 的敏感度并列报告，结论不依赖该选择。
- `posN`：**首个可解析的** `#### <数字>` 的字符起点除以生成总字符数，仅在该样本存在可解析 marker 时有定义（即 `committed` 一类）。越接近 0 越早提交。以字符而非 token 计；token 口径需 tokenizer，缺失时列直接省略而非估算。

> **注意：`early-cand%` 本身是干预的结果，按它分层属 post-treatment stratification，只能作为 consistent-with 证据，不构成 mediation。**
- `unparsed%`：出现 `####` 但答案不可解析的比例（= `marker_unparsed`）；括号内 `loop%` 是其中 marker ≥4 次的退化反复提交子集。
- `nomk%`：全文完全没有出现 `####` marker 的比例。

> **Qwen 的 `nomk%` 在全部 11 格均为 0.0**——它总是写出 `####`，其不可解析样本全部属于 `marker_unparsed`（典型形式 `70\n####\n\n`：marker 已发出但其后无数字）。这与 §4 记录的 Qwen commit 计数口径差异同源。**此前版本因分类器把带 1–3 个 marker 的样本误归入 `nomk%`，报出 1.0–16.7% 的虚假比例（全部 20 格共 469 题受影响，Qwen 337、Llama 132）；上表为修正后的值。**

> 在各模型内部，剂量不仅改变准确率，也系统性改变答案形成与提交的时序。

- **Llama**：`−6` 时过早回答最少、准确率最佳；到 `−8` 时答案候选和 commit 大幅提前，并伴随异常提交，准确率随之崩溃。
- **Qwen**：正向剂量逐渐推迟答案形成和提交；到 `+8` 后相关指标趋于饱和，因此准确率表现为平台，而非继续上升。

> **Llama 的 peak 与 Qwen 的 plateau，都与各自的 commitment timing 曲线相对应。** raw α 只用于模型内部比较，不能直接跨模型对齐。

#### 5.8.2 Comparing Transfer Curves in Standardized Entry Coordinates

为避免直接比较不可通约的 raw α，我们将入口变化转换为模型内标准化坐标：

$$
z=\frac{\bar{x}_{\text{prefill}}(\alpha)-\bar{x}_{\text{prefill}}(0)}
{\mathrm{SD}_{\alpha=0}}
$$

该坐标只表示干预在**各模型内部**造成的相对入口变化，不代表两模型接受了等量干预。

即使处于相似的强负向 entry displacement，两模型的行为仍明显不同：Llama 在 `α=−6` 时抢答率仅 19.3%、准确率为 79.67%；Qwen 在相同 raw α 下抢答率为 95.3%、准确率为 65.33%。

> **两个模型都表现出 entry gain，但 entry gain 如何转化为 commitment timing 与最终准确率，由模型特异的下游转换机制决定。**

#### 5.8.3 Does Commitment Explain Behavior Beyond Entry Gain?

在剂量曲线层级，commitment timing 比 entry gain 更能解释 Llama 的准确率变化：

| Accuracy curve predictor | Llama (9 cells) | Qwen (11 cells) |
|---|---:|---:|
| Entry gain `z` | 0.136 | 0.898 |
| Early candidate | **0.945** | 0.923 |
| `z` + early candidate | **0.964** | **0.981** |

表格里的数值是 **R²**，计算单位是“剂量 cell”，不是单道题。具体做法是：把每个剂量下的总体 accuracy 作为因变量，分别拟合：

- `accuracy ~ entry gain z`
- `accuracy ~ early-candidate rate`
- `accuracy ~ z + early-candidate rate`

Llama 的 entry gain 几乎无法解释其 peak–collapse 曲线，而 early candidate 可以；Qwen 的两个指标则都随剂量单调变化。由于只有 9/11 个 cell，这些 R² 仅为描述性证据，不构成因果中介分析。

逐题分析提供了更稳健的证据：在全部 20 个 cell 中，抢答样本的正确率均低 6–36 个百分点。控制题目难度与剂量后，抢答仍显著增加错误概率：

| Model | Early → P(error) | SE | t |
|---|---:|---:|---:|
| Llama | **+29.7 pp** | 2.8 pp | 10.8 |
| Qwen | **+20.0 pp** | 3.6 pp | 5.5 |

**逐题层级的回归分析**，研究“提前出现答案候选”是否更容易答错。
- `Early → P(error)`：控制**题目难度**和**剂量差异**后，抢答样本的错误率平均增加多少。
  - Llama：增加 **29.7 个百分点**
  - Qwen：增加 **20.0 个百分点**
- `SE`：该估计值的标准误，按题目聚类计算。数值越小，估计越稳定。
- `t`：估计值除以标准误。
  - Llama：`29.7 / 2.8 ≈ 10.8`
  - Qwen：`20.0 / 3.6 ≈ 5.5`
  - 两者都表明关联非常明确。

> 在同一道题、排除整体剂量差异后，过早形成答案的样本仍更容易出错：Llama 约高 **30 pp**，Qwen 约高 **20 pp**。

> **Commitment timing 能够解释 entry gain 未覆盖的行为差异，尤其是 Llama 的高剂量崩溃；这一结果是稳定的解释性关联，但不是因果 mediation。**

#### 5.8.4 Stopping and Loops: The −8 Collapse Reflects Premature Lock-In, Not Shorter Reasoning

| Llama α | Characters | `n_markers` mean / p90 | Post-commit proportion |
|---:|---:|---:|---:|
| −8 | 2300 | 6.9 / **2** | **0.957** |
| −6 | 2092 | 21.2 / 119 | 0.702 |
| 0 | 2186 | 23.6 / 127 | 0.781 |

- `n_markers mean / p90`：每个输出中 `####` 标记出现次数的统计。
  - `mean`：300 道题的平均出现次数。
  - `p90`：第 90 百分位数，即约 90% 的样本不超过这个次数。
  - 次数很高通常表示模型反复输出 `####`，可能出现循环或重复提交。
- `Post-commit proportion`：首次出现 `####` 之后的文本长度，占完整生成文本的比例。$\text{post-commit proportion}=\frac{\text{length after first commit}}
  {\text{total length}}$ 例如 `0.957` 表示 **95.7% 的输出发生在首次 commit 之后**，说明模型非常早就提交答案，之后仍继续生成大量内容。

Llama 在 `−8` 时并未明显缩短输出，但首次 commit 后的内容占比升至 **95.7%**。这说明模型几乎一开始便锁定答案，之后继续生成大量文字；性能崩溃来自**过早锁定后的退化生成**，而不是推理长度不足。

Qwen 呈现相反趋势：高剂量下重复 marker 和 post-commit 内容持续减少，而 early-candidate 与 commit rate 在 `+8` 左右趋于稳定。

> **Llama 的高剂量崩溃对应过早锁定，Qwen 的高剂量平台则对应 commitment dynamics 饱和。**

#### 5.8.5 Confidence Does Not Explain Premature Commitment

Llama 的前 20 个 decode steps 显示：

- 抢答样本的 `top1` confidence 并未更高（差值 `−0.058～+0.019`）。
- 在抢答样本中，答错者的 confidence 也不高于答对者（差值 `−0.085～−0.001`）。

> **Premature commitment 不是 over-confidence，而是 commitment timing 失调，并非置信度校准失败。**

Qwen 只有 7 个非同批次条件，无法进行对称检验，因此该结论目前仅适用于 Llama。数据配对、计算方法与代码细节见 `CLAUDE.md`。

#### 5.8.6 Conclusion and Evidence Boundaries

> **结果支持模型特异的 commitment transfer function：entry gain 并不直接决定推理表现。Llama 的极端负剂量触发过早且不稳定的 commitment，造成峰后崩溃；Qwen 的高正剂量则使 commitment dynamics 饱和，形成性能平台。**

相比寻找“最佳 α”，更有价值的功能变量是：

> **模型是否在合适的推理阶段形成 commitment。**

该变量可进一步检验能否预测答案正确性，以及能否跨任务迁移。

解释边界如下：

- `z` 仅为模型内标准化入口坐标，不代表两模型接受了等量干预。
- 曲线 R² 只有 9/11 个剂量点，是描述性拟合，不是 mediation。
- 逐题关系控制了题目与剂量效应，但仍是观察性关联，不构成因果证据。

主图呈现 entry gain、early candidate、commit position 与 accuracy 的对应关系。Post-commit release 因 Qwen 对照队列存在选择偏差，仅作为补充结果；实现与验收细节见 `CLAUDE.md`。

## 5.9 Commitment-Based Prediction and Cross-Task Workpoint Selection

### 5.9.1 Prediction and Evaluation Protocol

P2 使用 GSM8K 已有输出训练基于文本的 commitment predictor，并按题目进行五折交叉验证；同一道题的所有剂量始终位于同一折，以避免数据泄漏。模型输入包括 early-candidate、commit state、标准化 commit position（`posN`）及其可观测性，raw α 不作为特征。

冻结后的 GSM8K predictor 直接应用于 MATH，不使用 MATH accuracy 进行训练、调参或校准。Qwen 使用完整的 9 点剂量曲线，Llama 因仅有 `−4/0/+4`，只检验局部 steering 方向。主要 accuracy 口径为 `first_acc`。

具体协议版本、commit-state 编码、marker 适配、fold manifest、抽取器、填充与标准化方法、bootstrap 和 SHA256 provenance 统一记录于 `CLAUDE.md`。

### 5.9.2 P2A: Held-Out Correctness Prediction on GSM8K

**Table 5.9.2. Out-of-Sample Prediction Performance on GSM8K**

| Model | Commitment-only AUROC | Entry-only AUROC | Commitment − Entry | Combined − Commitment |
|---|---|---|---|---|
| Llama | **.687** [.656, .719] | .548 [.526, .571] | **+.139** [+.104, +.172] | −.001 [−.004, +.002] |
| Qwen | **.749** [.710, .787] | .628 [.601, .654] | **+.121** [+.084, +.156] | +.002 [−.002, +.007] |

两模型均通过预注册 gate，且在 GSM8K 内部校准良好（calibration slope：Llama .95，Qwen .98；`fig_p2a_calibration.png`）。

> **结论：commitment timing 能预测未见 GSM8K 题目的对错，而且明显优于 entry gain；在此基础上加入 entry gain，没有带来可检测的额外提升。**

这是预测证据，不是因果证据；“未检出额外提升”也不等于证明 entry gain 无用。完整统计口径与实现细节见 `CLAUDE.md`。

### 5.9.3 Retrospective Locked Transfer to MATH

**Table 5.9.3. Cross-Task Direction and Workpoint Selection on MATH**

| Model | Predicted Direction | Direction Match | Spearman ρ | Selected α | Observed Best α | Regret | Near-Optimal Set |
|---|---|---|---|---|---|---|---|
| Qwen (9 α; full curve) | Positive | **Correct** | **+.962** (n=9) | **+6** | +6 | **.000** | Hit [+4, +6] |
| Llama (3 α; local only) | Negative | **Correct** | +1.000 (n=3) | −4 | −4 | .000 | Hit [−4, 0] |

Qwen 不仅选中真实最佳工作点 `+6`，也识别出 `+8` 的性能回落。其预测分数为 `.83–.88`，实际 accuracy 为 `.54–.68`：绝对数值明显高估，但剂量排序与曲线变化基本一致（`fig_p2b_transfer.png`）。

> **结论：commitment predictor 无法直接估计 MATH 的准确率，但能够判断 steering 方向并选择合适的工作点。**

这是 retrospective locked transfer，并非真正的盲测；完整冻结顺序、校准边界与图表口径见 `CLAUDE.md`。

### 5.9.4 Evidence Scope and Conclusion

- 这是一次**规则冻结后的回顾性跨任务验证**，不是真正的盲测；仍需在从未查看准确率的数据集上进行预注册验证。
- Qwen 支持完整曲线与工作点判断；Llama 只有三个剂量点，仅支持局部方向判断。
- 跨任务迁移的主要是答案形成与提交时序，而不是 GSM8K 特有的 loop 行为；预测排序可以迁移，绝对概率校准不能迁移。

> **P2 总结：commitment timing 能预测未见 GSM8K 题目的对错，也能在 MATH 上选择 steering 方向和工作点，但尚不能视为真正的跨任务盲测。**

完整特征分布、抽取器差异、稳健性检查与统计边界见 `CLAUDE.md`。

## 5.10 Prospective Blind Transfer on GSM-Hard

本节在从未查看 accuracy 的 GSM-Hard 上，前瞻性检验由 GSM8K 冻结的 commitment predictor。剂量、predictor、工作点规则与成功标准均在生成前冻结，预测文件也在 gold 解封前固定。

**Table 5.10a — Predicted and Observed Dose Curves**

| Model      | Metric                 | −8    | −6        | −4      | 0     | +4    | +6    | +8        |
| ---------- | ---------------------- | ----- | --------- | ------- | ----- | ----- | ----- | --------- |
| Llama3-8B  | Predicted score        | .5554 | **.68834** | .68828 | .6303 | .5770 | —     | —         |
|            | Observed `first_acc`   | .1100 | **.2433** | .2400   | .1800 | .1700 | —     | —         |
| Qwen2.5-7B | Predicted score        | —     | —         | .6959   | .7038 | .6794 | .7182 | **.8552** |
|            | Observed `first_acc`   | —     | —         | .3433   | .3400 | .3467 | .4033 | **.5033** |

`Predicted score` 是 frozen logistic regression 对每题正确概率的输出在同一剂量内取平均，用于排序而非估计 GSM-Hard 的绝对准确率。

**Table 5.10b — Blind Workpoint Selection**

| Model      | Direction  | Selected | Observed best | Observed near-optimal | Regret  | ρ      |
| ---------- | ---------- | -------- | ------------- | --------------------- | ------- | ------ |
| Llama3-8B  | negative ✓ | −6       | −6            | {−6, −4}              | 0.00 pp | +1.000 |
| Qwen2.5-7B | positive ✓ | +8       | +8            | {+8}                  | 0.00 pp | +0.600 |

两条剂量曲线均可读（paired McNemar tests with Holm correction：Llama minimum adjusted p = 2.29e−07，Qwen = 1.35e−07；n=300）。

**Table 5.10c — Direct Transfer of GSM8K Workpoints**

| Model      | GSM8K workpoint | acc(α) | acc(0) | Δ             | discordant | McNemar p |
| ---------- | --------------- | ------ | ------ | ------------- | ---------- | --------- |
| Llama3-8B  | α = −6          | .2433  | .1800  | **+6.33 pp**  | 32 / 13    | .0066     |
| Qwen2.5-7B | α = +8          | .5033  | .3400  | **+16.33 pp** | 63 / 14    | 1.41e−08  |

这些工作点直接取自冻结的 GSM8K 结果，未使用 GSM-Hard predicted score。

**Table 5.10d — Llama3-8B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −8 | .110 | 66.0 | 67.7 | 10.3 | 13.3 | 8.7 | **0.0000** | **100.0** | 2194 |
| **−6** | **.243** | **28.7** | 54.7 | 5.3 | 34.0 | 6.0 | 0.2740 | 72.6 | 2168 |
| −4 | .240 | 28.0 | 53.0 | 4.3 | 34.7 | 8.0 | 0.2351 | 76.5 | 2134 |
| 0 | .180 | 45.7 | 54.3 | 4.0 | 28.0 | 13.7 | 0.2161 | 78.4 | 2072 |
| +4 | .170 | 60.0 | 44.7 | 6.3 | 27.7 | 21.3 | 0.1274 | 87.3 | 2078 |

**Table 5.10e — Qwen2.5-7B commitment panel (No-CoT, n=300/cell)**

| α | acc | early% | committed% | unparsed_nonloop% | loop% | no-marker% | posN med | post-commit% | chars med |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| −4 | .343 | 94.7 | 79.3 | 18.3 | 2.3 | 0.0 | 0.8062 | 19.4 | 1326 |
| 0 | .340 | 93.7 | 72.7 | 22.7 | 4.7 | 0.0 | 0.7680 | 23.2 | 1271 |
| +4 | .347 | 92.0 | 78.0 | 17.3 | 4.3 | 0.3 | 0.6791 | 32.1 | 1268 |
| +6 | .403 | 58.7 | 86.3 | 11.0 | 2.3 | 0.3 | 0.5969 | 40.3 | 1145 |
| **+8** | **.503** | **6.0** | **98.3** | 1.7 | 0.0 | 0.0 | 0.7765 | 22.3 | **875** |

各列含义如下：

| 指标 | 含义 |
|---|---|
| `α` | RSN steering 的干预剂量。`0` 是不干预，正负值表示沿相反方向 steering。 |
| `acc` | 该剂量下的真实准确率，即 `first_acc`。 |
| `early%` | 生成开头较早出现答案候选的样本比例，用于衡量是否存在过早回答倾向。 |
| `committed%` | 输出中存在可解析最终答案 marker，例如 `#### 42` 的样本比例。 |
| `unparsed_nonloop%` | 输出中出现 `####`，但后面没有可解析数字，并且不属于重复 marker loop 的比例。即“尝试提交，但格式无效”。 |
| `loop%` | `####` 或答案提交片段反复出现，形成退化循环的样本比例。 |
| `no-marker%` | 整段输出完全没有出现 `####` 的样本比例，即没有按照指定格式提交答案。 |
| `posN med` | 首个可解析 `#### <数字>` 在整段生成文本中的归一化位置中位数。`0` 表示几乎在开头提交，`1` 表示接近结尾才提交。只在存在可解析 commitment 的样本上定义。 |
| `post-commit%` | 第一次提交答案之后的文本占整段生成文本的比例。越高表示模型越早提交，随后仍继续生成大量文字。需与其有效样本分母一起解释。 |
| `chars med` | 每个样本生成文本字符数的中位数，用于描述输出长度。 |

```text
committed
+ unparsed_nonloop
+ loop
+ no-marker
= 100%
```

例如 Llama `α=−8`：

- `early%=66%`：很多样本很早出现答案候选；
- `posN med=0`：可解析答案通常在生成开头就出现；
- `post-commit%=100%`：几乎整段文字都生成在首次提交之后。

这表示“过早锁定后继续生成”，而不是“推理长度不足”。完整分母与抽取规则见 `CLAUDE.md`。

**Observed Commitment Patterns.**

以下分析均为 **exploratory**：十个 No-CoT cell 的 accuracy 已在该分析开始前解封，因此只能作为与 P1 一致的行为证据，不能构成 mediation。

- **高准确率对应较低的 early-candidate rate。** Llama 的近优区间 `{−6,−4}` 位于 early-candidate 低谷（28.7% / 28.0%）；Qwen `+8` 则从 `+6` 的 58.7% 骤降至 6.0%。这一方向与 GSM8K 上的结果一致。
- **Llama `−8` 表现为极端的过早锁定。** 在可解析 commitment 样本中，`posN med=0`、`post-commit=100%`，同时 early-candidate 升至 66.0%，accuracy 降至 .110。其生成长度并未明显缩短，说明性能崩溃不是因为“少生成”，而是因为答案形成得过早。
- **Qwen `+8` 的输出格式最稳定。** Parseable commitment 达到 98.3%，`loop=0%`、`unparsed_nonloop=1.7%`，生成长度也缩短至 875 字符，对应最高 accuracy `.503`。
- **两个模型具有不同的失败模式。** Llama 主要表现为 loop 与 no-marker failure；Qwen 几乎不 loop，no-marker 也接近于零（0–0.3%），其失败更多表现为写出 `####` 但没有可解析数字。这一模型差异同样延续了 GSM8K 上的观察。

**Qwen Commit-Position Rebound.**

Qwen 的 `posN med` 从 `+6` 的 0.597 回升至 `+8` 的 0.777。该回升在五个剂量都 committed 的共同子集上仍然存在（n=153；0.566→0.758），因此不是 committed coverage 改变造成的分母假象。

在该共同子集中，绝对 `commit_char` 从 274 延后至436，而总生成长度从 959 缩短至831。因此，`+8` 并不是简单地让正式 marker 更早出现；它主要减少了过早答案候选和不可解析提交，并使最终提交更加完整。可迁移规律是由 candidate timing、commit validity、loop/no-marker 和生成长度共同构成的 commitment regime，而不是单一 `posN` 的单调变化。

Llama 的五剂量共同 committed 子集仅 n=35，且由 `−8` 的极端行为强烈筛选，因此不作对应的共同子集推断。

**Generation-Budget Truncation (Llama).**

`cap%` = 重新分词后达到生成上限的样本比例（`>=767` of `max_new_tokens=768`；阈值不取精确等于，因离线重新分词在边界处有误差——Llama α=0 精确判定读 82.0%，`>=767` 读 94.0%，`>=760` 读 94.3%，760 以上的平台才是真实截断群体）。

| | −8 | −6 | −4 | 0 | +4 | +6 | +8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Llama cap%** | 91.3 | 92.3 | 96.0 | 94.0 | 95.3 | — | — |
| **Qwen cap%** | — | — | 20.3 | 21.0 | 22.7 | 19.3 | 13.3 |

**Llama 五格全部 91–96% 触及上限，decode 长度中位数恒为 768**；Qwen 仅 13–23%。因此上文「Llama `−8` 生成长度并未明显缩短」应理解为**天花板效应而非自然观测**——各剂量长度都被同一上限压平。这不削弱该处结论（性能崩溃不是因为少生成），但其依据应改为 `posN med=0` 与 `post-commit=100%` 这两项与长度无关的量：答案出现在生成开头，其后整段皆在提交之后。

**截断不否定「固定 768-token budget 下」的配对比较**——全部十格在同一预算、同一约定下逐题配对，estimand 是一致的。**但它会影响绝对准确率，也可能影响剂量差异**（Llama 各格 cap 率散布 4.7 pp；固定工作点检验所用的 `−6 vs 0` 一对差异不显著，18/23，exact McNemar p=.53，但这只覆盖该一对），**因此结果不能外推为不受生成预算限制的能力表现**。

CoT 生成更长而上限不变，故 CoT 结果若变差，截断是必须与 commitment 解读并列报告的替代解释，而非可径直归因于其一。**现阶段不提高上限**：那会改变已冻结的主问题。先完成 768-token 条件；若 Llama 结果为 null/negative，再单独建立 larger-budget sensitivity，且不得事后替换主结果。

**Conclusion.**

> 在前瞻性封存的 GSM-Hard 上，frozen commitment predictor 正确判断了两个模型的 steering 方向，并以 zero empirical regret 选中 observed near-optimal workpoint。直接迁移 GSM8K 工作点使 Llama 与 Qwen 的 accuracy 分别提高 6.33 pp 和 16.33 pp，但 absolute probability calibration 未能迁移。

结果表明 commitment timing 携带可迁移的工作点位置信息，但目前证据仅覆盖 GSM8K→GSM-Hard 的 near-domain transfer。

**Interpretation Boundaries.**

- **Tables 5.10a–c 不是独立证据。** 它们使用同一批 per-question correctness；固定工作点表检验的是既定 α 能否直接迁移，而不是提供独立重复。Tables 5.10d–e 使用相同输出进行 exploratory commitment analysis。
- **Llama 未实质区分 `−6` 与 `−4`。** 两者 predicted score 仅差约 0.00006，且都属于 observed near-optimal set。Qwen 的选择更明确，`+8` 比次优 predicted dose 高 0.137。
- **Qwen 并非完美排序。** `ρ=+0.600`；predictor 将 `−4` 排在 `0` 之上，但两者 observed accuracy 仅相差一道题。
- **绝对校准没有迁移。** Predicted score 系统性高于 observed accuracy（`.55–.86` vs `.11–.50`）；迁移的是 dose ordering，而非 absolute probability calibration。
- Commitment panel 与既有机制一致，但属于解封后的 exploratory analysis，不构成 causal mediation。
- 当前证据限于 GSM8K→GSM-Hard 的 near-domain transfer，且仍需生成多个剂量，尚未实现仅从 `α=0` 选择 steering 方向。

Protocol provenance、artifact hashes、evaluator validation 与完整统计见 `CLAUDE.md`。


## 6. Conclusion

从目前的 Llama3 与 Qwen2.5 结果来看，ThinkingCurve 可以形成一条相当完整、但需要分层表述的结论链：

> `α / prompt manipulation → G_prefill → decode slow/fast dynamics → commitment → state release / stopping → accuracy`

其中 α 确实是干预，但链条中各变量之间尚未完成 mediation，因此不能把每个箭头都写成已证明的因果关系。

1. **RSN 是一个可控的 task-entry gain axis。**  
   两个模型中，`G_prefill` 都随 α 近乎线性变化。Qwen 到 `+12` 仍保持 `R²≈0.9999`，说明后续平台或性能下降不是注入失效造成的。

2. **一次性的 prefill steering 足以改变整条生成路径。**  
   注入只发生在生成边界；到 `decode[0]` 时，入口偏移已回弹约 95–100%，但 commitment、accuracy、循环和停止行为仍明显改变。这支持 initial-condition / boundary-gating：α 改变起始状态，效应通过 KV cache、早期 token 选择和 autoregressive path dependence 延续。

3. **线性输入会转化为非线性的内部状态与行为。**  
   `G_prefill` 线性移动，但 `s_t`、commitment timing 和 accuracy 并不线性响应。因此 RSN steering 更像改变模型所在的 working regime，而不是简单地“gain 越高越好”。

4. **Commitment 是一个明确的 generation-state transition。**  
   首次 `####` 附近，`s_t`、`p_t` 和输出分布都会快速变化。Llama 与 Qwen 均出现 commit-locked 的负向 `p_t` dip，说明这一事件在两个模型中都对应快速状态转折。

5. **`s_t` level 是目前最稳定的 slow-state 行为读数。**  
   较高、较持续的 pre-commit `s_t` 通常与较晚 commitment、持续 processing/engagement 和 viable reasoning 同向变化。它比 task-entry gain 更接近实际的 commitment behavior。

6. **Commit 后存在 slow-state release。**  
   Llama 的 release 较明显；Qwen 在可分析的高剂量 cell 中也呈相同方向，但幅度较小——最大格 `+12` 为 Llama 的约 `0.84×`（`−0.233/−0.279`），较低两格 `+8/+10` 为 `0.53×/0.61×`——并且下降后仍停留在较高水平。因此这是 **same-sign but attenuated replication**。

7. **Thinking quality 与 stopping quality 是不同维度。**  
   首次答案正确不代表后续生成稳定。模型可能先输出正确答案，随后继续改写、重复或产生其他答案。因此 first-answer accuracy 不能代替 loop、answer switching、自然 EOS 和 stable completion。

**Llama3 only**

8. **Llama 存在非对称的最佳工作点。**  
   离散最佳剂量是 `α=−6`；`−8` 出现明显崩溃，正 α 端逐渐下降后趋平。这不是平滑、对称的标准 inverted-U，更准确的说法是 **asymmetric peaked working-point response**。

9. **Llama 的表现更接近 decode state，而不是入口 gain 大小。**  
   dose-level 上，accuracy 与 early `s_t` 的相关约为 `r=0.74`，高于与 `G_prefill` 的相关。可视为规律性证据，但只有 9 个 dose points，不能证明 `s_t` 中介 accuracy。

10. **Correct responses 具有不同的 slow-state trajectory。**  
    Correct 组在 commit 前通常维持更高、更持久的 `s_t`，commit 后释放更快。但这可能来自题目可解性、熟悉度或较早形成可行路径，不能写成“高 `s_t` 导致答对”。

11. **CoT 主要提高 engagement 与 output decisiveness。**  
    CoT 提高 task-entry gain，维持更高的 pre-commit `s_t`，降低 entropy、提高 top1/margin，并伴随更快的 post-commit release。

12. **α 与 CoT 的作用时间不同。**  
    α 主要控制 generation boundary；CoT 主要重塑后续 decode/commitment dynamics。两者在 task entry 和 output decisiveness 上大致可叠加，未观察到可靠的强交互。

13. **Persona 会重新分配推理阶段，而不只是整体平移信号。**  
    Expert persona 在 task entry 的 gain 更高，但 pre-commit 阶段排序可以反转，post-commit release 也不同。Persona 更像改变 engagement–commitment–release 的时间配置。

14. **RSN temporal readout 具有方向特异性。**  
    与 random-support、same-support orthogonal 和 off-support directions 相比，NMD/RSN 投影呈现更强的 commitment-locked signed organization。这支持 readout specificity，但不是 random-direction causal steering specificity。

**Qwen Only**

15. **Qwen 的主要变化是把“先答后推”改成“先推后答”。**  
    `α≤+4` 时首次 `####` 通常在 step 3 左右；`+6→+8` 后 commitment 大幅右移，accuracy 从约 `68%` 提升至 `86%`，loop rate 从 `13.0%` 降至 `2.7%`。

16. **Qwen 呈现上升后平台，而不是 Llama 式尖锐最佳点。**  
    accuracy 在 `+8/+10/+12` 为 `86.00/88.00/87.67%`；相对 commit 位置也趋平，但绝对 commit step 仍由 `110→134→163` 延后。即相对位置饱和，绝对延迟继续增加。

17. **Qwen 高剂量下出现 decode-response compression。**  
    入口 gain 继续线性增长，但 decode `s_t` 对每单位 α 的响应显著减弱。它是 compression，不是完全停止响应的 ceiling。

18. **这种 compression 更像固定 profile 的标量缩放。**  
    两个剂量区间的逐层 response 几乎共线：`cos=0.987`、`k=0.309`，97.4% response energy 可由固定 profile 的整体缩小解释。目前没有看到明显的 layer-profile rotation。

19. **RSN 比一般 null direction 更像干净的标量通道。**  
    RSN 的 profile residual 与 CV 均低于三个 null family。这个结果反对把高剂量压缩解释为明显的方向重分配，但仍属于 offline readout evidence。

20. **Qwen 复现 fast transition，但没有复现 residual-amplitude dose effect。**  
    `abs_mean/std` 在 `−8…+12` 基本不变；但 commit 当步的负向 dip 从 `−0.578` 加深到 `−1.322`。因此 α 改变的是 commit 瞬间的 signed transition，而不是整体放大 fast residual。

21. **Qwen 的 post-commit release 存在但较弱。**  
    在相同 `±20 token` 窗口中，Qwen `+8/+10/+12` release 为 `−0.148/−0.171/−0.233`，方向与 Llama 一致，但下降后仍维持较高的 slow-state plateau。

22. **Qwen 的 output decisiveness 呈现阶段依赖的 α 效应。**  
    Task entry 随正 α 增大而更 decisive（entropy 下降、top1/margin 上升）；commit 当步进一步 sharpen，但 commit 后 entropy 上升、top1/margin 下降。该 transition 在高剂量下减弱。由于只有 5 个 No-CoT 剂量且有效 cohort 偏向高剂量，目前只能确认单调趋势与 commit-locked transition，不能判断完整非线性曲线。

    CoT `+6` 未检出明显的 commit 前后 confidence transition，但这只是单个、结果筛选 cohort 中的观察，不能写成 CoT 消除了该机制。

**Cross Model**

23. **两个模型共享部分调节结构。**  
    两者都呈现：

    - α 线性控制 task-entry gain；
    - entry offset 在 decode 初期迅速释放；
    - 后续 commitment 与行为发生非线性变化；
    - commit 当步出现 fast residual transition；
    - commit 后出现 slow-state release；
    - α steering 同时改变 RSN dynamics 与 output decisiveness——因此 **α 不是只影响 wanting/engagement 的 selective intervention**。

    但两模型的剂量曲线，以及 commit 前后的变化幅度，并不相同。

24. **共同机制更可能是“adaptive calibration”，而不是共同剂量曲线。**  
    Llama 是尖锐的非对称工作点，Qwen 是上升后平台。这说明 RSN steering 的作用可能依赖模型基线、任务和生成策略，不存在当前数据支持的统一最佳 α。

25. **不能直接比较 raw α。**  
    Qwen 的 `+8` 与 Llama 的 `−6` 不是共同强度的干预。两模型使用不同 mask、层数与模型内 reference，因此不能根据最佳 α 的正负推断它们位于不同基线位置。

26. **“相同 working state、不同到达方向”仍是假说。**  
    Qwen 的 output decisiveness 已完成提取，但仅覆盖 7 个 H5 cell。`entropy/log(V)` 提高了词表尺度上的可比性，却不能消除 tokenizer、reference 与 cohort 差异。因此，“相同 working state、不同到达方向”仍是假说。

**null**

27. **`s_t` slope 没有支持 GSM8K 上的 vigor prediction。**  
    Llama 的 marginal slope–timing 关系为 null；控制 level 后的残余方向与预期相反。Qwen 可分析 cell 中 slope 也逐渐趋近 0。因此当前有意义的是 `s_t level`，不是 ramping slope。

28. **`p_t` frequency organization 没有稳定证据。**  
    zcr、spectral centroid、dominant frequency 等容易受到 answer format、重复尾巴和窗口长度影响。当前较可靠的是 amplitude 与 signed transition，不是频率重组。

29. **`p_t` 不能视为独立 fast channel。**  
    它是同一条 RSN 投影相对 EMA baseline 的残差，因此不能拿 `s_t` 与 `p_t` 当作两条独立证据累加。

30. **Commit 附近的 confidence change 可能包含格式效应。**  
    `####` 本身会改变 token distribution，因此 entropy spike、top1 dip 和 margin change 不能全部解释为实质 confidence 改变。

31. **目前没有 manifold reorganization 的证据（manifold pilot 已完成，2026-08-28）。**  
    一维结果更符合固定 RSN profile 的标量压缩，manifold 分析未推翻这一点：两模型的 entry displacement 都是线性单轴的（`AdaManifold.md`）。注意 `k=20` 是**分析上限，不是 intrinsic dimension**——PCA 只能显示线性低秩，且 top-20 仅覆盖约一半 α=0 方差，所以「内在维度」在本项目中从未被测量，不应作为待检项保留。

32. **目前没有证明 causal direction specificity。**  
    现有 random/orthogonal remask 是对同一批 hidden states 的重新投影，只证明 readout specificity。要证明只有 RSN steering 能产生行为变化，仍需真正注入 random/orthogonal directions。

---

**总体结论。**

> Llama 与 Qwen 都支持 α 对 RSN dynamics 和 output decisiveness 的联合调节；共同点是 commitment 附近存在多信号状态转换，差异则体现在行为曲线、fast-residual amplitude 和 transition 强度。现有结果支持 adaptive calibration，但仍不足以证明两模型到达同一个 working state。
