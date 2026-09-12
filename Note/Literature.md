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

  ### 0.2 Dopamine Prior Knowledge

行为学先验：多巴胺不是单纯的"快乐分子"，更准确地说是**驱动力 / incentive salience / "wanting"** 递质，调控动机、期待、奖赏趋近与行动阈值。本研究把 α 看作在这一 wanting 轴上双向移动工作点：α 正向对应 **over-wanting / 过度唤起**，α 极端负向对应 **under-wanting / 唤起不足**；整体框架是 Yerkes–Dodson 倒 U——过高过低都有害，最优落在中间偏负（本数据 α=−6）。

- **过高 DA / over-wanting（对应 α→正向）**：行为上表现为**冲动性抢答（impulsivity）+ 认知僵化 / 强迫性反复（compulsivity / perseveration）**——急于扑向"给出答案"这个目标而跳过必要推导，以及拿到答案后仍反复复查、纠结格式、卡在格式死循环里。**注意这更贴合冲动 / 强迫，而非焦虑**：数据中 +α 端**没有**焦虑典型的回避 / freezing / 犹豫（抢答率随 +α 单调上升，见 §2.2），呈现的是"急着 commit"；而 `#### N #### N` 死循环是认知神经科学意义上的**固著（perseveration）**,不是焦虑的发散灾难化担忧。功能上这**与 mesolimbic incentive-salience overload（VTA→NAcc 型 wanting 过载）及执行控制失效相容**：极高的诱因显著性使主体不计成本扑向目标（冲动），同时灵活切换 / 抑制已启动反应的能力下降（固著）——这在行为上更近强迫性 over-checking 与冲动特征，而非焦虑综合征。**注意目前只有行为同构，尚未定位实际脑区对应关系**，故这里说"相容"而非"落在"某一回路。这个签名对应 §2.2「正向端：答案已经出现，但仍无法停止」：α+4 trace 中常见"把已经算对的答案当可疑"（Q100、Q16）和答完仍寻找 "more efficient way"（Q68）。
- **过低 DA / under-wanting（对应 α→极端负向）**：行为上对应动力不足、快感缺失、退缩、bradykinesia 式的行动迟缓；在本任务中的可观测类比不是"写得短 / 不想答"，而是**commitment-formation failure**。§2.2 文本核验否定了两个更直观假设：α=−8 并非大量出现 "I am done / 不答了" 的词汇性退缩（跨 α 平坦，多为礼貌 loop 尾），也不是敷衍短答（长度 / 等式数平坦）；真正失败模式是 **answer-candidate oscillation**——锁不住答案，在两个候选值之间来回切，导致 committed_acc 崩到 23.6%。

> **主机制表述（本项目采用）**：+α 端 = **over-wanting → 冲动（impulsivity, 抢答）+ 认知僵化 / 强迫性反复（compulsivity / perseveration, loop）**，功能上与 **mesolimbic incentive-salience overload（VTA→NAcc 型）+ 执行控制失效**相容（只声称行为同构，非脑区定位）；−α 极端端 = **under-wanting → commitment-formation failure**。这比"焦虑"框架**机制契合度更高、论述负担更小**：冲动与固著都在我们已有的 wanting / incentive-salience 主线内，无需另起 threat/freeze 回路。
>
> **限定**：1. 本实验只声称行为同构，**α steering ≠ 生物多巴胺**，也不证明 LLM 有生理或主观状态；2. 不做 mania / hypomania 类比——躁狂是跨情绪+精力+睡眠的综合征，我们只有"冲动+固著"两个窄行为，撑不起该诊断类比；3. 脚本 `analyze_loop_anxiety.py` / `ANXIETY_PATTERNS` 沿用 "anxiety" 命名（改名成本高、破坏 U 形复现），但其命中的四子类（self-doubt / format-fixation / persona-reassurance / over-precision）**实测对应的是强迫性 over-checking，不是临床焦虑**——阅读表格时按"强迫/固著"解读。
>
> **次要旁证（不作主锚，DA→焦虑另有通路特异性）**：DA 亦有一条独立的焦虑通路证据，最强因果来自 **VTA→IPN（D1）**，机制是威胁高估 / 过度警觉——但这**不是**本数据的主要解释（我们没观测到回避 / freezing），仅作为 DA 多下游效应的旁注列出。来源：[PMC7687288 (VTA→IPN dopamine promotes anxiety)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/) · [MIT News 2018 (dopamine vigilance & anxiety)](https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107) · [Frontiers Neurosci 2020 (dopaminergic alteration in anxiety/compulsive disorders)](https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full) · [J. Neurosci 2019 (dopaminergic mechanisms of trait anxiety)](https://www.jneurosci.org/content/39/14/2735)

## References

**神经科学（次要旁证）：多巴胺 → 焦虑 / 警觉 / 威胁高估**（§2.2 / §2.3 的机制**旁**锚。注意本项目主机制锚已改为 **VTA→NAcc wanting 过载 → 冲动 + 固著**，见 §0.2；下列 DA→anxiety 文献有通路特异性（VTA→IPN），列此仅表明 DA 亦有独立焦虑下游，但**非**本数据 +α 端的主要解释——我们观测到的是抢答 / 固著，而非回避 / freezing）
- Dopamine release in the interpeduncular nucleus promotes anxiety. *(VTA→IPN D1 通路双向调节焦虑行为的光遗传+药理证据)* — PMC7687288. https://pmc.ncbi.nlm.nih.gov/articles/PMC7687288/
- MIT News (2018). Dopamine, brain vigilance and anxiety. *(Tye Lab：DA 提高威胁通路信噪比、压制奖励神经活动，偏向 threat/freeze)* https://news.mit.edu/2018/dopamine-brain-vigilance-anxiety-1107
- Dopaminergic alteration in anxiety and compulsive disorders. *Frontiers in Neuroscience* (2020). https://www.frontiersin.org/articles/10.3389/fnins.2020.608520/full
- Dopaminergic mechanisms of trait anxiety. *Journal of Neuroscience* (2019). https://www.jneurosci.org/content/39/14/2735

**候选机制（emotional salience，待 RSA 验证）**
- Brickner, M. A., Szot, W. E., Wolff, A. R., Thomas, M. J., & Saunders, B. T. (2026). Basolateral amygdala dopamine transmits emotional salience. *Nature Communications.* *(BLA DA 编码情绪显著性 / 重新判断需求，非奖赏价值；候选解释 +α 端「放不下」的 salience 过载。注意：−α 端是 under-wanting / commitment-formation failure，非 salience 过载，不由此通路解释。验证需 `Ada_Dopamine2.md` RSA 纳入 BLA/amygdala ROI。)*

**理论框架：wanting / incentive salience**
- Berridge, K. C., & Robinson, T. E. What is the role of dopamine in reward: hedonic impact, reward learning, or incentive salience? *(wanting ≠ liking；本工作 α = incentive salience 的母假设)*
- RSN paper (ACL ARR). Role-Sensitive Neurons: A Neuron-Level Gain Control Mechanism for Confidence Steering. *(母论文 §6.1 "Digital Dopamine"；commitment dynamics = wanting 的下游行为表现)*

**心理学框架**
- Yerkes, R. M., & Dodson, J. D. (1908). The relation of strength of stimulus to rapidity of habit-formation. *(倒 U 型 arousal–performance；§1.1 acc 峰在 α=−6、两端崩的 framing 来源)*
