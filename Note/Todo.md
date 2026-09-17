### Document writing
我们的要求是： 1）细节部分放到Claude.md不要在文档中出现 2）数据尽量保持完整 3）尽量合并表格（但是不要勉强，可以合并的合并） 4）内容和章节都可以重构或者合并 5）结论简洁 通俗易懂 目标是提升可读性 6）标题和表格写英文
给我md版本的文字就好 我会自己去替换
尽量不要删除原始数据，可以合并

---

### Note
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

Bandit.NatureCommunications2026.[Foraging models explain human exploration in uncertain tasks.](https://doi.org/10.1038/s41467-026-75773-4) 该研究发现，人类在动态 Bandit 中更接近 compare-to-threshold 策略：主要追踪当前选项是否仍值得继续，而非持续比较所有候选价值。这与 PV10 中模型反复采样 incumbent、却不响应低样本替代臂的行为相似，为 incumbent persistence 提供了“局部阈值决策”的替代解释；但我们尚未进行相应的模型拟合，且任务设定不同，因此只能视为行为结构上的参照，不能断言 LLM 使用了相同的 foraging-RL 机制。

---

#### Tmux
conda activate dopamine
conda activate roleplaying
conda deactivate

tmux new -s da
tmux attach -t da

cd /data1/paveen/Dopamine

git pull origin main
watch -n 1 nvidia-smi
export CUDA_VISIBLE_DEVICES=3

top -u $USER

##### 清理缓存
rm -rf /home/nas/d12922004/.cache/huggingface/hub
rm -rf /home/nas/d12922004/.hf_cache/huggingface/hub

##### 182/184/185/177/178
rsync -avzP d12922004@140.112.31.185:/data1/paveen/Dopamine/components/hidden_states_mean/{llama3,qwen2.5} /Users/paveenhuang/Downloads

rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.185:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

---
### Daily
09.17 周四 收拾行李；继续实验
09.18 周五 12：20 group meeting；台北-杭州萧山 机票 ✔
09.19 杭州逛逛
09.20 杭州逛逛
09.21 回家高铁*1 - Helene ✔
09.23、09.24 在家 需要去办理公证 + 爸妈护照 （身份证，户口本）
09.24 全曜回家 高铁*1  ✔
09.25 - 10.02 武夷山-成都-丽江 <川滇之间>
10.02 丽江
10.02晚上-10.03 成都市区 机票 ✔ 住宿 ✔
10.04 成都-武夷山Flight Home
10.04-10.10 Home
10.11 Flight Taipei

准备多益考试
看完瑜伽视频
看完徐玉兰视频

### TO DO
16. MATH补充完整 ✔ 
17. LogiQA working point ✖ 目前做不出来，不确定是因为选择题的形式问题还是逻辑推理无法迁移
18. BBH counting task ✔
19. 精简claude.md的内容 ✔
20. 整理GSM8K文档 ✔
21. 补充一下llama3-gsm8k-cot working point的结果 ✔
22. 思考关于working point，补充一下其余的点 (GSM 185/ Llama GSM8K 182/ LLama Math 182) ✔ 
23. CRUXEval Design ✔ 
24. 优化文档GSM8K ✔
25. 补充BBH CRUX LogiQA cot的结果 -> cot会有效果 ✔
25. 顺便丰富一下prediction的结果 ✔
26. multi-hop ProofWwiter OWA 无法得到有效的格式的答案 ✖
27. ZebraLogic WP 测试 ✖
28. 1-shot multi-hop ProofWwiter OWA ✖
29. FinQA ✖
30. GSM-Symbolic cot & non-cot ✔
31. ProofWriter check是不是格式问题导致Llama失效 -> 确认是格式问题 -> 是格式问题，用chat趋势ok ✔
32. 也修改一下CRUXEval的chat版本 cot & non-cot ✔
33. GSM-Symbolic行为学特征统计 ✔
34. Confidence neurons ✔
   1) 相关性分析：role neurons & confidence neurons; 相关性在11-19层上升 -> 实际上相关性非常的高，最高可以达到0.7左右
   2) confident & unconfident相关性分析：相关性在11-19最低，类似RSN
   3) overlap：band 内仅共享 46/180=25.6%，Jaccard 为 0.14；但是是显著高于随机
   4) shared-top 的单位贡献大约是 role-only/confidence-only 的 8 倍；是 neither-top 的 74 倍。
   5) overlap分析：整体 alignment 是广泛分布的，而非集中在极少数高贡献 neurons
35. cross-steering: MMLUE ✔
36. cross-steering: GSM8K ✔
---
37. 统一各个任务的行为统计指标 ✔
38. 缩小confidence alpha MMLUE: ACC + Behaivour ✔
39. 缩小confidence alpha GSM8K: : ACC + Behaivour ✔
38. 确认Loop的问题 -> Llama3补充chat template + steering的结果
   1) GSM8K ✔
   2) GSM-hard ✔
   3) MATH ✔
   4) 整理一份当前的结果 ✔
39. 试着理解chat template的影响的原因：注入点切回到answer <><><> Answer: MATH + GSM8K ✔
40. 确认Qwen上是不是存在相同的chat 现象 -> 基本可以确认RSN和Chat是类似的function ✔
41. chat - bare llama3 & qwen2.5 GSM8K/MATH/GSMHard -> 以目前的结果分析不出来 ✖
42. 采一下GSM8K/ MATH/ GSMHard/ MMLUE 上面的expert vs. non-expert HS ✔
43. 分析Role Matrix之间的关系 ✖ 除了部分top neurons重叠之外 几乎正交 没有相关性 
44. Manifold分析 reasoning RSN & Chat -> 类似top neurons之间的关系 ✔
45. 确认Reasoning RSN(RRSN)的expert与non-expert的关系；RRSN与MRSN之间的相关性 -> 还是和之前一样 核心重叠；✔
46. Steering RSM8K with RRSN ✔ 
47. Steering MMLUE with RRSN ✔ -> 结果很乱，和MRSN非常不一致
48. 看是不是RRSN的提取需要增加Abstention的提示 -> ARRSN 结果没有比较好 ✖ 
49. 换成全量的GSM8K RRSN 作为固定的RRSN ✔

48. 再讨论一下相似度表征这件事
46. Manifold reasoning Chat & MMLUE RSN
44. MMLUE confidence vector和这些之间的关系
42. 看一下cot 区分RSN COT Chat Confidence
40. 试着理解chat template的影响的原因：Base model 
42. 认知切换开关
43. 观察这些neurons的状态 应该要在认知指令的位置达到高峰
44. MMLUE 
45. 待补 Qwen Matched-Anchor α=0”

| Task | Existing Hidden-State Vectors |
|---|---|
| MMLU-E | Role-MRSN, Confidence-CSN |
| GSM8K | Role-RRSN, Chat, Role-Abstention-ARRSN |
| MATH | Role-RRSN, Chat |
| GSM-Hard | Role-RRSN, Chat |

---
若重点是“几何结构相似”，建议做一条很干净的 activation-manifold 分析链，而不是先做 neuron overlap。

1. 固定可比状态  
   同一批 prompt、同一 token 位置、同一 layer，收集两条件的 residual stream：
   $$ 
   X\in\mathbb{R}^{n\times d},\quad Y\in\mathbb{R}^{n\times d}.
   $$
   对 RSN，优先比较 prefill 最后一个 token、注入 token，或 teacher-forced 的相同生成位置；不能拿两条自由生成、token 已错位的轨迹直接比较。

2. 分层计算三类主指标  
   - **Centered linear CKA**：比较样本间几何关系是否一致，对正交变换及整体尺度较稳健。适合作为主报告指标。
   - **mCCA / SVCCA**：问“是否存在近似线性坐标变换，使两团 activation 对齐”；更接近你刚读的论文所说的线性表征相似。
   - **Orthogonal Procrustes residual**：先求最佳旋转 \(R\)，报告
    $$
     \frac{\lVert X-YR\rVert_F}{\lVert X\rVert_F}.
     $$
     它很直观：最优刚体对齐后，还剩多少结构差异。

3. 补充子空间而非只看坐标轴  
   对每层 activation 做 PCA，取解释 80–90% 方差的主子空间，计算：
   - principal angles；
   - shared explained variance / cross-projection $X\rightarrow {\rm span}(Y)$；
   - 前几个主成分的方差谱。  
   这能区分“整体流形相似但单 neuron basis 不同”与“主变化方向本身已经改变”。

4. 单独报告轴级关系  
   若同一模型、同一 residual basis 下比较 steering 前后，可补充 coordinate-wise Pearson/cosine、方向向量 cosine、rank correlation。它们回答的是“是否沿同一 neuron axes 改变”，不能替代 CKA/CCA。已有的 dense cosine、top-neuron overlap、Jaccard 和 contribution mass 正适合放在这一层。

5. 用严格 null 与不确定性  
   以 prompt 为 cluster bootstrap 单位给 CIs；加入 prompt-pair permutation、随机正交旋转或随机 neuron/direction 的 null。每层、每个 token 位置单独画曲线或热图，不把 layer 平均后才看结果。

我会把结论分成三档写：

- CKA/CCA 高、Procrustes residual 低、子空间夹角小：**几何上近似线性对齐**。
- CKA 高但 coordinate overlap 低：**流形相近，但不共享稀疏轴/神经元实现**。
- cosine 或 overlap 有富集，但 CKA/CCA 不高：**只有局部或方向性结构关联，不能说整体表征相近**。

对你当前 RSN 项目，最有价值的首个图大概是：`layer × token-position` 的 CKA / Procrustes residual 热图，再配一张 PCA 子空间夹角曲线。PCA/UMAP/t-SNE 可以作图，但不应作为正式相似性证据。

---
内部表征“相近”没有单一指标，最好分三层评估：

| 层次 | 要回答的问题 | 推荐指标 |
|---|---|---|
| 几何结构 | 两组 activation 的空间布局是否相似？ | centered CKA、mCCA/SVCCA、正交 Procrustes residual、子空间夹角 |
| 可读出功能 | 在 A 中线性可读的信息，B 中是否仍可读？ | 同一标签/状态的线性 probe；跨条件训练-测试（train A → test B） |
| 因果功能 | 两个表征是否能产生同一种干预效果？ | norm-matched cross-steering / cross-ablation 矩阵 |

对 RSN，最实用的最小方案是：

1. 固定同一批 prompts、模型、tokenizer 和 token 位置；在每层提取 residual stream（尤其是注入位点）。不要把不同输出长度或不同生成阶段的 states 直接混在一起比较。
2. 对 baseline 与 steering 条件，先算每层的 centered CKA 或 mCCA；它们评价的是“允许线性变换后的几何对应”，比逐神经元 cosine 更合适。
3. 再做 probe：例如用 baseline activation 训练“reason-first / premature candidate / answer correctness / confidence”等线性读出器，直接迁移到 steering activation。迁移仍有效，才说明该信息的编码方式近似保留。
4. 若比较的是两个方向或两个 neuron set，控制层数、神经元数、方向范数与实际注入强度；报告 direction cosine、top-neuron overlap/Jaccard、rank correlation 与 principal angles。
5. 最后用交叉干预裁决：用 A 的方向/神经元去复现 B 的行为 readout，并反向测试。只有在这一层成立，才能接近“功能上可替代”；前面所有相似性指标都只能说明结构关联。

你们当前的 role/confidence 与 RSN 结果就是很好的例子：层间 cosine、top-neuron overlap 和共享稀疏核心可以支持“结构关联”，但不能推出两者同一机制或可互换；还需要 norm-matched cross-steering 复现彼此的行为效应。

顺带说，这篇 logit-distance 论文补充了一个输出侧条件：若两模型在同一输入上的全词表 logit distance 小，在其强假设下可推出较高线性表征相似性；但对单模型局部 steering，它不能替代 activation-level 对齐与因果交叉验证。

---

因此下一步不需要马上继续做 Manifold，而应进入 causal steering：
1. 用 RRSN 在 GSM8K 做同方向 positive control；
2. 用 RRSN 在 MMLU-E 做 reverse transfer；
3. 与已有的 MRSN→MMLU-E、MRSN→GSM8K 构成完整的 direction × task 矩阵；
4. 固定当前共同 band、exact-NMD 数量，并做注入范数匹配，不能直接把相同 raw α 当作相同剂量。
届时能够区分三种情况：
- RRSN 同时作用于 reasoning 和 MMLU-E：支持共享的 task-general role/expertise component；
- RRSN 只作用于 reasoning：共享神经元存在，但具体权重决定任务功能；
- 两个方向交叉迁移不对称：支持“共享核心 + task-specific extension”，与当前 containment 非对称性一致。
唯一建议补充的统计备注：如果 pooled hypergeometric p 是把整个 band 当作一个全局 universe 直接计算，它并不是严格符合“每层固定选 k 个”的精确零分布。更严格的是逐层 hypergeometric 的卷积或 layer-wise permutation。不过目前 p 值极小、富集达 29–38 倍，这不会改变实质结论；可以把 pooled p 标为 uniform-coordinate reference，避免把它写成主要证据。
总之，这个结果没有制造新的“不一致”，反而给出了一个很清楚的解释框架：共享稀疏核心是真实的，但完整方向具有任务特异性；接下来由 cross-steering 判断共享部分是否具有共同功能。

---

1. 先补一个成本很低的 final RRSN–MRSN exact-NMD comparison  
   包括 aggregate mask overlap、Jaccard、weighted cosine、双向 energy containment。这个不算完整 Manifold。

2. `RRSN → GSM8K`  
   使用已经决定的 matched band：
   - Llama `[11,20)`
   - Qwen `[16,22)`

   这是 RRSN 的自身正对照。若这里没有效果，`RRSN → MMLU-E` 的 null 将无法解释。

3. `RRSN → MMLU-E`  

   检验反向跨任务迁移，并与已有的：
   - `MRSN → MMLU-E`
   - `MRSN → GSM8K`

   组成完整 2×2 cross-steering。比较时匹配 band、neuron count 和注入 norm，不能直接比较原始 α。

4. 最后做 induced-state Manifold  
   根据 steering 结果提出更精确的问题：

   - 双向迁移：两个不同 vector 是否汇聚到相同下游状态？
   - 仅各自在本任务有效：是否对应 task-specific manifolds？
   - MRSN 可迁移、RRSN 不可迁移：MRSN 是否更接近通用 interface/engagement direction？
   - RRSN 自身都无效：先检查 vector 构造或层段，不宜解释 Manifold。

所以我的明确判断是：

> **现在完整做 Manifold 容易得到又一组“结构相关但含义不确定”的结果；先完成 RRSN 的 causal steering，之后 Manifold 才能解释机制。**

当前主线可以简化为：

> **exact-NMD 静态核对 → RRSN→GSM8K → RRSN→MMLU-E → induced-state Manifold。**

---

- `MMLU-E RSN`：MMLU-E 上得到的 sparse vector；
- `Reasoning RSN`：GSM8K、MATH、GSM-Hard 三个方向取平均后，经过相同 sparse 流程得到的 vector；
- 当前真正未完成的是 **Reasoning RSN 的中间层 band 选择**。
### 1. 先完成逐层关系和 band 选择

先不做 steering，比较两个 sparse vector 的：

- 每层 cosine、norm、符号一致性；
- top-neuron overlap 和随机富集；
- 三个 reasoning task 在每层的一致性；
- split-half reliability；
- Reasoning RSN 对 MMLU-E RSN 的逐层 projection。

Reasoning band 不要根据 downstream accuracy 选择，而应根据提取集上的结构稳定性确定，例如选择一段连续层，使其同时满足：

- 三个 reasoning task 的方向一致性较高；
- split-half 稳定；
- sparse overlap 高于随机；
- 不由单个任务独占。

band 长度最好与 MMLU-E RSN 一致，便于后续 norm matching。除此之外，可以把“直接使用 MMLU-E 原 band”保留为 matched-band control。

### 2. 再完成 2×2 cross-steering

这里应改称 Reasoning RSN，而不是 GSM8K RSN：

| Steering vector | MMLU-E | GSM8K |
|---|---:|---:|
| MMLU-E RSN | 已有 | 已有 |
| Reasoning RSN | **待做** | **待做自身正对照** |

优先顺序：

1. `Reasoning RSN → GSM8K`：确认这个新 vector 自身确实具有 causal effect；
2. `Reasoning RSN → MMLU-E`：检验反向迁移；
3. 与已有的 `MMLU-E RSN → GSM8K/MMLU-E` 做 norm-matched 比较。

这里至少保留两个 Reasoning RSN 版本：

- **Matched-band**：使用 MMLU-E 相同层段，回答“同层条件下是否功能可互换”；
- **Native-band**：使用 reasoning 数据选出的稳定层段，回答“Reasoning RSN 自身最佳的功能范围是什么”。

如果只做 native-band，一旦结果不同，会混入 layer band 差异；只做 matched-band，则可能低估 Reasoning RSN。

### 3. Manifold 分成两层

在 steering 前，可以做静态的 Role-difference subspace：

- MMLU-E RSN 是否位于 reasoning 三任务形成的 subspace；
- Reasoning RSN 是否位于 MMLU-E sample-difference subspace；
- matched-band 与 native-band 的结论是否一致。

完成 cross-steering 后，再做更关键的 induced-state analysis：

> 两个初始 vector 即使方向不同，是否在模型中间层传播后产生相似的 hidden-state displacement？

这能区分：

- **shared injection direction**；
- **different inputs, convergent downstream state**；
- **different directions and different functions**。

所以目前最合理的主线是：

> **逐层关系 → 冻结 Reasoning band → Reasoning RSN 自身正对照 → 反向 MMLU-E transfer → induced-state manifold。**

其中真正具有决定性的实验仍是 `Reasoning RSN → MMLU-E`，但必须与 `Reasoning RSN → GSM8K` 一起看。否则反向 null 无法区分“不能跨任务”和“Reasoning RSN 本身没有被正确定位”。

---


## 接下来最合理的执行顺序

### 1. 先完成 Role–Confidence–Chat 三角关系

Todo 中“confidence vector 和这些之间的关系”应该是当前第一优先级。

必须在相同任务、token position、层区间和方向定义下比较：

- Role vs Confidence
- Role vs Chat
- Confidence vs Chat
- dense cosine、top-neuron overlap、asymmetric containment
- shared / Role-only / Confidence-only / Chat-only components

重点不是找“谁最像谁”，而是判断：

- Chat 是否包含一个 RSN/Confidence sparse core；
- RSN 是否更接近 Confidence，而只与 Chat 共享少量控制维度；
- 两个模型是否都呈现这一组织结构。

### 2. 补 matched-anchor hidden-state control

这是当前最关键的有效性缺口。现在的 `Chat−Bare` 同时混合了：

- Chat 上下文；
- 最后 token；
- token position；
- prompt length。

所以现有跨任务稳定方向还不能直接叫“认知状态开关”。

建议保持 Todo 中的三个位置：

- 相同问题正文的最后 token；
- 各条件实际 last-prefill token；
- teacher-force 相同短前缀后的共同 token。

同时增加 split-half reliability 或跨题目的 sample-level probe。当前报告主要基于 mean matrices；高跨任务 cosine 很有意思，但还没有回答这个方向是否能稳定地区分单个样本。

### 3. 做一个小型 Base–Instruct 对照

不需要扩成大 sweep。Llama 的 `Base/Instruct × Bare/Native Chat/Matched-anchor`、α=0 就够。

它回答的是：

> Chat state switch 是否主要是 post-training-associated？

不要表述为 RLHF-specific，因为公开模型无法拆开 SFT、rejection sampling 和 preference optimization。Base 的 Chat 条件也是 OOD diagnostic，不适合拿准确率作公平对比。

### 4. 最后做一个真正有判别力的因果实验

如果前面方向可靠，再做：

- `+d_chat → Bare`
- `−d_chat → Chat`
- norm-matched random direction
- orthogonal-to-chat direction

先测试完整 dense `d_chat` 能否改变 interface/output state。若成功，再拆分：

- Role–Chat shared sparse core
- Role-only
- Chat-only

这样可以直接区分：

1. Chat state 是否可被 hidden-state paste；
2. sparse shared core 是否负责 engagement/commitment；
3. dense Chat-only component 是否主要负责 assistant style 与 generation health。

主要 readouts 应该是 reason-first、early candidate、loop/truncation、natural EOS、abstention 和 commitment；accuracy 是下游结果，不应成为唯一判据。

---
3. 最后才做因果实验

如果方向稳定，再进行 norm-matched steering：

- 把 `Chat−Bare` direction 注入 Bare，看行为是否向 Chat 靠近；
- 反向注入 Chat，看是否向 Bare 返回；
- 再定位对跨任务方向贡献稳定的 neurons，并做消融/激活。

有一个重要限制：现在 Chat 和 Bare 的最后 token 不同，所以 `Chat−Bare` 同时包含接口状态和注入位置/token 差异。建议先分析现有数据；如果确实发现稳定方向，再补一个很便宜的 matched-anchor HS control。这个控制不需要重新生成答案，只需提取 prefill hidden states，却是以后声称“状态切换方向”前很重要的一步。

所以眼下最直接的任务是：**先完成两个模型各自的逐层 Chat–Bare direction、跨任务稳定性，以及与 signed RSN direction 的对齐分析。**

---

### 2. 做 Llama 的 Chat–Bare hidden-state 对比

这是现在机制价值最高的一步。使用 α=0，比较：

- Bare
- Native Chat
- Chat-matched

不要直接把不同长度序列逐 token 相减。建议记录三个位置：

1. 相同问题正文的最后一个 token；
2. 各条件实际的最后一个 prefill token；
3. 在三种条件后 teacher-force 同一个短前缀时，该共同 token 的 hidden state。

逐层计算：

- `‖h_chat − h_bare‖`
- `‖h_matched − h_bare‖`
- `‖h_matched − h_chat‖`
- Chat–Bare 差异方向在 GSM8K 与 MATH 之间的 cosine
- 用 GSM8K 训练 Chat/Bare linear probe，在 MATH 上测试

最关键的预测是：

> 如果 Chat-matched 的状态在中后层更接近 Native Chat，而不是与它共享最后 token 的 Bare，就说明整套角色上下文比最后一个 token 更能决定模型状态。

只保存选定位置的 HS 即可，不必保存完整序列，能大幅减少空间。

### 3. 用 Base–Instruct 对照连接 post-training

做一个 α=0 的简单 `2×2`：

| Model | Bare | Chat-formatted |
|---|---:|---:|
| Llama-3.1 Base | ✓ | ✓ |
| Llama-3.1 Instruct | ✓ | ✓ |

如果 Chat template 的行为和 HS 切换主要出现在 Instruct，而 Base 对这些 token 没有同等反应，就能支持：

> 这种状态切换是 post-training 学出来的，而不是 special tokens 天然具有的功能。

但公开模型不能干净拆分 SFT 与 DPO 的贡献，所以最多写成 **post-training-associated**，暂时不能特指 RLHF。Base 的 Chat 条件也是诊断性 OOD 对照，不能公平比较准确率。

### 4. 最后做因果状态切换

如果得到稳定的：

```text
d_chat = mean(h_chat − h_bare)
```

可以在 held-out task 上尝试：

- 向 Bare 注入 `+d_chat`：能否减少 loop、产生 Chat 式 reasoning-first 风格；
- 向 Chat 注入 `−d_chat`：能否恢复部分 Bare 风格；
- 加入 norm-matched random/orthogonal direction；
- 比较 `d_chat` 与 RSN、Confidence direction 的 cosine、neuron overlap 和 cross-steering。

这一步才真正连接你们的核心假设：

> RSN/Confidence neurons 可能不是完整的状态开关，而是广义 Chat/post-training 状态切换系统中的一个调节轴。

我的建议顺序是：**Qwen α=0 小试验与 Llama HS 同时准备 → Base/Instruct 对照 → hidden-state direction 的因果注入**。目前不建议马上定位“状态切换 neurons”；先证明存在稳定、跨任务、可干预的 state direction，再向 neuron 层下钻。
---
对，这其实是最直接的实验。做一个 α=0 的 `2×3`：

| Model | Bare | Native Chat | Chat-matched |
|---|---:|---:|---:|
| Llama-3.1-8B Base | ✓ | ✓ | ✓ |
| Llama-3.1-8B-Instruct | ✓ | ✓ | ✓ |

保持题目、解码参数和 token 序列一致。Base 没有可用 chat template 时，就手工序列化 Instruct 的同一套特殊 token。

重点不只看 accuracy，还看：

- 开场是直接答案还是 `To solve...`
- reasoning-first
- loop、truncation、natural EOS
- valid submission
- 生成长度和答案位置

结果很好解释：

- **只有 Instruct 在 Chat 下切换**：Chat token 的“状态开关”功能主要由 post-training 学会。
- **Base 也明显切换**：说明相关行为在 pre-training 中已经存在，Chat 结构能够直接调用它。
- **Base Chat 反而变差**：说明这些 token 对 Base 是 OOD，而 Instruct 通过 post-training 才赋予它们明确意义。
- **Base 出现 Chat 风格但能力没提升**：说明 template 能改变表达状态，但高质量 assistant policy 仍来自 post-training。

再进一步，才是真正的“state paste”：

```text
d_chat = h(Instruct, Chat) − h(Instruct, Bare)
```

把这个方向注入 Base，观察 Base 是否变得更像 Chat。  
**贴 template 测的是条件提示；贴 hidden-state direction 测的是状态本身的因果作用。**

所以我同意：先做这个 Base–Instruct 对照，比继续扩 benchmark 更直接。
---
### RSN vs Confidence
Role/Confidence cross-steering
建议测试的任务
第一阶段：MMLU-E
首先补齐上述 cross-steering matrix，继续测量：
- E-option rate；
- accuracy；
- 各学科类别结果；
- 正负 steering 的变化方向。
已有表格可以作为 Role direction × Role mask 的参照；如果原实验配置完全一致，就不需要重复运行。
第二阶段：GSM8K
如果不同组合在 MMLU-E 上具有相似功能，再迁移到 GSM8K，测量：
- commitment position；
- early-candidate rate；
- reasoning length；
- answer switching；
- accuracy。
MMLU-E 回答“是否都能调节 confidence”，GSM8K则回答“这种功能等价性是否能够迁移为相似的 reasoning commitment 效果”。
MATH 和其他任务暂时不需要加入。先完成 MMLU-E → GSM8K 两级验证，已经足够形成清晰的证据链。


2. **四种 mean 的关系**
   - Expert、Non-Expert、Confident、Unconfident 做逐层相关/距离矩阵。
   - 观察 Expert 是否更接近 Confident、Non-Expert 是否更接近 Unconfident。
   - 但 raw mean 会受到共同模型状态影响，所以只能作为辅助；核心仍是两个 difference direction 的比较。

3. **主结果与 divergent sensitivity**
   - 比较 all-paired confidence direction 与 divergent-only direction。
   - 目前 divergent 为 `13711/14042 ≈ 97.6%`，所以预计两者非常接近，主要用于证明结论不依赖筛选口径。

4. **之后才提取 confidence neurons**
   - 严格复用论文 NMD：每层 top `0.5%`，相同层区间。
   - 与 Role RSN 比较 Jaccard、随机期望以上的 overlap、符号一致性和 cross-projection。
   - 静态分析完成后再决定 cross-steering；只有“低重叠但功能相似”时，Manifold 才真正值得重开。

---

15. commitment regime 作为预测标的（直接预测调整的方向）
SAE ?
Model：ZGCM-1


---

建议把研究拆成两个相互独立的问题：

1. **功能差异**
   - confidence：confident vs unconfident
   - commitment：先推理再提交 vs 先提交再推理
   - 最好构造一个 `2×2` 分组，避免把“自信”误当成“提前提交”。
   - 控制正确性、题目难度、输出长度和任务，先在 GSM8K 做干净分析，再用 ProofWriter-Chat 检查跨领域保持性。

2. **不同 neuron discovery 方法的差异**
   - 在相同层、相同 neuron 数量和相同向量范数下比较 NMD、KL、LR、PCA 等方法。
   - Jaccard、排名相关、方向 cosine 和 manifold/subspace angle 只能说明结构是否相似。
   - 最关键的是做 **cross-steering matrix**：方法 A 找到的 neurons 能否改变方法 B 对应的行为，以及是否同时影响 confidence 和 commitment。功能可互换性比静态 overlap 更重要。

1. 先比较 `Expert − Non-Expert` 与 `Confident − Unconfident`
   - 相同问题、层、token position 和 sparsity；
   - 比较 neuron overlap、方向 cosine、cross-projection；
   - 最重要的是做双向 cross-steering，看两个方向能否相互控制 MSP 和 abstention。

2. 再加入 `late commitment − early commitment`
   - 在 GSM8K 中匹配正确性、难度和长度；
   - 比较它与前两个方向；
   - cross-steering 同时观察 commit position、accuracy 和 confidence。

3. 只有出现“neurons 重叠很低，但功能可以互换”后，再做 manifold/subspace 分析，研究不同稀疏方向是否汇聚到相同下游状态。
---

### P2. 补 causal direction control

- [ ] 构造与 RSN 匹配 norm、sparsity 和注入层的 random directions。
- [ ] 构造 orthogonal-to-RSN directions。
- [ ] 实际注入模型，比较 accuracy、commitment 与 Thinking Curve。
这一步回答的是“效果是否来自 RSN 方向本身”；现有 remask 只能支持 readout specificity。

---

# Brain

1. **现在做：Steingroever 健康常模对齐**

这条成本低，而且公开数据确实包含 617 名参与者的逐 trial 选择、收益与损失。不过数据混合了 95/100/150 trials 和三种 payoff scheme，因此必须只选与我们 IGT 协议完全匹配的子集。[Steingroever et al. 数据说明](https://openpsychologydata.metajnl.com/articles/jopd.ak)

建议检验：

- `net_block1–5` 学习曲线；
- net score 分布；
- deck preference；
- Wasserstein distance / RMSE；
- 按原始 study 做 held-out，而不是把所有参与者混在一起挑最佳 α。

它能支持的结论是：

> 某个 RSN 条件产生的 IGT 行为最接近健康人常模。

但不能写成：

> α=0 等于正常 dopamine 水平。

行为相似不能识别神经递质水平。因此建议称为 **human behavioural calibration**，而不是 dopamine-axis calibration。

2. **已有药理学方向对照：保留即可**

现在的定性文献结论已经够用了：

- α+ 与较早承诺、较高下注或 reward seeking 的方向相似；
- α− 与较保守行为的方向相似。

把它作为 Discussion 中的 correspondence table，不必继续寻找无法获得的三组相关系数。措辞停在：

> RSN manipulation exhibits behavioural correspondence with reported dopaminergic pharmacology effects.

不要上升为神经机制同源。

3. **fMRI RSA：有潜力，但暂不作为当前投稿必做项**

NARPS 确实有 108 名参与者、四个 mixed-gamble runs、trial timing、BIDS 数据和公开预处理结果，技术上适合开展共享刺激分析。[NARPS 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602933/)

但原计划需要两处修正：

- 必须把 NARPS 中完全相同的 gain/loss gamble 输入 LLM，才能构建可比较的 RDM；现有 MCQ/Betting 刺激不能直接与 fMRI trial 做 RSA。
- 即使 vmPFC/striatum RSA 显著，也只能说明 **representational correspondence**，不能“直接证明 RSN 操纵 reward representation”。
- Broca/Wernicke 不显著也不能证明“不是语言表征”；ROI 功效、刺激语言量和噪声都会造成 null。
- 还需要控制 gain、loss、accept/reject、RT 等 nuisance RDM，否则相关性可能只是两个系统都编码金额。

因此这会变成一个独立而完整的 brain–LLM 项目，不太可能只是附加分析。

我的建议优先级是：

1. 完成 Thinking Curve、GSM-Hard 和 causal direction control。
2. 并行做 Steingroever 健康常模的小型定量对齐。
3. 将药理学方向一致性放进 Discussion。
4. NARPS RSA 放入 future work；如果前面的行为对齐很漂亮，再考虑扩成后续论文。

所以这条线不应该删除。**最值得现在做的是健康人 IGT 行为对齐；fMRI RSA 很有价值，但不应成为当前 ACL ARR 的阻塞项。**

---
## TO DO
0. 测试一下qwen中间一点的mask ✖
1. Qwen25-7B ICG实验结果整理 ✔ 
2. 整理行为学的结果 ✔
3. Qwen GSM8k实验以及结果分析 ✔
4. Qwen MATH ✔
5. Qwen High-Dose in GSM8K ✔ 
6. MATH cot ✔ 
7. 复现qwen的thinking curve signal部分，没有存HS ✔
8. qwen的thinking curve 存HS ✔
9. 重新梳理一下ThinkingCurve的文档 ✔
9. qwen的具体分析 ✔
10. Qwen 的 output decisiveness: 从现有 7 个 H5 cell 提取 entropy/log(V)、top1、margin ✔
11. manifold llama3 实验以及结果整理 
11. manifold 补齐 Llama 全 α 曲线 ✔
12. manifold Qwen25 实验以及结果整理 ✔
13. manifold sentiity ✔
14. cross-model thinking curve + 再次整理thinking.md -> commitment regime ✔
15. GSM-Hard -> 1）最佳工作点可以复制；2）可以通过回答情况看来预测是否最优 ✔
15. GSM-Hard COT + alpha Vs. COT ✔
--- 
