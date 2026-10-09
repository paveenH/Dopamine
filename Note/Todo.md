### Document writing
要求：1）细节部分放到Claude.md不要在文档中出现 2）数据尽量保持完整 3）尽量合并表格（但是不要勉强，可以合并的合并） 4）内容和章节都可以重构或者合并 5）结论简洁 通俗易懂 目标是提升可读性 6）标题和表格写英文
给我md版本的文字就好 我会自己去替换
尽量不要删除原始数据，可以合并

---

### Note
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

Bandit.NatureCommunications2026.[Foraging models explain human exploration in uncertain tasks.](https://doi.org/10.1038/s41467-026-75773-4) 该研究发现，人类在动态 Bandit 中更接近 compare-to-threshold 策略：主要追踪当前选项是否仍值得继续，而非持续比较所有候选价值。这与 PV10 中模型反复采样 incumbent、却不响应低样本替代臂的行为相似，为 incumbent persistence 提供了“局部阈值决策”的替代解释；但我们尚未进行相应的模型拟合，且任务设定不同，因此只能视为行为结构上的参照，不能断言 LLM 使用了相同的 foraging-RL 机制。

---
```text

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

##### 182/184/185/177/178a
rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/llama3/  /Users/paveenhuang/Downloads/

rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/hidden_states/llama3/  /data1/paveen/Dopamine/components/hidden_states/llama3


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.185:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```

---
### TO DO
Dopamine：调控的到底是什么怎么定位？和confidence的区分？practical？

1. 重新看看willingness实验，统一模板，优化负向表达 ✔
2.  

---

### Daily
10.09 周五 重新组织行为学实验；看Paper
10.10 高铁到杭州
10.11 Flight Taipei

准备多益考试
看完瑜伽视频
看完徐玉兰视频

---

## Vector
- 提取位置：当前 RSN 是 Role 提示在任务入口引起的综合状态差异。可以比较“身份描述结束处”和“生成入口”的 HS，区分身份编码与任务整合后的影响；目前还不能确定换位置会更好。
- 与 confidence 的差异方向：可以提取 Role 中无法被 confidence 方向解释的 residual。它可能帮助定位额外的功能成分，但不能直接叫“更纯粹的 Role”，因为仍可能包含其他提示和任务效应。
- Manifold reasoning Chat & MMLUE RSN

## Representation

| 层次 | 要回答的问题 | 推荐指标 |
|---|---|---|
| 几何结构 | 两组 activation 的空间布局是否相似？ | centered CKA、mCCA/SVCCA、正交 Procrustes residual、子空间夹角 |
| 可读出功能 | 在 A 中线性可读的信息，B 中是否仍可读？ | 同一标签/状态的线性 probe；跨条件训练-测试（train A → test B） |
| 因果功能 | 两个表征是否能产生同一种干预效果？ | norm-matched cross-steering / cross-ablation 矩阵 |

| Layer | Input | Core metrics | 回答的问题 |
| --- | --- | --- | --- |
| Mean Direction | 每个实验的全量均值差 $r=\mu_{\text{expert}}-\mu_{\text{non-expert}}$ | signed cosine、norm、逐层 cosine | 两个角色方向是否指向相近的位置？ |
| Sparse Coordinates | 均值方向的逐层 top-$k$ 坐标 | overlap、Jaccard、enrichment、符号一致性、双向能量包含率 | 是否涉及相同神经元？共享部分有多重要？ |
| Role-Transition CKA | 两个实验在相同题目上的逐题配对差 $d_i=h_{i,\text{expert}}-h_{i,\text{non-expert}}$ | **Centered linear CKA**；题目配对打乱作为 null | 不同题目的 role 变化之间，整体几何关系是否相似？ |
| Role-Transition Subspace | 各实验的逐题配对差 $d_i$ | 对 $d_i-\bar d$ 拟合 PCA；比较子空间夹角、交叉重建率 | 除平均方向外，role 对不同题目的影响模式是否相似？ |
| Direction–Subspace | 一个实验的均值方向 $r$ 与另一个实验的 role-transition PCA 子空间 $V_k$ | 投影比例 $R^2=\lVert V_k^\top r\rVert^2/\lVert r\rVert^2$，及随机基线 | 一个实验的平均角色方向在多大程度上落入另一个实验的变化空间？ |

## Brain

**Steingroever 健康常模对齐**

公开数据确实包含 617 名参与者的逐 trial 选择、收益与损失。不过数据混合了 95/100/150 trials 和三种 payoff scheme，因此必须只选与我们 IGT 协议完全匹配的子集。[Steingroever et al. 数据说明](https://openpsychologydata.metajnl.com/articles/jopd.ak)

建议检验：

- `net_block1–5` 学习曲线；
- net score 分布；
- deck preference；
- Wasserstein distance / RMSE；
- 按原始 study 做 held-out，而不是把所有参与者混在一起挑最佳 α。

它能支持的结论是：某个 RSN 条件产生的 IGT 行为最接近健康人常模。行为相似不能识别神经递质水平。因此建议称为 **human behavioural calibration**，而不是 dopamine-axis calibration。


**已有药理学方向对照：保留即可**

现在的定性文献结论已经够用了：

- α+ 与较早承诺、较高下注或 reward seeking 的方向相似；
- α− 与较保守行为的方向相似。

把它作为 Discussion 中的 correspondence table，不必继续寻找无法获得的三组相关系数。措辞停在：

> RSN manipulation exhibits behavioural correspondence with reported dopaminergic pharmacology effects.

**fMRI RSA：有潜力，但暂不作为当前投稿必做项**

NARPS 确实有 108 名参与者、四个 mixed-gamble runs、trial timing、BIDS 数据和公开预处理结果，技术上适合开展共享刺激分析。[NARPS 数据说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC6602933/)

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
16. MATH 补充完整 ✔ 
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
45. 确认Reasoning RSN(RRSN)的expert与non-expert的关系；RRSN与MRSN之间的相关性 -> 还是和之前一样 核心重叠，相似度很低；✔
46. Steering RSM8K with RRSN ✔ 
47. Steering MMLUE with RRSN ✔ -> 结果很乱，和MRSN非常不一致
48. 看是不是RRSN的提取需要增加Abstention的提示 -> ARRSN 结果没有比较好 ✖ 
49. 换成全量的GSM8K RRSN 作为固定的RRSN ✔
50. 换成全量的GSM8K Chat 作为固定的ChatSN ✔
51. 删除之前所有关于RRSN的内容 ✔
52. GSM8K with anstention expert & non-expert performance ✔
   1) 当前的prompt控制不住输出，那么HS应该也是不对的 ✖
   2) 优化prompt v2 -> 有差异但是都会回答 ✖
53. 全量 1319 题 GSM8K Role 方向：GRSN & AGRSN ✔
54. 相似度 GRSN & AGRSN -> 相似度也不是很高 ✖
