### Document writing
要求：1）细节部分放到Claude.md不要在文档中出现 2）数据尽量保持完整 3）尽量合并表格（但是不要勉强，可以合并的合并） 4）内容和章节都可以重构或者合并 5）结论简洁 通俗易懂 目标是提升可读性 6）标题和表格写英文
给我md版本的文字就好 我会自己去替换
尽量不要删除原始数据，可以合并

---

### Note
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

Bandit.NatureCommunications2026.[Foraging models explain human exploration in uncertain tasks.](https://doi.org/10.1038/s41467-026-75773-4) 该研究发现，人类在动态 Bandit 中更接近 compare-to-threshold 策略：主要追踪当前选项是否仍值得继续，而非持续比较所有候选价值。这与 PV10 中模型反复采样 incumbent、却不响应低样本替代臂的行为相似，为 incumbent persistence 提供了“局部阈值决策”的替代解释；但我们尚未进行相应的模型拟合，且任务设定不同，因此只能视为行为结构上的参照，不能断言 LLM 使用了相同的 foraging-RL 机制。

---

### Daily

准备多益考试
看完瑜伽视频
看完徐玉兰视频

---
### TO DO

15. ProofWriter-OWA 补充实验 Chat zero-shot No-CoT ✔ -> Llama 未检出工作点收益；Qwen 严格分数上升主要伴随格式恢复，宽松标签正确率未见提升。
16. 复核 ProofWriter-OWA Qwen Chat 1-shot CoT `+8` ✔ -> 冻结严格收益 +6.67 pp（Holm p=.03027）；宽松标签 41.00%→48.00%，+7.00 pp（配对 p=.00985，独立 Holm m=3 p=.02956），不由严格 marker 格式恢复解释；主结果不变，见 ReasoningBare §6.5。
17. ProofWriter-OWA Chat zero-shot CoT ✔ -> Qwen +8 严格 +18.00 pp（Holm p=1.28×10⁻⁹），但宽松标签口径仅 +1.67 pp（p=.487）或 +3.67 pp（p=.0614）；260/300 在解释前给标签。Llama −6 未检出提升。
18. FinQA Chat 0-shot CoT ✔ -> Llama −6 未检出提升；Qwen +8 严格准确率下降 12.33 pp（Holm p=4.74×10⁻⁷），宽松数值评分无净变化，主要受 marker 格式影响。
19. 一些统计分析 ✔
20. Inducing answer-first ordering -> pushy chat GSM8K ✔ -> pushy 未诱发更多 early-candidate
21. Mistral实验补充 ✔ -> Mistral在Bare常无法回答GSM8K 在Chat上有趋势但是没有commitment change
22. Openchat/ Qwen3 GSM8K 实验补充 ✔
23. 把Qwen3 测完整 ✔ -> GSM8K 768/2048、GSM-Hard 与 MATH 完成；固定 `+10` 在两项迁移比较中均通过 Holm m=2，MATH 的 `+8` 高于 `+10`。
24. New model: OLMo-3-7B-Instruct ✖
25. New Model: granite-3.3-8b-instruct ✔

---
### TO DO
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
---
1. 重新看看willingness实验，统一模板，优化负向表达 -> v2，出现相反的结果 ✖
2. 优化willingness的表述，再做一版本的结果 -> v3 还是很难解释 ✖
3. 从MMLUE开始完善，弄清E选项降低的原因 -> 看起来结果还是很乱 ✖
4. 重新整理思路 ✔
5. 统一posN Qwen和llama的reasoning行为学分析指标 #### 的定义 ✔
6. infoquestion: RSN 改变的是模型“直接作答的倾向”，还是“根据信息是否足够，决定先问清楚再回答”的策略 ✔
6. 重新梳理当前的思路和之前想到的理论之间的关系 ✔
7. 完成 Norm-Matched Orthogonal Injection Controls ✔
8. GSM8K Base ✔
9. llama3-base 1-shot gsm8k -> 1-shot会破坏commit的自然规律 ✔
10. Qwen Base 的 early-candidate 小盲审 ✔
11. Off-Support Orthogonal Control ✔
12. Off-Support Orthogonal MMLUE ✔
13. Qwen2.5 expert vs non-expert ✔
14. Chat-CoT Saturation Check (GSM8K / GSM-Hard) ✔
