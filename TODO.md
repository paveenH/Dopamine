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
9. 重新梳理一下AdaptiveThinking的文档 ✔
9. qwen的具体分析 ✔
10. Qwen 的 output decisiveness: 从现有 7 个 H5 cell 提取 entropy/log(V)、top1、margin ✔
11. manifold llama3 实验以及结果整理 ✔
11. manifold 补齐 Llama 全 α 曲线 ✔
12. manifold Qwen25 实验以及结果整理 ✔
13. manifold sentiity ✔
14. cross-model thinking curve + 再次整理thinking.md -> commitment regime ✔
15. GSM-Hard -> 1）最佳工作点可以复制；2）可以通过回答情况看来预测是否最优 ✔
15. GSM-Hard COT + alpha Vs. COT ✔
--- 
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
   1. 相关性分析：role neurons & confidence neurons; 相关性在11-19层上升 -> 实际上相关性非常的高，最高可以达到0.7左右
   2. confident & unconfident相关性分析：相关性在11-19最低，类似RSN
   3. overlap：band 内仅共享 46/180=25.6%，Jaccard 为 0.14；但是是显著高于随机
   4. shared-top 的单位贡献大约是 role-only/confidence-only 的 8 倍；是 neither-top 的 74 倍。
   5. overlap分析：整体 alignment 是广泛分布的，而非集中在极少数高贡献 neurons
35. cross-steering: MMLUE ✔
36. cross-steering: GSM8K ✔
---
37. 统一各个任务的行为统计指标
38. 再确认一下Loop的问题
