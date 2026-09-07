现在我们的目标是将Dopamine写成一篇conference paper 或者journal 
RSN最初的内容已经发表，来源是RSNpaper，具体内容在/Users/paveenhuang/Downloads/Dopamine/ACLARR
现在我们的目标是想要论证，这一套机制类似人体的Dopamine系统机制
AdaDopamine.md记录了一些行为学实验
AdaDopamine_gsm8k.md是推理任务上的表现，gsm8k & math
AdaptiveThinking.md 这里是对内部的thinking Curve的一些观察 原本是计划找到一些类似激素水平变化的曲线，但是目前没有显著的效果
AdaManifold.md这里是和Manifold有关的一些研究，是AdaptiveThinking的一些拓展
其余是一些辅助文件：
AdaBandit.md 专门记录了Bandit实验（主线在AdaDopamine.md），但是没有找到合适的结果
AdaLogitsLens.md 对应RSNpaper时候做的一下研究
TODO是接下来的一些执行计划

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
rsync -avzP d12922004@140.112.31.185:/data1/paveen/Dopamine/components/qwen2.5 /Users/paveenhuang/Downloads

rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/benchmark/cruxeval_p4c_formal.json /Users/paveenhuang/Downloads

rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

---
Daily
09.07 周一 生理期推迟，确认公证需要的材料
09.08 周二
09.09 周三 组会（11:30）
09.11 seminar
09.19 台北-杭州萧山 机票 ✔
09.20 杭州逛逛
09.21 杭州逛逛
09.22 回家高铁*1 - Helene ⏸
09.23、09.24 在家 需要去办理公证 + 爸妈护照
09.24 全曜回家 高铁*1  ⏸
09.25 - 10.02 武夷山-成都-丽江 <川滇之间>
10.02 丽江
10.02晚上-10.03 成都市区 机票 ✔ 住宿 ⏸
10.04 成都-武夷山Flight Home
10.04-10.10 Home
10.11 Flight Taipei

准备多益考试
看完瑜伽视频
看完徐玉兰视频
---
Dopamine.Nature2026.[Endocannabinoids facilitate reward engagement through retrograde gain control.](https://doi.org/10.1038/s41586-026-10967-w) 该研究发现，伏隔核 D2R–Penk 神经元通过释放内源性大麻素 2-AG，逆向抑制 aPVT→NAc 的谷氨酸输入，从而以通路特异的增益控制维持奖励追求中的行为投入。该机制与 RSN 调节 engagement/commitment gain 的功能解释高度相关，也位于接受多巴胺调节的伏隔核奖赏回路中；但论文直接验证的是 `2-AG→CB1R` 通路，而非 dopamine，因此适合作为 neuromodulatory engagement gain control 的生物学参照，而不能作为 RSN≈dopamine 的直接证据。

---
Task Agent: task coding (claude)
Agent2: Task design & check (GPT) & write to claude when finishing 
Agent3: claude.md mataining (claude)
Agent4: Document (GPT)
---
组会内容（08.31）：
1）弄清楚neurons的差异：confident & unconfident // thinking & answer directly (先推理再提交、先提交再推理)
2）Manifold -> 不同方法找到的neurons之间的差异

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
29. FinQA ⏸
30. GSM-Symbolic ⏸
31. 考虑一下不同的neurons之间有什么差异

---
16. Ada-GSM8K部分需要一个同一的指标 （reason-first）
15. commitment regime 作为预测标的（直接预测调整的方向）

---

### P2. 补 causal direction control

- [ ] 构造与 RSN 匹配 norm、sparsity 和注入层的 random directions。
- [ ] 构造 orthogonal-to-RSN directions。
- [ ] 实际注入模型，比较 accuracy、commitment 与 Thinking Curve。
这一步回答的是“效果是否来自 RSN 方向本身”；现有 remask 只能支持 readout specificity。

---
---

## TO DO — ACL ARR


---

> **RSN/Thinking Curve 能否成为可预测、可迁移、可用于选择推理状态的指标。**

这会让工作从“有趣的机制现象”提升成“有实际用途的 reasoning calibration framework”。

## 一、预测模型对错：值得做，但要重新定义目标

现有结果其实还不能说 Thinking Curve 可以预测单题对错：

- correct 组通常有更高、更持续的 pre-commit `s_t`；
- 但 `G_prefill`、`p_t`、entropy/top1/margin 都没有稳定区分 correct/incorrect；
- Manifold feature 也没有提供稳定的增量预测价值。

这说明单独拿一个 `s_t` 或 PCA 指标做分类可能不够。[AdaptiveThinking.md](/Users/paveenhuang/Downloads/Dopamine/AdaptiveThinking.md:276) [AdaManifold.md](/Users/paveenhuang/Downloads/Dopamine/AdaManifold.md:231)

但我仍然认为值得继续，因为之前检验的是“单个信号是否直接预测 correctness”，还没有真正检验：

> **完整的 reasoning-state panel 是否能在答案提交前，增量预测这道题最终会不会做对。**

建议定义两个预测时间点：

1. **Early prediction**：生成前 20/50 tokens 后预测最终正确性。
2. **Pre-commit prediction**：模型即将首次提交答案前预测正确性。

特征可以包括：

- `s_t` level、变化量和稳定度；
- `p_t` amplitude；
- entropy、top1、margin；
- early-candidate 是否出现；
- 当前 reasoning length；
- commitment proximity；
- answer switching / instability proxy。

关键基线必须包括：

- 只用题目难度；
- 只用 entropy/top1/margin；
- 只用生成长度；
- 上述基线 + RSN/Thinking Curve features。

真正有价值的结果不是单纯 AUROC 高，而是：

> **加入 RSN dynamics 后，在 held-out questions 上比普通 confidence、difficulty 和 length 基线预测得更好。**

更强的验证是：

> 在 GSM8K 训练预测器，冻结后直接测试 MATH。

如果能跨任务保持预测力，Thinking Curve 才真正具有“reasoning monitor”的价值。

## 二、统一工作点：应该改成“统一功能工作区间”

我不建议假设 GSM8K 和 MATH 存在完全相同的 raw α 最优点，因为现有数据已经显示：

- Llama GSM8K：`−6` 是尖锐峰值；
- Llama MATH：目前只充分支持 `−4 > 0 > +4`，还没有完整覆盖 `−6`；
- Qwen GSM8K：`+8～+12` 平台；
- Qwen MATH：`+6` 左右最好，`+8` 在困难题上回落。

所以“统一 α”大概率不成立。但可能存在：

> **跨任务共享的 functional working-state region。**

例如这个区间可能表现为：

- 不在开头立刻给候选答案；
- 保留足够的 pre-commit computation；
- output distribution 已经足够明确；
- 但没有进入过度延迟、重复、改答案或计算压缩；
- commit 后能够正常 release 和停止。

也就是说，统一的不是剂量，而是：

> **commitment readiness 与 remaining computation 之间的平衡状态。**

## 三、最有价值的实验：用 GSM8K 的状态目标预测 MATH 的最佳剂量

我建议下一步直接做一个“跨任务工作点迁移”实验。

### 阶段 A：先用现有输出做零成本 pilot

在 GSM8K 上定义一个不使用 accuracy 的 functional-state score，例如组合：

- normalized commitment position；
- pre-commit reasoning span；
- early-candidate rate；
- answer switching；
- loop/stopping；
- output decisiveness。

然后冻结这个定义，直接应用到现有 MATH 各 α cell：

- 检查 GSM8K 的优良状态区间，在 MATH 上是否也对应较高 accuracy；
- 检查过早 commit 和过度 processing 是否在两个任务上都对应失败区。

这一步主要用现有文本结果，可以先判断假说有没有希望。

### 阶段 B：再补 MATH hidden states

如果文本层 pilot 支持，就只收集关键剂量的 MATH hidden states，不做完整大 sweep：

- Llama：`−8 / −6 / −4 / 0 / +4`
- Qwen：`0 / +4 / +6 / +8 / +12`

然后完成真正的迁移检验：

1. 在 GSM8K 上冻结 state representation 和目标区间；
2. 不看 MATH accuracy，只根据 MATH calibration subset 的内部状态选择最接近目标区间的 α；
3. 在独立 MATH test subset 上揭示 accuracy；
4. 比较 state-selected α、固定 α=0，以及直接照搬 GSM8K 最佳 raw α。

如果 state-selected α 能迁移，而 raw α 不能，这会是非常强的结果：

> **RSN 的价值不是提供一个通用剂量，而是提供一个可以跨任务识别和校准的 reasoning working state。**

## 最终可以形成的新故事

现在的故事是：

> RSN steering 改变 commitment dynamics，并产生模型与任务依赖的 performance curve。

下一阶段可以升级成：

> **Thinking Curve 提供一个在线 reasoning-state readout；该状态能够预测推理成功与失败，并允许我们不依赖目标任务标签、通过功能状态匹配将 GSM8K 上发现的工作区间迁移到 MATH。**

我会把优先级排成：

1. 现有 GSM8K/MATH 输出上的跨任务 functional-state pilot；
2. GSM8K 内部的 held-out correctness prediction；
3. GSM8K→MATH frozen predictor transfer；
4. 只有前三项出现稳定信号，才补 MATH hidden-state cells。

这条路线比继续证明干预有效更有“价值”，而且能够明确回答两个更大的问题：**能否提前知道模型会不会做错，以及能否把一个任务上的最佳推理状态迁移到另一个任务。**


# Brain

我觉得有价值，但建议拆成“现在做”与“后续做”。

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
