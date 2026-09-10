
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
09.10 週四 确认公证需要的材料，买药
09.11 週五 上午11：00和學弟開會；下午：seminar
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
Agent1: Task design & check (GPT) & write to claude when finishing 
Agent2: claude.md mataining (claude)
Agent3: Document (GPT)

---
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

Shared-only 与 exclusive-neuron 因果拆分
最后才考虑 Manifold

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

有一个文件口径必须固定：

- Confidence 主结果使用 `confidence_diff_8B.npy`
- 不要使用 `diff_mean_confidence_8B.npy`，它是较早的 divergent-only 文件。
- 旧 Role direction 建议由 `llama3_logits/diff_mean_8B.npy - none_diff_mean_8B.npy` 现场重建，或使用与之匹配的 `llama3_logits_8B_diff.npy`。
- 不建议直接使用 `llama3_8B_diff.npy`；我检查到它与上述旧均值之差并不一致，可能来自另一版本。

因此第一份正式产物应该是一张逐层 cosine/norm 图和一份数值表，用来先回答：

> Confidence direction 和原 Role direction 是同一方向、局部共享，还是基本独立？

---

16. Ada-GSM8K部分需要一个同一的指标 （reason-first）
15. commitment regime 作为预测标的（直接预测调整的方向）
SAE ?

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
