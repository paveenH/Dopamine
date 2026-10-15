


InfoQuest 即使完成，也主要增加一个行为读数，无法单独裁决上述解释。 如果接下来要优先回答“到底控制什么”，实验就必须让这些解释产生不同预测，而不能只再找一个 α 能改变的任务。我不应把行为证据的增加，等同于功能识别的推进。

### GSM8K Base

Qwen 的方向反转，是不是推理训练造成的。我看了驱动脚本，不用改代码就能跑。关键看 **Qwen2.5-7B Base 在 α=0 时会不会也"先答后算"**。Qwen-IT 的 early-candidate 是 96%。

| Qwen Base 的表现 | 含义 |
|---|---|
| α=0 已经先答后算，+8 也让它改成先推理 | 反转在预训练阶段就存在，不是 IT 造成的。Qwen2.5 的预训练里本来就有大量合成数学数据，所以这个结果完全可能出现 |
| α=0 先推理，或 +8 不起作用 | 反转和指令微调有关 |
| 格式本身就不健康（续写新题、循环、不写 `####`） | 这个接口在 Base 上读不出结果，需要改用 few-shot |

Llama Base 作为对照：它应该保持 Llama-IT 的方向，即 −6 让它更晚给出候选答案。

## 最小设计

- **模型：** `Qwen/Qwen2.5-7B` 和 `meta-llama/Llama-3.1-8B`，都是 Base。
- **mask：** 用 IT 上提取的 mask，不另外提取。Base 和 IT 层数、维度相同。原论文在 MMLU-E 上已经证明这样能起作用。
- **prompt：** 和 IT 的 Bare zero-shot 完全一样。这样 Base 和 IT 之间只有模型不同。先不用 few-shot，因为它会同时改变接口。
- **α：** 0 和冻结工作点（Qwen +8，Llama −6）。一次加载跑两格，每个模型 2 格，共 4 格。
- **读数：**
  - 先看格式健康：no-marker、loop、截断，以及有没有续写出新题；
  - 健康的话，再看 `early_candidate_rate` 和 `first_acc`。

## 启动命令

`--model` 只决定输出路径，所以设成 `qwen2.5_base` / `llama3_base` 就能和 IT 结果分开。`--hs` 不变，这样会读 IT 的 mask。

```bash
cd /data1/paveen/Dopamine
CUDA_VISIBLE_DEVICES=0 nohup python get_answer_regenerate_gsm8k.py \
  --model qwen2.5_base --model_dir Qwen/Qwen2.5-7B --hs qwen2.5 --size 7B \
  --type non --percentage 0.5 --mask_type nmd --configs 0-16-22 8-16-22 \
  --test_file benchmark/gsm8k_test_sample.json --ans_file answer_mdf_gsm8k \
  --suite default --fmt_wording plain --base_dir /data1/paveen/Dopamine/components \
  --roles neutral --max_new_tokens 768 --temperature 0 --batch_size 24 \
  > qwen_base_gsm8k.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python get_answer_regenerate_gsm8k.py \
  --model llama3_base --model_dir meta-llama/Llama-3.1-8B --hs llama3 --size 8B \
  --type non --percentage 0.5 --mask_type nmd --configs 0-11-20 neg6-11-20 \
  --test_file benchmark/gsm8k_test_sample.json --ans_file answer_mdf_gsm8k \
  --suite default --fmt_wording plain --base_dir /data1/paveen/Dopamine/components \
  --roles neutral --max_new_tokens 768 --temperature 0 --batch_size 24 \
  > llama_base_gsm8k.log 2>&1 &
```

输出会写到 `components/{qwen2.5_base,llama3_base}/answer_mdf_gsm8k/mdf_{0,8|-6}/`。启动后先 `cat` 两个 log，确认已经开始运行。

---
### 第二优先：角色提示与 steering 的对应检验（先做一格，GPU 开销小）

RSN 在定义上是 expert − non-expert 的表征差，所以最直接的检验不预设心理功能，只问：

> 在同一模型、同一任务上，+α 与 −α 的差异，是否复现 expert 与 non-expert 提示之间的差异？

**待检验的解释（"角色策略"）：** +α 把模型推向它自己学到的"专家行为"，而专家在某个情境下做什么是训练决定的。决策任务里两个模型学到的专家都"果断"，所以 Betting / CGT / InfoQuest 方向一致；数学推理里 Qwen2.5 可能因大量推理训练而把"专家"学成先推导再作答，所以只有 Qwen GSM8K 的方向和 Llama 相反。如果成立，"方向相反"不再是矛盾，而是角色效应本身因模型和任务而异。

#### 第一步：只做 Qwen GSM8K（决定性的一格）

整个方向矛盾集中在这里，先用它决定这条线是否值得扩展。

- **设计：** Qwen2.5-7B-Instruct，GSM8K Bare No-CoT，α=0，同一次运行内跑 `an expert` / `a non expert` / `neutral` 三个提示（role 字符串与 Llama `run_gsm8k.sh` 一致），n=300，逐题配对。neutral 同批重跑，不与旧 `mdf_0` 跨批比较。输出写入新目录 `answer_mdf_gsm8k_role/`，不覆盖已有 cell。
- **预先写定的预测（运行前冻结）：** Qwen expert 提示的 `early_candidate_rate` **低于** non-expert 提示，方向与 Qwen 自身 +α 一致、与 Llama 相反。
- **读数：** 主读数 `early_candidate_rate`（`earlycand-v1`，冻结检测器，不重调）；辅助读数 offline `first_acc`；expert vs non-expert 用配对 exact McNemar，并报配对 bootstrap 区间。
- **判定：**

| 结果 | 解读 |
|---|---|
| 方向与预测一致，区间不含 0 | 支持"角色策略"解释，进入第二步 |
| 方向相反 | 解释被否定；方向差异来自 steering 本身而非角色，需另找原因 |
| 差异接近 0 / 区间跨 0 | 提示层面的角色效应太弱，本检验不能下结论，**不计为支持** |

#### Llama 对照：离线重算

此前的 Llama GSM8K 角色数字（expert 58.0% / ec 54.3%；neutral 60.0% / 48.0%；non-expert 68.0% / 41.7%）来自 inline 字段，**不能直接引用**。用冻结抽取器（`analyze_first_last_acc`）和 `earlycand-v1` 在 `llama3/gsm8k/bare/mdf_0` 上重算后，与 Qwen 用同一个分析脚本出对照表。纯离线。

#### 第二步（仅当第一步支持时）：扩展到行为任务

- Betting 和 CGT 各补 expert、non-expert 提示，两个模型，只跑 α=0。
- **这些都是新协议，不是旧结果的补格：** Betting 的冻结 prompt 没有 role 位置，加入角色句会改变 prompt；Qwen CGT v5 已 CLOSED。两者都只能与同批重跑的 neutral 比较，不能与 §3.1 / §3.3 的冻结结果比较。
- 可选：在 Chat 接口下看角色提示是否同样不再改变输出组织，用于解释表 7a 的接口依赖。

**限制：** 角色提示与 steering 方向一致只说明两者对应，是 consistent-with 证据，不证明 steering 的效应"经由"角色表征产生；提示改变的是整个上下文，steering 只改一组神经元。

### 第三优先：草稿末尾新加的"核验工具"实验，暂缓

它的思路很好：把"是否发起核验"和"能否利用核验结果"分开。但我有三点顾虑：

1. **接口风险。** 调用工具基本需要 Chat 接口，而表 7a 显示 Llama 在 Chat 下基线已经先推理后作答，RSN 在那里没有效果。Llama 很可能直接得到无效结果。Qwen 才可能有信号。
2. **前车之鉴。** Fixed-Answer、Willingness、InfoQuest 这三条新接口实验，都在建立测量接口阶段卡住了。这个实验需要新的工具接口、新的注入位置和新的评分方式，是同类风险。
3. **它不解决你最关心的问题。** 草稿自己也写了：即使成功，也只建立推理内部的联系，不能把 Betting、CGT 连起来。

所以建议：只有在角色对应检验失败、而我们仍然需要一座行为与推理之间的桥时，再启动它。届时先在 Qwen 上做 α=0 的小规模试点。

## 顺带的小事

- 草稿里的回退子集准确率还是反推出来的近似值，用冻结抽取器逐题重算一遍，大约十分钟。
- 先提交这一轮指标统一的文档改动。
- C1 里 Qwen CGT 降为附录里的边界证据，主文的行为证据以 Betting 为主。
