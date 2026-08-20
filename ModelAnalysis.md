- All results are calculated by the softmax logits of the last token in the prompt.
- Roles in the prompts are *non task expert* and *task expert*
- non-task-expert

## **Hypothesis1: Last-Token Logits Predict Final Answers**

!image.png

| **Model** | **Acc** | **E** |
| --- | --- | --- |
| **hermes** | 0.958 | 0.936 |
| **llama3** | 0.993 | 0.971 |
| **qwen2.5** | 0.988 | 0.977 |
| **mistral** | 1.000 | 1.000 |
| **openchat** | 0.997 | 0.999 |
| **Zephyr** | 0.997 | 0.997 |

---

## ~~Hypothesis2: As long as there exists either explicit roleplaying data (e.g., ShareGPT, UltraChat) or implicit supervision via human feedback (e.g., RLHF, DPO), role-sensitive neurons (RSNs) can be activated and edited to control role behavior.~~

### Evaluation on role effect

- **Paired Win Ratio:** Expert Acc > Non Expert Acc
- **Accuracy Gain (AAG):** mean (expert - non_expert)
- **Max-Normalized Gain:** mean((expert - non_expert) / (1 - non_expert))

| **Model** | **Accuracy Gain<br>expert−non** | **Paired Win Ratio<br>expert−non** | **Max-Normalized Gain<br>expert−non** | $\Delta E$<br>non-expert change | **RolePlaying Data** | **Human Data** | **Level** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **llama3_base_8B** | 0.228 | 0.965 | 0.282 |  | 0 | 0 | 3 |
| **llama3_8B** | 0.244 | 0.982 | 0.393 |  | 0 | 1 | 1 |
| **hermes_8B** | 0.321 | 0.982 | 0.406 |  | 1 | 1 | 2 |
| **llama3_base_3B** | 0.171 | 0.982 | 0.195 |  | 0 | 0 | 3 |
| **llama3_3B** | 0.102 | 0.912 | 0.175 |  | 0 | 1 | 3 |
| **hermes_3B** | 0.131 | 0.947 | 0.193 |  | 1 | 0 | 3 |
| **mistral_base_7B** | 0.265 | 0.982 | 0.277 |  | 0 | 0 | 3 |
| **mistral_7B** | 0.290 | 0.982 | 0.364 |  | 1 | 0 | 1 |
| **openchat_7B** | 0.414 | 0.982 | 0.489 |  | 1 | 1 | 2 |
| **zephyr_7B** | 0.295 | 0.965 | 0.325 |  | 1 | 1 | 2 |
| **openhermes_7B** | 0.387 | 1.000       | 0.433   |  | 1 | 0 | 1 |
| **noushermes_7B** | 0.361 | 1.000        | 0.413 |  | 1 | 1 | 1 |
| **qwen2.5_base_7B** | 0.092 | 0.947 | 0.216 |  | 0 | 0 | 2 |
| **qwen2.5_7B** | 0.240 | 0.982 | 0.371 |  | 0 | 1 | 1 |
| **qwen2.5_base_3B** | 0.030 | 0.789 | 0.058 |  | 0 | 0 | 2 |
| **qwen2.5_3B** | 0.126 | 0.877 | 0.153 |  | 0 | 1 | 1 |
| **qwen3_base_8B** | 0.041 | 0.860 | 0.107 |  | 0 | 0 | 2 |
| **qwen3_8B** | 0.109 | 0.965 | 0.223 |  | 0 | 1 | 1 |
| **phi4mini_4B** | 0.046 | 0.789 | 0.100 |  | 0 | 0 | 4 |
| **deepseek_base_7B** | 0.013 | 0.632 | 0.018 |  | 0 | 0 | 4 |
| **deepseek_7B** | 0.021 | 0.684 | 0.029 |  | 0 | 1 | 4 |
| **deepseekRqwen_7B** | 0.102 | 0.965 | 0.108 |  | 0 | 0 | 4 |
| **falcon3_10B** | 0.009 | 0.544 | 0.008 |  | 0 | 0 | 4 |
| **falcon3_7B** | 0.006 | 0.544 | 0.013 |  | 0 | 0 | 4 |
| **gemma_7B** | 0.006 | 0.561 | 0.010 |  | 0 | 1 | 4 |
| **stablelm_12B** | 0.168 | 0.912 | 0.225 |  | 1 | 1 | 2 |
| **stablelm_base_12B** | 0.017 | 0.667 | 0.017 |  | 0 | 0 | 4 |

| **Model** | Performance<br>Acc / E (%) Non<br>Acc / E (%) Non + 4RSN<br>—————————<br>Acc / E (%) Exp<br>Acc / E (%) Exp + 4RSN | Data | Post Training | Note |
| --- | --- | --- | --- | --- |
| **Llama3-8B-base** | 20.67%	61.17%<br>29.11%	42.59%<br>—————————<br>43.46%	17.79%<br>48.17%	5.75% | meta-llama/Llama-3.1-8B<br>Grouped-Query Attention (GQA)<br>Llama 3.1 was pretrained on approximately 15 trillion tokens from public data sources. The training data includes both public instruction datasets and over 25 million synthetic examples, with a composition of **50% general knowledge, 25% math and reasoning, 17% code, and 8% multilingual content**. | No SFT<br>No post training<br>Reported Acc of MMLU: 66.7%<br>Highest Acc in RSN: 53.86% | 1. Base model具有roles之间的鉴别能力；<br>2. 中间层的提升不是很明显，Full的提升比较明显；**→ Base没有办法把差异内化到mid layer**<br>3. 用llama-IT的hs修改的效果会更好 |
| **Llama3-8B** | 38.65%	44.77%<br>63.22%	3.73%<br>—————————<br>63.01%	6.87%<br>65.74%	0.26% | meta-llama/Llama-3.1-8B-Instruct<br>**Fine-tuning data:** Combined ***human-written*** and synthetic data, selected using LLM-based classifiers to ensure high-quality prompts and responses, with a focus on safety.<br>**Refusals and Tone:** Special attention to refusal behavior and tone, including borderline and adversarial prompts, with safety responses adjusted to follow tone guidelines. | SFT → Rejection Sampling (RS) → DPO<br>Preference Data是自己标注的<br>Reported Acc of MMLU: 69.4%<br>Highest Acc in RSN: 66.10% | 1. **IT模型的角色分离已被中间层很好地编码**，不依赖于后层<br>2. 虽然 base 的“全层 RSN”迁移到 IT 有一定效果，但对中间层编辑却远不如 IT 自己的 RSN **→ Base没有办法把差异内化到mid layer** |
| **Hermes-8B** | 22.23% 69.98%<br>51.02% 23.92%<br>—————————<br>54.34% 19.79%<br>60.13% 3.66% | NousResearch/Hermes-3-Llama-3.1-8B<br>在Hermes 2的基础上优化roleplaying，Hermes 3 uses **ChatML** as the prompt format; System prompts allow steerability and interesting new ways to interact with an LLM, guiding rules, roles, and stylistic choices of the model.;<br>在数据中有引入**Roleplaying**，占比6.1%（24M token） | SFT based on **meta-llama/Llama-3.1-8B**<br>SFT + DPO (only the 8B model has DPO)<br>****Reported Acc of MMLU: 64.79%<br>Highest Acc in RSN: 60.44% | 1. Full会比Middle略微好一点；但是如果α拉到5倍的话，non expert就会比较好一点。总体来说差异不是很大；可以认为是都有效果<br>2. 从Role effect指标来看，直接在SFT的时候**引入role data更加能够提高模型对Role的敏感程度** |
| **Llama3-3B-base** | 13.88% 63.80%<br>16.63% 55.49%<br>—————————<br>30.95% 20.22%<br>32.76% 13.73% | meta-llama/Llama-3.2-3B<br>Grouped-Query Attention (GQA)<br>Base model<br>Training Data: A new mix of publicly available online data (9 trillion tokens). | No SFT No post training<br>Pretraining includes knowledge distillation using **token-level logits from Llama 3.1 8B/70B** to recover performance after pruning.<br>Reported Acc of MMLU: 58%<br>Highest Acc in RSN: 37.62% | 1. 中间层的提升不是很明显，Full的提升比较明显；**→ Base没有办法把差异内化到mid layer**<br>2. Base整体的表现是比IT差很多的，最高只到37%； |
| **Llama3-3B** | **42.62%	12.02%**<br>50.90%	3.71%<br>—————————<br>**52.39%	3.96%**<br>55.72%	0.61% | **meta-llama/Llama-3.2-3B-Instruct**<br>SFT data: combining **human-generated data** from our vendors with synthetic data to mitigate potential safety risks<br>Refusals and Tone: The model is trained to handle both benign and adversarial prompts with appropriate refusals and a controlled, safe tone. | SFT → Rejection Sampling (RS) → DPO<br>Reported Acc of MMLU: 63.4%<br>Highest Acc in RSN: 55.72% | 在切换到Logits计算之后，MDF的表现都下降了；<br>→ 3B 模型容量有限，经过 instruction tuning 后，所有“role effect”信号都被模型直接写死在参数里，不再有独立的、可以被 neuron edit 调用的 activation gap; RSN 向量的编辑，反而扰乱了已稳定的中间层特征表达。 |
| **Hermes-3B** | 34.01%	34.56%<br>40.12%	18.98%<br>—————————<br>47.10%	10.13%<br>49.11%	3.05% | **NousResearch/Hermes-3-Llama-3.2-3B**<br>强调 alignment to user、角色扮演（roleplaying）、多轮对话、推理能力等提升<br>在数据中有引入**Roleplaying**，占比6.1%（24M token） | SFT based on **Llama-3.2-3B**<br>full parameter fine-tune<br>No RLHF or DPO<br>Reported Acc of MMLU: \<br>Highest Acc in RSN: 53.78% | middle提升有效果，但是不如full；mid和full的差异比hermes-8B更大；<br>**→ 3B model没有办法从post training中学习到差异；但是可以直接从数据中学到差异，但是这里的差异还是没有办法完全内化到middle layer（有部分效果）** |
| **Mistral-7B-base** | 3.30% **94.86%**<br>8.49% 84.35%<br>—————————<br>29.83% 51.40%<br>39.31% 26.37% | **mistralai/Mistral-7B-v0.3**<br>Grouped-Query Attention (GQA) + Sliding Window Attention (SWA)<br>Trained on a mixture of publicly **available and synthetic** datasets across multiple languages (e.g., English, French, German, Italian, Spanish). The dataset was carefully deduplicated and filtered for quality. No proprietary or private data was used. Used temperature-based sampling to balance data proportions<br>**May be 8T tokens** | No SFT<br>No post training | 1. Full的效果还不错，但是mid几乎没有效果；<br>2. **IT RSN的效果超过base** model自身RSN； |
| **Mistral-7B** | 21.17%	72.71%<br>51.97%	18.80%<br>—————————<br>50.14%	24.68%<br>57.90%	1.46% | **mistralai/Mistral-7B-Instruct-v0.3**<br>Publicly available chat and instruction datasets (e.g., from Hugging Face); no proprietary data, safety labels, or preference data were used.<br>**May include roleplaying datasets such as UltraChat、ShareGPT、OpenAssistant** | Pre-train → SFT<br>**no DPO or PPO** | 1. 原本full会下降（generation）在这里是提升<br>2. mid > full |
| **Openchat-7B** | 18.31%	73.65%<br>56.86%	2.51%<br>—————————<br>59.68%	3.98%<br>**57.49%	0.25%** | **openchat/openchat_3.5<br>Data mixture**: A custom blend of high-quality instruction datasets (e.g. **OpenChat ShareGPT**, **Open-Orca** with FLAN answers, Capybara, GOAT, Glaive, MetaMathQA, MathInstruct, **OpenAssistant**, Feedback-Collection, etc.)   | Based on **mistralai/Mistral-7B-v0.1**<br>Single-stage C-RLFT (no separate SFT or RLHF), directly aligns with preference-labeled data.<br>Reported Acc of MMLU: 64.3% | 1. Non expert编辑的效果是理想的, mid > full;<br>2. Expert也存在mid >full，一点小问题是只有当α=3的时候，Expert的表现才能上升，α=4以及full都会造成acc下降；<br>3. 从SFT引入roles information来训练模型看起来是只有成效的；<br>4. 使用mistral-IT RSN的效果也不错；接近自身的RSN |
| **Zephyr-7B** | 13.11%	76.83%<br>38.97%	25.85%<br>—————————<br>42.59%	22.08%<br>51.42%	1.61% | **HuggingFaceH4/zephyr-7b-beta**<br>1. Initially fine-tuned on a filtered and preprocessed of the **UltraChat dataset**, which contains a diverse range of synthetic dialogues generated by ChatGPT → dSFT；<br>2. Aligned the model with TRL's DPOTrainer on the **UltraFeedback dataset**, which contains 64k prompts and model completions that are ranked by GPT-4. → AI feedback for dDPO | Based on **mistralai/Mistral-7B-v0.1**<br>dSFT+ **dDPO**（distilled Direct Preference Optimization）<br>Reported Acc of MMLU: 61.44%<br>Highest Acc in RSN: 51.46% | 1. full的结果是最高的；对于non expert来说，如果α取到5倍的话是最好的 (>full)；对于expert来说，α=4会比较好，但是低于full<br>2. 用mistral-it修正的也有效果，但是不如full |
| **OpenHermes-7B** | 11.88%	83.06%<br>47.79%	26.10%<br>—————————<br>50.60%	19.55%<br>59.14%	2.73% | **Teknium/OpenHermes-2.5-Mistral-7B**<br>Represents a continuation of the OpenHermes 2 series, with additional **code-heavy instruction datasets** (~7–14% of training data) added during tuning.<br>Trained on a curated corpus of **~1 million dialogue entries, primarily GPT-4–generated**, supplemented by high-quality open-source datasets.<br>Data transformation pipeline: public sources → filtered → ShareGPT format → ChatML via Axolotl | Based on **mistralai/Mistral-7B-v0.1**<br>Only SFT  | **结果理想**，mid > full，如果α=5会更好； |
| **NousHermes-7B** | 13.94%	80.45%<br>48.33%	25.53%<br>—————————<br>50.00%	21.23%<br>58.86%	3.84% | **NousResearch/Nous-Hermes-2-Mistral-7B-DPO**<br>Loaded from *OpenHermes‑2.5,* no additional large-scale SFT; focus on preference alignment.   | Based on *Teknium/OpenHermes-2.5-Mistral-7B*<br>Only **DPO** | **结果理想**，mid > full，如果α=5会更好；<br>结果与 OpenHermes-7B 基本一致，RSN 向量迁移兼容性极好 |
| **Qwen2.5-7B-base** | 57.02%	18.25%<br>67.24%	0.81%<br>—————————<br>66.19%	4.83%<br>68.10%	0.27% | **Qwen/Qwen2.5-7B**<br>Pretrained on **18T** tokens; includes **synthetic math/code/knowledge** generated by **Qwen2-72B-Instruct** and **Qwen2Math-72B-Instruct**; improved filtering and mixture. | No SFT and No RLHF<br>Reported Acc of MMLU: 74.16%<br>Highest Acc in RSN: 68.30% | 1. Base model本身的表现很好，E%不是很高；且可以通过full和mid都达到很好的修正效果，且mid接近甚至略好于full (non expert)<br>2. 但是如果从role effect来看，差异算是比较小；<br>3. **Qwen系类在计算logits的时候，base的表现都更好** |
| **Qwen2.5-7B** | 39.50%	50.57%<br>66.18%	9.41%<br>—————————<br>63.54%	14.16%<br>69.89%	2.10% | **Qwen/Qwen2.5-7B-Instruct**<br>SFT: Over 1 million high-quality examples across various tasks. Robust System Instruction in post-training improves robustness to diverse system prompts (role-play & condition setting.<br>DPO: Around 150k response pairs were created from SFT-generated outputs using execution feedback and answer matching, with both **human and automatic review**, especially for tasks like math, code, and reasoning.<br>GRPO: The data comes from both **open-source queries** and a set of more complex, high-quality **proprietary queries**. | SFT → DPO → Group Relative Policy Optimization（GRPO） | 1. **mdf effect: mid > full**<br>2. 和Base相比，主要non expert的E↑ Acc↓；Expert的上限和Base其实差不多<br>3. IT和Base的middle layer看起来是不一样的，而llama是差一层，mistral则各个模型完全一致；<br>4. 另外比较值得注意的地方是，其他model generation和logits的表现都差不多；但是qwen的non expert却差异非常大（53.27, 27.56-gen/39.50, 50.57-log）；可能是在generation的时候有一些生成的答案，在logits的时候被定义为了E |
| **Qwen2.5-3B-base** | 52.23%	7.02%<br>54.02%	5.34%<br>—————————<br>55.23%	4.28%<br>55.49%	3.86% | **Qwen/Qwen2.5-3B** | No SFT No post training<br>Reported Acc of MMLU:<br>Highest Acc in RSN:  | 1. 和7B同样的特征是E%很低；<br>2. 修改虽然有一些效果，但是本身空间有限 |
| **Qwen2.5-3B** | 25.95%	64.95%<br>45.30%	34.27%<br>—————————<br>38.50%	46.06%<br>53.51%	20.42% | **Qwen/Qwen2.5-3B-Instruct** | SFT → DPO → Group Relative Policy Optimization（GRPO）<br>Reported Acc of MMLU:<br>Highest Acc in RSN:  | 1. **编辑增益明显，mid > full**<br>2. 现象和7B类似，都是IT选择E更多；IT有增加Role effect；<br>3.如果继续增加α，这里的表现可以 > Base，看来这里整理出来的middle layer的强度比较小； |
| **Qwen3-8B-base** | 63.11%	13.05%<br>66.85%	7.37%<br>—————————<br>67.21%	7.00%<br>69.32%	4.14% | **Qwen/Qwen3-8B-Base**<br>Base model<br>The pre-training of Qwen3 uses a massive **36 trillion-token** dataset covering 119 languages and diverse domains such as STEM, coding, reasoning, books, and multilingual texts, constructed through PDF extraction, synthetic generation from domain-**specific Qwen2.5 models**, and enriched with fine-grained annotations for educational value, domain, and safety. | No SFT<br>No post training<br>Reported Acc of MMLU: 79.66% | 1. Mid ≈ full；<br>2. 和Qwen2.5系列类似，Base模型的E%反而更低； |
| **Qwen3-8B** | 52.51%	29.88%<br>62.44%	17.50%<br>—————————<br>63.38%	14.26%<br>67.73%	7.78% | **Qwen/Qwen3-8B**<br>The post-training of Qwen3 uses carefully curated multi-task datasets, including math and reasoning problems (for cold start and reasoning RL), thinking and non-thinking mode data (for fusion fine-tuning), and a diverse reward dataset covering over 20 tasks (for general capability reinforcement). | Long-CoT Cold Start (SFT) → Reasoning RL (GRPO) → Thinking Mode Fusion → General RL (GRPO)<br>+ Strong-to-Weak Distillation<br>* thinking must be used in the chat template, which is not included in  RSN experiments. | 1. Mid > full；<br>2. 整体上和Qwen2.5系列类似，non expert的E%↑使role之间的差异更明显；<br>3. 当前的最好的performance没有Base model高；<br>4. 如果继续增加α会出现Acc↓ E↑的状况；可以理解为已经到临界值 →**不确定其他model会不会出现这种现象？有的model似乎只是下降** |
| **Phi4mini-4B** | 57.50%	13.15%<br>62.75%	3.92%<br>—————————<br>60.09%	6.39%<br>62.88%	0.63%	 | **microsoft/Phi-4-mini-instruct<br>Training data:** Textbook-quality data, focusing on reasoning, mathematics, and code tasks.<br>SFT: It uses supervised fine-tuning on synthetic, **instruct-style datasets** built from high-quality educational content, **without any human preference data or reinforcement learning.** | SFT only | Phi3.5和phi4都是差异很小且无法通过RSN提升；<br>phi系列是完全没有人类偏好数据的model，也没有Roleplaying |
| **Deepseek-7B-base** | 27.98% 0.00%<br>29.30% 0.00% | **deepseek-ai/deepseek-llm-7b-base**<br>Trained on 2 trillion English + Chinese tokens using GQA/SwiGLU/RoPE/RMSNorm | No SFT | 完全不行 |
| **Deepseek-7B-chat** | 44.29% 10.31%<br>46.37% 4.41% | **deepseek-ai/deepseek-llm-7b-chat**<br>Fine-tuned on instruction-following chat data (referred to as “extra instruction data”). | **SFT + DPO** | 几乎不行 |
| **DeepseekRqwen-7B** | 6.07% 87.35%<br>16.25% 66.08% | **deepseek-ai/DeepSeek-R1-Distill-Qwen-7B**<br>蒸馏数据量约为 800k 样本，对 Reasoning、Math、Code 任务增强效果明显 | Based on Qwen‑2.5‑Math‑7B<br>使用 DeepSeek‑R1 生成的CoT数据作为蒸馏来源；没有预先进行SFT，直接对基础模型进行 RL‑增强蒸馏训练  | 几乎不行<br>有可能是因为reasoning model需要推理过程才能生出答案 |
| **Falcon3-7B** | 61.68%. 4.55%<br>62.32%. 2.84% | **tiiuae/Falcon3-7B-Instruct<br>Pretrained** on 14T tokens of datasets comprising of web, code, STEM, high quality and mutlilingual data<br>**Postrained** on 1.2 million samples of STEM, conversations, code, safety and function call data | SFT only | 几乎不行 |
| **Falcon3-10B** | 46.27%. 19.36%<br>47.17%.  15.43% | **tiiuae/Falcon3-10B-Instruct**<br>An enhanced version of the 7B model, created by increasing its depth through layer duplication and continuing pretraining with **2 trillion** high-quality tokens. This resulted in Falcon3-10B-Base. | SFT only | 几乎不行<br>但是10B会回答E |
| **Gemma-7B** | 49.94%  4.40%<br>50.56%  3.04% | **google/gemma-7b-it**<br>Pretraining: Web Documents, Code, Mathematics(6T) | **SFT + RLHF** | 几乎不行 |
| **stablelm-12B-Base** | 0.15%. 99.83%<br>1.83%. 97.85% | **stabilityai/stablelm-2-12b**<br>Architecture based on GPT-NeoX<br>**Dataset:** 2 trillion tokens, open-source datasets including Falcon RefinedWeb, RedPajama-Data (excluding Books3), The Pile (excluding Books3), StarCoder, and multilingual data from CulturaX (OSCAR corpora) |  | 完全拒答 |
| **stablelm-12B-chat** | 29.71%. 49.00%<br>46.55%. 15.52% | **stabilityai/stablelm-2-12b-chat<br>SFT:** UltraChat, WizardLM, SlimOrca, **ShareGPT**, Capybara, Deita, and MetaMathQA.<br>DPO:  UltraFeedback and Intel Orca Pairs. | DPO |  |
| **LLaDA** |  | **GSAI-ML/LLaDA-8B-Instruct** | Pre-train → SFT； |  |
| **LLaDA1.5** |  | **GSAI-ML/LLaDA-1.5** | Pre-train → SFT → VRPO |  |
| **Dream** |  | **Dream-org/Dream-v0-Instruct-7B** | SFT only |  |

### 1.  “RSN 被激活” 是什么？

模型成功激活了角色敏感子空间（Role-Sensitive Neurons, RSN），需同时满足以下两个条件：

1. 编辑后性能提升，且mid-layer注入效果优于或接近full-layer，即：mid >≈ full > original
    - 表明角色相关信息在中层已经被编码，可通过稀疏干预放大，不依赖深层输出层干扰。
2. 非专家样例在编辑后趋近于专家输出，即 edited NonExpert ≈ Expert
    - 包括 Accuracy 提升和 E-rate（拒答率）下降；
    - 表明 RSN 向量能有效调节角色风格和行为。

### 2. 将模型按训练信号分四象限

|  | Roleplaying data ❌ | Roleplaying data ✔ |
| --- | --- | --- |
| Human feedback ❌ | **Llama3-8B-base ③ 15T<br>Llama3-3B-base ③ 9T<br>Mistral-7B-base ③ 8T<br>Phi4mini-4B ④ 5T<br>Deepseek-7B-base ④ 2T<br>DeepseekRqwen-7B ④<br>Falcon3-7B ④ 14T<br>Falcon3-10B ④ 14T+2T<br>stablelm-12B-Base ④ 2T**<br>**Qwen2.5-7B-base ② 18T<br>Qwen2.5-3B-base ② 18T<br>Qwen3-8B-base  ② 36T**  | **Hermes-3B ③<br>Mistral-7B ① ～<br>OpenHermes-7B ①** 0.433<br>**** |
| Human feedback ✔ | **Gemma-7B ④ 6T**<br>**Deepseek-7B-chat ④**<br>**Llama3-3B ③** 0.175<br>**Llama3-8B ①** 0.393<br>**Qwen2.5-7B ①** 0.371<br>**Qwen2.5-3B ①** 0.153<br>**Qwen3-8B ①** 0.223<br>**** | **Hermes-8B ②<br>Openchat-7B ②**<br>**Zephyr-7B ②**<br>**stablelm-12B-chat ②**<br>**NousHermes-7B ①**  |

**①完美符合；
②几乎符合：**mid 接近 full
**③有差异但是无法通过mid layer来修正；
④没有Role effect**

备注

- ～ Mistral: Publicly available chat and instruction datasets (e.g., from Hugging Face);
**May include** roleplaying datasets such as UltraChat、ShareGPT、OpenAssistant
- - Qwen: 大量training data (18T & 36T)，在Qwen2.5的技术报告中有关于pretraining data部分提到 “More resilient to the diversity of system prompts, enhancing **role-play implementation** and condition-setting for chatbots.”
- 标记数字为 *Max-Normalized Gain*

### Information about Mistral

社区与官方资料都 **没有公开列出完整、逐个可下载的 “Mistral-7B 预训练或 SFT 数据集”**。目前仅能确认与推断的要点如下：

| **阶段** | **已公开信息** | **社区推测 / 旁证** | **可靠程度** |
| --- | --- | --- | --- |
| **预训练** | 官方博客称 “**完全基于公开网页数据**，经过严格去重与质量过滤；数据多语种，含代码” | 共同创始人在 X (推特) 提到 **规模≈8 T tokens**，含英语、法语、代码 等 | ★★☆☆☆（未正式论文或卡片确认） |
| **SFT / Instruct** | model card 仅写 “**variety of publicly available conversation datasets on Hugging Face**”，无具体列表 | 社区复现常用 **UltraChat、OpenOrca、ShareGPT、OpenAssistant** 等开源对话集合作 SFT；Zephyr、Conifer 等项目的微调流程也用这些集合并能复现效果 | ★★☆☆☆（间接证明，可操作但非官方） |
| **HF / RL 对齐** | 官方表示 **Mistral-7B-Instruct** “只是一个快速演示（quick demonstration），未使用任何专有偏好数据，也无额外 RLHF” | 社区普遍认为目前版本确无 DPO/RLHF；需要自行引入 UltraFeedback 之类数据才能做偏好对齐 | ★★★☆☆ |
- **预训练数据**：官方只给出 “公开网页 + 去重 + 多语种 + 代码” 的笼统描述；文件级明细与确切 token 数未公开。
- **SFT 数据**：官方称来自 “Hugging Face 公共指令数据集” ，但未列清单。社区能成功复现的做法是使用 UltraChat / ShareGPT / OpenOrca 等开放合集。

### Model catagories

- **The "Role-Agnostic" (无感区):**
    - *特征:* 缺乏 Role Data，Mid/Full 编辑都无效或微效。
    - *模型:* DeepSeek-Base, Falcon, Phi-4.
    - *原因:* 训练数据中严重缺乏 Role-playing 语料，或者架构/训练目标完全不支持多角色模拟。
- **The "Superficial Role-Player" (表面派):**
    - *特征:* Base 模型，或者无 Role Data 的 SFT 模型。
    - *表现:* Full Layer 编辑有效，Mid Layer 无效。
    - *原因:* 角色只是 Prompt 里的文字，没有内化为中间层的表征。
- **The "Deep Role-Player" (内化派):**
    - *特征:* 经过 Role Data SFT 的模型 (Hermes, OpenChat, Mistral-IT)。
    - *表现:* Mid Layer 编辑效果显著，甚至优于 Full。
    - *原因:* 角色已成为一种深层语义特征，可以通过稀疏干预被激活。

## Statistic

- **Acc Gain**

```bash
=== Point-biserial correlation (acc_gain ~ **RolePlaying**) ===
r_pb (RP) = 0.708, p = 0.000036

=== Point-biserial correlation (acc_gain ~ HumanFeedback) ===
r_pb (HF) = 0.307, p = 0.119252

=== Spearman correlation (robust, rank-based) ===
Spearman rho (RP) = 0.688, p = 0.000074
Spearman rho (HF) = 0.306, p = 0.120189

=== Welch t-test: acc_gain by RolePlaying (True vs False) ===
mean(True)=0.296, mean(False)=0.098, t=4.799, p=0.000428

=== Welch t-test: acc_gain by HumanFeedback (True vs False) ===
mean(True)=0.201, mean(False)=0.122, t=1.596, p=0.124371

Cohen's d (RP True - False): 2.114
Cohen's d (HF True - False): 0.625

=== OLS: acc_gain ~ RP + HF + RP:HF ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               acc_gain   R-squared:                       0.522
Model:                            OLS   Adj. R-squared:                  0.460
Method:                 Least Squares   F-statistic:                     8.386
Date:                Wed, 20 Aug 2025   Prob (F-statistic):           0.000603
Time:                        16:39:03   Log-Likelihood:                 27.299
No. Observations:                  27   AIC:                            -46.60
Df Residuals:                      23   BIC:                            -41.42
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.0850      0.028      3.087      0.005       0.028       0.142
RP             0.1843      0.062      2.994      0.006       0.057       0.312
HF             0.0361      0.045      0.797      0.434      -0.058       0.130
RP:HF          0.0063      0.083      0.076      0.940      -0.166       0.178
==============================================================================
Omnibus:                        1.699   Durbin-Watson:                   1.430
Prob(Omnibus):                  0.428   Jarque-Bera (JB):                1.266
Skew:                           0.307   Prob(JB):                        0.531
Kurtosis:                       2.135   Cond. No.                         6.58
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified
```

- 单变量相关性
    - RolePlaying (RP)
        - Point-biserial: r = 0.708, p < 0.001 → 强正相关且高度显著。
        - Spearman rho: ρ = 0.688, p < 0.001 → 排名相关性也强且稳健。
        
        说明 有无 **roleplaying data 和 acc gain 之间有显著关系**，有 role data 的模型在 role effect 指标上明显更强。
        
    - Human Feedback (HF)
        - Point-biserial: r = 0.307, p ≈ 0.12 → 正相关但不显著。
        - Spearman rho: ρ = 0.306, p ≈ 0.12 → 同样结果。
        
        人类反馈的作用趋势是正的，但统计上不显著，说明 **HF 单独并不能很好解释 acc gain**。
        
- **均值比较 (Welch t-test)**
    - RP
        - mean(True)=0.296 vs mean(False)=0.098
        - t = 4.799, p < 0.001
        - Cohen’s d = 2.114 (极大效应量)
        
        带有 roleplaying data 的模型在 acc gain 上远超没有的，效应很强。
        
    - HF
        - mean(True)=0.201 vs mean(False)=0.122
        - t = 1.596, p ≈ 0.124
        - Cohen’s d = 0.625 (中等效应量，但不显著)
        
        HF 组均值稍高，但差异没有达到显著水平。
        
- **多变量回归 (acc_gain ~ RP + HF + RP:HF)**
    - R² = 0.522 → 模型解释了约一半的 acc_gain 方差。
    - RP 系数 = 0.1843, p = 0.006 → roleplaying data 对 acc_gain 有显著正贡献。
    - HF 系数 = 0.0361, p = 0.434 → 人类反馈不显著。
    - RP×HF 交互项 = 0.0063, p = 0.94 → 没有交互作用。
    - 回归结果和前面的统计一致：
        - roleplaying data 是主要决定因素，解释力显著。
        - human feedback 单独不显著，也不会增强 roleplaying 的作用。
- **综合解读**
    - 结果强烈支持 Hypothesis2: “Explicit roleplaying data 激活了 role-sensitive neurons (RSNs)”。
    - Human feedback 并不足以激活 RSN（至少在当前样本下），更多是对其他维度（安全性、拒答、风格）的优化。
    - 最佳解释：Roleplaying data 是主要驱动力；HF 是次要或条件性因素。
