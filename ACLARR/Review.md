**Meta Review of Submission930 by Area Chair xNJG**

Meta Reviewby Area Chair xNJG26 Mar 2026, 19:45 (modified: 07 Apr 2026, 00:40)Area Chairs, Authors, Reviewers, Program Chairs[Revisions](https://openreview.net/revisions?id=oyjklMTXli)

**Metareview:**

I am following the AC's recommendation, since rebuttal and planned revisions address the reviewers' main concerns.

**Confidence:** 5: The area chair is absolutely certain

**Rating:** 6: Marginally above acceptance threshold

**Recommendation:** Findings

**Presentation Mode:** Poster
---
**Metareview:**

The paper studies the Role-Sensitive neurons (**RSN**) that govern the LLM willingness to answer. The authors show that role prompts (e.g. "you are an expert...") mostly change how readily the model answers, without affecting the underlying conditional accuracy.

Two research questions are tackled:

- How to extract such **RSNs** from a pre-trained model (steering vector)
- How they could be used to steer the model towards answering following a given role (which does not change the accuracy, but the abstention rate).

The authors show that only 0.5% of neurones drive the LLM confidence – hinting to the fact that they really control this specific behavior (qGrh).

he task setting is limited to QA tasks, while the validation and performance of RSNs on more general domains like reasoning tasks, including math reasoning or coding, remain underexplored (qGrh, HxKE). The authors provided new experimental results on GSM8K, showing that their results extend over QA.

(qGrh, HxKE) unclear on how RSN are identified with high confidence: suggests to include a pseudo-code or algorithm. The authors state in their answer they will do.

**Summary Of Reasons To Publish:**

Overall, the paper shows that "confidence" (even if this concept is not clearly defined for an LLM) can be controlled by a small subset of neurons (that generalize over datasets). This is one more evidence about the fact that many behavorial phenomena are controlled by some neurons within the LLM – while not presenting new methodology, this work contributes fairly to the field.

**Summary Of Suggested Revisions:**

The different clarification requested by the reviewers need to be adressed – in particular clarfiying the methodology and processing (pseudo-code). Including the experiments on GMS8K is also a requirement.

**Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

**Reported Issues:** No

**Publication Ethics Policy Compliance:** I did not use any generative AI tools for this review
---
**Official Review of Submission4767 by Reviewer qGrh**

Official Reviewby Reviewer qGrh09 Feb 2026, 21:40 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer qGrh, Commitment Readers[Revisions](https://openreview.net/revisions?id=CjPyl9P2pT)

**Paper Summary:**

This paper discovers and analyzes the neurons that control the output confidence in LLMs. It identifies the role-sensitive neurons (RSN) by comparing the probability that LLMs hesitate to reply on the MMLU benchmark given different prompting roles, pointing out the phenomenon of confidence-performance decoupling. Afterwards, it details how to extract such RSNs from a pre-trained model, and how they could be used to guide the model’s responses.

**Summary Of Strengths:**

1. Unlike the common practice in activation steering, this paper is based on the neuron-level. Besides, the ratio of neurons that constitute RSNs is relatively low, with only 0.5% neurons; the output confidence could be sharply steered.
2. The experiments are based on various backbone models, validating the generalizability of the effects of RSNs.
3. The RSNs could serve as a bidirectional switch, enhancing the model’s confidence with a positive coefficient while weakening it with a negative coefficient.

**Summary Of Weaknesses:**

1. The task setting is limited to QA tasks, while the validation and performance of RSNs on more general domains like reasoning tasks, including math reasoning or coding, remain underexplored.
2. It lacks details about the concrete resources and computation required to identify enough RSNs with a high confidence.
3. The current analyses are mostly based on the final tokens' probabilities, with a neglect of the intermediate reasoning steps, which may provide more fine-grained guidance on how to steer LLMs to the desired behavior, like admitting the unfamiliarity after some specious reasoning.

**Comments Suggestions And Typos:**

The interaction between the model’s confidence and performance could be formulated as an empirical rule, with more experiments on different model scales, task domains, and RSN selection criteria, etc.

**Confidence:** 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.

**Soundness:** 3.5

**Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.

**Overall Assessment:** 3.5 = Borderline Conference

**Ethical Concerns:**

There are no concerns with this submission

**Reproducibility:** 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.

**Datasets:** 1 = No usable datasets submitted.

**Software:** 1 = No usable software released.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

**Publication Ethics Policy Compliance:** I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

RESPONSE
**New GSM8K Open-Ended Experiments, Computational Efficiency, and Reasoning Propagation**

Official Commentby Authors ([Peiwen Huang](https://openreview.net/profile?id=~Peiwen_Huang2), [Tzu-Hung Huang](https://openreview.net/profile?id=~Tzu-Hung_Huang1), [Shou-De Lin](https://openreview.net/profile?id=~Shou-De_Lin1), [Chih-Hao Hsu](https://openreview.net/profile?id=~Chih-Hao_Hsu2))20 Feb 2026, 17:58 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer qGrh, Commitment Readers[Revisions](https://openreview.net/revisions?id=P0zZsXVaUW)

**Comment:**

We appreciate your constructive feedback and address your specific concerns below.

### **W1: Task Coverage Beyond QA (New GSM8K Experiments)**

We appreciate this suggestion. To address this, we first clarify our existing coverage and then present new experiments on open-ended reasoning.

**Existing Benchmarks.** Table 5 already includes reasoning-intensive benchmarks evaluated in a neutral setting: **MMLU-Pro**, **GPQA**, **AR-LSAT**, and **LogiQA**. Our focus on multiple-choice (MC) format was methodological: single-token probabilities provide a stable proxy for internal confidence (Plaut et al., 2024) , enabling precise measurement of confidence-performance decoupling.

**New Experiments: GSM8K.** To verify if RSNs influence open-ended reasoning, we conducted **zero-shot** experiments on GSM8K without explicit CoT prompts. We apply the RSN intervention **solely at the final prompt token** during prefill, then allow the model to generate the full reasoning chain freely. We analyzed 300 randomly sampled problems (strictly paired across conditions), focusing on the **Confidence Ratio** of the generated text.

**Table: Confidence & Representative Linguistic Markers on GSM8K**

| Model | Metric |  | Baseline |  |
| --- | --- | --- | --- | --- |
| **Llama3-8B-IT** | Confidence Ratio | 0.91 | 1.08 | **1.37** |
|  | Example (Assertive): "the answer is” | 106 | 148 | 200 |
| **Qwen3-8B-IT** | Confidence Ratio | 0.27 | 0.31 | **0.33** |
|  | Example (Self-correction): "wait” | 959 | 827 | 810 |

**Linguistic Marker Analysis.** The results demonstrate that RSNs successfully set a global "confidence state" that persists throughout the entire reasoning trajectory. We analyzed markers in the generated text: **Assertive** (e.g., "the answer is", "therefore"), **Hedging** (e.g., "maybe"), and **Self-correction** (e.g., "wait").

As shown in the Table, both models exhibit a strict monotonic trend (). While the Confidence Ratio incorporates our full dictionary of linguistic markers, specific examples like Llama3's assertive phrase count () clearly illustrate the systematic gain shift. This proves that a single-point RSN injection during prefill acts as a mechanistic "gain knob," regulating the model's decisiveness across the downstream chain.

### **W2: Computational Resources**

We appreciate this point and will add a dedicated paragraph in the revision. The RSN extraction pipeline is computationally lightweight:

- **Data:** We scan ~14k MMLU questions to find ~4,900 divergent pairs. No training labels or optimization is required.
- **Compute:** Requires scanning the dataset via **~28,000 forward passes** (2 roles 14k). For Llama3-8B, this takes **2–3 hours on a single A100**. There is **no gradient computation**, backpropagation, or iterative optimization.
- **Storage:** The resulting RSN vectors comprise only ~180 scalar values (20 neurons 9 layers for Llama3), occupying negligible memory.

### **W3: Intermediate Reasoning Steps**

We address it at two levels:

1. **Decision Bottleneck:** In MC, the final token is the architectural bottleneck where context maps to a decision, making it the precise locus for measuring "willingness to act."
2. **Propagation (New Evidence):** Our GSM8K results (W1) directly address this. Even with a single prefill intervention, the systematic shift in linguistic markers (e.g., Llama3 "answer is" counts: ) proves that RSNs set a global "confidence state" that governs the **entire downstream reasoning chain**.

**Future Directions:** We agree that tracking dynamic activation changes token-by-token is an exciting frontier and will highlight this in Section 8.

### **Comment: Empirical Scaling Rules**

Your suggestion to formulate the interaction between confidence and performance as an empirical rule is excellent and aligns with our long-term vision. Our current results already hint at such a rule: the optimal  depends on the model's baseline confidence regime (over-confident models benefit from negative steering; under-confident models benefit from positive steering). Formalizing this as a function of model scale, task difficulty, and baseline calibration is a compelling direction for future work, and we will discuss this explicitly in the revision.

We are committed to incorporating the GSM8K experiments and computational details in the final paper.
---
**Official Review of Submission4767 by Reviewer HxKE**

Official Reviewby Reviewer HxKE07 Feb 2026, 22:23 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer HxKE, Commitment Readers[Revisions](https://openreview.net/revisions?id=sE7FOPhX5E)

**Paper Summary:**

This paper asks why role prompts like “you are an expert” change LLM behavior, and argues the main effect is not better knowledge but a shift in willingness to answer: “expert” prompting reduces abstention and raises apparent confidence, while accuracy on answered questions changes much less. To explain this, the authors identify a small set of role-sensitive neurons (≈0.5%), mostly in mid layers, whose activations reliably differ across expert vs. non-expert settings; intervening on these neurons during inference can push the model toward being more decisive or more hesitant, effectively acting as a gain control for abstention. They also suggest this circuitry is largely present from pretraining, with instruction tuning mainly sharpening the role signal, and caution that making the model more decisive can also amplify unwarranted certainty when it lacks knowledge.

**Summary Of Strengths:**

- The finding provides a useful empirical message: Role prompts mostly change how readily the model answers, not the underlying conditional accuracy. It is important for how we interpret “expert” prompting in evaluations and applications.
- The paper identifies a small, sparse set of role-sensitive neurons concentrated in mid layers and links them to role-conditioned behavior.
- Causal evidence + practical control: Direct neuron-level interventions provide bidirectional control over hesitation/decisiveness, offering a concrete handle for abstention calibration.

**Summary Of Weaknesses:**

- Task coverage is narrow. Most experiments are on MMLU/ARC/CSQA-style QA benchmarks; it’s less convincing that the conclusions transfer to open-ended generation or interactive settings where uncertainty is expressed differently.
- Impact/novelty may be modest. The headline finding, role prompts mainly shift confidence/abstention, has been discussed before, and the neuron/activation steering angle may read as an incremental mechanistic case study rather than a new method with broad reach.

**Comments Suggestions And Typos:**

Suggestion: Make RSN extraction easier to reproduce: Include a short pseudocode/algorithm box for how you build divergent pairs, select top neurons, construct the steering vector, and choose alpha.

No typos are found

**Confidence:** 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.

**Soundness:** 2.5

**Excitement:** 2.5

**Overall Assessment:** 2.5 = Borderline Findings

**Ethical Concerns:**

There are no concerns with this submission

**Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.

**Datasets:** 3 = Potentially useful: Someone might find the new datasets useful for their work.

**Software:** 3 = Potentially useful: Someone might find the new software useful for their work.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

**Publication Ethics Policy Compliance:** I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

RESPONSE
**Extending to Open-Ended Generation (GSM8K) and Clarifying Low-Level Mechanistic Impact**

Official Commentby Authors ([Peiwen Huang](https://openreview.net/profile?id=~Peiwen_Huang2), [Tzu-Hung Huang](https://openreview.net/profile?id=~Tzu-Hung_Huang1), [Shou-De Lin](https://openreview.net/profile?id=~Shou-De_Lin1), [Chih-Hao Hsu](https://openreview.net/profile?id=~Chih-Hao_Hsu2))20 Feb 2026, 18:13 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer HxKE, Commitment Readers[Revisions](https://openreview.net/revisions?id=ePjQgrB1of)

**Comment:**

We sincerely thank you for recognizing the empirical value of our neuron-level analysis. We address your specific concerns below.

### **W1: Task Coverage & Open-Ended Generation**

To demonstrate RSN effects beyond multiple-choice formats, we conducted new **zero-shot** experiments on **GSM8K** (open-ended math) without explicit Chain-of-Thought (CoT) prompting.

**Experimental Setup.** Consistent with our multiple-choice (MC) experiments, we apply the RSN intervention **solely at the final prompt token** during prefill, allowing free generation of the reasoning chain. We evaluate 300 paired, randomly sampled problems across conditions. Our analysis focuses on the **Confidence Ratio** derived from linguistic markers.

**Table: Confidence Ratio & Representative Linguistic Markers on GSM8K**

| Model | Metric |  | Baseline |  |
| --- | --- | --- | --- | --- |
| **Llama3-8B-IT** | Confidence Ratio | 0.91 | 1.08 | **1.37** |
|  | Example (Assertive): "the answer is” | 106 | 148 | 200 |
| **Qwen3-8B-IT** | Confidence Ratio | 0.27 | 0.31 | **0.33** |
|  | Example (Self-correction): "wait” | 959 | 827 | 810 |

**Linguistic Marker Analysis.** We analyzed categories of markers: **Assertive** (e.g., "the answer is," "therefore"), **Hedging** (e.g., "maybe"), and **Self-correction** (e.g., "wait"). As shown in the table, both models exhibit a **strict monotonic trend** (). For instance, Llama3's assertive phrases increase systematically (). This confirms a single prefill intervention sets a global "confidence state" governing the entire downstream chain.

We note that interactive, multi-turn settings remain an important future direction; however, since our intervention modulates the model's internal belief state (Section 5.4), we expect this modulation to propagate to any output modality.

### **W2: Novelty and Mechanistic Impact**

While we acknowledge prior behavioral observations (e.g., Xu et al., 2025), our neural-level steering discovery is far from an "incremental" case study. It represents a critical breakthrough transitioning the field from black-box observation to white-box control.

**From Behavioral Observation to Low-Level Control.** Observing that "role prompts shift confidence" is merely behavioral. Just as medical intervention requires targeting specific neural circuits, we achieve precise **low-level control**. By isolating the exact neural substrate (~0.5% sparse sub-network) and enabling bidirectional causal intervention, we directly manipulate the model's mechanistic "gain knob," enabling three major theoretical contributions:

1. **First Experimental Verification:** While prior works merely hypothesized that role-playing functions as a confidence bias without altering knowledge, we are the **first to experimentally verify** this mechanism at the physical layer. We prove that this behavior is driven by a distinct, prompt-agnostic neural substrate.
2. **The Necessity of Decoupled Evaluation:** We prove that standard evaluation metrics conflate true capability with decisiveness. By establishing the right criteria (isolating Conditional Accuracy via explicit abstention), we demonstrate that LLMs often possess correct latent knowledge but suppress it due to a low-gain state. RSNs explicitly disentangle the "willingness to act" from epistemic truth.
3. **Physical Evidence for Alignment Theory:** As connected in our final chapter (Section 7), our low-level findings directly substantiate high-level alignment theories. Our cross-model transfer experiments show that IT-extracted RSN vectors successfully steer Base models (e.g., Llama3-Base accuracy leaps from 20.67% to 47.47%). This provides physical, neuron-level evidence for the Superficial Alignment Hypothesis: instruction tuning acts primarily as a "signal sharpener" for pre-existing gain control circuitry, rather than creating the capability de novo.

**Practical Utility.** Beyond theory, RSNs offer concrete tools: (a) **Prompt-free abstention control** without fragile prompt engineering; (b) **Safety monitoring** for "unwarranted certainty" before hallucinations (Appendix A.13).

### **Suggestion: Algorithm Box for Reproducibility**

We fully agree with this excellent suggestion. In the revision, we will include a formal **Algorithm 1 (pseudocode)** in the main text to ensure reproducibility. This algorithm explicitly formalizes the four-step pipeline:

1. **Divergent Pair Collection:** Identifying input pairs that yield conflicting answers under Expert/Non-Expert roles.
2. **Mean Activation Shift:** Computing the activation difference vector at the final prompt token.
3. **Sparse Neuron Selection:** Applying hard thresholding (Top-%) to isolate RSNs.
4. **Inference-Time Intervention:** Adding the scaled RSN vector solely at the prefill stage.
---
**Official Review of Submission4767 by Reviewer 3ErX**

Official Reviewby Reviewer 3ErX03 Feb 2026, 17:20 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 3ErX, Commitment Readers[Revisions](https://openreview.net/revisions?id=9Qsrf4su3i)

**Paper Summary:**

The authors claim that role prompting alters the model’s confidence without affecting knowledge representation or retrieval. They identify a set of neurons responsible for modulating the model’s decisiveness and show that these neurons can be steered to control confidence. Finally, they show that steering vectors extracted from instruction-tuned (IT) models are compatible with the corresponding base models, implying that instruction tuning does not alter the mechanism underlying confidence.

**Summary Of Strengths:**

- The overall direction and the question undertaken are interesting and important.

**Summary Of Weaknesses:**

The main weakness of this paper is a lack of logical coherence in several places.

### **Presentation**

- I did not fully understand “Observation 2.”
    - Is MMLU-E an established benchmark? How are conditional accuracy, abstention, and the E-ratio computed?
- In Section 4.3, the authors refer to improved decisiveness, but Figure 3a reports accuracy. They later mention that increasing degrades performance, but I do not see this in the plots. Also, what is the “role-induced divergence” discussed in “Localization (Layers)” and why is correlation used to demonstrate it? I did not understand this subsection.

### **Methodology**

- The MSP metric appears to be misused. The authors of the original paper ([1]) note that “the prediction probability from a softmax distribution has a poor direct correspondence to confidence”. After extracting MSP, they fit a classifier, but the authors of this paper do not go that far.
- Why does Figure 2 suggest that knowledge retrieval abilities are unchanged? What explains the accuracy gap between “Non-Expert” and “Expert”? I am especially curious because you use only these two roles in most of the other experiments.
- In Section 4.1, the authors write: “dense vectors inevitably mix confidence signals with semantic content.” Why? Why do you assume neuron-level editing will yield a more fine-grained effect? What about superposition, as discussed extensively in Anthropic’s work? Why can there not be a “semantic” content vector represented as a combination of signals across many neurons, and a “confidence” vector orthogonal to it?
    - Section 4.3 concludes by showing that confidence is encoded in a low-dimensional region. Can you show that this region is not lower-dimensional than the set of neurons you selected? (For example, consider a single direction in a 2D space in which the data are distributed.)
- Section 4.2, Step 1: Why is it necessary to collect divergent pairs where the answer changes? If accuracy is decoupled from confidence, as you claim, then whether pairs diverge should not matter for controlling decisiveness.
- Section 5.1: I do not think I understand the authors’ notions of accuracy, confidence, and decoupling. Given what was written earlier, the chain “This intervention recovers the model’s accuracy” → “sufficient causal substrate for confidence modulation” is incoherent.

[1] Hendrycks et al. “A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks”

**Comments Suggestions And Typos:**

See the “Weaknesses” section. I did not finish reading the paper due to a set of confusions that accumulated by the middle of it. I would be happy to reconsider my score after the authors’ response and clarifications.

**Confidence:** 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.

**Soundness:** 2 = Poor: Some of the main claims are not sufficiently supported. There are major technical/methodological problems.

**Excitement:** 2 = Potentially Interesting: this paper does not resonate with me, but it might with others in the *ACL community.

**Overall Assessment:** 2.5 = Borderline Findings

**Ethical Concerns:**

There are no concerns with this submission

**Needs Ethics Review:** No

**Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.

**Datasets:** 1 = No usable datasets submitted.

**Software:** 1 = No usable software released.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.

**Publication Ethics Policy Compliance:** I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits

RESPONSE
**Clarifying the Decoupling Logic, Evaluation Metrics, and Methodological Choices**

Official Commentby Authors ([Peiwen Huang](https://openreview.net/profile?id=~Peiwen_Huang2), [Tzu-Hung Huang](https://openreview.net/profile?id=~Tzu-Hung_Huang1), [Shou-De Lin](https://openreview.net/profile?id=~Shou-De_Lin1), [Chih-Hao Hsu](https://openreview.net/profile?id=~Chih-Hao_Hsu2))20 Feb 2026, 18:24 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 3ErX, Commitment Readers[Revisions](https://openreview.net/revisions?id=hPMPgScScD)

**Comment:**

Thank you for your rigorous review.

Our paper uses **three distinct metrics**:

- **Overall Accuracy**: Standard metric; abstaining ("E) I am not sure") counts as *incorrect*.
- **Conditional Accuracy**: Accuracy on questions the model actually answered (i.e., disregarding the cases when the model answered “E”). Reflects knowledge quality when it acts.
- **E-ratio**: Fraction of questions where the model abstains.

**Confidence-performance decoupling** means role prompts primarily shift the E-ratio (willingness to act) while Conditional Accuracy remains stable, proving the Expert/Non-Expert gap is driven by differential **abstention**, not knowledge.

### **1. MMLU-E and Observation 2**

**MMLU-E is our own framework** — we add "E) I am not sure" to standard MMLU. We will define MMLU-E formally at its first mention.

Observation 2 reveals that role-playing modulates the **expression** of knowledge rather than its **underlying representation**. For instance, while the Non-Expert Llama3-8B-IT abstains on **44.8%** of questions with a conditional accuracy () of **69.3%**, the Expert reduces abstention to **6.9%** while maintaining a stable  of **67.2%**. This stability demonstrates that role identity functions as a **decision threshold regulator** rather than a knowledge booster.

### **2. Figure 3a, Degradation, Role-Induced Divergence**

**(a) Figure 3a shows accuracy, not E-ratio.** We will add a dual-axis panel with E-ratio in the revision.

**(b) Degradation at high ρ is hard to see** because the y-axis is compressed. We will add a dedicated Expert-only plot where degradation is unambiguous.

**(c) Role-induced divergence.** We compute layer-wise Pearson correlations between Expert and Non-Expert hidden states. The sharp correlation drop in layers 11–19 precisely isolates where the model differentiates roles and modulates confidence.

### **3. MSP Metric**

We appreciate this careful cross-referencing. Two important contextual distinctions:

**(a)** We use MSP only as a **relative metric** comparing conditions on the same question — relative ranking is sufficient, not absolute calibration. Plaut et al. (2024) confirm MSP reliably discriminates correct from incorrect answers in LLMs.

**(b)** We **triangulate** confidence via three independent signals: (1) Internal logits (MSP), (2) External abstention (E-ratio from MMLU-E), (3) Verbalized self-evaluation (Appendix Figs. 12/13). All three yield consistent conclusions.

### **4. Figure 2 and the Accuracy Gap**

**Figure 2** illustrates that while **MSP** increases **strictly monotonically** across the role hierarchy (**Expert > Student > Person > Non-Expert**), **Overall Accuracy** fluctuates **non-monotonically**, with the ‘Person’ occasionally outperforming the ‘Expert’. This suggests expert prompts primarily boost **confidence** rather than the fundamental capability of retrieving knowledge. **Table 1** further corroborates this, showing the Non-Expert's **Conditional Accuracy** (69.3%) is actually slightly higher than the Expert's (67.2%). Ultimately, Expert accuracy reflects higher willingness, not superior knowledge; Non-Experts achieve equivalent performance when encouraged to answer. We focus on the **Expert vs. Non-Expert** pair to maximize SNR and control lexical confounders, differing only by the token "non".

### **5. Dense vs. Sparse Vectors**

**Our claim is empirical:** sparse editing yields superior accuracy while modifying drastically fewer neurons. From Table 2, PCA/ICV (Liu et al., 2024; dense, top-1 principal component) reaches only 51.47% for Non-Expert and *degrades* Expert performance (to 60.21%). In contrast, RSN achieves 63.22% and 65.74%. If a cleanly orthogonal 1D confidence direction existed, PCA would find it. Its failure proves sparse intervention is necessary to isolate target behavior.

### **6. Why Divergent Pairs?**

Correct—not logically required; consistent samples also work. We use divergent pairs as a **SNR optimization**: the same effect requires α = 7.0 with consistent-sample vectors vs. α = 4.0 with divergent-pair vectors.

| Setting | ACC (%) | E-ratio (%) |
| --- | --- | --- |
| Non-Expert Original | 46.0 | 34.8 |
| + Consistent Vector (α=7) | **65.2** | 2.5 |
| Expert Original | 65.5 | 3.4 |
| + Consistent Vector (α=7) | **66.9** | 0.2 |

### **7. Section 5.1: The Decoupling Logic**

The apparent contradiction resolves as:

1. Non-Expert has the knowledge but suppresses it (high abstention → low Overall Accuracy).
2. RSN injection adds **no new knowledge**; it only lowers the decision threshold.
3. The model already knew the answers — forcing a decision surfaces latent knowledge → accuracy recovers.

"Recovers accuracy" is the causal proof of "modulates confidence." The gap was caused by gating, not knowledge deficiency. We will restructure early sections to present the three-metric framework. We hope these clarifications enable a complete reading and would be grateful for your reconsideration.

**Official Comment by Reviewer 3ErX**

Official Commentby Reviewer 3ErX22 Feb 2026, 18:44 (modified: 18 Mar 2026, 00:21)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 3ErX, Commitment Readers[Revisions](https://openreview.net/revisions?id=yHvCm4dS2b)

**Comment:**

- What do you mean by “knowledge retrieval”? I see your logic in comparing “Person” and “Expert” in Figure 2 to conclude that knowledge is preserved while confidence differs (though see my comment about MSP). However, you then conclude that this holds for all roles. That is not the case if one compares the “non-Expert” and “Expert” roles.
- At the end of Observation 2, you write that it “was previously suppressed by a conservative confidence gate.” Isn’t that a reflection of improved retrieval?
- I understand that the questions are the same, but when you condition on a role, you create several separate datasets. A threshold on MSP would correctly discriminate samples within each specific dataset, but I do not see how this carries over to the inter-dataset level.
- I understand that images are not allowed during rebuttal, but you could at least attach a table. Currently, I have to take your word for it.
- Regarding — I would not expect decompressing the axis to reveal degradation. The range of values for the Expert in Fig. 3a is around 2%. Why is this considered degradation?
- FV found a dense direction that also restores accuracy (62.11 vs. 63.22 for RSN does not seem like a large difference). Why is it a problem that abstention dropped to 0, which you describe as aggressive? Also, how do FV’s metrics change if it is projected onto the space of the neurons you selected?To clarify, it is interesting to see that a specific subset of neurons is responsible for the behavior. My issue is with the statement “dense vectors **inevitably** mix confidence signals with semantic content” — I do not see sufficient grounds for that.
- Regarding point 7: sorry, I still do not get it. I am either right or wrong; my confidence does not change that. I understand how it decouples. But despite stating in Observation 1 (as I understand it) that you also agree, you couple them here by saying that one (“recovers accuracy”) implies the other (“modulates confidence”).

Also, thank you for sounding human in your response :)


