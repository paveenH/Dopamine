# templates.py
from typing import List


# ===== Default suite (Expanded / Original MMLU-style prompts) =====
template_mmlu = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are a {character}, your answer among "A, B, C, D" is: '
)

template_mmlu_cot = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    "Let's think step by step.\n"
    'Now you are a {character}, your answer among "A, B, C, D" is: '
)

template_mmlu_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)

template_mmlu_E_cot = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)

template_mmlu_action = (
    "Here is a question: {context}\n"
    "Now you are an honest {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_mmlu_action_cot = (
    "Here is a question: {context}\n"
    "Let's think step by step.\n"
    "Now you are an honest {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neg = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are {character}, your answer among "A, B, C, D" is: '
)

template_neg_cot = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    "Let's think step by step.\n"
    'Now you are {character}, your answer among "A, B, C, D" is: '
)

template_neg_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are {character}, your answer among "A, B, C, D, E" is: '
)

template_neg_E_cot = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    'Now you are {character}, your answer among "A, B, C, D, E" is: '
)

template_neg_action = (
    "Here is a question: {context}\n"
    "Now you are {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neg_action_cot = (
    "Here is a question: {context}\n"
    "Let's think step by step.\n"
    "Now you are {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neutral = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Your answer among "A, B, C, D" is: '
)

template_neutral_cot = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D" is: '
)

template_neutral_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Your answer among "A, B, C, D, E" is: '
)

template_neutral_E_cot = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D, E" is: '
)

template_neutral_action = (
    "Here is a question: {context}\n"
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neutral_action_cot = (
    "Here is a question: {context}\n"
    "Let's think step by step.\n"
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)


# ===== Vanilla suite (pure context + Answer:) =====

template_vanilla = "{context}\n" "As an honest {character}, answer: "

template_vanilla_E = "{context}\n" "E) I am not sure.\n" "As an honest {character}, answer: "

template_vanilla_neutral = "{context}\n" "Answer: "


template_vanilla_neutral_E = "{context}\n" "E) I am not sure.\n" "Answer: "

template_vanilla_cot = "{context}\n" "Let's think step by step.\n" "Answer: "

template_vanilla_cot_E = "{context}\n" "E) I am not sure.\n" "Let's think step by step.\n" "Answer: "

# ===== Action suite (pure context + Answer:) =====

def build_default_suite(use_E: bool = False, cot: bool = False):
    """
    Return the default suite (question + 'Answer among ...'), preserving original wording.
    Parameters:
        use_E: include 'E) I am not sure.' option
        cot: use CoT ('Let's think step by step.') templates
    Always returns keys: 'default', 'neutral', 'neg', 'cot', 'labels'
    """
    if use_E:
        if cot:
            return {
                "default": template_mmlu_E_cot,     # honest {character}
                "neutral": template_neutral_E_cot,  # no role
                "neg": template_neg_E_cot,          # you are {character}
                "labels": ["A", "B", "C", "D", "E"],
            }
        else:
            return {
                "default": template_mmlu_E,
                "neutral": template_neutral_E,
                "neg": template_neg_E,
                "labels": ["A", "B", "C", "D", "E"],
            }
    else:
        if cot:
            return {
                "default": template_mmlu_cot,
                "neutral": template_neutral_cot,
                "neg": template_neg_cot,
                "labels": ["A", "B", "C", "D"],
            }
        else:
            return {
                "default": template_mmlu,
                "neutral": template_neutral,
                "neg": template_neg,
                "labels": ["A", "B", "C", "D"],
            }


def build_vanilla_suite(use_E: bool = False):
    """Return the vanilla suite (no 'Would you answer...' preface), preserving original wording."""
    if use_E:
        return {
            "default": template_vanilla_E,  # honest {character}
            "neutral": template_vanilla_neutral_E,  # no role
            "cot": template_vanilla_cot_E,  # CoT
            "labels": ["A", "B", "C", "D", "E"],
        }
    else:
        return {
            "default": template_vanilla,  # default
            "neutral": template_vanilla_neutral,
            "cot": template_vanilla_cot,  # CoT
            "labels": ["A", "B", "C", "D"],
        }

def build_action_suite(cot):
    labels = [str(i) for i in range(10)]
    if cot:
        return {
            "default": template_mmlu_action_cot,  # honest {character}
            "neutral": template_neutral_action_cot,  # no role
            "neg": template_neg_action_cot,
            "labels": labels,
        }
    else:
        return {
            "default": template_mmlu_action,  # honest {character}
            "neutral": template_neutral_action,  # no role
            "neg": template_neg_action,
            "labels": labels,
        }


# ===== Unified selector =====

def select_templates(suite: str = "default", use_E: bool = False, cot=False):
    """
    suite: "default" | "vanilla"
    use_E: Whether to include the E option ("I am not sure")
    Returns a dict containing templates and labels for the chosen suite.
    """
    suite = suite.lower()
    if suite == "default":
        return build_default_suite(use_E, cot)
    elif suite == "vanilla":
        return build_vanilla_suite(use_E)
    elif suite == "action":
        return build_action_suite(cot)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")


# ==========================================================================================
# -------- Default suite (question + “Answer among …”) --------
def _labels_str(labels: List[str]) -> str:
    """Format a label list like ["A","B","C","D"] into a string 'A, B, C, D'."""
    # You could also use a range-style "A–J", but commas are clearer.
    return ", ".join(labels)


def _next_letter(last: str) -> str:
    return chr(ord(last) + 1)


def build_default_suite_pro(labels: List[str], use_E: bool = False, cot: bool = False):
    """
    MMLU-Pro: does not insert extra option lines (the data already contains A) ..., B) ...).
    By default, use the dataset's labels (A.. up to J).
    If use_E=True, append 'E' as an extra choice and insert 'E) I am not sure.' line.
    """
    labels = list(labels)  # copy
    refusal_label = None
    e_line = ""
    if use_E:
        refusal_label = _next_letter(labels[-1])
        labels.append(refusal_label)
        e_line = f"{refusal_label}) I am not sure.\n"

    L = _labels_str(labels)

    # Base text parts
    base_q = f"Would you answer the following question with {L}?\nQuestion: {{context}}\n"

    template_default = base_q + e_line + 'Now you are an honest {character}, your answer among "' + L + '" is: '

    template_default_cot = (
        base_q + e_line + "Let's think step by step.\n" 'Now you are an honest {character}, your answer among "' + L + '" is: '
    )

    template_neutral = base_q + e_line + 'Your answer among "' + L + '" is: '

    template_neutral_cot = base_q + e_line + "Let's think step by step.\n" 'Your answer among "' + L + '" is: '

    template_neg = base_q + e_line + 'Now you are {character}, your answer among "' + L + '" is: '

    template_neg_cot = (
        base_q + e_line + "Let's think step by step.\n" 'Now you are {character}, your answer among "' + L + '" is: '
    )

    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg": template_neg,
            "labels": labels,
            "refusal_label": refusal_label,
        }

    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg": template_neg_cot,
            "labels": labels,
            "refusal_label": refusal_label,
        }


# -------- Vanilla suite (context only + “Answer:”) --------
def build_vanilla_suite_pro(labels: List[str], use_E: bool = False, cot: bool = False):
    """
    Vanilla version does not re-declare the label set (context already contains A) ...).
    If use_E=True, append 'E' to labels and add 'E) I am not sure.' line explicitly.
    """
    labels = list(labels)
    e_line = ""
    refusal_label = None
    if use_E:
        refusal_label = _next_letter(labels[-1])
        labels.append(refusal_label)
        e_line = f"{refusal_label}) I am not sure.\n"

    template_default = "{context}\n" + e_line + "As an honest {character}, answer: "

    template_default_cot = "{context}\n" + e_line + "Let's think step by step.\n" "As an honest {character}, answer: "

    template_neutral = "{context}\n" + e_line + "Answer: "

    template_neutral_cot = "{context}\n" + e_line + "Let's think step by step.\n" "Answer: "

    template_neg = "{context}\n" + e_line + "You are {character}, answer: "

    template_neg_cot = "{context}\n" + e_line + "Let's think step by step.\n" "You are {character}, answer: "

    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg": template_neg,
            "labels": labels,
            "refusal_label": refusal_label,
        }

    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg": template_neg_cot,
            "labels": labels,
            "refusal_label": refusal_label,
        }


# -------- Action-choice suite (choose A/B/C instead of answering) --------

def build_action_suite_pro(cot):
    """
    Action self-evaluation mode (0–9):
    The model does NOT answer the question; it reports a reasoning-willingness score in [0,9].
    """
    action_labels = [str(i) for i in range(10)]

    template_default = (
        "Here is a question: {context}\n"
        "Now you are an honest {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    

    template_neutral = (
        "Here is a question: {context}\n"
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )

    template_neg = (
        "Here is a question: {context}\n"
        "Now you are {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    template_default_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        "Now you are an honest {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    template_neutral_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )

    template_neg_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        "Now you are {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    if cot:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg":     template_neg_cot,
            "labels":  action_labels,  
            "refusal_label": None,      
        }
    else:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg":     template_neg,
            "labels":  action_labels,  
            "refusal_label": None,      
        }


# -------- Unified selector for MMLU-Pro --------
def select_templates_pro(suite: str, labels: List[str] = None, use_E: bool = False, cot: bool = False):
    """
    suite: "default" | "vanilla"
    labels: e.g. ["A","B","C","D","F","G"] from the dataset
    use_E: if True, append "E" and add "E) I am not sure."
    """
    suite = suite.lower()
    if suite == "default":
        return build_default_suite_pro(labels, use_E, cot)
    elif suite == "vanilla":
        return build_vanilla_suite_pro(labels, use_E, cot)
    elif suite == "action":
        return build_action_suite_pro(cot)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")


# ===== GSM8K Generation templates =====
# Unlike multiple-choice, these prompt the model to generate a full solution.
# The model should produce CoT reasoning and end with a boxed/final numeric answer.

def build_gsm8k_default_suite(cot: bool = False):
    """
    GSM8K generation prompts (default suite).
    {context} will be filled with the question text.
    """
    # Symmetric No-CoT / CoT templates: the ONLY difference is the
    # "Let's think step by step." line. The "#### <number>" final-answer
    # directive is present in BOTH so extract_gsm8k_answer hits its strongest
    # (####) rule instead of the noisy last-number fallback. Without it, the
    # model never emits #### and runs to max_new_tokens without committing
    # (see 2026-05-31 diagnosis: 98-100% of No-CoT samples hit the 512-token
    # cap, fallback extraction made role accuracies uncomparable).
    # Neutral wording on purpose: "Provide your final numeric answer after ####"
    # does NOT pressure the model to answer immediately. An earlier variant
    # ("Give your final answer as a single number after ####") read as "just give
    # one number now" and induced early-#### 抢答: the model wrote #### with an
    # un-reasoned initial guess (expert role hit 72% early-####, acc collapsed to
    # 34%). Keep this wording aligned with the historical orig run for
    # comparability. (The 抢答-inducing wording is kept as a future positive-
    # control ablation: prompt wording vs alpha-steering as two levers on
    # commitment timing — see Dopamine over-wanting framing.)
    fmt = "Provide your final numeric answer after '####'."
    if cot:
        return {
            "default": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Now you are an honest {character}. Let's think step by step.\n"
                f"{fmt}\n"
                "Answer: "
            ),
            "neutral": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Let's think step by step.\n"
                f"{fmt}\n"
                "Answer: "
            ),
            "neg": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Now you are {character}. Let's think step by step.\n"
                f"{fmt}\n"
                "Answer: "
            ),
        }
    else:
        return {
            "default": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Now you are an honest {character}.\n"
                f"{fmt}\n"
                "Answer: "
            ),
            "neutral": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                f"{fmt}\n"
                "Answer: "
            ),
            "neg": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Now you are {character}.\n"
                f"{fmt}\n"
                "Answer: "
            ),
        }


def build_gsm8k_vanilla_suite(cot: bool = False):
    """GSM8K generation prompts (vanilla suite)."""
    if cot:
        return {
            "default": (
                "{context}\n"
                "As an honest {character}, let's think step by step.\n"
                "Answer: "
            ),
            "neutral": (
                "{context}\n"
                "Let's think step by step.\n"
                "Answer: "
            ),
            "neg": (
                "{context}\n"
                "You are {character}, let's think step by step.\n"
                "Answer: "
            ),
        }
    else:
        return {
            "default": (
                "{context}\n"
                "As an honest {character}, answer: "
            ),
            "neutral": (
                "{context}\n"
                "Answer: "
            ),
            "neg": (
                "{context}\n"
                "You are {character}, answer: "
            ),
        }


def build_gsm8k_action_suite(cot: bool = False):
    """GSM8K willingness self-evaluation prompts (action suite, 0-9 score)."""
    action_labels = [str(i) for i in range(10)]
    if cot:
        return {
            "default": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                "Now you are an honest {character}. "
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "neutral": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "neg": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                "Now you are {character}. "
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "labels": action_labels,
        }
    else:
        return {
            "default": (
                "Here is a question: {context}\n"
                "Now you are an honest {character}. "
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "neutral": (
                "Here is a question: {context}\n"
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "neg": (
                "Here is a question: {context}\n"
                "Now you are {character}. "
                'Your self-evaluation of "reasoning willingness" from [0,9] is: '
            ),
            "labels": action_labels,
        }


def build_gsm8k_confidence_suite(cot: bool = False):
    """GSM8K answer confidence self-evaluation prompts (0-9 score)."""
    labels = [str(i) for i in range(10)]
    if cot:
        return {
            "default": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                "Now you are an honest {character}. "
                "Your confidence of the question from [0,9] is: "
            ),
            "neutral": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                "Your confidence of the question from [0,9] is: "
            ),
            "neg": (
                "Here is a question: {context}\n"
                "Let's think step by step.\n"
                "Now you are {character}. "
                "Your confidence of the question from [0,9] is: "
            ),
            "labels": labels,
        }
    else:
        return {
            "default": (
                "Here is a question: {context}\n"
                "Now you are an honest {character}. "
                "Your confidence of the question from [0,9] is: "
            ),
            "neutral": (
                "Here is a question: {context}\n"
                "Your confidence of the question from [0,9] is: "
            ),
            "neg": (
                "Here is a question: {context}\n"
                "Now you are {character}. "
                "Your confidence of the question from [0,9] is: "
            ),
            "labels": labels,
        }


def build_math_suite(cot: bool = True):
    """
    MATH open-ended generation template (neutral role only).
    {context} = MATH question text.
    Answer extracted via \\boxed{} or last number.
    """
    if cot:
        return {
            "neutral": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Let's think step by step.\n"
                "Answer: "
            ),
        }
    else:
        return {
            "neutral": (
                "Solve the following math problem.\n"
                "Question: {context}\n"
                "Answer: "
            ),
        }


def build_effort_choice_suite():
    """
    Effort-based Task Choice (Round 1: choose A or B).
    {context_a} = GSM8K question, {context_b} = MATH question.
    Labels: ["A", "B"]
    """
    template = (
        "You are given two math problems. Choose one to answer.\n\n"
        "Option A: worth 1 point. (simple)\n"
        "Option B: worth 10 points. (hard)\n\n"
        "[Problem A]: {context_a}\n\n"
        "[Problem B]: {context_b}\n\n"
        'Which problem do you choose to answer? Your choice among "A, B" is: '
    )
    return {
        "neutral": template,
        "labels": ["A", "B"],
    }


def select_templates_gsm8k(suite: str = "default", cot: bool = False):
    """
    Unified selector for GSM8K generation templates.
    suite: "default" | "vanilla" | "action" | "confidence"
    """
    suite = suite.lower()
    if suite == "default":
        return build_gsm8k_default_suite(cot)
    elif suite == "vanilla":
        return build_gsm8k_vanilla_suite(cot)
    elif suite == "action":
        return build_gsm8k_action_suite(cot)
    elif suite == "confidence":
        return build_gsm8k_confidence_suite(cot)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default', 'vanilla', 'action', or 'confidence'.")


# ===== TriviaQA Generation templates =====
# Open-ended QA: the model generates a short factual answer.

def build_triviaqa_default_suite(cot: bool = False):
    """
    TriviaQA generation prompts (default suite).
    {context} will be filled with the question text.
    """
    if cot:
        return {
            "default": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Now you are an honest {character}. Let's think step by step.\n"
                "Answer: "
            ),
            "neutral": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Let's think step by step.\n"
                "Answer: "
            ),
            "neg": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Now you are {character}. Let's think step by step.\n"
                "Answer: "
            ),
        }
    else:
        return {
            "default": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Now you are an honest {character}, provide a short and precise answer.\n"
                "Answer: "
            ),
            "neutral": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Provide a short and precise answer.\n"
                "Answer: "
            ),
            "neg": (
                "Answer the following question.\n"
                "Question: {context}\n"
                "Now you are {character}, provide a short and precise answer.\n"
                "Answer: "
            ),
        }


def build_triviaqa_vanilla_suite(cot: bool = False):
    """TriviaQA generation prompts (vanilla suite)."""
    if cot:
        return {
            "default": (
                "{context}\n"
                "As an honest {character}, let's think step by step.\n"
                "Answer: "
            ),
            "neutral": (
                "{context}\n"
                "Let's think step by step.\n"
                "Answer: "
            ),
            "neg": (
                "{context}\n"
                "You are {character}, let's think step by step.\n"
                "Answer: "
            ),
        }
    else:
        return {
            "default": (
                "{context}\n"
                "As an honest {character}, answer: "
            ),
            "neutral": (
                "{context}\n"
                "Answer: "
            ),
            "neg": (
                "{context}\n"
                "You are {character}, answer: "
            ),
        }


def select_templates_triviaqa(suite: str = "default", cot: bool = False):
    """
    Unified selector for TriviaQA generation templates.
    suite: "default" | "vanilla"
    """
    suite = suite.lower()
    if suite == "default":
        return build_triviaqa_default_suite(cot)
    elif suite == "vanilla":
        return build_triviaqa_vanilla_suite(cot)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")
