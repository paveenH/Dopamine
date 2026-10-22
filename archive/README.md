# archive/

Code of CLOSED / shelved / dropped experiment lines, moved here on 2026-10-22. Nothing was edited, only moved (`git mv`, history kept).

Lines: Bandit pv1-PV11 (+ frozen manifests/state banks), CGT-simultaneous / CGT-sequential, behavioural probes (SciWorld, Reversal, CRT, Trait, Capitulation, effort-choice, GPQA delay), SocialIQA / HaluEval / TruthfulQA-gen, Phase-2 closed loop, Manifold pilot, ZebraLogic-Easy.

These scripts import `utils`, `llms`, `template` from the repo root, so they do NOT run from here. To re-run one, `git mv` it (and its launcher) back to the root. Directory layout is preserved for `manifold/` and `zebralogic/`.

Added later on 2026-10-22: `inspect_p4c_preflight.py` (P4c format preflight) and `finqa/run_finqa_eval_preflight.sh` (finqa-v0 preflight scoring), both preflight leftovers of closed lines.

Kept at the root on purpose because active code imports them: `bandit_pv6_episode.py` and `bandit_reference.py` (chat helpers used by the Native Chat runners), `get_answer_cgt.py` (used by `get_answer_igt.py`).
