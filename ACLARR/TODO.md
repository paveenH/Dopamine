# Camera-Ready TODO (Due: April 19, 2026)

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---

## Task 1: Update Figure 3a (Sparsity) to dual-axis version
**File**: `main.tex` — Figure 3 (fig:rsn-characterization), subfigure (a)
- [x] Change `figure5-mean.png` → `figure5-dual.png` in `\includegraphics`
- [x] Update subfigure caption to mention dual-axis (Accuracy + E-ratio)
- [x] Update the figure's overall caption to reflect the new content

---

## Task 2: Add Algorithm pseudocode (AC requirement)
**File**: `main.tex` — Section 4 Method (after subsection 4.2)
- [x] Add `\usepackage{algorithm}` and `\usepackage{algpseudocode}` to preamble
- [x] Insert Algorithm 1 box with 4 steps

---

## Task 3: Add GSM8K results (AC requirement)
**File**: `main.tex` — Section 5.4 (Validation: Internal State Dynamics)
- [x] Add paragraph introducing GSM8K open-ended evaluation
- [x] Add Confidence Ratio table (Llama3 / Qwen3 × α=-4 / Neutral / α=+4)
- [x] Add brief interpretation
- [x] Accuracy data excluded (model-dependent trends)

---

## Task 4: Add computational resources paragraph
**File**: `main.tex` — Section 4 Method (after Algorithm 1)
- [x] ~28,000 forward passes, 2–3 hours on A100, no gradient, ~180 scalar values/layer

---

## Task 5: Formally define MMLU-E at first mention
**File**: `main.tex` — Section 3, Observation 2
- [x] Added explicit definition of MMLU-E and E-ratio

---

---

## Task 6: Post-review fixes (2026-04-10)
- [x] Add Acknowledgments section (NSTC grant 114-2221-E-002-134-MY3 + NTU TCE)
- [x] Move Acknowledgments before Limitations (ACL convention)
- [x] Replace anonymized URL with GitHub link: https://github.com/paveenH/RSN
- [x] Update `paper8_mirror_human` bib entry to ACL Findings 2025 inproceedings format (Xu et al.)
- [x] Remove residual `[cite: 844, 845, 846]` and `[cite: 951, 954]` markers from Appendix A.8
- [x] Revise GSM8K paragraph: add LaTeX formula for Confidence Ratio, clarify Neutral setting
- [x] Add Appendix: GSM8K Linguistic Marker Dictionary (full marker list + detailed counts table)
- [x] Regenerate figure5-dual.png at 600 dpi (was 300 dpi)

---

## Already Completed
- [x] Switched from `\usepackage[review]{acl}` to `\usepackage{acl}` (camera-ready mode)
- [x] Added author names and NTU CSIE affiliation
- [x] Fixed typo: `acting primarily as``signal sharpener''` → proper spacing
- [x] Generated dual-axis figure5-dual.png (figure5-mean.png = old version, figure5-dual.png = new)
