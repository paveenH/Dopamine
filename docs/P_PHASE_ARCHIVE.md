# P-phase archive — result tables, artifact hashes, runbooks

Split out of `CLAUDE.md` on 2026-09-02 to shrink the always-loaded context. P2,
P3 and the P3 supplement are all **COMPLETE + FROZEN**; their frozen wording,
口径 traps and boundaries stay in `CLAUDE.md`, and their narrative results live
in `AdaDopamine_gsm8k.md` §5.1–5.6. What moved here is the material a future
session does not need in context to avoid a mistake: the numeric result tables
already published in the results document, the per-artifact SHA256 lists (also
in the `docs/*_manifest.json` / `docs/*_result_*.json` files), and the
reproduction runbooks.

Nothing was deleted; every block below is byte-identical to what stood in
`CLAUDE.md`.

---

## P2 — commitment prediction + cross-task workpoint transfer

Protocol `docs/PREREG_P2.md` (commit `37713c8`, protocol sha256 `8fddcf12…`).
Offline in `RoleAnswer/p2/`, `python3.10`, no server, no GPU.

### P2A frozen result details

- **Held-out performance.** Question-grouped five-fold OOF AUROC is `.687
  [.656, .719]` for Llama and `.749 [.710, .787]` for Qwen with the
  commitment-only predictor. Entry-only AUROC is `.548 [.526, .571]` and `.628
  [.601, .654]`, respectively. The paired question-cluster bootstrap difference
  `commitment − entry` is `+.139 [+.104, +.172]` for Llama and `+.121
  [+.084, +.156]` for Qwen.

- **Incremental entry-gain comparison.** The paired `combined − commitment`
  difference is `−.001 [−.004, +.002]` for Llama and `+.002 [−.002, +.007]`
  for Qwen. Both intervals include zero. Frozen wording is **"no detectable
  incremental predictive gain"**, never "entry gain is useless" or "entry gain has
  no information".

- **Gate and calibration.** The preregistered gate is commitment-only AUROC 95% CI
  lower bound `> .5`; both models pass (Llama `.6561`, Qwen `.7098`). GSM8K OOF
  calibration slopes are `.95` and `.98`, respectively, and are shown in
  `fig_p2a_calibration.png`. These results establish out-of-sample prediction on the
  training task, not causal mediation or intervention specificity.


### P2B frozen result details

- **Qwen full-curve transfer.** Across 9 MATH doses, the frozen predictor selects the
  positive direction and `alpha=+6`; observed `first_acc` also peaks at `+6`. Spearman
  `rho=+.962` (`n=9`), regret is `.000`, and the selection falls in the near-optimal set
  `[+4,+6]`. The predicted rollover at `+8` is also the observed accuracy decline.

- **Llama local-direction transfer.** Only three MATH cells exist (`−4/0/+4`), so the
  frozen evaluation is local direction only. It selects the negative direction and
  `alpha=−4`; observed `first_acc` is locally best at `−4`, regret is `.000`, and the
  selection falls in `[−4,0]`. The resulting `rho=+1.000` at `n=3` is mechanically weak
  evidence and must not be presented beside Qwen's nine-point correlation as an
  equivalent full-curve result.

- **Freeze and figure interpretation.** `p2b_predictions.json` was frozen at SHA256
  `4e52b079…` before MATH accuracy was read. In `fig_p2b_transfer.png`, panel A retains
  the uncalibrated raw gap (Qwen predicted score `.83–.88` versus observed accuracy
  `.54–.68`, roughly `.25` high), while panel B plots change relative to `alpha=0` and
  shows the matched positive direction, `+6` peak, and `+8` decline. No prediction was
  rescaled or recalibrated using MATH labels. The supported claim is retrospective
  locked workpoint selection, not blind validation and not transferable absolute
  accuracy calibration.


### Artifacts and reproduction

- Artifacts + SHA256 in `p2/p2_freeze_manifest.json`: `PREREG_P2.md` `929004f1…`,
  `p2b_predictions.json` `4e52b079…`, `p2b_evaluation.json` `0e11cb0d…`,
  `p2_predictor_llama.json` `b9ee07e4…`, `p2_predictor_qwen.json` `9aad950a…`,
  `fig_p2a_calibration.png` `7f135253…`, `fig_p2b_transfer.png` `f380461e…`.

```bash
# Reproduce, in order. From RoleAnswer/, python3.10, no GPU.
python3.10 p2/build_p2_folds.py --check   # fold manifest reproduces exactly
python3.10 p2/run_p2_audit.py             # label firewall + exhaustive partition,
                                          # reproduces P1's 177/66/57 + loop 52
python3.10 p2/run_p2a.py                  # 5-fold OOF, cluster bootstrap, gates
python3.10 p2/freeze_p2_predictor.py      # refit on all GSM8K, freeze artifacts
python3.10 p2/run_p2b_predict.py          # MATH predictions; REFUSES to overwrite
python3.10 p2/run_p2b_eval.py             # unlocks accuracy; REFUSES without predictions
python3.10 p2/plot_p2.py                  # both required figures
```

Both `run_p2b_predict.py` and `build_p2_folds.py` refuse to overwrite an existing
frozen file: re-running the pipeline end-to-end requires deliberately deleting
them, which is the point — a silent regeneration would let a later fit
masquerade as the locked prediction.

---

## P3 — blind cross-task validation on GSM-Hard

Protocol `docs/PREREG_P3.md` (`p3-v1`, tag `p3-prereg-v1`, commit `1c0b865`).
Result record `docs/p3_result_20260830.json`. Gold was unsealed once, on
2026-08-30; **the main analysis 口径 is CLOSED** and anything further is
exploratory.

```bash
# Server, from /data1/paveen/Dopamine. Download writes to components/benchmark/;
# a non-existent --out_dir is refused rather than created, so a wrong working
# directory fails before the download instead of writing to the wrong tree.
python data_gsm_hard.py --revision 960448f73503112d4226baeb8eb41d3fb5ae2506

# Format preflight, then the ten cells. One model per card.
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_llama3.sh --preflight > p3_pre_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_qwen25.sh --preflight > p3_pre_qwen.log 2>&1 &
# cat the log immediately -- a wrong PY exits 127 before anything runs and the
# nohup log looks empty.
```

---

## P3 supplement — CoT condition transfer

Protocol `docs/PREREG_P3_SUPPLEMENT.md` (`p3-supp-v1`, tag `p3-supp-frozen`).
Result record `docs/p3_supp_result_20260830.json`.

```bash
# Server, from /data1/paveen/Dopamine. One model per card; both cells of a model
# stay together. cat the log immediately -- a wrong PY exits 127 before anything
# runs and the nohup log looks empty.
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_cot_llama3.sh --preflight > p3_cot_pre_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_cot_qwen25.sh  --preflight > p3_cot_pre_qwen.log  2>&1 &
# then, only after the format check passes:
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm_hard_cot_llama3.sh --full > p3_cot_full_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm_hard_cot_qwen25.sh  --full > p3_cot_full_qwen.log  2>&1 &
```

```bash
# Offline (RoleAnswer/, python3.10, no GPU). Refuses to overwrite a frozen file,
# so re-running the chain requires deliberately deleting it.
python3.10 p3/freeze_p3_supp_predictions.py   # stage 1; already run, fcf8b9c9b8fa8b70
python3.10 p3/freeze_p3_supp_commit.py        # stage 2; already run, 6a16d4d862edbdaa
                                              # applies the frozen P2 predictor to the
                                              # four CoT cells and seals the commitment
                                              # side BEFORE any accuracy is read
python3.10 p3/run_p3_supp_eval.py             # unlocks CoT accuracy; REFUSES to start
                                              # without the stage-2 file
python3.10 p3/commit_panel_gsm_hard.py        # commitment + cap% panel (EXPLORATORY)
```
