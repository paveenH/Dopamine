# GSM8K Norm-Matched Orthogonal Injection v1

**Status:** code and protocol prepared; no GPU generation or result exists yet. Freeze before any formal cell starts.

## Question and conditions

At each model's frozen neutral Bare No-CoT GSM8K workpoint, test whether the observed benefit depends on the RSN weight direction. The three controls use the prebuilt `ortho_gauss_same` masks: same per-layer support as the model's NMD mask, equal per-layer norm, and orthogonal in float64 to the corresponding dense role-difference. Seed selects a direction only; it is not a generation replicate. No direction selection, sign flip, dose change, or replacement is allowed.

| Model | Model id | α | saved mask rows / decoder layers | Cells |
|---|---|---:|---|---|
| Llama3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | −6 | `[11,20)` / 10–18 | baseline, RSN, orthogonal seeds 1–3 |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | +8 | `[16,22)` / 15–20 | baseline, RSN, orthogonal seeds 1–3 |

All conditions use the same 300-item Bare No-CoT dataset, plain neutral prompt, greedy decoding, 768-token budget, batch 24, output-side prefill-only injection at the last prefill token (`tail=1`). Injection is `α × mask` exactly once. Expected fire counts are 2700 / 1800 for nonzero Llama / Qwen cells and 0 for baseline. Prompt token IDs are recorded from the actual tokenizer; token 220 is not assumed.

The server-side `verify` command checks existing artifacts only and must stop on missing files or failed support, finite-value, band, norm, or dense-difference orthogonality checks. It does not rebuild masks. Original means are stored with embedding row 0; mask row `i` is checked against dense-difference row `i+1`. Float64 checks are the acceptance criterion; the report also records bf16 norm and dot-product errors because actual injection is approximate after dtype conversion.

## Matching, output, and analysis

Run each model's five conditions sequentially on one machine and one pinned GPU. Run Llama first. Smoke is the first four fixed questions and writes only under `smoke/`; it tests the same five conditions and cannot select seeds. Formal outputs live under `components/{llama3,qwen2.5}/gsm8k_orthogonal_v1/{baseline,rsn,seed1,seed2,seed3}/`. Existing cells are skipped only when the full provenance manifest matches; an incomplete or conflicting cell stops without overwrite. Keep every row, raw decoded text, generated token count, and stop reason.

The 300-question set is the original GSM8K evaluation sample and may have contributed to the historical direction-construction data. Unless source manifests prove otherwise, report this as a **direction-specificity check on the original workpoint data**, not held-out validation.

Primary metric: frozen offline `first_acc` from `analyze_first_last_acc.py` (first `####` plus the production fallback). Auxiliary metrics use the frozen `early_candidate_detector.py` (`earlycand-v1`) and report candidate coverage, answer coverage, truncation, loops, empty outputs, and generated lengths. Reproduce the published reference count before interpreting new cells. Pair rows by question identity and exact question text; missing/duplicate pairs are errors.

For each model, report all five cell health/accuracy summaries, RSN and each control versus baseline, and the three direct RSN-versus-control paired contrasts. The six direct contrasts (two models × three seeds) are the fixed primary family: item-paired bootstrap, B=10,000, seed 0, with raw 95% CIs; two-sided exact McNemar for accuracy; Holm adjustment across all six p-values. Show each seed separately. If averaging the three controls, resample common question IDs jointly; do not treat 900 rows as independent. Three directions do not represent the full random-direction distribution.

If the matched RSN reference fails to reproduce the original workpoint, report that first. A null control result does not establish that all other directions fail; control degradation alone does not identify confidence, wanting, or another psychological mechanism.

## Server commands

On the server after syncing these files, from `/data1/paveen/Dopamine`:

```bash
# Llama first. Pin one GPU and leave it fixed for verify, smoke, and full.
export CUDA_VISIBLE_DEVICES=0
bash run_gsm8k_orthogonal_v1.sh verify llama3
bash run_gsm8k_orthogonal_v1.sh preflight llama3
bash run_gsm8k_orthogonal_v1.sh smoke llama3
bash run_gsm8k_orthogonal_v1.sh full llama3
bash run_gsm8k_orthogonal_v1.sh status llama3

# Only after Llama is complete, use one fixed Qwen GPU.
export CUDA_VISIBLE_DEVICES=0
bash run_gsm8k_orthogonal_v1.sh verify qwen2.5
bash run_gsm8k_orthogonal_v1.sh preflight qwen2.5
bash run_gsm8k_orthogonal_v1.sh smoke qwen2.5
bash run_gsm8k_orthogonal_v1.sh full qwen2.5
bash run_gsm8k_orthogonal_v1.sh status qwen2.5
```

Archive/sync the two `gsm8k_orthogonal_v1` result trees back to the analysis machine without merging them into legacy trees. Run the offline scorer from `/Users/paveenhuang/Documents/RSNResult/RoleAnswer` against the six formal cell JSON files and the frozen Llama/Qwen GSM8K references; inspect every manifest before scoring. The analysis command and scorer-provenance file are maintained alongside that frozen scorer directory.

On the analysis machine, set `SERVER` to the SSH host alias for the GPU machine, then sync each model tree and its two historical reference files. The reference files are the existing neutral No-CoT `mdf_0` and workpoint JSONs (`mdf_-6` for Llama, `mdf_8` for Qwen); the commands pass those files directly. The per-model command validates all five manifests against one another, checks 300 result rows, row order, result hashes, prompt/data/environment settings and injection counts, then writes direct control and baseline contrasts. Only the three RSN-versus-direction tests per model enter the fixed six-test Holm family; baseline contrasts are descriptive with paired bootstrap CIs.

```bash
cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer
mkdir -p ./gsm8k_orthogonal_v1/{llama3,qwen2.5}
rsync -av "$SERVER:/data1/paveen/Dopamine/components/llama3/gsm8k_orthogonal_v1/" ./gsm8k_orthogonal_v1/llama3/
rsync -av "$SERVER:/data1/paveen/Dopamine/components/qwen2.5/gsm8k_orthogonal_v1/" ./gsm8k_orthogonal_v1/qwen2.5/
mkdir -p ./gsm8k_orthogonal_v1/references
rsync -av "$SERVER:/data1/paveen/Dopamine/components/llama3/answer_mdf_gsm8k/mdf_0/gsm8k_8B_answers_20_11_20.json" ./gsm8k_orthogonal_v1/references/llama3_baseline.json
rsync -av "$SERVER:/data1/paveen/Dopamine/components/llama3/answer_mdf_gsm8k/mdf_-6/gsm8k_8B_answers_20_11_20.json" ./gsm8k_orthogonal_v1/references/llama3_rsn.json
rsync -av "$SERVER:/data1/paveen/Dopamine/components/qwen2.5/answer_mdf_gsm8k/mdf_0/gsm8k_7B_answers_17_16_22.json" ./gsm8k_orthogonal_v1/references/qwen25_baseline.json
rsync -av "$SERVER:/data1/paveen/Dopamine/components/qwen2.5/answer_mdf_gsm8k/mdf_8/gsm8k_7B_answers_17_16_22.json" ./gsm8k_orthogonal_v1/references/qwen25_rsn.json
python3 analyze_gsm8k_orthogonal_v1.py --model llama3 --result_root ./gsm8k_orthogonal_v1/llama3 --baseline_reference ./gsm8k_orthogonal_v1/references/llama3_baseline.json --rsn_reference ./gsm8k_orthogonal_v1/references/llama3_rsn.json --out orthogonal_v1_llama3.json
python3 analyze_gsm8k_orthogonal_v1.py --model qwen2.5 --result_root ./gsm8k_orthogonal_v1/qwen2.5 --baseline_reference ./gsm8k_orthogonal_v1/references/qwen25_baseline.json --rsn_reference ./gsm8k_orthogonal_v1/references/qwen25_rsn.json --out orthogonal_v1_qwen25.json
python3 combine_gsm8k_orthogonal_v1.py --llama orthogonal_v1_llama3.json --qwen orthogonal_v1_qwen25.json --out orthogonal_v1_primary_contrasts.json
```
