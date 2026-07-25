# Generic-direction null — server runbook (§4.6 upgrade)

Upgrades §4.6 from a **support-selection null** (diff_random) toward a **direction-specificity**
test, using two orthogonal-Gaussian null families. Zero-GPU (offline re-projection only).

## What was added (already committed locally)

- `detection/nmd.py`: two new `--mask_type` values
  - `ortho_gauss_same` — NMD's own 20 positions/layer, Gaussian weights **⊥ dense role-diff Δ_l**, per-layer norm-matched to NMD.
  - `ortho_gauss_off` — 20 random positions/layer **disjoint** from NMD, same ⊥ + norm-match.
  - Both run in float64; built-in acceptance guards assert per-layer `|cos(m_l, Δ_l)| < 1e-5`,
    exact norm-match, exactly `top_k` nonzeros, and the correct support relation. The run aborts
    (set -e) if any guard fails — so a clean finish IS the validation.
- `run_generic_null.sh`: builds 10 seeds × 2 families and re-projects the 7 frozen conditions.
- `analyze_rsn_specificity.py` (local workspace): `--null_family {diff_random,ortho_gauss_same,ortho_gauss_off}`
  + `--null_root`. Frozen metrics unchanged (`s_pre_mean`, `p_post_mean`, LOO-RMS).

## Run on the server

```bash
cd <RSN repo on server>              # where detection/nmd.py + run_generic_null.sh live
git pull                             # get the two new mask types + the script
bash run_generic_null.sh             # ~ builds 20 masks + 20 re-projection sets
```

Watch for the guard line per seed:
```
[ortho guard] max |cos(m_l, Δ_l)| over layers = X.XXe-18
```
If any seed raises an AssertionError it stops — re-run just that seed after checking the mask dir.

Outputs on server:
```
$BASE_DIR/mask/llama3_non_logits/ortho_gauss_same_0.5_11_20_8B_seed{1..10}.npy
$BASE_DIR/mask/llama3_non_logits/ortho_gauss_off_0.5_11_20_8B_seed{1..10}.npy
$BASE_DIR/llama3_ortho_same/seed{1..10}/random_signal_*.json
$BASE_DIR/llama3_ortho_off/seed{1..10}/random_signal_*.json
```

## Pull to the local analysis workspace

Into `~/Documents/RSNResult/RoleAnswer/`:

```bash
# masks (analysis reads them for norm/reference)
rsync -av <server>:$BASE_DIR/mask/llama3_non_logits/ortho_gauss_*_seed*.npy \
      llama3/dopamine/mask/llama3_non_logits/

# signal JSON (the null draws)
rsync -av <server>:$BASE_DIR/llama3_ortho_same/  llama3/dopamine/ortho_same/
rsync -av <server>:$BASE_DIR/llama3_ortho_off/   llama3/dopamine/ortho_off/
```

## Analyze (frozen metrics, only the null family switches)

```bash
cd ~/Documents/RSNResult/RoleAnswer
python3.10 analyze_rsn_specificity.py --null_family ortho_gauss_same --null_root llama3/dopamine/ortho_same
python3.10 analyze_rsn_specificity.py --null_family ortho_gauss_off  --null_root llama3/dopamine/ortho_off
# figures:
python3.10 analyze_rsn_specificity.py --plot --null_family ortho_gauss_same --null_root llama3/dopamine/ortho_same
python3.10 analyze_rsn_specificity.py --plot --null_family ortho_gauss_off  --null_root llama3/dopamine/ortho_off
```

## Read the 2×2 (per the agreed decision)

| same-support | off-support | interpretation |
|---|---|---|
| collapses | collapses | NMD support **and** role-diff weight structure jointly matter |
| holds | collapses | mainly the **NMD neuron support** is special |
| collapses | collapses | support alone insufficient; role-diff weight/sign structure is key |
| — | **holds** | warning: any sparse direction hits a global mode → NMD direction-specificity weakened |

"holds" = the primary signed effect (`s_pre_mean`/`p_post_mean`) stays at pctile 0%/100% across
contrasts and LOO-RMS pctile ≈100%, as it does vs diff_random. "collapses" = NMD falls inside the
null (pctiles drift off the extremes).

10 seeds → exploratory distribution (p floor 1/11 ≈ 0.091). Only if the result is **ambiguous**
extend seeds; only if support-vs-weight needs disentangling add same-support sign-shuffle.
