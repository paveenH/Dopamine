
```text
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

rsync -avzP \
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/{qwen2.5_base,llama3_base}" \
  /Users/paveenhuang/Downloads/


rsync -avzP \
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/llama3_base" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/


##### GSM8K off-support orthogonal controls (gsm8k-orthogonal-off-v1) -- code prepared, no GPU result
# 1) sync code. New files are UNTRACKED -> commit+push them first (local), then pull on the server:
cd /Users/paveenhuang/Downloads/Dopamine
git add get_answer_gsm8k_orthogonal.py verify_gsm8k_orthogonal_masks.py run_gsm8k_orthogonal_off_v1.sh \
  test_gsm8k_orthogonal_off.py docs/PREREG_GSM8K_ORTHOGONAL_OFF_V1.md CLAUDE.md Note/code.md
git commit -m "Add gsm8k-orthogonal-off-v1" && git push origin main
# (or, without git: rsync -av the same files to the server repo dir)
# on the server (pin a GPU; hardware is recorded, not enforced)
cd /data1/paveen/Dopamine && git pull origin main
# 2) read-only mask checks (no generation, no mask is built/changed)
bash run_gsm8k_orthogonal_off_v1.sh verify llama3
bash run_gsm8k_orthogonal_off_v1.sh verify qwen2.5
# 3) technical checks (1 item, then 4 items; isolated dirs)
export CUDA_VISIBLE_DEVICES=0
bash run_gsm8k_orthogonal_off_v1.sh preflight llama3
bash run_gsm8k_orthogonal_off_v1.sh smoke llama3
bash run_gsm8k_orthogonal_off_v1.sh preflight qwen2.5
bash run_gsm8k_orthogonal_off_v1.sh smoke qwen2.5
# 4) formal off-support cells (seed1-3); then the same for qwen2.5
bash run_gsm8k_orthogonal_off_v1.sh full llama3
bash run_gsm8k_orthogonal_off_v1.sh status llama3
bash run_gsm8k_orthogonal_off_v1.sh full qwen2.5
bash run_gsm8k_orthogonal_off_v1.sh status qwen2.5
# 4b) ONLY if the analyzer reports INCOMPARABLE reference (re-runs baseline, rsn, on-support seed1-3 -> .../ref)
bash run_gsm8k_orthogonal_off_v1.sh refs llama3 && bash run_gsm8k_orthogonal_off_v1.sh status llama3 refs
# 5) sync results (analysis box)
cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer && mkdir -p gsm8k_orthogonal_off_v1/{llama3,qwen2.5}
rsync -av "$SERVER:/data1/paveen/Dopamine/components/llama3/gsm8k_orthogonal_off_v1/" ./gsm8k_orthogonal_off_v1/llama3/
rsync -av "$SERVER:/data1/paveen/Dopamine/components/qwen2.5/gsm8k_orthogonal_off_v1/" ./gsm8k_orthogonal_off_v1/qwen2.5/
# 6) offline analysis (needs numpy+scipy). Old section-7 trees as reference:
O=AdarResult
python3 analyze_gsm8k_orthogonal_off_v1.py --model llama3 --off_root ./gsm8k_orthogonal_off_v1/llama3 --ref_root $O/llama3/gsm8k/orthogonal --ref_kind old --out off_v1_llama3.json
python3 analyze_gsm8k_orthogonal_off_v1.py --model qwen2.5 --off_root ./gsm8k_orthogonal_off_v1/qwen2.5 --ref_root $O/qwen2.5/gsm8k/orthogonal --ref_kind old --out off_v1_qwen25.json
python3 combine_gsm8k_orthogonal_off_v1.py --llama off_v1_llama3.json --qwen off_v1_qwen25.json --out off_v1_primary.json
# (rerun references instead: --ref_root ./gsm8k_orthogonal_off_v1/<model>/ref --ref_kind rerun)
# local validator test (needs numpy+torch): python3 test_gsm8k_orthogonal_off.py
