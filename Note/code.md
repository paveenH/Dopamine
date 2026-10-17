
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


rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/hidden_states/llama3  /data1/paveen/Dopamine/components/hidden_states/llama3


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/



##### GSM8K Llama Base 1-shot (gsm8k-base-1shot-v1) — code prepared 2026-10-17, NOT RUN
# (zero-shot Base cells were already run; the earlier "NOT RUN" status is superseded)
git pull origin main
export CUDA_VISIBLE_DEVICES=0
bash run_gsm8k_base_1shot.sh prep          # downloads + freezes the demo on the server (refuses to overwrite)
bash run_gsm8k_base_1shot.sh prep-check    # optional: re-derive and compare
bash run_gsm8k_base_1shot.sh check         # real prompts: one BOS, final token, gold-leak scan, fires
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_base_1shot.sh run > base_1shot.log 2>&1 &
tail -f base_1shot.log                     # cat it right after launch; EXIT_STATUS line at the end
ls components/llama3_base/answer_mdf_gsm8k_1shot_v1/*/run_meta_gsm8k_*.json

rsync -avzP \
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/llama3_base/answer_mdf_gsm8k_1shot_v1/" \
  /Users/paveenhuang/Documents/RSNResult/RoleAnswer/AdarResult/llama3_base/gsm8k_1shot_v1/answer_mdf_gsm8k_1shot_v1/
rsync -avzP d12922004@140.112.31.184:/data1/paveen/Dopamine/components/benchmark/gsm8k_1shot_demo_v1.json \
  /Users/paveenhuang/Documents/RSNResult/RoleAnswer/AdarResult/llama3_base/gsm8k_1shot_v1/

cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer
python3.10 analyze_gsm8k_base_1shot.py --root AdarResult/llama3_base/gsm8k_1shot_v1 \
  --zeroshot_root AdarResult/llama3_base/gsm8k --dump /tmp/base_1shot_newq.txt
