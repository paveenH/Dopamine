
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
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/{qwen2.5,llama3}" \
  /Users/paveenhuang/Downloads/


rsync -avzP d12922004@140.112.31.182:/data1/paveen/Dopamine/components/hidden_states/llama3  /data1/paveen/Dopamine/components/hidden_states/llama3


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/
```

##### GSM8K Base check (gsm8k-base-v0) -- 2026-10-16
# server (GPU not enforced; set CUDA_VISIBLE_DEVICES to the card you want, it is recorded in run_meta)
cd /data1/paveen/Dopamine && git pull origin main && mkdir -p logs
CUDA_VISIBLE_DEVICES=0 bash run_gsm8k_base.sh qwen  check 2>&1 | tee logs/base_check_qwen.log    # --alpha 8
CUDA_VISIBLE_DEVICES=1 bash run_gsm8k_base.sh llama check 2>&1 | tee logs/base_check_llama.log   # --alpha -6
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_base.sh qwen  run > logs/base_qwen.log  2>&1 & echo $! > logs/base_qwen.pid
CUDA_VISIBLE_DEVICES=1 nohup bash run_gsm8k_base.sh llama run > logs/base_llama.log 2>&1 & echo $! > logs/base_llama.pid
sleep 30; head -30 logs/base_qwen.log logs/base_llama.log
ps -p $(cat logs/base_qwen.pid) $(cat logs/base_llama.pid)
grep EXIT_STATUS components/qwen2.5_base/base_run_record_*.txt components/llama3_base/base_run_record_*.txt
cat components/*_base/answer_mdf_gsm8k/*/run_meta_gsm8k_*.json   # fires: qwen 0/1800, llama 0/2700; n_generated 300
# local (after run; python3.10 on the analysis box)
S=/Users/paveenhuang/Documents/RSNResult/RoleAnswer
rsync -avzP d12922004@<host>:/data1/paveen/Dopamine/components/qwen2.5_base/ $S/qwen2.5_base/
rsync -avzP d12922004@<host>:/data1/paveen/Dopamine/components/llama3_base/  $S/llama3_base/
cd $S
python3.10 analyze_gsm8k_base.py --model qwen2.5_base --root qwen2.5_base --dump /tmp/qwen_base_cont.txt
python3.10 analyze_gsm8k_base.py --model llama3_base  --root llama3_base  --dump /tmp/llama_base_cont.txt

