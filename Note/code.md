
```text
#### Tmux
conda activate dopamine26
conda activate roleplaying
conda activate RolePlaying
conda deactivate

conda env list
ls /data1/paveen/conda_envs

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
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/{qwen2.5,llama3}" \
  /Users/paveenhuang/Downloads/


rsync -avzP \
  "d12922004@140.112.31.185:/data1/paveen/Dopamine/components/olmo3" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


```bash
cd /data1/paveen/Dopamine && git pull
 
nohup env CUDA_VISIBLE_DEVICES=6 WORKPOINT="8 10" \
  bash run_gsm8k_qwen3_bare_budget2048.sh \
  > gsm8k_qwen3_budget2048_a8_a10.log 2>&1 &
sleep 10; cat gsm8k_qwen3_budget2048_a8_a10.log

```
