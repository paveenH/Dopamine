
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
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/qwen3_base" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


```bash

cd /data1/paveen/Dopamine && git pull && mkdir -p tmp  
export HUGGINGFACE_HUB_TOKEN=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
CUDA_VISIBLE_DEVICES=3 nohup bash run_gsm8k_base_qwen3_granite.sh qwen3 > tmp/gsm8k_base_qwen3.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup bash run_gsm8k_base_qwen3_granite.sh granite33 > tmp/gsm8k_base_granite33.log 2>&1 &
sleep 10; cat tmp/gsm8k_base_qwen3.log tmp/gsm8k_base_granite33.log

CUDA_VISIBLE_DEVICES=5 nohup bash run_gsm8k_base_qwen3_granite.sh granite33 > tmp/gsm8k_base_granite33.log 2>&1 &
sleep 10; tail -c 400 tmp/gsm8k_base_granite33.log



```
