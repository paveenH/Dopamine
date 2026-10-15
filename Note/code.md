
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
