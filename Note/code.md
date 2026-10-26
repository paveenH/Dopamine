
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
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/qwen3" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.182:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


```bash
cd /data1/paveen/Dopamine
git pull

ls components/answer/granite33 components/hidden_states/granite33/mmlue 2>&1 | head
nohup env CUDA_VISIBLE_DEVICES=3 \
  bash run_mmlue_granite33_roles.sh > mmlue_granite33_roles.log 2>&1 &
sleep 10; cat mmlue_granite33_roles.log



```
