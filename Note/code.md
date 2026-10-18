
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


rsync -avzP \
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/llama3_base" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```

---

## MMLU-E off-support controls (mmlue-orthogonal-off-v1) — NOT YET RUN
```bash
# 0) local -> server code sync (repo files only)
rsync -avzP get_answer_regenerate_logits.py run_mmlue_orthogonal_off_v1.sh \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/

# 1) on server (conda env roleplaying, `python`): read-only mask checks, then smoke, then full
bash run_mmlue_orthogonal_off_v1.sh verify llama3
bash run_mmlue_orthogonal_off_v1.sh verify qwen2.5
CUDA_VISIBLE_DEVICES=0 bash run_mmlue_orthogonal_off_v1.sh smoke llama3     # -> mmlue_orthogonal_off_v1_smoke
CUDA_VISIBLE_DEVICES=1 bash run_mmlue_orthogonal_off_v1.sh smoke qwen2.5
CUDA_VISIBLE_DEVICES=0 nohup bash run_mmlue_orthogonal_off_v1.sh full llama3 > mmlue_off_llama3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_mmlue_orthogonal_off_v1.sh full qwen2.5 > mmlue_off_qwen25.log 2>&1 &
head mmlue_off_llama3.log mmlue_off_qwen25.log
bash run_mmlue_orthogonal_off_v1.sh status llama3     # expect 342

# 2) sync results (local)
for m in llama3 qwen2.5; do
  mkdir -p ~/Documents/RSNResult/RoleAnswer/AdarResult/$m/mmlue/orthogonal_off_v1
  rsync -avzP d12922004@140.112.31.184:/data1/paveen/Dopamine/components/$m/mmlue_orthogonal_off_v1/ \
    ~/Documents/RSNResult/RoleAnswer/AdarResult/$m/mmlue/orthogonal_off_v1/
done

# 3) analysis (local, python3.10)
cd ~/Documents/RSNResult/RoleAnswer
python3.10 analyze_mmlue_orthogonal_off_v1.py --audit_refs
python3.10 analyze_mmlue_orthogonal_off_v1.py
```

