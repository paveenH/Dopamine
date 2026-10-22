
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
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/qwen2.5" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


```bash
cd /data1/paveen/Dopamine && git pull

# 1. 两卡并行：Llama(0,−6) 与 Qwen(0,+8)，每个模型只加载一次，断开终端后继续运行
nohup env LLAMA_GPU=1 QWEN_GPU=2 bash run_proofwriter_owa_zeroshot_cot_4cell.sh > zs_cot_4cell.log 2>&1 &
sleep 10; cat zs_cot_4cell.log zs_cot_llama3_*.log zs_cot_qwen25_*.log   # 确认两个进程都已启动

# 2. 两个进程都结束后（zs_cot_4cell.log 末行出现 done），离线统计，不用 GPU
bash run_proofwriter_owa_zeroshot_cot_4cell_eval.sh

---

cd /data1/paveen/Dopamine && git pull
CUDA_VISIBLE_DEVICES=0 bash finqa/run_finqa_chat.sh llama3  check
CUDA_VISIBLE_DEVICES=1 bash finqa/run_finqa_chat.sh qwen2.5 check
# smoke（30 题，两剂量）：只看格式、停止行为和注入；不要据准确率改提示或选剂量
CUDA_VISIBLE_DEVICES=0 bash finqa/run_finqa_chat.sh llama3  smoke
CUDA_VISIBLE_DEVICES=1 bash finqa/run_finqa_chat.sh qwen2.5 smoke
# 正式（两卡并行，断线后继续）
CUDA_VISIBLE_DEVICES=0 nohup bash finqa/run_finqa_chat.sh llama3  run > finqa_chat_llama.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash finqa/run_finqa_chat.sh qwen2.5 run > finqa_chat_qwen.log 2>&1 &
# 四格同步回本机或在同一目录结构下后：
DETECTOR_DIR=~/Documents/RSNResult/RoleAnswer bash finqa/run_finqa_chat.sh both eval

```
