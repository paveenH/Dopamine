
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
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/{qwen2.5,llama3}" \
  /Users/paveenhuang/Downloads/


rsync -avzP \
  "d12922004@140.112.31.182:/data1/paveen/Dopamine/components/qwen2.5" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


---**服务器命令**（先 `git pull`，`LLAMA_GPU` 和 `QWEN_GPU` 换成你空闲的两张不同的卡）：
```bash
cd /data1/paveen/Dopamine && git pull

# 1. 两卡并行：Llama(0,−6) 与 Qwen(0,+8)，每个模型只加载一次，断开终端后继续运行
nohup env LLAMA_GPU=1 QWEN_GPU=2 bash run_proofwriter_owa_zeroshot_cot_4cell.sh > zs_cot_4cell.log 2>&1 &
sleep 10; cat zs_cot_4cell.log zs_cot_llama3_*.log zs_cot_qwen25_*.log   # 确认两个进程都已启动

# 2. 两个进程都结束后（zs_cot_4cell.log 末行出现 done），离线统计，不用 GPU
bash run_proofwriter_owa_zeroshot_cot_4cell_eval.sh
```
