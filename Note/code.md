
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
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/llama3" \
  /Users/paveenhuang/Downloads/


rsync -avzh --partial --info=progress2 \
  --exclude '/hidden_states' \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/components/ \
  /data1/paveen/Dopamine/components/

```


---

##### Chat-CoT Saturation Check (chat-cot-sat-v1): 8 cells, 代码准备完成, 尚无真实结果

```bash
cd /data1/paveen/Dopamine

# 卡 0：Llama（先 GSM-Hard，再 GSM8K）
CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  bash run_chat_cot_sat.sh llama3 gsm_hard run > ccs_llama3_gsm_hard.log 2>&1
  bash run_chat_cot_sat.sh llama3 gsm8k   run > ccs_llama3_gsm8k.log   2>&1
' > ccs_gpu0.log 2>&1 &

# 卡 1：Qwen（先 GSM8K，再 GSM-Hard）
CUDA_VISIBLE_DEVICES=3 nohup bash -c '
  bash run_chat_cot_sat.sh qwen2.5 gsm8k    run > ccs_qwen_gsm8k.log     2>&1
  bash run_chat_cot_sat.sh qwen2.5 gsm_hard run > ccs_qwen_gsm_hard.log  2>&1
' > ccs_gpu1.log 2>&1 &

sleep 10; head -5 ccs_llama3_gsm_hard.log 
sleep 10; head -5 ccs_qwen_gsm8k.log

# 5. 状态 / 日志检查
for M in llama3 qwen2.5; do for T in gsm8k gsm_hard; do bash run_chat_cot_sat.sh $M $T status; done; done
grep -n "FATAL\|steering_fires\|wrote\|\[tokens\]" ccs_*.log

# 6. 本机: 同步结果 (目标路径 = 分析脚本默认读取位置)
R=/Users/paveenhuang/Documents/RSNResult/RoleAnswer/AdarResult
for M in llama3 qwen2.5; do for T in gsm8k gsm_hard; do
  mkdir -p $R/$M/$T/chat_cot_sat_v1
  rsync -avzP d12922004@140.112.31.184:/data1/paveen/Dopamine/components/$M/${T}_chat_cot_sat_v1/ $R/$M/$T/chat_cot_sat_v1/
done; done

# 7. 本机: 离线分析 (任一 provenance 检查失败则不写输出; 不覆盖已有输出)
cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer
python3 analyze_chat_cot_sat_v1.py            # -> chat_cot_sat_v1_result.{json,csv,md}
```
