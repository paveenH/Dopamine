
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
# 0. 只加载 tokenizer，不需要 GPU：看提示和注入位点，两个模型都跑
bash run_proofwriter_owa_zeroshot.sh check llama3
bash run_proofwriter_owa_zeroshot.sh check qwen2.5

# 1. No-CoT α=0 两格（格式门槛）
CUDA_VISIBLE_DEVICES=0 nohup bash run_proofwriter_owa_zeroshot.sh gate llama3  > zs_gate_llama3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_proofwriter_owa_zeroshot.sh gate qwen2.5 > zs_gate_qwen.log 2>&1 &
# 启动后先 cat 日志确认进程活着
bash run_proofwriter_owa_zeroshot.sh gate-eval llama3     # 不读 gold，写 gate JSON；没过 exit 3
bash run_proofwriter_owa_zeroshot.sh gate-eval qwen2.5

# 2. 哪个模型过了门槛，才跑它自己的工作点（没过会拒绝）
CUDA_VISIBLE_DEVICES=0 nohup bash run_proofwriter_owa_zeroshot.sh workpoint llama3  > zs_wp_llama3.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_proofwriter_owa_zeroshot.sh workpoint qwen2.5 > zs_wp_qwen.log 2>&1 &

# 3. 四格齐了再评分（读 gold，Holm m=2）
bash run_proofwriter_owa_zeroshot.sh eval

# 后续阶段，不会自动启动：同样的命令末尾加 cot，例如
#   bash run_proofwriter_owa_zeroshot.sh gate llama3 cot

```
