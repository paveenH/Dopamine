
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

```

**6. 回到本机:同步结果并分析**(`$SERVER` 换成你的 ssh 主机别名)
```bash
cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer
mkdir -p gsm8k_orthogonal_off_v1/{llama3,qwen2.5}
rsync -av "$SERVER:/data1/paveen/Dopamine/components/llama3/gsm8k_orthogonal_off_v1/" ./gsm8k_orthogonal_off_v1/llama3/
rsync -av "$SERVER:/data1/paveen/Dopamine/components/qwen2.5/gsm8k_orthogonal_off_v1/" ./gsm8k_orthogonal_off_v1/qwen2.5/
O=AdarResult
python3 analyze_gsm8k_orthogonal_off_v1.py --model llama3 --off_root ./gsm8k_orthogonal_off_v1/llama3 --ref_root $O/llama3/gsm8k/orthogonal --ref_kind old --out off_v1_llama3.json
python3 analyze_gsm8k_orthogonal_off_v1.py --model qwen2.5 --off_root ./gsm8k_orthogonal_off_v1/qwen2.5 --ref_root $O/qwen2.5/gsm8k/orthogonal --ref_kind old --out off_v1_qwen25.json
python3 combine_gsm8k_orthogonal_off_v1.py --llama off_v1_llama3.json --qwen off_v1_qwen25.json --out off_v1_primary.json
```
分析需要本机有 numpy 和 scipy(没有就先装)。

**7. 仅当分析器报 `INCOMPARABLE reference` 时**:回服务器补跑参照(baseline、RSN 和三个 on-support seed),然后同步 `…/ref/` 并用 `--ref_kind rerun` 重新分析。
```bash
bash run_gsm8k_orthogonal_off_v1.sh refs llama3 && bash run_gsm8k_orthogonal_off_v1.sh status llama3 refs
bash run_gsm8k_orthogonal_off_v1.sh refs qwen2.5 && bash run_gsm8k_orthogonal_off_v1.sh status qwen2.5 refs
```

注意:
- 这是第一次真实运行新增的 runner 检查(BOS、尾 token、RSN hash)。`preflight` 若在加载模型后报错,把报错原文发给我,不要绕过。
- 后台命令启动后一定要立刻看日志。