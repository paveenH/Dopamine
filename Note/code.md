
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


---

##### Role Prompt–RSN correspondence (gsm8k-role-correspondence-v1, Qwen2.5, α=0, 3 roles)

```bash
# 1. 本机 -> 服务器同步代码 (已有 mask / benchmark / gsm8k_test_sample.json 不动)
cd /Users/paveenhuang/Downloads/Dopamine
rsync -avzP get_answer_regenerate_gsm8k.py verify_role_correspondence_cell.py \
  run_gsm8k_role_correspondence_qwen25.sh \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/

# 2. 服务器: 技术检查 (不写数据)
cd /data1/paveen/Dopamine
bash run_gsm8k_role_correspondence_qwen25.sh --check

# 3. 服务器: 正式运行 (一次加载, 三角色, 900 条; 已存在的 cell 仅在校验通过时跳过)
CUDA_VISIBLE_DEVICES=0 nohup bash run_gsm8k_role_correspondence_qwen25.sh --run \
  > role_corr_qwen25.log 2>&1 &
sleep 20; cat role_corr_qwen25.log      # 立刻确认已启动 (解释器/路径错误会静默死)

# 4. 日志检查: 应见 [prompt-audit] 三行 (final token 单一, 解码为空格), 末尾 "finished + verified"
grep -n "prompt-audit\|FATAL\|steering_fires\|run_meta\|verified" role_corr_qwen25.log
bash run_gsm8k_role_correspondence_qwen25.sh --verify

# 5. 本机: 同步结果
rsync -avzP \
  "d12922004@140.112.31.184:/data1/paveen/Dopamine/components/qwen2.5/answer_gsm8k_role_correspondence_v1/" \
  /Users/paveenhuang/Documents/RSNResult/RoleAnswer/AdarResult/qwen2.5/gsm8k/role_correspondence_v1/

rsync -avzP d12922004@140.112.31.184:/data1/paveen/Dopamine/components/benchmark/gsm8k_test_sample.json \
  /Users/paveenhuang/Documents/RSNResult/RoleAnswer/benchmark/

# 6. 本机: 离线分析 (Llama 已有结果 + Qwen 新 cell; 输出不覆盖)
cd /Users/paveenhuang/Documents/RSNResult/RoleAnswer
python3 analyze_role_correspondence_v1.py --test_file benchmark/gsm8k_test_sample.json
#   -> role_correspondence_v1_result.{json,md}
```

状态: 代码准备完成 / Llama 离线重算完成 / **Qwen 尚未运行**。
