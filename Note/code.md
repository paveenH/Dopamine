
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


---

##### Chat-CoT Saturation Check (chat-cot-sat-v1): 8 cells, 代码准备完成, 尚无真实结果

```bash
# 1. 本机 -> 服务器: 同步代码
cd /Users/paveenhuang/Downloads/Dopamine
rsync -avzP chat_cot_sat_lib.py get_answer_chat_cot_sat.py run_chat_cot_sat.sh \
  d12922004@140.112.31.184:/data1/paveen/Dopamine/

# 2. 服务器: 技术检查 (加载模型, 校验全部 prompt/token/mask/已有 cell; 不写任何文件). 4 个组合各一次
cd /data1/paveen/Dopamine
for M in llama3 qwen2.5; do for T in gsm8k gsm_hard; do
  CUDA_VISIBLE_DEVICES=0 bash run_chat_cot_sat.sh $M $T check 2>&1 | tail -8
done; done

# 3. 服务器: smoke (8 题 x 2 alpha -> *_smoke 目录; 验证 fires 与生成; 不会串联正式运行)
CUDA_VISIBLE_DEVICES=0 bash run_chat_cot_sat.sh llama3 gsm_hard smoke

# 4. 服务器: 正式运行 (每个模型x任务一次加载, 先 alpha=0 再工作点). 可在不同卡/机器并行
CUDA_VISIBLE_DEVICES=0 nohup bash run_chat_cot_sat.sh llama3 gsm_hard run > ccs_llama3_gsm_hard.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup bash run_chat_cot_sat.sh llama3 gsm8k   run > ccs_llama3_gsm8k.log   2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup bash run_chat_cot_sat.sh qwen2.5 gsm8k  run > ccs_qwen_gsm8k.log     2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash run_chat_cot_sat.sh qwen2.5 gsm_hard run > ccs_qwen_gsm_hard.log 2>&1 &
sleep 30; head -5 ccs_*.log                       # 立刻确认已启动

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

状态: 代码准备完成 / 尚无真实结果。
