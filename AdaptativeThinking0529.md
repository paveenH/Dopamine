# AdaptativeThinking — 2026-05-29

---
待驗證的事情：
1. 重新整理近期GSM8K相關的code，重新備份一次code;
2. 重新核對一次
1. 確認新的template的準確率是一致的
2. 確認新的指標，需要重新跑HS + 結果

## 0. Template Update

| | 舊 No-CoT | 舊 CoT |
|---|---|---|
| 標題 | `Solve the following math problem.` | `Solve the following math problem **step by step**.` |
| 格式指示 | `Provide your final numeric answer after '####'.` | （無） |
| 推理提示 | （無） | `Let's think step by step.` |

修正後（對稱）——唯一變量是 `Let's think step by step.` 一行：

```
No-CoT:  Solve the following math problem.
         Question: {context}
         Answer:

CoT:     Solve the following math problem.
         Question: {context}
         Let's think step by step.
         Answer:
```

## 1. Signal Comparation

Phase 1b signal-proxy validation，重跑於對稱模板下。三項驗證：

1. **RSN curve** — expert / non_expert / neutral (No-CoT) 的 EMA trajectory 是否分得開
2. **NMD vs Random mask + other metrics** — 比較 NMD projection 與 random sparse projection 的 role gap（late-tonic gap, Cohen's d）；並算 entropy / top1_prob / margin / info_gain
3. **Multi-role multi-metric** — neutral (CoT & No-CoT) 納入，cross-metric correlation matrix

Roles：`expert` ("an expert") / `non_expert` ("a non expert") / `primary_teacher` ("a primary school teacher") / `neutral`。
Setup：Llama3-8B, GSM8K, 300 samples/condition, greedy bs=1, EMA α=0.95, layer 11–20, NMD mask `nmd_0.5_11_20_8B.npy`。

### 1.1 Baseline accuracy (symmetric template)

| Condition | acc | note |
|---|---|---|
| neutral No-CoT | _TBD_ | new anchor，與舊模板數字不可比 |
| neutral CoT | _TBD_ | |
| expert No-CoT | _TBD_ | |
| non_expert No-CoT | _TBD_ | |
| primary_teacher No-CoT | _TBD_ | |


