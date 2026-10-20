**Preference/reward metrics of the same policies from Tasks 1-3 (their own held-out sets).**

| Policy | Source | Metric | Value |
|---|---|---|---|
| SFT | - | reference policy | n/a |
| DPO | results/task1_dpo/eval/standard/summary.json | held-out pref. accuracy | 0.633 |
| DPO | results/task1_dpo/eval/standard/summary.json | held-out RM score | 1.390 |
| DPO | results/task1_dpo/eval/standard/summary.json | KL/token | 0.000 |
| PPO | results/task2_ppo/eval/standard/summary.json | held-out raw RM | 1.809 |
| PPO | results/task2_ppo/eval/standard/summary.json | KL/token | 0.000 |
| GRPO | results/task3_grpo/eval/standard/summary.json | held-out raw RM | 1.690 |
| GRPO | results/task3_grpo/eval/standard/summary.json | KL/token | 0.000 |
