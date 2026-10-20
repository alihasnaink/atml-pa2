**Matched short forks (8 updates, beta_KL=0.1) for each epsilon. Stability statistic: per-update KL(old||new) on the update's own rollout (mean and max).**

| Condition | Updates | Gen. tokens | Train reward (mean) | Clip frac. (last epoch, mean) | Affected frac. (mean) | Update KL old->new mean | Update KL old->new max | Max |log ratio| | Policy grad-norm CV | Held-out reward | Held-out KL/token | Held-out len | Held-out entropy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fork eps=0.05, beta_KL=0.10 | 8 | 1643 | 1.574 | 0.000 | 0.000 | 0.000 | 0.001 | 0.044 | 1.459 | 1.739 | 0.000 | 278.160 | 0.980 |
| Fork eps=0.20, beta_KL=0.10 | 8 | 1643 | 1.574 | 0.000 | 0.000 | 0.000 | 0.001 | 0.044 | 1.459 | 1.739 | 0.000 | 278.160 | 0.980 |
| Fork eps=0.50, beta_KL=0.10 | 8 | 1643 | 1.574 | 0.000 | 0.000 | 0.000 | 0.001 | 0.044 | 1.459 | 1.739 | 0.000 | 278.160 | 0.980 |
