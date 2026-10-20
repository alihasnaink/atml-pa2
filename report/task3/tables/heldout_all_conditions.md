**Held-out evaluation (rl_prompt_pool_eval, identical sampling/seed, cap 768 tokens).**

| Condition | Reward (eff.) | Raw RM | KL/token | KL/seq | Entropy | Len mean ± std | Len median [IQR] | EOS rate | Trunc. rate | Repetition | corr(RM, len) | corr(RM, KL) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT reference | 1.758 ± 1.207 | 1.758 | 0.000 | 0.000 | 0.955 | 275.2 ± 243.8 | 217 [53-447] | 0.930 | 0.070 | 0.014 | 0.360 | n/a |
| GRPO midpoint (start) | 1.764 ± 1.269 | 1.764 | 0.000 | 0.010 | 0.966 | 284.6 ± 249.4 | 232 [54-462] | 0.925 | 0.075 | 0.014 | 0.259 | -0.073 |
| Standard GRPO (20 updates) | 1.690 ± 1.341 | 1.690 | 0.000 | 0.054 | 0.970 | 285.7 ± 254.3 | 238 [52-471] | 0.925 | 0.075 | 0.008 | 0.367 | 0.351 |
| Fork canonical GRPO (8 upd.) | 1.786 ± 1.281 | 1.786 | 0.000 | 0.022 | 0.974 | 279.7 ± 249.3 | 208 [52-480] | 0.920 | 0.080 | 0.016 | 0.357 | 0.119 |
| Fork Dr. GRPO (8 upd.) | 1.738 ± 1.343 | 1.738 | 0.000 | 0.029 | 0.963 | 282.4 ± 248.0 | 229 [46-482] | 0.930 | 0.070 | 0.019 | 0.235 | 0.197 |
