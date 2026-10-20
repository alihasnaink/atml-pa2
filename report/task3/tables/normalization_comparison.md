**Canonical GRPO vs Dr. GRPO normalization (8 matched updates). Gradient statistics use each unmasked completion's own policy-term gradient norm before the update; short/long split at the pooled median (78 tokens).**

| Condition | Held-out raw RM | Held-out KL/token | Held-out len mean | Held-out len median | Held-out trunc. | Held-out entropy | Train len mean | Gen. tokens | Completions (unmasked) | Grad norm short (<= 78 tok) | Grad norm long (> 78 tok) | Long/short grad ratio | Long share of grad norm | corr(len, grad norm) | corr(len, grad norm / |A|) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fork canonical GRPO (8 upd.) | 1.786 | 0.000 | 279.725 | 208.000 | 0.080 | 0.974 | 229.375 | 7340 | 26 | 0.502 | 0.107 | 0.213 | 0.175 | -0.688 | -0.622 |
| Fork Dr. GRPO (8 upd.) | 1.738 | 0.000 | 282.395 | 229.000 | 0.070 | 0.963 | 228.875 | 7324 | 25 | 0.019 | 0.046 | 2.373 | 0.687 | 0.632 | 0.925 |
