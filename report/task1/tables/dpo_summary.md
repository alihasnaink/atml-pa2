**Task 1 DPO summary. Standard and length-balanced runs: 1 epoch on all usable pairs; beta forks: short runs on the first 600 usable standard pairs (different budget). Pair metrics on the held-out pairs; RM/KL/length on sampled responses to the held-out prompts.**

| Condition | Train data | Budget | Train pairs | Opt. steps | beta | Held-out DPO loss | Pref. acc. | Margin m | Chosen reward | Rejected reward | KL/token | KL/seq | RM score | Len mean ± std | Len median [IQR] | Trunc. rate | Word-limit compl. | Train min | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT (no DPO) | - | - | - | - | - | n/a | n/a | n/a | n/a | n/a | 0.000 | 0.000 | 1.278 ± 1.145 | 155.2 ± 106.2 | 199 [38-256] | 0.457 | 0.450 | - | - |
| Standard DPO | standard | 1 epoch | 1443 | 91 | 0.100 | 0.674 | 0.633 | 0.439 | -0.051 | -0.095 | 0.000 | 0.037 | 1.390 ± 1.143 | 156.5 ± 104.8 | 206 [41-256] | 0.446 | 0.475 | 22.400 | 6.650 |
| Short fork beta=0.03 | standard | short fork | 600 | 38 | 0.030 | 0.691 | 0.616 | 0.155 | -0.005 | -0.009 | -0.000 | -0.018 | 1.356 ± 1.131 | 155.4 ± 105.5 | 194 [41-256] | 0.450 | 0.500 | 9.300 | 6.650 |
| Short fork beta=0.10 | standard | short fork | 600 | 38 | 0.100 | 0.687 | 0.630 | 0.140 | -0.015 | -0.029 | -0.000 | -0.027 | 1.328 ± 1.115 | 155.6 ± 105.3 | 194 [39-256] | 0.446 | 0.500 | 9.400 | 6.650 |
| Short fork beta=0.30 | standard | short fork | 600 | 38 | 0.300 | 0.676 | 0.630 | 0.140 | -0.035 | -0.077 | -0.000 | -0.027 | 1.363 ± 1.121 | 156.0 ± 105.4 | 215 [39-256] | 0.446 | 0.500 | 9.400 | 6.650 |
| Length-balanced DPO | length-balanced | 1 epoch | 1432 | 90 | 0.100 | 0.675 | 0.626 | 0.487 | -0.133 | -0.181 | 0.002 | 0.375 | 1.436 ± 1.185 | 156.4 ± 104.7 | 186 [41-256] | 0.450 | 0.475 | 16.700 | 6.650 |
