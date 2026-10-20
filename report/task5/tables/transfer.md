**Out-of-domain comparison on the fixed SVAMP subset (n=100; same protocol).**

| Policy | Exact acc. | Acc. - SFT [95% CI] | Judge win rate vs SFT (W/T/L) | Strict format | Has #### final | Multiple #### | Trunc. rate | Tokens mean ± std | Tokens correct / wrong |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 0.420 | ref. | 0.5 (ref.) | 0.450 | 0.550 | 0.000 | 0.000 | 163.3 ± 70.9 | 134 / 184 |
| RLVR | 0.460 | +0.040 [+0.000, +0.090] | 0.470 (0.08/0.78/0.14) | 0.470 | 0.580 | 0.000 | 0.000 | 164.4 ± 72.1 | 145 / 181 |
| RLAIF | 0.480 | +0.060 [+0.020, +0.110] | 0.460 (0.07/0.78/0.15) | 0.500 | 0.620 | 0.000 | 0.000 | 164.3 ± 73.0 | 146 / 182 |
