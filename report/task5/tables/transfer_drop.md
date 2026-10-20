**In-domain vs transfer: drop from the GSM8K metric.**

| Policy | GSM8K acc. | SVAMP acc. | Acc. drop (SVAMP - GSM8K) | Acc. - SFT on SVAMP | Win rate vs SFT GSM8K / SVAMP | Strict format GSM8K / SVAMP | Tokens GSM8K / SVAMP |
|---|---|---|---|---|---|---|---|
| SFT | 0.273 | 0.420 | 0.147 | 0.000 | ref. | 0.323 / 0.450 | 275.1 / 163.3 |
| RLVR | 0.293 | 0.460 | 0.167 | 0.040 | 0.507 / 0.470 | 0.340 / 0.470 | 275.8 / 164.4 |
| RLAIF | 0.290 | 0.480 | 0.190 | 0.060 | 0.503 / 0.460 | 0.330 / 0.500 | 276.6 / 164.3 |
