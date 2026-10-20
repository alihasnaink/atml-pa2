**In-domain comparison on the fixed GSM8K subset (n=300; greedy decoding, max 512 new tokens; judge Qwen/Qwen2.5-3B-Instruct). Win rate = (wins + 0.5 ties)/n against SFT on the same prompt; strict format = response ends with '#### <number>'.**

| Policy | Exact acc. | Acc. - SFT [95% CI] | Judge win rate vs SFT (W/T/L) | Strict format | Has #### final | Multiple #### | Trunc. rate | Tokens mean ± std | Tokens correct / wrong |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 0.273 | ref. | 0.5 (ref.) | 0.323 | 0.353 | 0.003 | 0.020 | 275.1 ± 86.4 | 221 / 296 |
| RLVR | 0.293 | +0.020 [+0.000, +0.043] | 0.507 (0.03/0.96/0.01) | 0.340 | 0.363 | 0.000 | 0.023 | 275.8 ± 87.9 | 221 / 299 |
| RLAIF | 0.290 | +0.017 [+0.003, +0.033] | 0.503 (0.02/0.97/0.01) | 0.330 | 0.363 | 0.000 | 0.023 | 276.6 ± 87.1 | 226 / 297 |
