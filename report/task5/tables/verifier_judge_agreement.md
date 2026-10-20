**Verifier-judge agreement on policy responses to the same prompt. 'Verifier separates' = exactly one response is correct; then the judge agrees, ties, or prefers the wrong one. When both are correct/wrong the verifier ties, and the judge may still break the tie.**

| Dataset | Pair | Pairs | Verifier separates | Judge agrees | Judge ties | Judge prefers wrong | Both correct (n) / judge breaks tie | Both wrong (n) / judge breaks tie |
|---|---|---|---|---|---|---|---|---|
| GSM8K (in-domain) | all pairs | 900 | 28 | 0.107 | 0.893 | 0.000 | 243 / 0.004 | 629 / 0.041 |
| GSM8K (in-domain) | SFT vs RLVR | 300 | 12 | 0.167 | 0.833 | 0.000 | 79 / 0.000 | 209 / 0.048 |
| GSM8K (in-domain) | SFT vs RLAIF | 300 | 5 | 0.200 | 0.800 | 0.000 | 82 / 0.000 | 213 / 0.033 |
| GSM8K (in-domain) | RLVR vs RLAIF | 300 | 11 | 0.000 | 1.000 | 0.000 | 82 / 0.012 | 207 / 0.043 |
| SVAMP (transfer) | all pairs | 300 | 14 | 0.143 | 0.571 | 0.286 | 129 / 0.178 | 157 / 0.236 |
| SVAMP (transfer) | SFT vs RLVR | 100 | 6 | 0.167 | 0.500 | 0.333 | 41 / 0.171 | 53 / 0.226 |
| SVAMP (transfer) | SFT vs RLAIF | 100 | 6 | 0.000 | 0.667 | 0.333 | 42 / 0.167 | 52 / 0.250 |
| SVAMP (transfer) | RLVR vs RLAIF | 100 | 2 | 0.500 | 0.500 | 0.000 | 46 / 0.196 | 52 / 0.231 |
