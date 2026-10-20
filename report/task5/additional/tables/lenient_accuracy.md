**Cumulative accuracy when the answer is also extracted from looser formats (tier 1 = the official exact verifier; each later column adds one fallback). A gap between tier 1 and tier 4 means formatting failures, not wrong maths. Supplementary to the official metric.**

| Dataset | Policy | 1 strict #### (official) | 2 + loose #### | 3 + \boxed{} | 4 + last-line number | No number found |
|---|---|---|---|---|---|---|
| GSM8K | SFT | 0.273 | 0.357 | 0.430 | 0.673 | 6 |
| GSM8K | RLVR | 0.293 | 0.360 | 0.437 | 0.673 | 5 |
| GSM8K | RLAIF | 0.290 | 0.363 | 0.443 | 0.687 | 4 |
| SVAMP | SFT | 0.420 | 0.450 | 0.490 | 0.780 | 0 |
| SVAMP | RLVR | 0.460 | 0.480 | 0.530 | 0.790 | 0 |
| SVAMP | RLAIF | 0.480 | 0.490 | 0.530 | 0.780 | 0 |
