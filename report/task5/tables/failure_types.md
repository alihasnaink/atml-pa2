**Failure types (fractions). wrong_final_gold_in_text: the gold number appears in the response but the designated final is wrong (can be incidental for small gold numbers); no_final: no parsable '#### <number>'.**

| Dataset | Policy | correct | wrong_final | wrong_final_gold_in_text | no_final | no_final_truncated |
|---|---|---|---|---|---|---|
| GSM8K (in-domain) | SFT | 0.273 | 0.070 | 0.010 | 0.627 | 0.020 |
| GSM8K (in-domain) | RLVR | 0.293 | 0.063 | 0.007 | 0.613 | 0.023 |
| GSM8K (in-domain) | RLAIF | 0.290 | 0.060 | 0.013 | 0.613 | 0.023 |
| SVAMP (transfer) | SFT | 0.420 | 0.070 | 0.060 | 0.450 | 0.000 |
| SVAMP (transfer) | RLVR | 0.460 | 0.060 | 0.060 | 0.420 | 0.000 |
| SVAMP (transfer) | RLAIF | 0.480 | 0.070 | 0.070 | 0.380 | 0.000 |
