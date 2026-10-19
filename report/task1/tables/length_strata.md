**Standard vs length-balanced DPO on the length-stratified held-out set (paired bootstrap 95% CI).**

| Model | Stratum | n | Pref. acc. | Margin m | DPO loss | Chosen reward | Rejected reward |
|---|---|---|---|---|---|---|---|
| Standard DPO | Preferred longer | 76 | 0.461 | -0.068 | 0.699 | -0.100 | -0.093 |
| Standard DPO | Length matched | 80 | 0.575 | 0.186 | 0.686 | -0.053 | -0.072 |
| Standard DPO | Rejected longer | 79 | 0.658 | 0.640 | 0.665 | -0.063 | -0.127 |
| Length-balanced DPO | Preferred longer | 76 | 0.434 | -0.459 | 0.721 | -0.207 | -0.161 |
| Length-balanced DPO | Length matched | 80 | 0.587 | 0.337 | 0.679 | -0.109 | -0.143 |
| Length-balanced DPO | Rejected longer | 79 | 0.734 | 1.425 | 0.630 | -0.113 | -0.256 |
| Balanced - Standard (paired) | Preferred longer | 76 | -0.026 [-0.092, +0.026] | r=0.91 | both 0.41 | only std 0.05 | only bal 0.03 |
| Balanced - Standard (paired) | Length matched | 80 | +0.013 [-0.062, +0.087] | r=0.90 | both 0.53 | only std 0.05 | only bal 0.06 |
| Balanced - Standard (paired) | Rejected longer | 79 | +0.076 [+0.000, +0.165] | r=0.95 | both 0.63 | only std 0.03 | only bal 0.10 |
