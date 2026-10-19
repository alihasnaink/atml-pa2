# Report material

Everything here is generated from saved, machine-readable results. Each number traces to a file in
`results/` and a command in the top-level README.

> Per the course AI-usage policy, the interpretation and prose in the PDF report must be your own.
> This folder holds evidence and pointers only. It does not contain written answers.

## Task 1: DPO (`report/task1/`)

Regenerate with `python -m task1_dpo.make_report --config configs/dpo.yaml` (after `bash task1_dpo/run_all.sh`).

```
report/task1/
  tables/        *.csv (machine readable), *.md (reading), *.tex (paste into the paper)
  figures/       *.png and *.pdf
  qualitative/   automatically selected candidate examples (read, then pick)
  dpo_loss_validation.json   dataset_length_stats.json   length_analysis.json
```

Raw per-example results are in `results/task1_dpo/`:
`eval/<condition>/{summary.json, pairs_heldout.jsonl, pairs_stratified.jsonl, generations_heldout.jsonl, generations_word_limit.jsonl}`
and `train_logs/<run>.jsonl` (per optimizer step), `train_logs/<run>_summary.json` (budget, IDs, time, VRAM).
Conditions: `sft`, `standard`, `beta_0.03`, `beta_0.10`, `beta_0.30`, `length_balanced`.

### Setup facts to state in the report

| Item | Value | Source |
|---|---|---|
| Policy / reference | Qwen2.5-1.5B-Instruct + fresh LoRA (r=8, alpha=16, dropout 0.05, q/v proj); the reference is the same model with the adapter disabled | `configs/base.yaml` |
| Standard DPO | 1 epoch on the standard train file, beta=0.10, lr 2e-5, batch 2 x accum 8 (effective 16), max len 768, grad-clip 1.0, seed 6304 | `configs/dpo.yaml` |
| Beta forks | beta in {0.03, 0.10, 0.30}, first 600 usable standard pairs, otherwise identical (same LoRA init and data order). **Different budget from the standard run.** | `train_logs/beta_*_summary.json` |
| Length-balanced DPO | Same recipe as standard (1 epoch) on the supplied balanced file (500 pairs per stratum) | `train_logs/length_balanced_summary.json` |
| Prompt filter | Pairs whose chat-formatted prompt leaves < 64 of the 768 tokens for the response are dropped, with the same rule for every file. Dropped IDs are saved. | `train_logs/*_summary.json`, `eval/*/summary.json` |
| Fixed defect | `dpo_loss` used `beta*(policy_margin + ref_margin)`; fixed to `beta*(policy_margin - ref_margin)` | `dpo_loss_validation.json` |
| Pair metrics | Summed response-token log-probs (teacher forcing); accuracy = fraction with m > 0; held-out loss at each condition's own beta | `task1_dpo/evaluate.py` |
| Decoding | Every condition uses the same held-out prompts, temperature 0.7, top-p 0.9, max 256 new tokens and seed 6304; the word-limit prompts get 4 samples each | `eval/*/summary.json` -> `decoding` |
| KL | Course sampled-response estimator log pi - log pi_ref on each policy's own samples. Main figure: token-averaged (`KL/token`); per-sequence sum also given (`KL/seq`) | `summary.json` -> `heldout_generations` |
| Length | Generated tokens incl. EOS, excluding padding: mean, std, median, IQR, truncation rate | same |
| Length-stratum rule (for describing the standard data) | \|Δ\|/max <= 0.10 counts as matched; Δ/max >= 1/6 (ratio >= 1.2) counts as clearly longer; anything else is "intermediate". Agreement with the supplied labels is in `dataset_length_stats.json` | `task1_dpo/analyze_length.py` |

### Required evidence → files

| Required evidence | Files |
|---|---|
| DPO summary (standard + 3 beta forks: loss/acc, KL, RM score, length stats, budgets) | `tables/dpo_summary.*`, `figures/beta_sweep.*`, `figures/training_curves.*` |
| Length-confounding diagnostics (per-stratum accuracy, standard vs balanced; generated length; word-limit compliance) | `tables/length_strata.*`, `figures/length_strata_accuracy.*`, `tables/word_limit.*`, `tables/word_limit_per_prompt.*`, `tables/paired_generation_differences.*`, `figures/generated_length_and_word_limit.*` |
| Qualitative: preference/reward vs quality disagreement | `qualitative/reward_vs_quality_candidates.md`, `qualitative/beta_examples.md` |
| Qualitative: length bias / instruction compliance | `qualitative/word_limit_examples.md`, `qualitative/length_pair_disagreements.md` |
| Implementation validation | `dpo_loss_validation.json`. The first logged training step should show loss ≈ log 2 (`results/task1_dpo/train_logs/standard.jsonl`) |

### Research questions → evidence to examine

**RQ1: How does beta change preference fitting, KL and reward-model score? Is the relationship monotonic?**
`tables/dpo_summary.*` rows `Short fork beta=*` (Pref. acc., Held-out DPO loss, Margin m, KL/token, KL/seq, RM score, length);
`figures/beta_sweep.*`; train-side dynamics in `figures/training_curves.*`. Compare only the forks with each other
(same budget). The standard run is a different budget and is drawn as a separate marker.

**RQ2: How much of the length behavior comes from the dataset itself? Does balancing change which examples are learned or how long the model responds?**
Data side: `tables/dataset_length_stats.*` (fraction chosen-longer, length difference, score-gap/length correlation, stratum composition).
Policy side: `tables/length_strata.*` (per-stratum accuracy, paired CI, overlap: both / only standard / only balanced),
`figures/margin_vs_length_difference.*` (margin vs length-difference correlation),
`tables/heldout_accuracy_by_length_rule.*`, `tables/paired_generation_differences.*`,
`tables/word_limit.*`, `figures/generated_length_and_word_limit.*`. Also `raw_likelihood_accuracy` in `eval/*/summary.json`.

**RQ3: Cases where the aligned policy has a stronger preference/reward signal but a worse response.**
Read `qualitative/reward_vs_quality_candidates.md` (A: higher RM score but much longer/truncated than SFT;
B: word-limit violations that out-score a compliant SFT answer) and `qualitative/length_pair_disagreements.md`.
Verify each case by hand before using it. Look up the full text by `prompt_id` in `results/task1_dpo/eval/*/generations_*.jsonl`.

## Task 2: PPO (`report/task2/`)

Regenerate with `python -m task2_ppo.make_report --config configs/ppo.yaml` (after `bash task2_ppo/run_all.sh`).
Raw results are in `results/task2_ppo/`:
- `train_logs/<run>.jsonl`: one record per update, including per-epoch details and the sampled response.
- `train_logs/<run>_summary.json`: prompt IDs, generated tokens, wall-clock time, peak VRAM and hyperparameters.
- `eval/<condition>/{summary.json, generations.jsonl}`
- `clipping_cached.json`
- `ppo_validation.json`

Conditions: `sft`, `midpoint`, `standard`, `fork_eps{0.05,0.20,0.50}_kl0.10`, `fork_eps0.20_kl{0.00,0.20}`.
`fork_eps0.20_kl0.10` is shared by the clipping and KL studies.

### Setup facts to state in the report

| Item | Value | Source |
|---|---|---|
| Start state | Supplied `ppo_midpoint_policy` LoRA and `ppo_midpoint_value` critic (Qwen2.5-0.5B, plus a fresh zero-init LoRA and a trainable head), restored identically for every run | `configs/ppo.yaml`, `common/models.py` |
| Reference / reward | Base Qwen2.5-1.5B-Instruct (policy adapter disabled); frozen 8-bit `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback` | `configs/base.yaml` |
| Standard run | 20 updates, 1 prompt per update, 2 PPO epochs, eps=0.20, beta_KL=0.10, gamma=1, lambda=0.95, policy lr 3e-6, critic LoRA/head lr 1e-4/3e-4, grad-clip 1.0, max response 512 | `train_logs/standard_summary.json` |
| Forks | 8 updates from the same midpoint; same prompt sequence and per-update sampling seed; only eps or beta_KL changes | `task2_ppo/forks.py` |
| Reward | Terminal learned reward, minus 1.0 when the response hits the cap without EOS (the release `missing_eos_penalty`, which matches the cached `effective_terminal_reward`). Token rewards are r_t = -beta_KL (log pi - log pi_ref) plus the terminal reward | `continue_train.py` |
| Advantages | GAE on critic values with V = 0 after the last token, whitened over the update's valid tokens | `ppo.py`, `continue_train.py` |
| Prompts | Training uses train-pool prompts of at most 256 chat tokens in a seeded fixed order (dropped IDs are saved). Held-out evaluation uses all 200 eval prompts, untruncated, at temperature 0.7 / top-p 0.9, seed 6304, max 768 tokens | `train_logs/*_summary.json`, `eval/*/summary.json` |
| Precision | Policy base in fp16 with fp32 LoRA and loss scaling; critic in fp32; dropout disabled during PPO, so epoch 1 starts at ratio = 1 | `configs/ppo.yaml` |
| Fixed defect | `ppo_policy_loss` used `torch.maximum`; fixed to `torch.minimum` (gradient-geometry table in `ppo_validation.json`) | `ppo_validation.json` |
| Clip vs affected fraction | Clip fraction: rho outside [1-eps, 1+eps]. Affected fraction: clipping binds (A>0 and rho>1+eps, or A<0 and rho<1-eps), so the token's gradient is zero. Both use the same response mask | `ppo.py::clip_diagnostics` |
| Stability statistic | Per-update KL(old \|\| new) on the update's own rollout (mean and max over updates); also the max \|log ratio\| and the coefficient of variation of the policy grad norm | `tables/clipping_forks.*` |

### Required evidence → files

| Required evidence | Files |
|---|---|
| Standard continuation diagnostics (reward, KL, policy/value loss, entropy, clip fraction, grad norm, length; peak VRAM, wall-clock time) | `figures/standard_continuation_diagnostics.*`, `tables/standard_continuation_per_update.*`, `tables/standard_continuation_compute.*`. Critic behavior is the `explained_variance` column |
| Clipping study: cached clip/affected fraction per eps | `tables/clipping_cached_batch.*`, `figures/clipping_cached_batch.*`, `clipping_cached.json`. Check `tables/cached_batch_reconstruction.*` first |
| Clipping study: matched forks (held-out reward, KL, length, stability statistic) | `tables/clipping_forks.*`, `figures/clipping_forks_trajectories.*` |
| KL-pressure study: held-out reward, KL, entropy, length per beta_KL | `tables/kl_forks.*`, `figures/kl_forks_trajectories.*`, `figures/kl_forks_paired_differences.*`, `tables/kl_forks_per_update.csv` |
| Qualitative: reward and quality agree / disagree | `qualitative/reward_vs_quality_candidates.md`, `qualitative/kl_fork_examples.md`, `qualitative/training_rollouts.md` |
| Implementation validation | `ppo_validation.json`, `tables/cached_batch_reconstruction.*` |

### Research questions → evidence to examine

**RQ1: How does eps change the fraction of updates constrained by clipping, and the stability of the short continuation?**
Use the step-0 and after-step columns of `tables/clipping_cached_batch.*` and the eps curve in `figures/clipping_cached_batch.*`.
For the forks, use `tables/clipping_forks.*` (clip/affected fraction, update KL mean and max, max |log ratio|, grad-norm CV) and
`figures/clipping_forks_trajectories.*`. If the step-0 ratios are close to 1, compare the reconstruction MAE
(`mae_midpoint_vs_cached_old`) with the band widths before interpreting the step-0 fractions.

**RQ2: When KL pressure is weakened, which observable changes first?**
Look at the per-update trajectories (`figures/kl_forks_trajectories.*`) and the paired per-update differences against the
beta_KL=0.20 fork (`figures/kl_forks_paired_differences.*`, `tables/kl_forks_per_update.csv`). The forks share the prompt and
sampling seed at each update, so the differences are paired. Then check the held-out confirmation in `tables/kl_forks.*`.

**RQ3: How informative is the learned reward as the policy moves away from the reference? Is there evidence for or against overoptimization?**
Use `figures/reward_vs_kl.*` (condition means plus per-response scatter), `tables/reward_by_kl_quartile.*`,
`tables/paired_vs_midpoint.*` (reward change against length, truncation and repetition change, with CIs),
`corr(RM, len)` and `corr(RM, KL)` in `tables/heldout_all_conditions.*`, and the agree/disagree candidates. Verify each candidate by hand.
