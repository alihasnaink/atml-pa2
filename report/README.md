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
