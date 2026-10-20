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

## Task 3: GRPO (`report/task3/`)

Regenerate with `python -m task3_grpo.make_report --config configs/grpo.yaml` (after `bash task3_grpo/run_all.sh`).
Raw results are in `results/task3_grpo/`:
- `train_logs/<run>.jsonl`: one record per update, with the K rewards, advantages, lengths, truncation and in-loss flags, per-completion gradient norms and responses.
- `train_logs/<run>_summary.json`: prompt IDs, generated tokens, wall-clock time, peak VRAM and hyperparameters.
- `eval/<condition>/{summary.json, generations.jsonl}`
- `group_size.json`
- `grpo_validation.json`

Conditions: `sft`, `midpoint`, `standard`, `fork_grpo`, `fork_dr_grpo`.

### Setup facts to state in the report

| Item | Value | Source |
|---|---|---|
| Start state | Supplied `grpo_midpoint_policy` LoRA (50 TRL steps, K=4, max length 512 per `checkpoints/grpo_midpoint_metadata.json`), restored identically for every run | `configs/grpo.yaml` |
| Standard run | 20 updates, 1 prompt per update, K=4 completions, 1 policy epoch, lr 5e-6, eps=0.20, beta=0.10 (k3 KL term in the loss), max completion 512, truncated completions masked from the loss, grad-clip 1.0 | `train_logs/standard_summary.json` |
| Reward / advantages | Raw learned reward (`yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`, 8-bit); A_k = (r_k - mean_group) / (std_group + 1e-6), computed per prompt group. Truncated completions stay in the group statistics but contribute no loss tokens | `grpo.py`, `continue_train.py` |
| Normalization forks | 8 updates each from the same midpoint; same prompts, sampling seeds, K, reward, beta, eps and cap. `grpo`: sequence weight 1/T_k (token weight 1/(K T_k)). `dr_grpo`: constant 1/512 (token weight 1/(K*512)). Advantages are unchanged | `compare_normalization.py`, `grpo_validation.json` |
| Length-conditioned gradient statistic | Before each fork update, the policy-term gradient norm of every completion on its own (its share of the batch loss), against its length. Reported as short-vs-long mean grad norm (split at the pooled median), the long share of the total, and corr(length, grad norm), also per unit of advantage | `tables/normalization_comparison.*` |
| Group-size regrouping (equal generations) | Each prompt's 8 cached completions, ordered by `generation_index`, are split into 8/K consecutive groups. All K use the same 192 completions of 24 prompts. The all-subsets average is a robustness check | `group_size.json` -> `regrouping_rule` |
| Difficulty bins (defined once) | Tertiles of the prompt's mean reward over its 8 cached completions (lowest third = hard) | `group_size.json` -> `difficulty_rule`, `bins` |
| Informative group | Within-group reward std > 1e-6 (tolerance of the released helper). For a continuous reward this is rarely violated, so near-zero rates (std < 0.1 / 0.25 / 0.5) are reported as a sensitivity check | `tables/group_size_*` |
| Prompts / evaluation | Same rules as Task 2: training uses train prompts of at most 256 chat tokens in a seeded order; held-out uses all 200 eval prompts at temperature 0.7 / top-p 0.9, seed 6304, max 768 tokens | `eval/*/summary.json` |
| Precision | fp16 base with fp32 LoRA and loss scaling; dropout set to 0, so the single policy epoch starts at ratio exactly 1 | `continue_train.py` |
| Fixed defect | `group_relative_advantages` used batch-level mean/std; fixed to per-group. With the bug, a zero-variance group got non-zero advantages, and shifting one prompt's rewards changed every advantage | `grpo_validation.json` |

### Required evidence → files

| Required evidence | Files |
|---|---|
| Standard GRPO continuation diagnostics (reward, KL, within-group std, uninformative fraction, policy loss, grad norm, entropy, length; peak VRAM, wall-clock time) | `figures/standard_continuation_diagnostics.*`, `tables/standard_continuation_per_update.*`, `tables/standard_continuation_compute.*` |
| Equal-generation group-size study (informative rate, mean within-group std, variance of the relative signal, overall and per difficulty bin; binning rule) | `tables/group_size_equal_generations.*`, `tables/group_size_by_difficulty.*`, `figures/group_size_by_difficulty.*`, `figures/prompt_difficulty_vs_std.*` |
| Canonical vs Dr. GRPO (held-out reward, KL, length, and a length-conditioned gradient statistic) | `tables/normalization_comparison.*`, `figures/normalization_comparison.*`, `tables/heldout_all_conditions.*` |
| Qualitative examples (normalization / group informativeness) | `qualitative/group_informativeness_examples.md`, `qualitative/training_group_examples.md`, `qualitative/normalization_examples.md` |
| Implementation validation | `grpo_validation.json` |

### Research questions → evidence to examine

**RQ1: How does group size change the probability of a useful relative signal, and which difficulty regimes benefit most from larger K?**
Use `tables/group_size_equal_generations.*` (informative rate, mean std, Var(r - group mean), near-zero-std fractions, all-subsets columns)
and `tables/group_size_by_difficulty.*` / `figures/group_size_by_difficulty.*` (the same metrics for the hard, medium and easy bins).
`figures/prompt_difficulty_vs_std.*` shows how reward spread depends on difficulty.

**RQ2: Does sequence normalization change response length or how gradient magnitude is spread across short and long completions? Is that associated with quality?**
Use the gradient columns of `tables/normalization_comparison.*` and the scatter in `figures/normalization_comparison.*`
(grad norm vs length); the per-token weight tables in `grpo_validation.json` show the mechanism. For length,
use the train/held-out length columns and the held-out boxplot. For quality, use the held-out raw RM and paired differences
(`tables/paired_vs_midpoint.*`), plus `qualitative/normalization_examples.md`.

**RQ3: With the critic removed, what is the main source of instability or sample inefficiency in the observed run?**
Use `tables/signal_and_sample_efficiency.*` (uninformative groups, masked truncated completions, zero-advantage completions,
tokens in loss / generated, group-std range, grad-norm CV, corr(group std, grad norm)), the reward-dot and masked-count panels in
`figures/standard_continuation_diagnostics.*`, and `qualitative/training_group_examples.md`.

## Task 4: Safety calibration (`report/task4/`)

Produced by `python -m task4_safety.evaluate_safety --config configs/feedback.yaml`. Run it after the Kaggle GPU stage,
and run it again after labeling the manual audit sheet; the agreement tables and figures appear only once labels exist.
Raw results are in `results/task4_safety/`:
- `generated_<policy>.jsonl`: response, tokens, truncation and reward-model score.
- `judged_<policy>.jsonl`: plus judge label, confidence and rationale tag.
- `manual_audit_{ids,sheet,key}.csv` and `manual_audit_guide.md`
- `safety_metrics.json`

### Setup facts to state in the report

| Item | Value | Source |
|---|---|---|
| Policies (fixed by the manual) | SFT = untouched Qwen2.5-1.5B-Instruct; DPO = Task 1 Step 1 standard; PPO = Task 2 Step 1 standard (20 updates); GRPO = Task 3 Step 1 standard (20 updates). No ablation checkpoints | `configs/feedback.yaml` -> `policies` |
| Prompts | Fixed XSTest CSV: 450 prompts = 250 SAFE (10 types x 25) + 200 UNSAFE (8 contrast types x 25), fixed order | `data/xstest_safety_prompts.csv` |
| Decoding | Greedy (deterministic), one response per policy and prompt, max 256 new tokens, the same for all policies | `generate_responses.py`, `safety_max_new_tokens` |
| AI judge | Fixed Qwen2.5-3B-Instruct (4-bit), greedy, the supplied prompt and parser unchanged. Unparseable outputs count as AMBIGUOUS (`parse_failures` column). Confidence is used only for auditing | `judge_responses.py`, `manifests/source_v2_task4_safety_judge_spec.json` |
| Rates | safe-answer = P(SAFE_ANSWER \| SAFE); over-refusal = P(OVER_REFUSAL \| SAFE); unsafe-compliance = P(UNSAFE_COMPLIANCE \| UNSAFE); justified-refusal = P(JUSTIFIED_REFUSAL \| UNSAFE); ambiguous = P(AMBIGUOUS) overall and per class. "Class-inconsistent" = a label that contradicts the XSTest class (e.g. JUSTIFIED_REFUSAL on a SAFE prompt) | judge spec, `evaluate_safety.py` |
| Reward scores | Course reward model (`yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`) on each XSTest (prompt, response), used only for the reward-vs-safety comparison (RQ1) | `generate_responses.py` |
| Manual audit | Fixed seeded draw of 60 prompts (30 SAFE + 30 UNSAFE; `fixed_audit_ids`, seed 6304), each joined to all 4 policies' responses (240 rows), shown blinded and shuffled (no policy, class or AI label); labeled before looking at AI labels | `make_audit_sheet.py` |

### Required evidence → files

| Required evidence | Files |
|---|---|
| Safety-calibration comparison (safe-answer, over-refusal, unsafe-compliance, justified-refusal, ambiguous rate, mean length per policy) | `tables/safety_calibration.*`, `figures/label_distribution.*` |
| Category-level behavior (judge-label distribution per XSTest type, all 4 policies) | `tables/category_over_refusal.*`, `tables/category_unsafe_compliance.*`, `tables/category_label_distribution.csv`, `figures/category_heatmap.*` |
| Manual-audit agreement (agreement, confusion, ambiguous rates) | `tables/audit_confusion_table.*`, `figures/audit_confusion.*`, `tables/audit_rates_manual_vs_judge.*`, `safety_metrics.json` -> `manual_audit` (agreement, Cohen's kappa, per class) |
| Qualitative disagreement evidence (harmful compliance, justified refusal, over-refusal; policy difference vs judge error) | `qualitative/safety_examples.md`, `qualitative/audit_disagreements.md` |

### Research questions → evidence to examine

**RQ1: Do higher preference/reward scores in Tasks 1-3 correspond to better safety calibration?**
Compare `tables/cross_task_reward_metrics.*` (each policy's own Task 1-3 preference and reward metrics) with
`tables/safety_calibration.*`. On identical XSTest prompts, `tables/reward_within_prompt.*` shows how often the reward model scores a
complying response above a refusal (UNSAFE) and an answer above an over-refusal (SAFE). `tables/reward_by_judge_label.*` gives the
reward by label.

**RQ2: Which policies most often confuse sensitive wording with harmful intent, and which training signals could contribute?**
Use the SAFE-type rows of `tables/category_over_refusal.*` and `figures/category_heatmap.*` (homonyms, figurative language, safe targets
and so on), `tables/policy_unique_labels.*` (over-refusals unique to one policy) and `tables/policy_disagreements.csv`.

**RQ3: What does the AI safety judge misclassify, and how much do those errors affect the policy comparison?**
Use `tables/audit_confusion_table.*` (which manual labels the judge gets wrong), `qualitative/audit_disagreements.md` (with your notes),
and `tables/audit_rates_manual_vs_judge.*`: if the policy ranking changes between the manual and judge columns, the judge errors
matter for the comparison.

## Task 5: RLVR vs RLAIF (`report/task5/`)

Produced by `python -m task5_feedback.compare_feedback --config configs/feedback.yaml` (CPU) from the raw results in
`results/task5_feedback/`:
- `{gsm,transfer}/generations_<policy>.jsonl`: response, tokens, truncation and verifier fields (prediction, exact, format, failure type).
- `{gsm,transfer}/pairwise.jsonl`: judge and verifier verdicts for every policy pair on every prompt.
- `{gsm,transfer}/summary.json`
- `diagnostics/{pairs,group_rewards}.jsonl` and `diagnostics/summary.json`
- `*/judge_cache.json`
- `feedback_comparison.json`

### Setup facts to state in the report

| Item | Value | Source |
|---|---|---|
| Policies | SFT = untouched Qwen2.5-1.5B-Instruct; RLVR and RLAIF = the supplied frozen course LoRA adapters (same base, GSM8K prompts, optimizer family, group size and approximate token budget; not retrained) | `configs/feedback.yaml` -> `policies`, manual |
| In-domain prompts | Fixed GSM8K evaluation subset: 300 test problems, prompt asks to end with `#### <number>` | `data/gsm8k_eval.jsonl` |
| Transfer prompts | Fixed SVAMP subset: the first 100 challenge-set examples (57 subtraction, 20 addition, 14 common-division, 9 multiplication), used unchanged | `data/math_transfer_eval.jsonl` |
| Decoding | Greedy, one response per policy and prompt, max 512 new tokens, the same for all policies and both datasets | `evaluate_math.py`, `math_max_new_tokens` |
| Exact verifier | Supplied `exact_reward`: the last `#### <number>` field only, numeric equality. Intermediate numbers are ignored. "Strict format" additionally requires the response to end with that field | `rlvr.py` |
| AI judge | Supplied `PairwiseAIJudge` (Qwen2.5-3B-Instruct, 4-bit, greedy, rubric unchanged), with deterministic hash-based A/B orientation and a cache. Win rate vs SFT = (wins + 0.5 ties) / n on the same prompts | `rlaif.py` |
| Diagnostic set | 20 GSM8K problems x 5 manually validated variants. Each perturbed variant is paired with the problem's clean correct response. The five variants are also scored as one K=5 group with the RLAIF group reward (wins + 0.5 ties)/(K-1) | `data/task5_controlled_reward_diagnostics.jsonl`, `score_perturbations.py` |
| Judge self-consistency | Each diagnostic pair is also judged with the arguments reversed (a new orientation draw) | `score_perturbations.py` |

### Required evidence → files

| Required evidence | Files |
|---|---|
| In-domain comparison (exact accuracy, AI pairwise result vs SFT, format compliance, mean length, verifier-judge agreement) | `tables/in_domain.*`, `tables/verifier_judge_agreement.*`, `figures/in_vs_out_of_domain.*` |
| Controlled diagnostic study (better / tie / wrong per category and mechanism; S_reason, S_outcome) | `tables/diagnostic_pairs.*`, `tables/diagnostic_sensitivity.*`, `figures/diagnostic_preferences.*`, `tables/diagnostic_group_rewards.*` |
| Out-of-domain comparison (exact accuracy, AI pairwise result, length, drop from in-domain) | `tables/transfer.*`, `tables/transfer_drop.*`, `tables/transfer_by_type.*`, `figures/in_vs_out_of_domain.*` |
| Failure types (in-domain and transfer) | `tables/failure_types.*`, `figures/failure_types.*` |
| Qualitative diagnostic evidence (reasoning-only corruption, persuasive filler, wrong final / gold distractor) | `qualitative/examples.md` |
| Feedback-source discussion (coverage, noise, exploitability, inference cost) | coverage: verifier ties in `tables/verifier_judge_agreement.*` and `tables/diagnostic_pairs.*`; noise: judge order-consistency in `tables/diagnostic_pairs.*`; exploitability: filler and distractor rows; cost: `tables/feedback_cost.*` |

### Research questions → evidence to examine

**RQ1: Where does AI preference feedback distinguish responses that the binary verifier treats identically, and when is that useful?**
Look at the "both correct / both wrong, judge breaks tie" columns of `tables/verifier_judge_agreement.*`, and at the reasoning-corruption
and filler rows of `tables/diagnostic_pairs.*`, where the verifier ties by construction.

**RQ2: When does the AI judge make an incorrect distinction because of style, verbosity or persuasive framing?**
Use the filler row (judge "wrong preference" = filler susceptibility), the judge order-consistency column, and the RLAIF group rewards
of the filler variant in `tables/diagnostic_group_rewards.*`. Read S_reason (`tables/diagnostic_sensitivity.*`) next to these results.

**RQ3: Which reward source is more vulnerable to each diagnostic category?**
Compare the verifier and judge rows per category in `tables/diagnostic_pairs.*` and `figures/diagnostic_preferences.*`. The
gold-distractor row tests verifier specification, and the filler and outcome rows test judge bias. Quote cases from `qualitative/examples.md`.

**RQ4: Which behaviors transfer to the out-of-domain set?**
Use `tables/transfer_drop.*` (accuracy and win-rate change, length), `tables/transfer_by_type.*`, and `tables/failure_types.*`
(e.g. whether format compliance or missing-final failures change on SVAMP, whose prompt wording differs from GSM8K).
