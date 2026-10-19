"""Length-confounding study.

1. Train one DPO model from the original initialization on the supplied
   length-balanced subset (standard one-epoch recipe) and evaluate it with the
   common protocol (skipped if already done).
2. Describe the length structure of every preference file (a property of the
   DATA): how often the preferred response is longer, length difference, and
   the composition under the length-stratum rule.
3. If the standard DPO and SFT evaluations exist, compare standard vs
   length-balanced DPO (properties of the learned POLICY): per-stratum
   preference accuracy, which pairs are learned by which model, generated
   response length and word-limit compliance (paired bootstrap CIs).
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.metrics import paired_bootstrap, safe_corr
from common.models import load_tokenizer
from task1_dpo.evaluate import evaluate_condition
from task1_dpo.train import run_training

STRATA = ("preferred_longer", "length_matched", "rejected_longer")


def length_stratum(chosen_tokens: int, rejected_tokens: int) -> str:
    """Rule matching the supplied strata: |diff|/max <= 0.10 is matched, a ratio >= 1.2
    (|diff|/max >= 1/6) is 'clearly longer'; anything in between is 'intermediate'."""
    rel = (chosen_tokens - rejected_tokens) / max(chosen_tokens, rejected_tokens, 1)
    if abs(rel) <= 0.10:
        return "length_matched"
    if rel >= 1 / 6:
        return "preferred_longer"
    if rel <= -1 / 6:
        return "rejected_longer"
    return "intermediate"


def dataset_length_stats(cfg, tokenizer):
    out = {}
    for key in ("dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"):
        rows = read_jsonl(cfg["paths"][key])
        c_len, r_len, score_gap, strata = [], [], [], []
        for row in rows:
            yc, yr = preference_responses(row)
            c = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
            r = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
            c_len.append(c)
            r_len.append(r)
            score_gap.append(row["score_chosen"] - row["score_rejected"])
            strata.append(length_stratum(c, r))
        c_len, r_len = np.array(c_len), np.array(r_len)
        diff = c_len - r_len
        stats = {
            "file": cfg["paths"][key],
            "n": len(rows),
            "frac_chosen_longer": float((diff > 0).mean()),
            "frac_rejected_longer": float((diff < 0).mean()),
            "chosen_tokens_mean": float(c_len.mean()),
            "rejected_tokens_mean": float(r_len.mean()),
            "length_diff_mean": float(diff.mean()),
            "length_diff_median": float(np.median(diff)),
            "corr_score_gap_vs_length_diff": safe_corr(score_gap, diff),
            "rule_strata": {s: float(np.mean([x == s for x in strata])) for s in STRATA + ("intermediate",)},
        }
        if "length_stratum" in rows[0]:  # check the rule against the supplied labels
            stats["rule_agrees_with_supplied_label"] = float(np.mean([x == row["length_stratum"] for x, row in zip(strata, rows)]))
        out[key] = stats
    return out


def learned_overlap(std_pairs, bal_pairs):
    """Per stratum: which pairs each model ranks correctly (margin > 0)."""
    bal = {r["prompt_id"]: r for r in bal_pairs}
    out = {}
    for s in STRATA:
        rows = [(r, bal[r["prompt_id"]]) for r in std_pairs if r["length_stratum"] == s and r["prompt_id"] in bal]
        a = np.array([x["margin"] > 0 for x, _ in rows])
        b = np.array([y["margin"] > 0 for _, y in rows])
        out[s] = {
            "n": len(rows),
            "both_correct": float((a & b).mean()),
            "only_standard": float((a & ~b).mean()),
            "only_balanced": float((~a & b).mean()),
            "neither": float((~a & ~b).mean()),
            "accuracy_diff_balanced_minus_standard": paired_bootstrap(a, b),
            "margin_corr_standard_vs_balanced": safe_corr([x["margin"] for x, _ in rows], [y["margin"] for _, y in rows]),
        }
    return out


def heldout_by_rule_stratum(pairs):
    """Preference accuracy on the ordinary held-out set, split by the length rule."""
    out = {}
    for s in STRATA + ("intermediate",):
        sel = [r for r in pairs if length_stratum(r["chosen_tokens"], r["rejected_tokens"]) == s]
        out[s] = {"n": len(sel), "preference_accuracy": float(np.mean([r["margin"] > 0 for r in sel])) if sel else None}
    return out


def paired_generation_diff(a_rows, b_rows, field, key=("prompt_id", "sample")):
    ka = {tuple(r.get(k) for k in key): r[field] for r in a_rows}
    common = [(ka[tuple(r.get(k) for k in key)], r[field]) for r in b_rows if tuple(r.get(k) for k in key) in ka]
    return paired_bootstrap([x for x, _ in common], [y for _, y in common])


def compare_policies(cfg):
    eval_dir = repo_path(cfg["results_dir"]) / "eval"
    names = ["sft", "standard", "length_balanced"]
    if not all((eval_dir / n / "summary.json").exists() for n in names):
        return None
    summaries = {n: load_json(eval_dir / n / "summary.json") for n in names}
    strat = {n: read_jsonl(eval_dir / n / "pairs_stratified.jsonl") for n in names}
    heldout = {n: read_jsonl(eval_dir / n / "pairs_heldout.jsonl") for n in names}
    gens = {n: read_jsonl(eval_dir / n / "generations_heldout.jsonl") for n in names}
    wl = {n: read_jsonl(eval_dir / n / "generations_word_limit.jsonl") for n in names}

    out = {"per_stratum": {}, "generation": {}, "word_limit": {}, "heldout_by_rule_stratum": {}}
    for n in names:
        out["per_stratum"][n] = summaries[n]["stratified_pairs"]
        out["generation"][n] = summaries[n]["heldout_generations"]
        out["word_limit"][n] = {k: summaries[n]["word_limit"][k] for k in
                                ("compliance_rate", "words_over_limit_mean", "length_words", "length_tokens", "reward_mean", "per_prompt_compliance")}
        out["heldout_by_rule_stratum"][n] = heldout_by_rule_stratum(heldout[n])
        lengths_diff = [r["chosen_tokens"] - r["rejected_tokens"] for r in strat[n]]
        out["per_stratum"][n]["all_strata_corr_margin_vs_length_diff"] = safe_corr([r["margin"] for r in strat[n]], lengths_diff)

    out["learned_overlap_standard_vs_balanced"] = learned_overlap(strat["standard"], strat["length_balanced"])
    for a, b in (("sft", "standard"), ("sft", "length_balanced"), ("standard", "length_balanced")):
        tag = f"{b}_minus_{a}"
        out["generation"][tag] = {
            "response_tokens": paired_generation_diff(gens[a], gens[b], "response_tokens", key=("prompt_id",)),
            "reward": paired_generation_diff(gens[a], gens[b], "reward", key=("prompt_id",)),
        }
        out["word_limit"][tag] = {
            "compliance": paired_generation_diff(wl[a], wl[b], "compliant"),
            "words": paired_generation_diff(wl[a], wl[b], "words"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    adapter = cfg["length_output"]
    if not (repo_path(adapter) / "adapter_config.json").exists():
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=adapter)
    if not (results / "eval" / "length_balanced" / "summary.json").exists():
        evaluate_condition(args.config, adapter, "length_balanced")

    tokenizer = load_tokenizer(cfg["base_model"])
    save_json(results / "dataset_length_stats.json", dataset_length_stats(cfg, tokenizer))
    print("Saved dataset length statistics")

    comparison = compare_policies(cfg)
    if comparison is None:
        print("SFT/standard evaluations not found yet; rerun after task1_dpo.evaluate for 'sft' and 'standard'.")
        return
    save_json(results / "length_analysis.json", comparison)
    print("Saved length analysis to", results / "length_analysis.json")


if __name__ == "__main__":
    main()
