"""Common Task 1 evaluation protocol, applied identically to every condition.

For one policy (a DPO adapter, or the untouched SFT model when no adapter is given):
  1. held-out pairs (dpo_standard_eval): DPO loss, preference accuracy, margins
  2. length-stratified pairs (dpo_length_stratified_eval): same, per pair
  3. generations on the held-out prompts: reward-model score, response length,
     sampled KL from the reference policy
  4. generations on the common word-limit prompts: word-limit compliance

Everything is written to results/task1_dpo/eval/<name>/ (per-example JSONL + summary.json).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from peft import PeftModel

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import preference_accuracy, safe_corr, word_count, word_limit_compliance, parse_word_limit
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_fitting_rows, make_collate, to_device


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    standard_rows, standard_dropped = filter_fitting_rows(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tokenizer, cfg)
    strat_rows, strat_dropped = filter_fitting_rows(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer, cfg)
    return {
        "cfg": cfg,
        "rows": standard_rows,
        "stratified_rows": strat_rows,
        "dropped": {"standard_eval": standard_dropped, "stratified_eval": strat_dropped},
        "word_limit_rows": read_jsonl(cfg["paths"]["word_limit_prompts"]),
        "tokenizer": tokenizer,
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


# ---------------------------------------------------------------- pair metrics

@torch.no_grad()
def score_pairs(model, tokenizer, rows, cfg):
    """Sequence log-probs (summed over response tokens) of chosen/rejected under policy and reference."""
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    is_peft = isinstance(model, PeftModel)
    bs = int(cfg["eval_pair_batch_size"])
    out = []
    for start in range(0, len(rows), bs):
        batch_rows = rows[start:start + bs]
        chosen, rejected = (to_device(b, device) for b in collate(batch_rows))
        pc, _, mask_c = response_sequence_logprobs(model, chosen)
        pr, _, mask_r = response_sequence_logprobs(model, rejected)
        if is_peft:
            with reference_mode(model):
                rc = response_sequence_logprobs(model, chosen)[0]
                rr = response_sequence_logprobs(model, rejected)[0]
        else:  # the SFT model is its own reference
            rc, rr = pc, pr
        for j, row in enumerate(batch_rows):
            margin = (pc[j] - rc[j]) - (pr[j] - rr[j])
            out.append({
                "prompt_id": row["prompt_id"],
                "source_index": row["source_index"],
                "length_stratum": row.get("length_stratum"),
                "chosen_tokens": int(mask_c[j].sum()),
                "rejected_tokens": int(mask_r[j].sum()),
                "policy_chosen_logp": float(pc[j]),
                "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]),
                "ref_rejected_logp": float(rr[j]),
                "margin": float(margin),
            })
    return out


def summarize_pairs(records, beta):
    if not records:
        return {"n": 0}
    t = lambda k: torch.tensor([r[k] for r in records], dtype=torch.float64)
    loss, diag = dpo_loss(t("policy_chosen_logp"), t("policy_rejected_logp"), t("ref_chosen_logp"), t("ref_rejected_logp"), beta)
    length_diff = [r["chosen_tokens"] - r["rejected_tokens"] for r in records]
    return {
        "n": len(records),
        "dpo_loss": float(loss),
        "preference_accuracy": float(diag["preference_accuracy"]),
        "margin_mean": float(diag["margin_mean"]),
        "margin_std": float(np.std([r["margin"] for r in records])),
        "chosen_reward_mean": float(diag["chosen_reward_mean"]),
        "rejected_reward_mean": float(diag["rejected_reward_mean"]),
        # Raw (not reference-relative) likelihood preference of the policy; summed
        # log-probs are strongly length dependent, so this is a length diagnostic.
        "raw_likelihood_accuracy": preference_accuracy(t("policy_chosen_logp"), t("policy_rejected_logp")),
        "corr_margin_vs_length_diff": safe_corr([r["margin"] for r in records], length_diff),
    }


# ---------------------------------------------------------------- generations

@torch.no_grad()
def generate_and_score(model, tokenizer, reward, items, cfg):
    """items: dicts with 'messages' plus id fields. Returns one record per generated response."""
    rm_model, rm_tok = reward
    gen = cfg["generation"]
    is_peft = isinstance(model, PeftModel)
    set_seed(int(cfg["seed"]))  # identical sampling stream for every condition
    gbs, sbs = int(cfg["eval_generation_batch_size"]), int(cfg["eval_score_batch_size"])
    records = []
    for start in range(0, len(items), gbs):
        batch = items[start:start + gbs]
        prompts = [it["messages"] for it in batch]
        g = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]),
        )
        # Sampled-response KL estimator: log pi(y_t) - log ref(y_t) on the policy's own samples.
        kl_sum, logp_sum = [], []
        for s in range(0, len(batch), sbs):
            sl = slice(s, s + sbs)
            args = (g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
            pol = response_token_logprobs(model, *args)[0]
            if is_peft:
                with reference_mode(model):
                    ref = response_token_logprobs(model, *args)[0]
            else:
                ref = pol
            mask = g["response_mask"][sl]
            kl_sum += ((pol - ref) * mask).sum(-1).tolist()
            logp_sum += (pol * mask).sum(-1).tolist()
        rewards = []
        for s in range(0, len(batch), sbs):
            rewards += score_reward_pairs(rm_model, rm_tok, prompts[s:s + sbs], g["responses"][s:s + sbs],
                                          max_length=int(cfg["reward_max_length"])).tolist()
        for j, it in enumerate(batch):
            rec = {k: v for k, v in it.items() if k != "messages"}
            rec.update({
                "prompt": prompts[j][-1]["content"],
                "response": g["responses"][j],
                "response_tokens": g["response_lengths"][j],
                "truncated": g["truncated"][j],
                "reward": rewards[j],
                "kl_sum": kl_sum[j],
                "logp_sum": logp_sum[j],
            })
            records.append(rec)
        print(f"  generated {len(records)}/{len(items)}", flush=True)
    return records


def length_stats(values):
    v = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(q50),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "max": float(v.max())}


def summarize_generations(records):
    tokens = [r["response_tokens"] for r in records]
    rewards = [r["reward"] for r in records]
    return {
        "n": len(records),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        # Token-averaged KL over all response tokens == common.metrics.sampled_kl on the whole set.
        "kl_token_mean": float(sum(r["kl_sum"] for r in records) / max(1, sum(tokens))),
        "kl_sequence_mean": float(np.mean([r["kl_sum"] for r in records])),
        "length_tokens": length_stats(tokens),
        "length_words": length_stats([word_count(r["response"]) for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "corr_reward_vs_length": safe_corr(rewards, tokens),
    }


def summarize_word_limit(records):
    out = summarize_generations(records)
    out["compliance_rate"] = float(np.mean([r["compliant"] for r in records]))
    out["words_over_limit_mean"] = float(np.mean([r["words"] / r["word_limit"] for r in records]))
    per_prompt = {}
    for r in records:
        per_prompt.setdefault(r["prompt_id"], []).append(r["compliant"])
    out["per_prompt_compliance"] = {k: float(np.mean(v)) for k, v in per_prompt.items()}
    return out


# ---------------------------------------------------------------- driver

def evaluate_condition(config_path: str, adapter: str | None, name: str, beta: float | None = None):
    timer = wall_timer()
    b = load_evaluation_bundle(config_path, adapter)
    cfg, model, tok = b["cfg"], b["policy"], b["tokenizer"]
    beta = float(cfg["beta"] if beta is None else beta)
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    print(f"[eval {name}] adapter={adapter} beta={beta} heldout={len(b['rows'])} stratified={len(b['stratified_rows'])}", flush=True)

    pairs = score_pairs(model, tok, b["rows"], cfg)
    write_jsonl(out_dir / "pairs_heldout.jsonl", pairs)
    strat = score_pairs(model, tok, b["stratified_rows"], cfg)
    write_jsonl(out_dir / "pairs_stratified.jsonl", strat)
    print(f"[eval {name}] pair metrics done", flush=True)

    items = [{"prompt_id": r["prompt_id"], "source_index": r["source_index"],
              "messages": prompt_messages_from_preference(r)} for r in b["rows"]]
    gens = generate_and_score(model, tok, b["reward"], items, cfg)
    write_jsonl(out_dir / "generations_heldout.jsonl", gens)

    wl_items = [{"prompt_id": r["prompt_id"], "sample": k, "messages": r["messages"]}
                for r in b["word_limit_rows"] for k in range(int(cfg["word_limit_samples"]))]
    wl = generate_and_score(model, tok, b["reward"], wl_items, cfg)
    for r in wl:
        r["word_limit"] = parse_word_limit(r["prompt"])
        r["words"] = word_count(r["response"])
        r["compliant"] = word_limit_compliance(r["prompt"], r["response"])
    write_jsonl(out_dir / "generations_word_limit.jsonl", wl)

    train_summary_path = repo_path(cfg["results_dir"]) / "train_logs" / f"{name}_summary.json"
    train_info = None
    if train_summary_path.exists():
        t = load_json(train_summary_path)
        train_info = {k: t[k] for k in ("dataset", "n_train_pairs", "optimizer_steps", "epochs", "wall_clock_s", "peak_vram_gb")}

    summary = {
        "name": name,
        "adapter": adapter,
        "beta": beta,
        "train": train_info,
        "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["max_generation_tokens"]), "seed": int(cfg["seed"])},
        "dropped_eval_prompt_ids": b["dropped"],
        "heldout_pairs": summarize_pairs(pairs, beta),
        "stratified_pairs": {
            s: summarize_pairs([r for r in strat if r["length_stratum"] == s], beta)
            for s in ("preferred_longer", "length_matched", "rejected_longer")
        },
        "heldout_generations": summarize_generations(gens),
        "word_limit": summarize_word_limit(wl),
        "eval_wall_clock_s": timer(),
    }
    save_json(out_dir / "summary.json", summary)
    print(f"[eval {name}] acc={summary['heldout_pairs']['preference_accuracy']:.3f} "
          f"KL={summary['heldout_generations']['kl_token_mean']:.4f} "
          f"reward={summary['heldout_generations']['reward_mean']:.3f} "
          f"len={summary['heldout_generations']['length_tokens']['mean']:.1f} "
          f"wl={summary['word_limit']['compliance_rate']:.2f} ({timer()/60:.1f} min)", flush=True)

    del b, model
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit (or 'none') for the SFT baseline")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (defaults to config beta)")
    args = ap.parse_args()
    adapter = None if args.adapter in (None, "", "none") else args.adapter
    evaluate_condition(args.config, adapter, args.name, args.beta)


if __name__ == "__main__":
    main()
