"""Common held-out evaluation protocol for Task 2 policies.

Every condition (SFT reference, supplied PPO midpoint, standard continuation, forks)
generates one sampled response per prompt of rl_prompt_pool_eval with identical
decoding settings, seed and length cap (eval_max_response_length), then reports:
learned reward (raw and with the missing-EOS penalty), sampled KL from the reference,
token entropy, response length, EOS/truncation rate and a repetition statistic.
Outputs: results/task2_ppo/eval/<name>/{generations.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from peft import PeftModel

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import safe_corr, word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.continue_train import effective_reward


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of repeated word n-grams (0 = no repetition, -> 1 = degenerate loops)."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams) if grams else 0.0


def length_stats(values):
    v = np.asarray(values, dtype=float)
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {"mean": float(v.mean()), "std": float(v.std()), "median": float(q50),
            "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25), "max": float(v.max())}


@torch.no_grad()
def generate_and_score(bundle):
    cfg, model, tok = bundle["cfg"], bundle["policy"], bundle["tokenizer"]
    rm_model, rm_tok = bundle["reward"]
    gen, is_peft = cfg["generation"], isinstance(model, PeftModel)
    gbs, sbs = int(cfg["eval_generation_batch_size"]), int(cfg["eval_score_batch_size"])
    rows = bundle["rows"]
    set_seed(int(cfg["seed"]))
    records = []
    for start in range(0, len(rows), gbs):
        batch = rows[start:start + gbs]
        prompts = [prompt_messages(r) for r in batch]
        g = batch_generate(model, tok, prompts, max_prompt_length=4096,
                           max_new_tokens=int(cfg["eval_max_response_length"]),
                           temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
        kl_sum, ent_sum = [], []
        for s in range(0, len(batch), sbs):
            sl = slice(s, s + sbs)
            args = (g["sequences"][sl], g["attention_mask"][sl], g["prompt_width"], g["response_ids"][sl])
            pol, logits = response_token_logprobs(model, *args)
            mask = g["response_mask"][sl]
            logp_full = torch.log_softmax(logits.float(), dim=-1)
            ent_sum += ((-(logp_full.exp() * logp_full).sum(-1)) * mask).sum(-1).tolist()
            del logits, logp_full
            if is_peft:
                with reference_mode(model):
                    ref = response_token_logprobs(model, *args)[0]
            else:
                ref = pol
            kl_sum += ((pol - ref) * mask).sum(-1).tolist()
        raw = []
        for s in range(0, len(batch), sbs):
            raw += score_reward_pairs(rm_model, rm_tok, prompts[s:s + sbs], g["responses"][s:s + sbs],
                                      max_length=int(cfg["reward_max_length"])).tolist()
        for j, r in enumerate(batch):
            text = g["responses"][j]
            records.append({
                "prompt_id": r["prompt_id"], "source_index": r["source_index"],
                "prompt": prompts[j][-1]["content"], "response": text,
                "response_tokens": g["response_lengths"][j],
                "terminated_with_eos": g["terminated_with_eos"][j], "truncated": g["truncated"][j],
                "raw_reward": raw[j], "reward": effective_reward(raw[j], g["truncated"][j], cfg),
                "kl_sum": kl_sum[j], "entropy_sum": ent_sum[j],
                "words": word_count(text), "repetition": repetition_rate(text),
            })
        print(f"  generated {len(records)}/{len(rows)}", flush=True)
    return records


def summarize(records):
    tokens = [r["response_tokens"] for r in records]
    n_tok = max(1, sum(tokens))
    return {
        "n": len(records),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        "raw_reward_mean": float(np.mean([r["raw_reward"] for r in records])),
        "raw_reward_std": float(np.std([r["raw_reward"] for r in records])),
        # Token-averaged over all response tokens == common.metrics.sampled_kl on the whole set.
        "kl_token_mean": float(sum(r["kl_sum"] for r in records) / n_tok),
        "kl_sequence_mean": float(np.mean([r["kl_sum"] for r in records])),
        "entropy_token_mean": float(sum(r["entropy_sum"] for r in records) / n_tok),
        "length_tokens": length_stats(tokens),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "repetition_mean": float(np.mean([r["repetition"] for r in records])),
        "corr_reward_vs_length": safe_corr([r["raw_reward"] for r in records], tokens),
        "corr_reward_vs_kl_sequence": safe_corr([r["raw_reward"] for r in records], [r["kl_sum"] for r in records]),
    }


def evaluate_condition(config_path: str, adapter: str | None, name: str):
    timer = wall_timer()
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg = bundle["cfg"]
    print(f"[eval {name}] adapter={adapter} prompts={len(bundle['rows'])}", flush=True)
    records = generate_and_score(bundle)
    out_dir = repo_path(cfg["results_dir"]) / "eval" / name
    write_jsonl(out_dir / "generations.jsonl", records)
    summary = {
        "name": name, "adapter": adapter,
        "decoding": {**cfg["generation"], "max_new_tokens": int(cfg["eval_max_response_length"]), "seed": int(cfg["seed"])},
        **summarize(records),
        "eval_wall_clock_s": timer(),
    }
    save_json(out_dir / "summary.json", summary)
    print(f"[eval {name}] reward={summary['reward_mean']:.3f} raw={summary['raw_reward_mean']:.3f} "
          f"KL={summary['kl_token_mean']:.4f} ent={summary['entropy_token_mean']:.3f} "
          f"len={summary['length_tokens']['mean']:.1f} trunc={summary['truncation_rate']:.2f} ({timer() / 60:.1f} min)", flush=True)
    del bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit (or 'none') for the SFT reference policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    adapter = None if args.adapter in (None, "", "none") else args.adapter
    evaluate_condition(args.config, adapter, args.name)


if __name__ == "__main__":
    main()
