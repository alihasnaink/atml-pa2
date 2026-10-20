"""Step 1 - common safety evaluation: one deterministic (greedy) response per XSTest prompt
for each of the four fixed policies, with identical decoding and maximum response length.

The fixed policies (manual, Task 4):
  sft  - untouched Qwen2.5-1.5B-Instruct (no adapter)
  dpo  - Task 1 Step 1 standard DPO adapter      (outputs/task1_dpo/standard)
  ppo  - Task 2 Step 1 standard PPO continuation (outputs/task2_ppo/standard)
  grpo - Task 3 Step 1 standard GRPO continuation (outputs/task3_grpo/standard)

Each response is also scored with the fixed course reward model (same scorer as Tasks 1-3)
so that reward and safety behaviour can be compared on identical prompts (RQ1).
Outputs: results/task4_safety/generated_<policy>.jsonl (fixed XSTest order).
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import wall_timer
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def output_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task4_safety"


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"{policy_name}: adapter {adapter} not found. Unpack the Task 1-3 artifacts first.")
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok, trunc in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
            })
    del model
    clear_gpu()
    return records


@torch.no_grad()
def add_reward_scores(cfg, records, batch_size: int = 8):
    """Learned reward of every (prompt, response) with the course reward model."""
    rm, rm_tok = load_reward_model(cfg)
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        scores = score_reward_pairs(rm, rm_tok, [[{"role": "user", "content": r["prompt"]}] for r in chunk],
                                    [r["response"] for r in chunk], max_length=1024)
        for r, s in zip(chunk, scores.tolist()):
            r["reward"] = float(s)
    del rm
    clear_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", help="subset of policies (e.g. to split work across GPUs)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    names = args.policies or list(policy_specs(cfg))
    print("Policies:", names)
    print("XSTest rows:", len(load_xstest(cfg)))
    for name in names:
        path = output_dir(cfg) / f"generated_{name}.jsonl"
        if path.exists():
            print(f"[{name}] {path} exists, skipping")
            continue
        timer = wall_timer()
        records = generate_for_policy(cfg, name, batch_size=int(cfg.get("safety_generation_batch_size", 4)))
        records = add_reward_scores(cfg, records)
        write_jsonl(path, records)
        print(f"[{name}] {len(records)} responses -> {path} ({timer() / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
