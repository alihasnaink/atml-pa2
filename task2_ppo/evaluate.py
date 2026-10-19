"""Common held-out evaluation protocol for Task 2 policies.

Every condition (SFT reference, supplied PPO midpoint, standard continuation, forks)
generates one sampled response per prompt of rl_prompt_pool_eval with identical
decoding settings, seed and length cap (eval_max_response_length), then reports:
learned reward (raw and with the missing-EOS penalty), sampled KL from the reference,
token entropy, response length, EOS/truncation rate and a repetition statistic.
Outputs: results/task2_ppo/eval/<name>/{generations.jsonl, summary.json}.
The protocol itself lives in common/rl_utils.py (shared with Task 3).
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from common.rl_utils import evaluate_policy, load_eval_bundle


def load_evaluation_bundle(config_path: str, adapter: str | None):
    return load_eval_bundle(load_yaml(config_path), adapter)


def evaluate_condition(config_path: str, adapter: str | None, name: str):
    return evaluate_policy(config_path, adapter, name)


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
