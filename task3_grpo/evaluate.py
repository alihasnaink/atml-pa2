"""Common held-out evaluation protocol for Task 3 policies (SFT reference, GRPO midpoint,
standard continuation, normalization forks): one sampled response per rl_prompt_pool_eval
prompt with identical decoding/seed/cap, scored for learned reward, sampled KL from the
reference, entropy, length, truncation and repetition. The protocol is shared with Task 2
(common/rl_utils.py). Outputs: results/task3_grpo/eval/<name>/{generations.jsonl, summary.json}.
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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory; omit (or 'none') for the SFT reference policy")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    adapter = None if args.adapter in (None, "", "none") else args.adapter
    evaluate_condition(args.config, adapter, args.name)


if __name__ == "__main__":
    main()
