"""Length-normalization study: canonical GRPO vs Dr. GRPO-style normalization.

Two matched short continuations (`fork_updates` updates) from the identical supplied
midpoint. Prompts, per-update sampling seeds, K, reward, beta, eps, generation settings
and the update budget are identical; only `loss_type` changes:
  grpo     per-sequence 1/T_k normalization (token weight 1 / (K * T_k))
  dr_grpo  constant 1/max_completion_length (token weight 1 / (K * L))
Each update also logs every completion's own gradient norm (length-conditioned gradient
statistic). Both forks are then evaluated with the common held-out protocol.
Finished runs/evaluations are skipped. Tables/figures: task3_grpo.make_report.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_condition

LOSS_TYPES = ("grpo", "dr_grpo")


def fork_name(loss_type: str) -> str:
    return f"fork_{loss_type}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    for loss_type in LOSS_TYPES:
        name = fork_name(loss_type)
        adapter = f"outputs/task3_grpo/{name}"
        if not (repo_path(adapter) / "adapter_config.json").exists():
            run_grpo(args.config, output=adapter, updates=int(cfg["fork_updates"]), loss_type=loss_type,
                     run_name=name, per_sequence_grads=True)
        else:
            print(f"[{name}] adapter exists, skipping training")
        if not (repo_path(cfg["results_dir"]) / "eval" / name / "summary.json").exists():
            evaluate_condition(args.config, adapter, name)
        else:
            print(f"[{name}] evaluation exists, skipping")


if __name__ == "__main__":
    main()
