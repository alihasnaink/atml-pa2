"""Matched short PPO forks shared by the clipping and KL studies.

Each fork restarts from the identical supplied policy/value midpoint, runs
`fork_updates` updates on the same fixed prompt sequence with the same per-update
sampling seeds, and changes only (clip_epsilon, kl_beta). The fork with the release
defaults (eps=0.20, beta=0.10) is shared by both studies and trained once.
Finished forks/evaluations are skipped, so drivers can be re-run safely.
"""
from __future__ import annotations

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_condition


def fork_name(eps: float, beta: float) -> str:
    return f"fork_eps{eps:.2f}_kl{beta:.2f}"


def run_fork(config_path: str, eps: float, beta: float) -> str:
    cfg = load_yaml(config_path)
    name = fork_name(eps, beta)
    adapter = f"outputs/task2_ppo/{name}"
    if not (repo_path(adapter) / "adapter_config.json").exists():
        run_ppo(config_path, output=adapter, updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=beta, run_name=name)
    else:
        print(f"[{name}] adapter exists, skipping training")
    if not (repo_path(cfg["results_dir"]) / "eval" / name / "summary.json").exists():
        evaluate_condition(config_path, adapter, name)
    else:
        print(f"[{name}] evaluation exists, skipping")
    return name
