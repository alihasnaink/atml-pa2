"""Reward-overoptimization / KL-pressure study.

Matched short continuations from the exact same PPO midpoint for each beta_KL in
kl_values (clip_epsilon fixed at the release value), each followed by the common
held-out evaluation. Trajectories (reward, KL, entropy, length per update) are in
results/task2_ppo/train_logs/; held-out metrics in results/task2_ppo/eval/.
The tables/figures for the report are built by task2_ppo.make_report.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.forks import run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--kl-values", type=float, nargs="+", help="override the config list (e.g. to split work across GPUs)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.kl_values or [float(b) for b in cfg["kl_values"]]
    print("KL beta conditions:", betas)
    print("Fork update budget:", cfg["fork_updates"])
    for beta in betas:
        run_fork(args.config, float(cfg["clip_epsilon"]), beta)


if __name__ == "__main__":
    main()
