"""Regularization-strength study: short-run DPO forks for each beta.

Every fork starts from the same original Qwen2.5-1.5B-Instruct initialization with
a fresh LoRA adapter (same seed => identical LoRA init and data order), trains on
the same first `short_ablation_examples` usable pairs of the standard train file,
and is then evaluated with the common protocol in task1_dpo.evaluate. Only beta
changes. Finished adapters/evaluations are skipped, so the script can be resumed.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import evaluate_condition
from task1_dpo.train import run_training


def beta_run_name(beta: float) -> str:
    return f"beta_{beta:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+", help="override the config beta list (e.g. to split across GPUs)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n)

    for beta in betas:
        name = beta_run_name(beta)
        adapter = f"outputs/task1_dpo/{name}"
        if not (repo_path(adapter) / "adapter_config.json").exists():
            run_training(args.config, name, output_path=adapter, beta=beta, max_examples=n)
        else:
            print(f"[{name}] adapter exists, skipping training")
        if not (repo_path(cfg["results_dir"]) / "eval" / name / "summary.json").exists():
            evaluate_condition(args.config, adapter, name, beta=beta)
        else:
            print(f"[{name}] evaluation exists, skipping")


if __name__ == "__main__":
    main()
