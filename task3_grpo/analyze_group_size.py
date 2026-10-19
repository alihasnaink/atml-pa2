"""Equal-generation group-size study on the supplied K=8 completion/reward cache (no training).

Regrouping rule (equal total generations): each prompt's 8 cached completions, ordered by
generation_index, are split into 8/K consecutive groups of size K, so K=2, 4 and 8 all use
the same 192 completions of the same 24 prompts (96, 48 and 24 groups). As a robustness
check, every metric is also averaged over all C(8, K) subsets of each prompt's completions.

Prompt-difficulty bins (defined once): tertiles of the prompt's mean reward over its 8
cached completions (lowest third = "hard", highest third = "easy").

Metrics per K (overall and per difficulty bin):
  informative-group rate: group reward std > 1e-6 (tolerance of group_relative_advantages)
  mean within-group reward std
  variance of the group-relative signal r - mean_group(r) (pooled over completions), and of the
  normalized advantage used for training
  fraction of completions that carry gradient: non-zero advantage AND not truncated
  (truncated completions are masked from the GRPO loss)
  sensitivity: fraction of groups whose std is below 0.1 / 0.25 / 0.5 reward units
Output: results/task3_grpo/group_size.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

TOL = 1e-6
NEAR_ZERO = (0.1, 0.25, 0.5)


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions (ordered by generation_index) are split into 8 // k
    consecutive groups, e.g. K=2 -> (0,1), (2,3), (4,5), (6,7). Every K therefore uses the
    same 8 x n_prompts completions; only how they are compared changes.
    """
    if 8 % k:
        raise ValueError(f"K={k} must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        for start in range(0, 8, k):
            groups.append({"prompt": pid, "rows": rows[start:start + k]})
    return groups


def group_metrics(groups):
    """Metrics over a list of groups (each a list of cache rows)."""
    stds, centered, normalized, carries = [], [], [], []
    for rows in groups:
        r = torch.tensor([float(x["reward"]) for x in rows], dtype=torch.float64)
        std = float(r.std(unbiased=False))
        stds.append(std)
        centered += (r - r.mean()).tolist()
        a = group_relative_advantages(r, torch.zeros(len(r), dtype=torch.long))
        normalized += a.tolist()
        carries += [abs(float(ai)) > 0 and not bool(x.get("clipped_at_max", False)) for ai, x in zip(a, rows)]
    stds = np.array(stds)
    return {
        "n_groups": len(groups),
        "n_generations": len(centered),
        "informative_rate": float((stds > TOL).mean()),
        "uninformative_rate": float((stds <= TOL).mean()),
        "mean_within_group_std": float(stds.mean()),
        "var_relative_signal": float(np.var(centered)),
        "var_normalized_advantage": float(np.var(normalized)),
        "mean_abs_normalized_advantage": float(np.mean(np.abs(normalized))),
        "frac_completions_with_gradient": float(np.mean(carries)),
        **{f"frac_groups_std_below_{t}": float((stds < t).mean()) for t in NEAR_ZERO},
    }


def all_subsets_metrics(by_prompt, k, prompts):
    """Average over every size-K subset of each prompt's 8 completions (lower-variance estimate)."""
    groups = [list(c) for p in prompts for c in combinations(by_prompt[p][:8], k)]
    m = group_metrics(groups)
    return {key: m[key] for key in ("informative_rate", "mean_within_group_std", "var_relative_signal",
                                    "frac_completions_with_gradient")} | {"n_subsets": len(groups)}


def difficulty_bins(by_prompt, n_bins):
    means = {p: float(np.mean([x["reward"] for x in rows[:8]])) for p, rows in by_prompt.items()}
    order = sorted(means, key=means.get)  # lowest mean reward first = hardest
    names = ["hard", "medium", "easy"] if n_bins == 3 else [f"bin{i + 1}" for i in range(n_bins)]
    bins = {}
    for i, chunk in enumerate(np.array_split(np.array(order), n_bins)):
        bins[names[i]] = {"prompts": chunk.tolist(),
                          "mean_reward_range": [means[chunk[0]], means[chunk[-1]]]}
    return bins, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    bins, prompt_mean = difficulty_bins(by_prompt, int(cfg.get("difficulty_bins", 3)))
    out = {
        "regrouping_rule": regroup_equal_generation_budget.__doc__.strip(),
        "difficulty_rule": "tertiles of each prompt's mean reward over its 8 cached completions (lowest = hard)",
        "informative_tolerance": TOL,
        "n_prompts": len(by_prompt),
        "total_generations": sum(min(8, len(v)) for v in by_prompt.values()),
        "truncated_completions": sum(bool(x.get("clipped_at_max")) for v in by_prompt.values() for x in v[:8]),
        "bins": bins,
        "per_k": {},
        "per_k_per_bin": {},
        "per_k_all_subsets": {},
        "per_prompt": [],
    }
    for k in [int(k) for k in cfg["group_sizes"]]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        out["per_k"][str(k)] = group_metrics([g["rows"] for g in groups])
        out["per_k_all_subsets"][str(k)] = all_subsets_metrics(by_prompt, k, list(by_prompt))
        out["per_k_per_bin"][str(k)] = {
            b: {**group_metrics([g["rows"] for g in groups if g["prompt"] in set(info["prompts"])]),
                "all_subsets": all_subsets_metrics(by_prompt, k, info["prompts"])}
            for b, info in bins.items()
        }
        m = out["per_k"][str(k)]
        print(f"K={k}: groups {m['n_groups']} informative {m['informative_rate']:.3f} "
              f"mean std {m['mean_within_group_std']:.3f} var(r-mu) {m['var_relative_signal']:.3f}")
    for p, rows in by_prompt.items():
        r = np.array([x["reward"] for x in rows[:8]])
        out["per_prompt"].append({
            "source_index": p, "prompt_id": rows[0].get("prompt_id"), "mean_reward": prompt_mean[p],
            "std_k8": float(r.std()), "n_truncated": int(sum(bool(x.get("clipped_at_max")) for x in rows[:8])),
            "bin": next(b for b, info in bins.items() if p in info["prompts"]),
        })
    save_json(repo_path(cfg["results_dir"]) / "group_size.json", out)
    print("Saved group-size study to", repo_path(cfg["results_dir"]) / "group_size.json")


if __name__ == "__main__":
    main()
