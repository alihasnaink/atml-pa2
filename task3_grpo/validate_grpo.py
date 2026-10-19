"""Validate the GRPO helpers in task3_grpo.grpo against the equations in the manual.

Runs on CPU in seconds and needs no model. Writes results/task3_grpo/grpo_validation.json.
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def released_buggy_advantages(rewards, eps=1e-6):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps_clip, beta, L = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    checks = {}

    # 1) Advantages are normalized within each prompt group (manual: (r - mu_group) / (sigma_group + eps)).
    K, G = 4, 5
    rewards = torch.randn(G * K, generator=g, dtype=torch.float64) + torch.arange(G).repeat_interleave(K) * 2.0
    rewards[:K] = 1.5  # group 0 has identical rewards -> uninformative
    gid = torch.arange(G).repeat_interleave(K)
    adv = group_relative_advantages(rewards, gid)
    manual = torch.zeros_like(rewards)
    for i in range(G):
        r = rewards[gid == i]
        manual[gid == i] = (r - r.mean()) / (r.std(unbiased=False) + 1e-6)
    buggy = released_buggy_advantages(rewards)
    checks["within_group_normalization"] = {
        "max_abs_diff_vs_manual": float((adv - manual).abs().max()),
        "released_buggy_max_abs_diff_vs_manual": float((buggy - manual).abs().max()),
        "pass": float((adv - manual).abs().max()) < 1e-9,
    }
    checks["zero_std_group_gets_zero_advantage"] = {
        "fixed_group0": adv[:K].tolist(), "released_buggy_group0": buggy[:K].tolist(),
        "pass": bool((adv[:K] == 0).all()),
    }
    shifted = rewards.clone()
    shifted[gid == 3] += 10.0  # an "easier" prompt must not change any advantage
    checks["invariant_to_per_group_reward_shift"] = {
        "fixed_max_change": float((group_relative_advantages(shifted, gid) - adv).abs().max()),
        "released_buggy_max_change": float((released_buggy_advantages(shifted) - buggy).abs().max()),
        "pass": float((group_relative_advantages(shifted, gid) - adv).abs().max()) < 1e-9,
    }
    sums = [float(adv[gid == i].sum()) for i in range(G)]
    checks["advantages_sum_to_zero_per_group"] = {"group_sums": sums, "pass": max(abs(s) for s in sums) < 1e-9}

    # 2) Sequence normalization of the clipped objective (ratio = 1, beta = 0 isolates the policy term).
    lengths = [3, 7, 12, 5]
    T = max(lengths)
    mask = torch.zeros(K, T, dtype=torch.float64)
    for k, n in enumerate(lengths):
        mask[k, :n] = 1
    seq_adv = torch.tensor([1.0, -0.5, 0.8, -1.3], dtype=torch.float64)
    logp = (-torch.rand(K, T, generator=g, dtype=torch.float64)) * mask
    for loss_type in ("grpo", "dr_grpo"):
        new = logp.clone().requires_grad_(True)
        loss, diag = grpo_policy_loss(new, logp, seq_adv, mask, logp, eps_clip, 0.0, loss_type, L)
        loss.backward()
        denom = torch.tensor(lengths, dtype=torch.float64) if loss_type == "grpo" else torch.full((K,), float(L), dtype=torch.float64)
        expected_loss = -(seq_adv * torch.tensor(lengths, dtype=torch.float64) / denom).mean()
        expected_grad = (-seq_adv / (K * denom))[:, None] * mask  # d loss / d log pi_t for every valid token
        checks[f"{loss_type}_normalization"] = {
            "loss": float(loss.detach()), "expected_loss": float(expected_loss),
            "per_token_weight_by_length": {n: float(new.grad[k, 0] / -seq_adv[k]) for k, n in enumerate(lengths)},
            "pass": abs(float(loss.detach() - expected_loss)) < 1e-12 and float((new.grad - expected_grad).abs().max()) < 1e-12,
        }

    # 3) Clipped objective is the pessimistic min (same geometry as PPO) for positive / negative advantages.
    new = torch.log(torch.tensor([[1.5], [0.5], [0.5], [1.5]], dtype=torch.float64)).requires_grad_(True)
    old = torch.zeros(4, 1, dtype=torch.float64)
    a = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float64)
    loss, _ = grpo_policy_loss(new, old, a, torch.ones(4, 1, dtype=torch.float64), new.detach(), eps_clip, 0.0, "grpo", L)
    loss.backward()
    zero_grad = [abs(float(x)) < 1e-12 for x in new.grad.flatten()]
    checks["clipping_geometry"] = {"zero_grad_[A>0,rho>1+e | A>0,rho<1-e | A<0,rho<1-e | A<0,rho>1+e]": zero_grad,
                                   "pass": zero_grad == [True, False, True, False]}

    # 4) k3 KL estimator: zero when policy == reference, positive otherwise.
    ref = logp - 0.3 * mask
    _, d0 = grpo_policy_loss(logp, logp, seq_adv, mask, logp, eps_clip, beta, "grpo", L)
    _, d1 = grpo_policy_loss(logp, logp, seq_adv, mask, ref, eps_clip, beta, "grpo", L)
    checks["kl_term"] = {"kl_policy_eq_ref": float(d0["sampled_kl"]), "kl_shifted_ref": float(d1["sampled_kl"]),
                         "pass": abs(float(d0["sampled_kl"])) < 1e-12 and float(d1["sampled_kl"]) > 0}

    # 5) Truncated completions are removed from the loss entirely.
    m2 = mask_truncated_sequences(mask, [False, True, False, True])
    checks["mask_truncated"] = {"tokens_kept_per_sequence": m2.sum(-1).tolist(),
                                "pass": m2.sum(-1).tolist() == [lengths[0], 0.0, lengths[2], 0.0]}

    out = {"clip_epsilon": eps_clip, "kl_beta": beta, "max_completion_length": L,
           "all_pass": all(c["pass"] for c in checks.values()), "checks": checks,
           "defect": "group_relative_advantages normalized over the whole batch; fixed to per-group (per-prompt) mean/std"}
    save_json(f"{cfg['results_dir']}/grpo_validation.json", out)
    for name, c in checks.items():
        print(f"{name:40s} {'PASS' if c['pass'] else 'FAIL'}")
    if not out["all_pass"]:
        raise SystemExit("GRPO validation failed")


if __name__ == "__main__":
    main()
