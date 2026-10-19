"""Validate the PPO helpers in task2_ppo.ppo against the equations in the manual.

Runs on CPU in seconds and needs no model. Writes results/task2_ppo/ppo_validation.json.
"""
from __future__ import annotations

import argparse

import torch

from common.data import load_yaml
from common.logging_utils import save_json
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def released_buggy_objective(ratio, adv, eps):
    return torch.maximum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)


def naive_gae(rewards, values, lengths, gamma, lam):
    """Literal A_t = sum_k (gamma*lam)^k delta_{t+k}, with V = 0 after the last valid token."""
    adv = torch.zeros_like(rewards)
    for b, n in enumerate(lengths):
        for t in range(n):
            total = 0.0
            for k in range(n - t):
                nxt = values[b, t + k + 1] if t + k + 1 < n else 0.0
                delta = rewards[b, t + k] + gamma * nxt - values[b, t + k]
                total += (gamma * lam) ** k * delta
            adv[b, t] = total
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    eps, gamma, lam, beta = float(cfg["clip_epsilon"]), float(cfg["gamma"]), float(cfg["gae_lambda"]), float(cfg["kl_beta"])
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    checks = {}

    # 1) Clipped surrogate equals the per-token min of the manual, and is pessimistic.
    n = 4000
    old = -3 * torch.rand(n, generator=g, dtype=torch.float64)
    new = old + 0.6 * torch.randn(n, generator=g, dtype=torch.float64)
    adv = torch.randn(n, generator=g, dtype=torch.float64)
    mask = torch.ones(n, dtype=torch.float64)
    loss, ratio, cf = ppo_policy_loss(new, old, adv, mask, eps)
    r = torch.exp(new - old)
    manual = torch.stack([r * adv, r.clamp(1 - eps, 1 + eps) * adv]).min(0).values
    checks["matches_manual_min_surrogate"] = {
        "fixed_objective": float(-loss), "manual_objective": float(manual.mean()),
        "released_buggy_objective": float(released_buggy_objective(r, adv, eps).mean()),
        "pass": abs(float(-loss) - float(manual.mean())) < 1e-10,
    }
    checks["objective_is_pessimistic_lower_bound"] = {
        "fixed_le_unclipped_everywhere": bool((manual <= r * adv + 1e-12).all()),
        "buggy_le_unclipped_everywhere": bool((released_buggy_objective(r, adv, eps) <= r * adv + 1e-12).all()),
        "pass": bool((manual <= r * adv + 1e-12).all()),
    }

    # 2) Gradient geometry: zero gradient exactly where clipping binds, unclipped gradient elsewhere.
    cases = {  # (ratio, advantage) -> expected d(loss)/d(new_logp) is zero?
        "A>0, rho>1+eps (binds)": (1.5, 1.0, True),
        "A>0, rho<1-eps (free)": (0.5, 1.0, False),
        "A<0, rho<1-eps (binds)": (0.5, -1.0, True),
        "A<0, rho>1+eps (free)": (1.5, -1.0, False),
    }
    geo = {}
    for name, (rho, a, should_be_zero) in cases.items():
        nl = torch.tensor([float(torch.log(torch.tensor(rho)))], dtype=torch.float64, requires_grad=True)
        l, _, _ = ppo_policy_loss(nl, torch.zeros(1, dtype=torch.float64), torch.tensor([a], dtype=torch.float64), torch.ones(1, dtype=torch.float64), eps)
        l.backward()
        nb = nl.detach().clone().requires_grad_(True)
        lb = -released_buggy_objective(torch.exp(nb), torch.tensor([a], dtype=torch.float64), eps).mean()
        lb.backward()
        geo[name] = {"fixed_grad": float(nl.grad), "released_buggy_grad": float(nb.grad),
                     "pass": (abs(float(nl.grad)) < 1e-12) == should_be_zero}
    checks["clipping_gradient_geometry"] = {**geo, "pass": all(v["pass"] for v in geo.values())}

    # 3) Clip / affected fractions match their definitions.
    d = clip_diagnostics(r, adv, mask, eps)
    outside = ((r < 1 - eps) | (r > 1 + eps)).double().mean()
    binds = (((r > 1 + eps) & (adv > 0)) | ((r < 1 - eps) & (adv < 0))).double().mean()
    checks["clip_and_affected_fraction_definitions"] = {
        **d, "manual_clip_fraction": float(outside), "manual_affected_fraction": float(binds),
        # fractions are accumulated in float32 inside clip_diagnostics
        "pass": abs(d["clip_fraction"] - float(outside)) < 1e-6 and abs(d["affected_fraction"] - float(binds)) < 1e-6
                and abs(float(cf) - float(outside)) < 1e-6,
    }

    # 4) GAE matches the literal discounted sum, with padding and the zero bootstrap.
    lengths = [7, 3, 5]
    B, T = len(lengths), max(lengths)
    m = torch.zeros(B, T, dtype=torch.float64)
    for b, L in enumerate(lengths):
        m[b, :L] = 1
    rew = torch.randn(B, T, generator=g, dtype=torch.float64) * m
    val = torch.randn(B, T, generator=g, dtype=torch.float64) * m
    a_fast, ret = compute_gae(rew, val, m, gamma, lam)
    a_ref = naive_gae(rew, val, lengths, gamma, lam)
    checks["gae_matches_literal_sum"] = {
        "max_abs_diff": float((a_fast - a_ref).abs().max()),
        "returns_equal_adv_plus_values": bool(torch.allclose(ret * m, (a_fast + val) * m)),
        "pass": float((a_fast - a_ref).abs().max()) < 1e-10,
    }

    # 5) KL-shaped reward: -beta*(logp - ref) per token, task reward on the last valid token only.
    task_r = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    lp = torch.randn(B, T, generator=g, dtype=torch.float64) * m
    rp = torch.randn(B, T, generator=g, dtype=torch.float64) * m
    sr = shaped_rewards(task_r, lp, rp, m, beta)
    expected = -beta * (lp - rp) * m
    for b, L in enumerate(lengths):
        expected[b, L - 1] += task_r[b]
    checks["kl_shaped_rewards"] = {"max_abs_diff": float((sr - expected).abs().max()),
                                   "pass": float((sr - expected).abs().max()) < 1e-12}

    # 6) Advantage whitening over valid tokens only.
    an = normalize_advantages(a_fast, m)
    valid = an[m.bool()]
    checks["advantage_normalization"] = {"mean": float(valid.mean()), "std": float(valid.std(unbiased=False)),
                                         "padding_zero": bool((an[~m.bool()] == 0).all()),
                                         "pass": abs(float(valid.mean())) < 1e-9 and abs(float(valid.std(unbiased=False)) - 1) < 1e-6}

    out = {"clip_epsilon": eps, "gamma": gamma, "gae_lambda": lam, "kl_beta": beta,
           "all_pass": all(c["pass"] for c in checks.values()), "checks": checks,
           "defect": "ppo_policy_loss used torch.maximum(surr1, surr2); fixed to torch.minimum (pessimistic clipped surrogate)"}
    save_json(f"{cfg['results_dir']}/ppo_validation.json", out)
    for name, c in checks.items():
        print(f"{name:42s} {'PASS' if c['pass'] else 'FAIL'}")
    if not out["all_pass"]:
        raise SystemExit("PPO validation failed")


if __name__ == "__main__":
    main()
