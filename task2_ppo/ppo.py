from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    L_clip = E_t[ min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t) ];  loss = -L_clip.

    Fix of the released defect: the starter took torch.maximum of the two
    surrogates. PPO's pessimistic bound needs the MINIMUM; with the maximum,
    ratios that move in the advantage direction are never clipped (unbounded
    updates) while ratios moving against it get zero gradient.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(outside, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def clip_diagnostics(ratio, advantage, mask, eps):
    """Token fractions for the clipping study (same response mask for both).

    clip_fraction:     rho outside [1-eps, 1+eps] before clipping (manual definition).
    affected_fraction: clipping actually binds in the min(), i.e. the clipped term is
                       selected and the token's policy gradient is zero
                       (A > 0 and rho > 1+eps, or A < 0 and rho < 1-eps).
    """
    above, below = ratio > 1.0 + eps, ratio < 1.0 - eps
    affected = (above & (advantage > 0)) | (below & (advantage < 0))
    return {
        "clip_fraction": float(masked_mean((above | below).float(), mask)),
        "affected_fraction": float(masked_mean(affected.float(), mask)),
        "clip_fraction_pos_adv": float(masked_mean((above | below).float(), mask * (advantage > 0))),
        "clip_fraction_neg_adv": float(masked_mean((above | below).float(), mask * (advantage < 0))),
    }


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
