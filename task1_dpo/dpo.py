from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    L = -E[ log sigmoid( beta * ( [log pi(y+|x) - log ref(y+|x)]
                                 - [log pi(y-|x) - log ref(y-|x)] ) ) ]

    Fix of the released defect: the starter used ``policy_margin + ref_margin``.
    The reference log-ratio must be SUBTRACTED, otherwise the loss is not
    log(2) at initialization (policy == reference) and the policy is pulled
    towards the *negative* of the reference margin instead of being measured
    relative to it.
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # m_theta in the manual: preference margin relative to the reference policy.
    margin = policy_margin - ref_margin
    logits = beta * margin

    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "margin_mean": margin.detach().mean(),
        # Implicit DPO rewards beta * log(pi / ref) for chosen and rejected responses.
        "chosen_reward_mean": (beta * (policy_chosen_logp - ref_chosen_logp)).detach().mean(),
        "rejected_reward_mean": (beta * (policy_rejected_logp - ref_rejected_logp)).detach().mean(),
        "preference_accuracy": (margin > 0).float().mean().detach(),
    }
