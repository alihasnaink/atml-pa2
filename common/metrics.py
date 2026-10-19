from __future__ import annotations

import re
import numpy as np
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def sampled_kl(policy_logp: torch.Tensor, ref_logp: torch.Tensor, mask: torch.Tensor):
    return masked_mean(policy_logp - ref_logp, mask)


def sample_entropy(sampled_logp: torch.Tensor, mask: torch.Tensor):
    return -masked_mean(sampled_logp, mask)


def mean_response_length(mask: torch.Tensor):
    return float(mask.sum(-1).float().mean().item())


def preference_accuracy(chosen_logp, rejected_logp):
    return float((chosen_logp > rejected_logp).float().mean().item())


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def parse_word_limit(prompt: str):
    patterns = [
        r"(?:at most|no more than|under|within)\s+(\d+)\s+words?",
        r"(?:in|use)\s+(\d+)\s+words?\s+(?:or fewer|max(?:imum)?)",
        r"(?:maximum|max)\s+(?:of\s+)?(\d+)\s+words?",
    ]
    lower = str(prompt).lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return int(m.group(1))
    return None


def word_limit_compliance(prompt: str, response: str):
    limit = parse_word_limit(prompt)
    if limit is None:
        return None
    return float(word_count(response) <= limit)


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """Mean of (b - a) over paired items with a 95% bootstrap CI."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {"mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": len(d)}
