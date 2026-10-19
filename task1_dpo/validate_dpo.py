"""Validate task1_dpo.dpo.dpo_loss against the DPO equation in the manual.

Runs on CPU in seconds and needs no model. Writes the checks (including what
the released buggy objective would have produced) to results/task1_dpo/.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from common.data import load_yaml
from common.logging_utils import save_json
from task1_dpo.dpo import dpo_loss


def reference_loss(pc, pr, rc, rr, beta):
    """Literal transcription of the manual equation, one example at a time."""
    vals = []
    for i in range(len(pc)):
        z = beta * ((pc[i] - rc[i]) - (pr[i] - rr[i]))
        vals.append(-math.log(1.0 / (1.0 + math.exp(-z))))
    return sum(vals) / len(vals)


def released_buggy_loss(pc, pr, rc, rr, beta):
    """The objective as released in the starter code (reference margin added)."""
    return -F.logsigmoid(beta * ((pc - pr) + (rc - rr))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    beta = float(cfg["beta"])
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    checks = {}

    # 1) Matches the manual equation on random sequence log-probs.
    pc, pr, rc, rr = (-50 * torch.rand(64, generator=g, dtype=torch.float64) for _ in range(4))
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    ref = reference_loss(pc.tolist(), pr.tolist(), rc.tolist(), rr.tolist(), beta)
    checks["matches_manual_equation"] = {
        "fixed_loss": float(loss), "manual_loss": ref,
        "released_buggy_loss": float(released_buggy_loss(pc, pr, rc, rr, beta)),
        "pass": abs(float(loss) - ref) < 1e-9,
    }

    # 2) At initialization the policy equals the reference => loss = log 2, margin 0.
    loss0, diag0 = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta)
    checks["policy_equals_reference_gives_log2"] = {
        "fixed_loss": float(loss0), "log2": math.log(2.0),
        "released_buggy_loss": float(released_buggy_loss(pc, pr, pc, pr, beta)),
        "fixed_accuracy": float(diag0["preference_accuracy"]),
        "pass": abs(float(loss0) - math.log(2.0)) < 1e-9,
    }

    # 3) Invariance: adding the same constant to policy and reference log-probs of a
    #    response must not change the loss (only log-ratios matter).
    shift = 7.0
    loss_s, _ = dpo_loss(pc + shift, pr, rc + shift, rr, beta)
    checks["invariant_to_shared_shift"] = {
        "fixed_delta": float(loss_s - loss),
        "released_buggy_delta": float(released_buggy_loss(pc + shift, pr, rc + shift, rr, beta)
                                      - released_buggy_loss(pc, pr, rc, rr, beta)),
        "pass": abs(float(loss_s - loss)) < 1e-9,
    }

    # 4) Gradient direction: raising log pi(y+) lowers the loss, raising log pi(y-) raises it.
    pc_v, pr_v = pc.clone().requires_grad_(True), pr.clone().requires_grad_(True)
    l, _ = dpo_loss(pc_v, pr_v, rc, rr, beta)
    l.backward()
    checks["gradient_signs"] = {
        "all_dL_dchosen_negative": bool((pc_v.grad < 0).all()),
        "all_dL_drejected_positive": bool((pr_v.grad > 0).all()),
        "pass": bool((pc_v.grad < 0).all() and (pr_v.grad > 0).all()),
    }

    # 5) Accuracy diagnostic equals the manual definition (m_theta > 0).
    m = (pc - rc) - (pr - rr)
    checks["accuracy_definition"] = {
        "diag": float(diag["preference_accuracy"]), "manual": float((m > 0).double().mean()),
        "pass": float(diag["preference_accuracy"]) == float((m > 0).double().mean()),
    }

    out = {"beta": beta, "all_pass": all(c["pass"] for c in checks.values()), "checks": checks,
           "defect": "starter computed beta*(policy_margin + ref_margin); fixed to beta*(policy_margin - ref_margin)"}
    save_json(f"{cfg['results_dir']}/dpo_loss_validation.json", out)
    for name, c in checks.items():
        print(f"{name:40s} {'PASS' if c['pass'] else 'FAIL'}")
    if not out["all_pass"]:
        raise SystemExit("DPO objective validation failed")


if __name__ == "__main__":
    main()
