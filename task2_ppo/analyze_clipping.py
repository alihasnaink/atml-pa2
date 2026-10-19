"""Clipping study.

Part A - fixed cached rollout batch (no new generation):
  1. Rebuild the 32 cached rollouts (prompt from rl_prompt_pool_eval + response tokens)
     and check the reconstruction against the cache (current/reference log-probs, values).
  2. Advantages/returns from the cached rewards, cached old/ref log-probs (KL shaping
     with the release beta_KL) and cached critic values via GAE, whitened over the batch.
  3. Step 0: ratio = pi_midpoint / pi_old on the cached tokens; for every epsilon report
     the clipped surrogate, clip fraction (rho outside the band) and affected-token
     fraction (clipping binds) using the same response mask. A fine epsilon grid is
     saved for a curve.
  4. For each required epsilon, `cache_update_steps` full-batch PPO steps on the cached
     batch from the midpoint, measuring the same quantities before every step and after
     the last one (the immediate effect of epsilon on an actual update).
Part B - matched short continuation forks for each epsilon (beta_KL fixed), evaluated
  with the common held-out protocol.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, load_value_model, reference_mode, trainable_parameters
from task2_ppo.continue_train import response_values
from task2_ppo.forks import run_fork
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_rollouts(rows, cfg, tokenizer, device):
    """Token ids for prompt + cached response (+ EOS when the rollout terminated)."""
    prompts = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    built, mismatched = [], []
    for row in rows:
        messages = prompt_messages(prompts[row["prompt_id"]])
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        resp_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos"):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        if len(resp_ids) != len(row["old_logprobs"]):
            mismatched.append(row["prompt_id"])
            continue
        seq = torch.tensor([prompt_ids + resp_ids], device=device)
        built.append({"row": row, "seq": seq, "attn": torch.ones_like(seq), "pw": len(prompt_ids),
                      "rid": seq[:, len(prompt_ids):], "T": len(resp_ids)})
    return built, mismatched


def pad(list_of_1d, T):
    out = torch.zeros(len(list_of_1d), T)
    for i, x in enumerate(list_of_1d):
        out[i, :len(x)] = torch.as_tensor(x, dtype=torch.float32).flatten()[:T]
    return out


def batch_stats(new_logp, old, start, adv, mask, eps):
    ratio = torch.exp(new_logp - old)
    unclipped = masked_mean(ratio * adv, mask)
    clipped = masked_mean(torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv), mask)
    valid = ratio[mask.bool()]
    return {
        "surrogate": float(clipped), "unclipped_surrogate": float(unclipped),
        **clip_diagnostics(ratio, adv, mask, eps),
        "ratio_mean": float(valid.mean()), "ratio_std": float(valid.std()),
        "ratio_p01": float(valid.quantile(0.01)), "ratio_p99": float(valid.quantile(0.99)),
        "kl_old_new": float(masked_mean(old - new_logp, mask)),        # sampled KL(old || current)
        "kl_midpoint_new": float(masked_mean(start - new_logp, mask)),  # drift from the step-0 policy
    }


def cached_batch_study(cfg):
    timer = wall_timer()
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    policy.eval()
    value_model = load_value_model({**cfg, "dtype": cfg.get("value_dtype", cfg["dtype"])},
                                   cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    device = next(policy.parameters()).device
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    built, mismatched = rebuild_rollouts(rows, cfg, tok, device)
    print(f"Cached PPO rollouts: {len(rows)} (rebuilt {len(built)}, token mismatches {len(mismatched)})", flush=True)

    # --- reconstruction check and current (midpoint) log-probs
    cur, recon = [], []
    with torch.no_grad():
        for b in built:
            args = (b["seq"], b["attn"], b["pw"], b["rid"])
            lp = response_token_logprobs(policy, *args)[0][0].float().cpu()
            with reference_mode(policy):
                ref = response_token_logprobs(policy, *args)[0][0].float().cpu()
            v_full = response_values(value_model, b["seq"], b["attn"], b["pw"], b["T"] + 1)[0].cpu()
            r = b["row"]
            cv = torch.as_tensor(r["values"]).float() if "values" in r else None
            recon.append({
                "prompt_id": r["prompt_id"], "tokens": b["T"],
                "mae_midpoint_vs_cached_old": float((lp - r["old_logprobs"].float()).abs().mean()),
                "mae_reference_vs_cached_ref": float((ref - r["ref_logprobs"].float()).abs().mean()),
                # critic alignment: V at the position predicting token t (used in training) vs. one later
                "mae_value_predicting_pos": float((v_full[:b["T"]] - cv).abs().mean()) if cv is not None else None,
                "mae_value_token_pos": float((v_full[1:b["T"] + 1] - cv).abs().mean()) if cv is not None else None,
            })
            cur.append(lp)

    T = max(b["T"] for b in built)
    mask = pad([torch.ones(b["T"]) for b in built], T)
    old = pad([b["row"]["old_logprobs"] for b in built], T)
    ref = pad([b["row"]["ref_logprobs"] for b in built], T)
    values = pad([b["row"]["values"] for b in built], T)
    start = pad(cur, T)
    reward = torch.tensor([float(b["row"].get("effective_terminal_reward", b["row"].get("raw_terminal_reward"))) for b in built])

    rewards = shaped_rewards(reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_n = normalize_advantages(adv, mask)

    # --- step 0: pure geometry of epsilon on the fixed batch
    grid = sorted(set(np.round(np.linspace(0.01, 1.0, 100), 3).tolist() + [float(e) for e in cfg["clip_values"]]))
    step0_grid = [{"eps": e, **batch_stats(start, old, start, adv_n, mask, e)} for e in grid]

    # --- a few real PPO steps on the cached batch for each required epsilon
    params = trainable_parameters(policy)
    init = [p.detach().clone() for p in params]
    n_tok = float(mask.sum())
    steps = int(cfg["cache_update_steps"])
    per_eps = {}
    for eps in [float(e) for e in cfg["clip_values"]]:
        for p, p0 in zip(params, init):
            p.data.copy_(p0)
        opt = torch.optim.AdamW(params, lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
        trace = []
        for step in range(steps + 1):
            train = step < steps
            new_rows = []
            for i, b in enumerate(built):
                args = (b["seq"], b["attn"], b["pw"], b["rid"])
                with torch.set_grad_enabled(train):
                    lp = response_token_logprobs(policy, *args)[0][0]
                    if train:
                        T_i = b["T"]
                        loss, _, _ = ppo_policy_loss(lp, old[i, :T_i].to(device), adv_n[i, :T_i].to(device),
                                                     mask[i, :T_i].to(device), eps)
                        scaler.scale(loss * (T_i / n_tok)).backward()  # token-weighted full-batch mean
                new_rows.append(lp.detach().float().cpu())
            stats = batch_stats(pad(new_rows, T), old, start, adv_n, mask, eps)
            if train:
                scaler.unscale_(opt)
                stats["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            trace.append({"step": step, **stats})
            print(f"  eps={eps:.2f} step {step}: clip {stats['clip_fraction']:.4f} affected {stats['affected_fraction']:.4f} "
                  f"surrogate {stats['surrogate']:.4f} kl_mid {stats['kl_midpoint_new']:.5f}", flush=True)
        per_eps[f"{eps:.2f}"] = trace

    out = {
        "n_rollouts": len(rows), "n_rebuilt": len(built), "token_mismatch_prompt_ids": mismatched,
        "prompt_ids": [b["row"]["prompt_id"] for b in built],
        "response_tokens_total": int(n_tok),
        "kl_beta_for_shaping": float(cfg["kl_beta"]), "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]),
        "policy_learning_rate": float(cfg["policy_learning_rate"]), "cache_update_steps": steps,
        "reconstruction": {
            "per_row": recon,
            **{k: float(np.mean([r[k] for r in recon if r[k] is not None]))
               for k in ("mae_midpoint_vs_cached_old", "mae_reference_vs_cached_ref", "mae_value_predicting_pos", "mae_value_token_pos")
               if any(r[k] is not None for r in recon)},
        },
        "advantage_raw_mean": float(masked_mean(adv, mask)), "advantage_raw_std": float(adv[mask.bool()].std()),
        "step0": {f"{float(e):.2f}": next(s for s in step0_grid if abs(s["eps"] - float(e)) < 1e-9) for e in cfg["clip_values"]},
        "step0_grid": step0_grid,
        "update_steps": per_eps,
        "wall_clock_s": timer(),
    }
    save_json(repo_path(cfg["results_dir"]) / "clipping_cached.json", out)
    print("Saved cached-batch clipping study", flush=True)
    del policy, value_model
    clear_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true", help="only run the cached-batch study")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required epsilon values:", cfg["clip_values"])
    if not (repo_path(cfg["results_dir"]) / "clipping_cached.json").exists():
        cached_batch_study(cfg)
    else:
        print("Cached-batch study exists, skipping")
    if not args.skip_forks:
        for eps in cfg["clip_values"]:
            run_fork(args.config, float(eps), float(cfg["kl_beta"]))


if __name__ == "__main__":
    main()
