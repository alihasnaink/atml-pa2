from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_utils import effective_reward, rollout_prompt_order, token_entropy
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


# ---------------------------------------------------------------- critic helpers (also used by analyze_clipping)

def response_values(value_model, sequences, attention_mask, prompt_width, steps):
    """V(s_t) for response token t = critic output at the position that predicts token t."""
    v = token_values(value_model, sequences, attention_mask)
    return v[:, prompt_width - 1: prompt_width - 1 + steps].float()


def explained_variance(values, returns, mask):
    v, r = values[mask.bool()], returns[mask.bool()]
    var_r = r.var(unbiased=False)
    return float(1 - (r - v).var(unbiased=False) / var_r) if var_r > 0 else float("nan")


# ---------------------------------------------------------------- setup

def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    policy.eval()  # no LoRA dropout: old/new log-probs come from the same deterministic policy
    value_model = load_value_model(
        {**cfg, "dtype": cfg.get("value_dtype", cfg["dtype"])},
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    value_model.eval()
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = rollout_prompt_order(read_jsonl(cfg["paths"]["rl_prompt_train"]), tokenizer, cfg)

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "dropped_prompt_ids": dropped,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ---------------------------------------------------------------- one PPO update

def ppo_update(bundle, row, update_idx, eps, beta, scaler):
    cfg, tok, policy, value_model = bundle["cfg"], bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    gen = cfg["generation"]
    messages = prompt_messages(row)

    # 1) On-policy rollout (seeded per update so every fork sees the same sampling stream).
    set_seed(int(cfg["seed"]) + update_idx)
    g = batch_generate(policy, tok, [messages], max_prompt_length=4096,
                       max_new_tokens=int(cfg["max_response_length"]),
                       temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
    pw = g["prompt_width"]
    T = max(1, g["response_lengths"][0])
    # clone(): generate() ran under inference_mode, whose tensors cannot enter autograd.
    seq, attn = g["sequences"][:, :pw + T].clone(), g["attention_mask"][:, :pw + T].clone()
    rid, mask = g["response_ids"][:, :T].clone(), g["response_mask"][:, :T].clone()
    truncated = g["truncated"][0]

    # 2) Old/reference log-probs, entropy, learned reward, critic values.
    with torch.no_grad():
        old_logp, logits = response_token_logprobs(policy, seq, attn, pw, rid)
        entropy = float(token_entropy(logits, mask))
        del logits
        with reference_mode(policy):
            ref_logp = response_token_logprobs(policy, seq, attn, pw, rid)[0]
        raw = float(score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], [messages], g["responses"],
                                       max_length=int(cfg["reward_max_length"]))[0])
        reward = effective_reward(raw, truncated, cfg)
        values = response_values(value_model, seq, attn, pw, T)

    # 3) KL-shaped token rewards, GAE advantages/returns.
    rewards = shaped_rewards(torch.tensor([reward], device=mask.device), old_logp, ref_logp, mask, beta)
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv_used = normalize_advantages(adv, mask) if cfg.get("normalize_advantages", True) else adv

    # 4) PPO epochs: clipped policy update and critic regression on the same rollout.
    pol_params, val_params = trainable_parameters(policy), trainable_parameters(value_model)
    epochs = []
    for epoch in range(int(cfg["ppo_epochs"])):
        new_logp = response_token_logprobs(policy, seq, attn, pw, rid)[0]
        pol_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_used, mask, eps)
        diag = clip_diagnostics(ratio, adv_used, mask, eps)
        scale_before = scaler.get_scale()
        scaler.scale(pol_loss).backward()
        scaler.unscale_(bundle["policy_optimizer"])
        pol_gnorm = torch.nn.utils.clip_grad_norm_(pol_params, float(cfg["max_grad_norm"]))
        scaler.step(bundle["policy_optimizer"])
        scaler.update()
        bundle["policy_optimizer"].zero_grad(set_to_none=True)

        v_new = response_values(value_model, seq, attn, pw, T)
        val_loss = float(cfg["value_coef"]) * value_mse_loss(v_new, returns, mask)
        val_loss.backward()
        val_gnorm = torch.nn.utils.clip_grad_norm_(val_params, float(cfg["max_grad_norm"]))
        bundle["value_optimizer"].step()
        bundle["value_optimizer"].zero_grad(set_to_none=True)

        epochs.append({
            "policy_loss": float(pol_loss.detach()), "value_loss": float(val_loss.detach()),
            "clip_fraction": float(clip_frac), "affected_fraction": diag["affected_fraction"],
            "ratio_mean": float(masked_mean(ratio, mask)), "ratio_max": float(ratio[mask.bool()].max()),
            "ratio_min": float(ratio[mask.bool()].min()),
            "policy_grad_norm": float(pol_gnorm), "value_grad_norm": float(val_gnorm),
            "skipped_step": bool(scaler.get_scale() < scale_before),
        })

    # 5) How far did this update move the policy on its own rollout? (stability diagnostic)
    with torch.no_grad():
        after_logp = response_token_logprobs(policy, seq, attn, pw, rid)[0]
        shift = after_logp - old_logp

    mean = lambda k: sum(e[k] for e in epochs) / len(epochs)
    return {
        "prompt_id": row["prompt_id"],
        "response_tokens": T,
        "terminated_with_eos": g["terminated_with_eos"][0],
        "truncated": truncated,
        "raw_reward": raw,
        "reward": reward,
        "kl": float(sampled_kl(old_logp, ref_logp, mask)),             # token-mean, course estimator
        "kl_sequence": float(((old_logp - ref_logp) * mask).sum()),
        "entropy": entropy,
        "policy_loss": mean("policy_loss"),
        "value_loss": mean("value_loss"),
        "policy_grad_norm": mean("policy_grad_norm"),
        "value_grad_norm": mean("value_grad_norm"),
        "clip_fraction": mean("clip_fraction"),
        "affected_fraction": mean("affected_fraction"),
        # The first epoch starts at ratio = 1, so the last epoch carries the clipping signal.
        "clip_fraction_last_epoch": epochs[-1]["clip_fraction"],
        "affected_fraction_last_epoch": epochs[-1]["affected_fraction"],
        "value_mean": float(masked_mean(values, mask)),
        "return_mean": float(masked_mean(returns, mask)),
        "explained_variance": explained_variance(values, returns, mask),
        "update_kl_old_new": float(masked_mean(-shift, mask)),   # sampled KL(old || new) on the rollout
        "update_max_abs_log_ratio": float(shift[mask.bool()].abs().max()),
        "skipped_steps": sum(e["skipped_step"] for e in epochs),
        "epochs": epochs,
        "response": g["responses"][0],
    }


# ---------------------------------------------------------------- driver

def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    if int(cfg["prompts_per_update"]) != 1:
        raise ValueError("This loop implements the release setting prompts_per_update=1 (one rollout per update).")
    n_updates, eps, beta = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    log_dir = repo_path(cfg["results_dir"]) / "train_logs"
    log_path = log_dir / f"{run_name}.jsonl"
    log_path.unlink(missing_ok=True)

    device = next(bundle["policy"].parameters()).device
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    print(f"[{run_name}] updates={n_updates} eps={eps} kl_beta={beta} usable_prompts={len(bundle['prompt_rows'])} "
          f"dropped_long_prompts={len(bundle['dropped_prompt_ids'])}", flush=True)

    generated = 0
    for u in range(n_updates):
        row = bundle["prompt_rows"][u % len(bundle["prompt_rows"])]
        rec = ppo_update(bundle, row, u, eps, beta, scaler)
        generated += rec["response_tokens"]
        rec.update({"run": run_name, "update": u + 1, "clip_epsilon": eps, "kl_beta": beta,
                    "generated_tokens_cum": generated, "elapsed_s": elapsed(),
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0})
        append_jsonl(log_path, rec)
        print(f"[{run_name}] update {u + 1}/{n_updates} reward {rec['reward']:.3f} kl {rec['kl']:.4f} "
              f"ent {rec['entropy']:.3f} len {rec['response_tokens']} clip {rec['clip_fraction_last_epoch']:.3f} "
              f"vloss {rec['value_loss']:.3f} ev {rec['explained_variance']:.2f} t {rec['elapsed_s'] / 60:.1f}m", flush=True)

    bundle["policy"].save_pretrained(str(out))
    bundle["value_model"].save_pretrained(str(out / "value"))
    summary = {
        "run": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta,
        "adapter": str(out.relative_to(repo_path("."))) if out.is_relative_to(repo_path(".")) else str(out),
        "prompt_ids": [bundle["prompt_rows"][u % len(bundle["prompt_rows"])]["prompt_id"] for u in range(n_updates)],
        "dropped_long_prompt_ids": bundle["dropped_prompt_ids"],
        "generated_tokens": generated,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "hyperparameters": {k: cfg[k] for k in ("prompts_per_update", "ppo_epochs", "policy_learning_rate",
                                                "value_lora_learning_rate", "value_head_learning_rate", "value_train_mode",
                                                "gamma", "gae_lambda", "value_coef", "missing_eos_penalty",
                                                "max_prompt_length", "max_response_length", "max_grad_norm", "seed")},
        "generation": cfg["generation"],
    }
    save_json(log_dir / f"{run_name}_summary.json", summary)
    print(f"[{run_name}] saved to {out} ({summary['wall_clock_s'] / 60:.1f} min, peak {summary['peak_vram_gb']:.1f} GB)", flush=True)

    del bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
