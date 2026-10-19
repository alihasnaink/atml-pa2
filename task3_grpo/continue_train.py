from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from common.rl_utils import rollout_prompt_order, token_entropy
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_std,
    grpo_policy_loss,
    is_informative,
    mask_truncated_sequences,
)


def disable_dropout(model):
    """Zero every dropout so old/new log-probs come from the same deterministic policy
    (policy_epochs=1 then starts at ratio exactly 1) while gradient checkpointing stays on."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts, dropped = rollout_prompt_order(read_jsonl(cfg["paths"]["rl_prompt_train"]), tokenizer, cfg)
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "dropped_prompt_ids": dropped,
        "optimizer": optimizer,
    }


def grad_norm(params, scale: float) -> float:
    total = sum(float((p.grad.float() / scale).pow(2).sum()) for p in params if p.grad is not None)
    return total ** 0.5


def grpo_update(bundle, row, update_idx, loss_type, scaler, per_sequence_grads=False):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    K, L = int(cfg["num_generations"]), int(cfg["max_completion_length"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    gen = cfg["generation"]
    messages = prompt_messages(row)
    params = trainable_parameters(policy)

    # 1) K completions for the same prompt (seeded per update: every fork sees the same sampling stream).
    set_seed(int(cfg["seed"]) + update_idx)
    g = batch_generate(policy, tok, [messages] * K, max_prompt_length=4096, max_new_tokens=L,
                       temperature=float(gen["temperature"]), top_p=float(gen["top_p"]), do_sample=bool(gen["do_sample"]))
    pw, Tm = g["prompt_width"], max(g["response_lengths"])
    # clone(): generate() ran under inference_mode, whose tensors cannot enter autograd.
    seq, attn = g["sequences"][:, :pw + Tm].clone(), g["attention_mask"][:, :pw + Tm].clone()
    rid, gen_mask = g["response_ids"][:, :Tm].clone(), g["response_mask"][:, :Tm].clone()
    truncated = g["truncated"]
    loss_mask = mask_truncated_sequences(gen_mask, truncated) if cfg.get("mask_truncated_completions", True) else gen_mask

    # 2) Rewards and group-relative advantages (truncated completions still count in the group statistics).
    with torch.no_grad():
        rewards = score_reward_pairs(bundle["reward_model"], bundle["reward_tokenizer"], [messages] * K, g["responses"],
                                     max_length=int(cfg["reward_max_length"])).float().to(gen_mask.device)
        gid = torch.zeros(K, dtype=torch.long, device=rewards.device)
        adv = group_relative_advantages(rewards, gid)
        std = group_reward_std(rewards, gid)[0]

        # 3) Old (= current, policy_epochs=1) and reference log-probs, entropy.
        old_logp, logits = response_token_logprobs(policy, seq, attn, pw, rid)
        entropy = float(token_entropy(logits, gen_mask))
        del logits
        with reference_mode(policy):
            ref_logp = response_token_logprobs(policy, seq, attn, pw, rid)[0]

    # 4) Optional length-conditioned gradient statistic: each completion's own policy-term gradient norm.
    seq_grad_norms = []
    if per_sequence_grads:
        for k in range(K):
            if float(loss_mask[k].sum()) == 0:
                seq_grad_norms.append(0.0)
                continue
            sl = slice(k, k + 1)
            new_k = response_token_logprobs(policy, seq[sl], attn[sl], pw, rid[sl])[0]
            term, _ = grpo_policy_loss(new_k, old_logp[sl], adv[sl], loss_mask[sl], ref_logp[sl], eps, 0.0, loss_type, L)
            scaler.scale(term / K).backward()  # this completion's share of the batch policy term
            seq_grad_norms.append(grad_norm(params, scaler.get_scale()))
            bundle["optimizer"].zero_grad(set_to_none=True)

    # 5) One clipped GRPO step on all K completions.
    new_logp = response_token_logprobs(policy, seq, attn, pw, rid)[0]
    loss, diag = grpo_policy_loss(new_logp, old_logp, adv, loss_mask, ref_logp, eps, beta, loss_type, L)
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(bundle["optimizer"])
    gnorm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
    scaler.step(bundle["optimizer"])
    scaler.update()
    bundle["optimizer"].zero_grad(set_to_none=True)

    lengths = g["response_lengths"]
    return {
        "prompt_id": row["prompt_id"],
        "rewards": rewards.tolist(),
        "advantages": adv.tolist(),
        "reward": float(rewards.mean()),
        "group_reward_std": float(std),
        "informative": bool(is_informative(std)),
        "kl": float(sampled_kl(old_logp, ref_logp, gen_mask)),   # course estimator, all generated tokens
        "kl_k3": float(diag["sampled_kl"]),                      # k3 term inside the loss (loss tokens)
        "entropy": entropy,
        "policy_term": float(diag["policy_term"]),
        "loss": float(loss.detach()),
        "grad_norm": float(gnorm),
        "clip_fraction": float(diag["clip_fraction"]),
        "ratio_mean": float(diag["ratio_mean"]),
        "response_tokens": lengths,
        "response_tokens_mean": sum(lengths) / K,
        "truncated": truncated,
        "n_masked": sum(truncated) if cfg.get("mask_truncated_completions", True) else 0,
        "in_loss": [bool(x > 0) for x in loss_mask.sum(-1).tolist()],
        "tokens_generated": int(gen_mask.sum()),
        "tokens_in_loss": int(loss_mask.sum()),
        "seq_grad_norms": seq_grad_norms,
        "skipped_step": bool(scaler.get_scale() < scale_before),
        "responses": g["responses"],
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", per_sequence_grads: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if int(cfg["prompts_per_update"]) != 1 or int(cfg["policy_epochs"]) != 1:
        raise ValueError("This loop implements the release setting prompts_per_update=1, policy_epochs=1.")

    n_updates = int(cfg["updates"])
    log_dir = repo_path(cfg["results_dir"]) / "train_logs"
    log_path = log_dir / f"{run_name}.jsonl"
    log_path.unlink(missing_ok=True)
    device = next(bundle["policy"].parameters()).device
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    print(f"[{run_name}] updates={n_updates} loss_type={loss_type} K={cfg['num_generations']} "
          f"usable_prompts={len(bundle['prompt_rows'])} dropped_long_prompts={len(bundle['dropped_prompt_ids'])}", flush=True)

    generated = 0
    for u in range(n_updates):
        row = bundle["prompt_rows"][u % len(bundle["prompt_rows"])]
        rec = grpo_update(bundle, row, u, loss_type, scaler, per_sequence_grads)
        generated += rec["tokens_generated"]
        rec.update({"run": run_name, "update": u + 1, "loss_type": loss_type, "generated_tokens_cum": generated,
                    "elapsed_s": elapsed(),
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0})
        append_jsonl(log_path, rec)
        print(f"[{run_name}] update {u + 1}/{n_updates} reward {rec['reward']:.3f} std {rec['group_reward_std']:.3f} "
              f"kl {rec['kl']:.4f} ent {rec['entropy']:.3f} len {rec['response_tokens_mean']:.0f} masked {rec['n_masked']} "
              f"gnorm {rec['grad_norm']:.3f} t {rec['elapsed_s'] / 60:.1f}m", flush=True)

    bundle["policy"].save_pretrained(str(out))
    summary = {
        "run": run_name, "updates": n_updates, "loss_type": loss_type,
        "adapter": str(out.relative_to(repo_path("."))) if out.is_relative_to(repo_path(".")) else str(out),
        "prompt_ids": [bundle["prompt_rows"][u % len(bundle["prompt_rows"])]["prompt_id"] for u in range(n_updates)],
        "dropped_long_prompt_ids": bundle["dropped_prompt_ids"],
        "generated_tokens": generated,
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "hyperparameters": {k: cfg[k] for k in ("prompts_per_update", "policy_epochs", "num_generations", "learning_rate",
                                                "clip_epsilon", "kl_beta", "max_prompt_length", "max_completion_length",
                                                "mask_truncated_completions", "max_grad_norm", "seed")},
        "generation": cfg["generation"],
    }
    save_json(log_dir / f"{run_name}_summary.json", summary)
    print(f"[{run_name}] saved to {out} ({summary['wall_clock_s'] / 60:.1f} min, peak {summary['peak_vram_gb']:.1f} GB)", flush=True)
    del bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--per-sequence-grads", action="store_true", help="also log each completion's gradient norm")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.per_sequence_grads)


if __name__ == "__main__":
    main()
