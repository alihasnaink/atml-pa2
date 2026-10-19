from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_fitting_rows(rows, tokenizer, cfg):
    """Keep pairs whose chat-formatted prompt leaves at least `min_response_tokens`
    of the `max_sequence_length` budget for the response.

    encode_prompt_response() refuses prompts longer than the budget (some
    UltraFeedback prompts are ~1.9k tokens), and a prompt that only leaves a
    handful of response tokens would compare two nearly empty responses.
    The same rule is applied to every train and eval file, and the dropped
    prompt IDs are saved with each run.
    """
    limit = int(cfg["max_sequence_length"]) - int(cfg["min_response_tokens"])
    kept, dropped = [], []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
        )
        (kept if len(prompt_ids) <= limit else dropped).append(row)
    return kept, [r["prompt_id"] for r in dropped]


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows, dropped_ids = filter_fitting_rows(read_jsonl(path), tokenizer, cfg)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dropped_ids": dropped_ids,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader, optimizer = bundle["cfg"], bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    log_dir = repo_path(cfg["results_dir"]) / "train_logs"
    log_path = log_dir / f"{run_name}.jsonl"
    log_path.unlink(missing_ok=True)

    accum = int(cfg["grad_accum_steps"])
    params = trainable_parameters(model)
    device = next(model.parameters()).device
    # The frozen base runs in fp16 on T4 (no bf16); loss scaling keeps small
    # gradients from underflowing on their way back to the fp32 LoRA weights.
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    print(f"[{run_name}] beta={beta} examples={len(bundle['rows'])} dropped={len(bundle['dropped_ids'])} "
          f"micro_batches/epoch={len(loader)} accum={accum}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, seen, window = 0, 0, []
    n_micro = len(loader)
    for epoch in range(int(cfg["epochs"])):
        for i, (chosen, rejected) in enumerate(loader):
            chosen, rejected = to_device(chosen, device), to_device(rejected, device)

            # Frozen reference = the same base model with the LoRA adapter disabled.
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)
            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

            # Average over the micro-batches of this optimizer step (the last group may be smaller).
            group_start = (i // accum) * accum
            group_size = min(accum, n_micro - group_start)
            scaler.scale(loss / group_size).backward()

            seen += chosen["input_ids"].shape[0]
            window.append({"loss": float(loss.detach()), **{k: float(v) for k, v in diag.items()}})

            if (i + 1) % accum == 0 or (i + 1) == n_micro:
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                record = {k: sum(w[k] for w in window) / len(window) for k in window[0]}
                record.update({
                    "run": run_name, "step": step, "epoch": epoch, "examples_seen": seen,
                    "grad_norm": float(grad_norm), "loss_scale": float(scaler.get_scale()),
                    # GradScaler skips the update (and lowers the scale) when gradients overflow.
                    "skipped_step": bool(scaler.get_scale() < scale_before),
                    "elapsed_s": elapsed(),
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                })
                append_jsonl(log_path, record)
                print(f"[{run_name}] step {step} seen {seen} loss {record['loss']:.4f} "
                      f"acc {record['preference_accuracy']:.3f} margin {record['margin_mean']:.3f} "
                      f"gnorm {record['grad_norm']:.3f} t {record['elapsed_s']/60:.1f}m", flush=True)
                window = []

    model.save_pretrained(str(output))
    summary = {
        "run": run_name,
        "beta": beta,
        "dataset": bundle["dataset_path"],
        "adapter": str(output.relative_to(repo_path("."))) if output.is_relative_to(repo_path(".")) else str(output),
        "n_train_pairs": len(bundle["rows"]),
        "max_examples": max_examples,
        "train_prompt_ids": [r["prompt_id"] for r in bundle["rows"]],
        "dropped_prompt_ids": bundle["dropped_ids"],
        "optimizer_steps": step,
        "skipped_steps": sum(1 for r in read_jsonl(log_path) if r["skipped_step"]),
        "epochs": int(cfg["epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "learning_rate": float(cfg["learning_rate"]),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "min_response_tokens": int(cfg["min_response_tokens"]),
        "lora": cfg["lora"],
        "seed": int(cfg["seed"]),
        "wall_clock_s": elapsed(),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    save_json(log_dir / f"{run_name}_summary.json", summary)
    print(f"[{run_name}] saved adapter to {output} ({summary['wall_clock_s']/60:.1f} min)", flush=True)

    del bundle, model, optimizer
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
