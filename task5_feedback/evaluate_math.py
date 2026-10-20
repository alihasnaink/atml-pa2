"""Steps 1 and 3 - in-domain (GSM8K) and out-of-domain (SVAMP) comparison of SFT, RLVR and RLAIF.

For one dataset (`--dataset gsm` or `--dataset transfer`):
  1. generate one deterministic (greedy) response per prompt for every policy, same decoding and
     maximum length (`math_max_new_tokens`) for all three;
  2. score every response with the exact final-answer verifier (accuracy, format compliance, failure type);
  3. compare responses on the same prompt with the fixed pairwise AI judge, for every policy pair
     (SFT-RLVR, SFT-RLAIF, RLVR-RLAIF). The SFT pairs give the RLAIF pairwise win rate against SFT;
     all pairs give the verifier-judge agreement.

Outputs (results/task5_feedback/<dataset>/):
  generations_<policy>.jsonl   responses + verifier fields (fixed dataset order)
  pairwise.jsonl               one row per (prompt, policy pair): judge verdict, verifier verdict
  judge_cache.json             the judge's persistent cache (one file per dataset, so two GPUs never share it)
  summary.json                 per-policy metrics, win rates vs SFT, agreement, judge cost
"""
from __future__ import annotations

import argparse
import itertools
import re
import time
from collections import Counter

import numpy as np

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json, wall_timer
from common.metrics import paired_bootstrap
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal

POLICIES = ("sft", "rlvr", "rlaif")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_STRICT_END_RE = re.compile(r"####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$")


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def output_dir(cfg, dataset: str):
    return repo_path(cfg["results_dir"]) / "task5_feedback" / dataset


def mentions_gold(text: str, gold) -> bool:
    """True if the gold number appears anywhere in the text (e.g. as an intermediate or discarded candidate)."""
    return any(numerically_equal(n.replace(",", ""), gold) for n in _NUMBER_RE.findall(str(text)))


def verifier_fields(response: str, gold, truncated: bool) -> dict:
    pred = extract_designated_final(response)
    correct = exact_reward(response, str(gold))
    if correct:
        failure = "correct"
    elif pred is None:
        failure = "no_final_truncated" if truncated else "no_final"
    elif mentions_gold(response, gold):
        failure = "wrong_final_gold_in_text"   # the right number was reached/mentioned, but not designated
    else:
        failure = "wrong_final"
    return {
        "pred_final": pred,
        "exact": float(correct),
        "has_final": pred is not None,                                   # verifier can parse a `#### <number>`
        "strict_format": bool(_STRICT_END_RE.search(response.strip())),  # response *ends* with `#### <number>`
        "n_finals": len(re.findall(r"####", response)),
        "failure_type": failure,
    }


def generate_for_policy(cfg, rows, name: str, batch_size: int):
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_frozen_policy(cfg, name)
    records = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        gen = batch_generate(
            model,
            tokenizer,
            [prompt_messages(r) for r in chunk],
            max_prompt_length=int(cfg.get("math_max_prompt_length", 512)),
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for row, response, n_tok, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
            records.append({
                "prompt_id": str(row.get("prompt_id", row["source_index"])),
                "policy": name,
                "question": row["question"],
                "gold_final": str(row["gold_final"]),
                **({"svamp_type": row["svamp_type"]} if "svamp_type" in row else {}),
                "response": response,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
                **verifier_fields(response, row["gold_final"], bool(trunc)),
            })
        print(f"  [{name}] {len(records)}/{len(rows)}", flush=True)
    del model
    clear_gpu()
    return records


def verifier_pref(a: dict, b: dict) -> str:
    """'A' / 'B' / 'TIE' in the same convention as the judge (A = first response)."""
    return "A" if a["exact"] > b["exact"] else "B" if b["exact"] > a["exact"] else "TIE"


def judge_pairs(judge, gens: dict):
    """Judge every policy pair on every prompt. Returns rows and the measured cost of uncached judge calls."""
    rows, new_calls, seconds = [], 0, 0.0
    for p, q in itertools.combinations(POLICIES, 2):
        for a, b in zip(gens[p], gens[q]):
            assert a["prompt_id"] == b["prompt_id"]
            n_before, t0 = len(judge.cache), time.perf_counter()
            verdict = judge.compare(a["question"], a["response"], b["response"])
            if len(judge.cache) > n_before:
                new_calls += 1
                seconds += time.perf_counter() - t0
            rows.append({
                "prompt_id": a["prompt_id"], "policy_a": p, "policy_b": q,
                "judge": verdict, "verifier": verifier_pref(a, b),
                "exact_a": a["exact"], "exact_b": b["exact"],
                "identical": a["response"].strip() == b["response"].strip(),
            })
        print(f"  judged {p} vs {q}", flush=True)
    return rows, {"new_judge_calls": new_calls, "judge_seconds": seconds,
                  "seconds_per_call": seconds / new_calls if new_calls else None}


def rate(counter: Counter, key, n):
    return counter[key] / n if n else float("nan")


def win_rate(pair_rows, base: str, policy: str):
    """Pairwise result of `policy` against `base` on the same prompts: (wins + 0.5 ties) / n."""
    rows = [r for r in pair_rows if (r["policy_a"], r["policy_b"]) == (base, policy)]
    c = Counter(r["judge"] for r in rows)   # A = base, B = policy
    n = len(rows)
    return {"n": n, "win": rate(c, "B", n), "tie": rate(c, "TIE", n), "loss": rate(c, "A", n),
            "win_rate": (c["B"] + 0.5 * c["TIE"]) / n if n else float("nan"),
            "identical_responses": sum(r["identical"] for r in rows)}


def agreement(pair_rows):
    """How the judge behaves when the verifier separates the pair, and when the verifier ties."""
    sep = [r for r in pair_rows if r["verifier"] != "TIE"]
    both_right = [r for r in pair_rows if r["verifier"] == "TIE" and r["exact_a"] == 1]
    both_wrong = [r for r in pair_rows if r["verifier"] == "TIE" and r["exact_a"] == 0]
    out = {"n_pairs": len(pair_rows), "n_verifier_separates": len(sep),
           "judge_agrees": float(np.mean([r["judge"] == r["verifier"] for r in sep])) if sep else float("nan"),
           "judge_tie_when_verifier_separates": float(np.mean([r["judge"] == "TIE" for r in sep])) if sep else float("nan"),
           "judge_prefers_wrong": float(np.mean([r["judge"] not in ("TIE", r["verifier"]) for r in sep])) if sep else float("nan"),
           "overall_three_way_agreement": float(np.mean([r["judge"] == r["verifier"] for r in pair_rows])) if pair_rows else float("nan")}
    for name, sub in [("both_correct", both_right), ("both_wrong", both_wrong)]:
        out[f"n_{name}"] = len(sub)
        out[f"judge_breaks_tie_{name}"] = float(np.mean([r["judge"] != "TIE" for r in sub])) if sub else float("nan")
    return out


def policy_summary(records):
    n = len(records)
    tok = np.array([r["response_tokens"] for r in records], dtype=float)
    return {
        "n": n,
        "accuracy": float(np.mean([r["exact"] for r in records])),
        "has_final_rate": float(np.mean([r["has_final"] for r in records])),
        "strict_format_rate": float(np.mean([r["strict_format"] for r in records])),
        "multiple_finals_rate": float(np.mean([r["n_finals"] > 1 for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "tokens_mean": float(tok.mean()), "tokens_std": float(tok.std()), "tokens_median": float(np.median(tok)),
        "tokens_mean_correct": float(np.mean([r["response_tokens"] for r in records if r["exact"]] or [np.nan])),
        "tokens_mean_wrong": float(np.mean([r["response_tokens"] for r in records if not r["exact"]] or [np.nan])),
        "failure_types": dict(Counter(r["failure_type"] for r in records)),
    }


def summarize(cfg, dataset, gens, pair_rows, cost, gen_minutes):
    out = {"dataset": dataset, "n_prompts": len(gens["sft"]), "max_new_tokens": int(cfg["math_max_new_tokens"]),
           "decoding": "greedy", "judge_model": cfg["ai_judge_model"],
           "policies": {n: policy_summary(gens[n]) for n in POLICIES},
           "accuracy_vs_sft": {n: paired_bootstrap([r["exact"] for r in gens["sft"]], [r["exact"] for r in gens[n]],
                                                   seed=int(cfg["seed"])) for n in POLICIES if n != "sft"},
           "pairwise_vs_sft": {n: win_rate(pair_rows, "sft", n) for n in POLICIES if n != "sft"},
           "rlaif_vs_rlvr": win_rate(pair_rows, "rlvr", "rlaif"),
           "agreement_all_pairs": agreement(pair_rows),
           "agreement_by_pair": {f"{p}_vs_{q}": agreement([r for r in pair_rows if (r["policy_a"], r["policy_b"]) == (p, q)])
                                 for p, q in itertools.combinations(POLICIES, 2)},
           "judge_cost": cost, "generation_minutes": gen_minutes}
    if "svamp_type" in gens["sft"][0]:
        types = sorted({r["svamp_type"] for r in gens["sft"]})
        out["accuracy_by_type"] = {n: {t: float(np.mean([r["exact"] for r in gens[n] if r["svamp_type"] == t]))
                                       for t in types} for n in POLICIES}
        out["type_counts"] = dict(Counter(r["svamp_type"] for r in gens["sft"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, _ = load_math_evaluation(args.config, args.dataset)
    outdir = output_dir(cfg, args.dataset)
    print(f"{args.dataset}: {len(rows)} prompts; policies {list(POLICIES)}")

    gens, gen_minutes = {}, {}
    for name in POLICIES:
        path = outdir / f"generations_{name}.jsonl"
        if path.exists():
            print(f"[{name}] {path} exists, reusing")
        else:
            adapter = policy_specs(cfg)[name]
            if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
                raise FileNotFoundError(f"{name}: {adapter} not found. Run python -m scripts.download_assets first.")
            timer = wall_timer()
            write_jsonl(path, generate_for_policy(cfg, rows, name, int(cfg.get("math_generation_batch_size", 16))))
            gen_minutes[name] = timer() / 60
            print(f"[{name}] -> {path} ({gen_minutes[name]:.1f} min)", flush=True)
        gens[name] = read_jsonl(path)

    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    pair_rows, cost = judge_pairs(judge, gens)
    write_jsonl(outdir / "pairwise.jsonl", pair_rows)
    summary = summarize(cfg, args.dataset, gens, pair_rows, cost, gen_minutes)
    save_json(outdir / "summary.json", summary)
    for n in POLICIES:
        s = summary["policies"][n]
        print(f"[{n}] acc={s['accuracy']:.3f} strict_format={s['strict_format_rate']:.3f} tokens={s['tokens_mean']:.0f}")
    for n, w in summary["pairwise_vs_sft"].items():
        print(f"[{n} vs sft] judge win rate={w['win_rate']:.3f} (W/T/L {w['win']:.2f}/{w['tie']:.2f}/{w['loss']:.2f})")
    print("verifier-judge agreement:", summary["agreement_all_pairs"])
    print("->", outdir / "summary.json")


if __name__ == "__main__":
    main()
