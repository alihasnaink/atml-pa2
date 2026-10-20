"""Step 2 - reward sensitivity and robustness on the Controlled Reward Diagnostic Set.

Every problem has five manually validated variants. Each perturbed variant is paired with the problem's
clean_correct response, which is the diagnostically better (or, for filler, at least as good) response:

  pair                         perturbation held fixed / changed            what it measures
  corrupt_reasoning_correct_final  same correct final, reasoning degraded   reasoning sensitivity  S_reason
  good_reasoning_wrong_final       reasoning ~fixed, designated final wrong outcome sensitivity    S_outcome
  persuasive_filler_correct        same content + irrelevant persuasion     filler susceptibility (tie/clean is right)
  gold_distractor_wrong_final      gold number mentioned, final wrong       distractor robustness

Both mechanisms score every pair:
  verifier  exact_reward (binary): better if R(clean) > R(variant), tie if equal, wrong if lower;
  judge     the fixed pairwise AI judge: better if it prefers clean, tie on TIE, wrong if it prefers the variant.
The judge is also asked with the arguments reversed (its A/B orientation comes from a hash of the
arguments, so this re-asks the comparison) to measure its self-consistency.
Finally the five variants of each problem are scored as one K=5 group with the RLAIF group reward
r = (wins + 0.5 ties) / (K - 1), the signal the RLAIF policy was trained on.

Outputs (results/task5_feedback/diagnostics/): pairs.jsonl, group_rewards.jsonl, judge_cache.json, summary.json
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
REFERENCE = "clean_correct"
PAIRS = {  # perturbed variant -> analysis family
    "corrupt_reasoning_correct_final": "reasoning",
    "good_reasoning_wrong_final": "outcome",
    "persuasive_filler_correct": "filler",
    "gold_distractor_wrong_final": "distractor",
}
GROUP_ORDER = [REFERENCE, *PAIRS]   # clean first, so the controlled pairs are the group's (clean, x) calls


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome(pref_for_clean: str) -> str:
    """Map a preference ('A' = clean, 'B' = variant, 'TIE') to better / tie / wrong."""
    return {"A": "better", "TIE": "tie", "B": "wrong"}[pref_for_clean]


def verifier_pref(r_clean: float, r_variant: float) -> str:
    return "A" if r_clean > r_variant else "B" if r_variant > r_clean else "TIE"


def rates(outcomes):
    n = len(outcomes)
    return {"n": n, **{k: sum(o == k for o in outcomes) / n for k in ("better", "tie", "wrong")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"

    # Verifier on every response, checked against the staff-validated expected reward.
    exact = {(pid, v): exact_reward(g[v]["response"], str(g[v]["gold_final"])) for pid, g in groups.items() for v in g}
    mismatches = [(pid, v) for (pid, v), r in exact.items() if r != float(groups[pid][v]["expected_exact_reward"])]
    print("Verifier vs expected_exact_reward mismatches:", mismatches or "none")

    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    pair_rows, group_rows, seconds, new_calls = [], [], 0.0, 0

    def ask(problem, a, b):
        nonlocal seconds, new_calls
        n_before, t0 = len(judge.cache), time.perf_counter()
        verdict = judge.compare(problem, a, b)
        if len(judge.cache) > n_before:
            new_calls += 1
            seconds += time.perf_counter() - t0
        return verdict

    for pid, g in sorted(groups.items(), key=lambda kv: int(kv[0])):
        clean = g[REFERENCE]
        problem = clean["question"]
        for variant, family in PAIRS.items():
            v = g[variant]
            j = ask(problem, clean["response"], v["response"])
            j_rev = {"A": "B", "B": "A", "TIE": "TIE"}[ask(problem, v["response"], clean["response"])]
            vp = verifier_pref(exact[(pid, REFERENCE)], exact[(pid, variant)])
            pair_rows.append({"problem_id": pid, "variant": variant, "family": family,
                              "exact_clean": exact[(pid, REFERENCE)], "exact_variant": exact[(pid, variant)],
                              "verifier": vp, "verifier_outcome": outcome(vp),
                              "judge": j, "judge_outcome": outcome(j),
                              "judge_reversed": j_rev, "judge_consistent": j == j_rev,
                              "clean_chars": len(clean["response"]), "variant_chars": len(v["response"])})
        responses = [g[v]["response"] for v in GROUP_ORDER]
        rewards = judge.group_rewards(problem, responses)   # same caching -> controlled pairs are reused
        group_rows += [{"problem_id": pid, "variant": v, "rlaif_group_reward": r, "exact": exact[(pid, v)]}
                       for v, r in zip(GROUP_ORDER, rewards)]
        print(f"  problem {pid} done", flush=True)

    write_jsonl(outdir / "pairs.jsonl", pair_rows)
    write_jsonl(outdir / "group_rewards.jsonl", group_rows)

    by_variant = {}
    for variant, family in PAIRS.items():
        sub = [r for r in pair_rows if r["variant"] == variant]
        by_variant[variant] = {
            "family": family,
            "verifier": rates([r["verifier_outcome"] for r in sub]),
            "judge": rates([r["judge_outcome"] for r in sub]),
            "judge_reversed": rates([outcome(r["judge_reversed"]) for r in sub]),
            "judge_order_consistency": float(np.mean([r["judge_consistent"] for r in sub])),
            "verifier_judge_same_outcome": float(np.mean([r["verifier_outcome"] == r["judge_outcome"] for r in sub])),
        }
    group = {v: {"rlaif_group_reward_mean": float(np.mean([r["rlaif_group_reward"] for r in group_rows if r["variant"] == v])),
                 "rlaif_group_reward_std": float(np.std([r["rlaif_group_reward"] for r in group_rows if r["variant"] == v])),
                 "exact_mean": float(np.mean([r["exact"] for r in group_rows if r["variant"] == v]))}
             for v in GROUP_ORDER}
    summary = {
        "n_problems": len(groups),
        "verifier_expected_mismatches": [list(m) for m in mismatches],
        "pairs": by_variant,
        # S_reason = Pr[R(clean) > R(reason-corrupt)], S_outcome = Pr[R(correct final) > R(wrong final)]
        "S_reason": {m: by_variant["corrupt_reasoning_correct_final"][m]["better"] for m in ("verifier", "judge")},
        "S_outcome": {m: by_variant["good_reasoning_wrong_final"][m]["better"] for m in ("verifier", "judge")},
        "filler_susceptibility": {m: by_variant["persuasive_filler_correct"][m]["wrong"] for m in ("verifier", "judge")},
        "distractor_robustness": {m: by_variant["gold_distractor_wrong_final"][m]["better"] for m in ("verifier", "judge")},
        "judge_order_consistency_overall": float(np.mean([r["judge_consistent"] for r in pair_rows])),
        "group_rewards": group,
        "judge_cost": {"new_judge_calls": new_calls, "judge_seconds": seconds,
                       "seconds_per_call": seconds / new_calls if new_calls else None},
    }
    save_json(outdir / "summary.json", summary)
    for variant, s in by_variant.items():
        print(f"{variant:34s} verifier B/T/W {s['verifier']['better']:.2f}/{s['verifier']['tie']:.2f}/{s['verifier']['wrong']:.2f}"
              f"   judge B/T/W {s['judge']['better']:.2f}/{s['judge']['tie']:.2f}/{s['judge']['wrong']:.2f}"
              f"   judge order-consistency {s['judge_order_consistency']:.2f}")
    print("S_reason:", summary["S_reason"], " S_outcome:", summary["S_outcome"])
    print("->", outdir / "summary.json")


if __name__ == "__main__":
    main()
