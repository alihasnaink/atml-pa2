"""Combine the in-domain, controlled-diagnostic and transfer results into the RLVR-vs-RLAIF comparison (CPU).

Inputs: results/task5_feedback/{gsm,transfer,diagnostics}/ (from evaluate_math.py and score_perturbations.py).
Outputs: results/task5_feedback/feedback_comparison.json and report/task5/{tables,figures,qualitative}/.
"""
from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.report_utils import fmt, quote, save_fig, write_table
from task5_feedback.evaluate_math import POLICIES
from task5_feedback.rlvr import exact_reward
from task5_feedback.score_perturbations import PAIRS

LABEL = {"sft": "SFT", "rlvr": "RLVR", "rlaif": "RLAIF"}
DATASET = {"gsm": "GSM8K (in-domain)", "transfer": "SVAMP (transfer)"}
VARIANT = {"clean_correct": "clean correct", "corrupt_reasoning_correct_final": "corrupt reasoning, correct final",
           "good_reasoning_wrong_final": "good reasoning, wrong final", "persuasive_filler_correct": "persuasive filler, correct",
           "gold_distractor_wrong_final": "gold distractor, wrong final"}
FAILURES = ["correct", "wrong_final", "wrong_final_gold_in_text", "no_final", "no_final_truncated"]
COLORS = {"better": "#2a7f62", "tie": "#b9b7af", "wrong": "#c8553d"}
POLICY_COLORS = {"sft": "#8a8984", "rlvr": "#2c6db5", "rlaif": "#d08a2c"}


def ci(d):
    return f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def wtl(w):
    return f"{w['win_rate']:.3f} ({w['win']:.2f}/{w['tie']:.2f}/{w['loss']:.2f})"


def policy_table(s):
    rows = []
    for n in POLICIES:
        p = s["policies"][n]
        rows.append({"Policy": LABEL[n], "Exact acc.": p["accuracy"],
                     "Acc. - SFT [95% CI]": "ref." if n == "sft" else ci(s["accuracy_vs_sft"][n]),
                     "Judge win rate vs SFT (W/T/L)": "0.5 (ref.)" if n == "sft" else wtl(s["pairwise_vs_sft"][n]),
                     "Strict format": p["strict_format_rate"], "Has #### final": p["has_final_rate"],
                     "Multiple ####": p["multiple_finals_rate"], "Trunc. rate": p["truncation_rate"],
                     "Tokens mean ± std": f"{p['tokens_mean']:.1f} ± {p['tokens_std']:.1f}",
                     "Tokens correct / wrong": f"{fmt(p['tokens_mean_correct'], 0)} / {fmt(p['tokens_mean_wrong'], 0)}"})
    return pd.DataFrame(rows)


def transfer_table(gsm, svamp):
    rows = []
    for n in POLICIES:
        g, t = gsm["policies"][n], svamp["policies"][n]
        wg = gsm["pairwise_vs_sft"].get(n, {"win_rate": 0.5})["win_rate"]
        wt = svamp["pairwise_vs_sft"].get(n, {"win_rate": 0.5})["win_rate"]
        rows.append({"Policy": LABEL[n], "GSM8K acc.": g["accuracy"], "SVAMP acc.": t["accuracy"],
                     "Acc. drop (SVAMP - GSM8K)": t["accuracy"] - g["accuracy"],
                     "Acc. - SFT on SVAMP": t["accuracy"] - svamp["policies"]["sft"]["accuracy"],
                     "Win rate vs SFT GSM8K / SVAMP": "ref." if n == "sft" else f"{wg:.3f} / {wt:.3f}",
                     "Strict format GSM8K / SVAMP": f"{g['strict_format_rate']:.3f} / {t['strict_format_rate']:.3f}",
                     "Tokens GSM8K / SVAMP": f"{g['tokens_mean']:.1f} / {t['tokens_mean']:.1f}"})
    return pd.DataFrame(rows)


def agreement_table(summaries):
    rows = []
    for ds, s in summaries.items():
        for key, a in [("all pairs", s["agreement_all_pairs"]), *s["agreement_by_pair"].items()]:
            rows.append({"Dataset": DATASET[ds], "Pair": key if key == "all pairs" else " vs ".join(LABEL[x] for x in key.split("_vs_")),
                         "Pairs": a["n_pairs"], "Verifier separates": a["n_verifier_separates"],
                         "Judge agrees": a["judge_agrees"], "Judge ties": a["judge_tie_when_verifier_separates"],
                         "Judge prefers wrong": a["judge_prefers_wrong"],
                         "Both correct (n) / judge breaks tie": f"{a['n_both_correct']} / {fmt(a['judge_breaks_tie_both_correct'])}",
                         "Both wrong (n) / judge breaks tie": f"{a['n_both_wrong']} / {fmt(a['judge_breaks_tie_both_wrong'])}"})
    return pd.DataFrame(rows)


def failure_table(summaries):
    rows = []
    for ds, s in summaries.items():
        for n in POLICIES:
            f, total = s["policies"][n]["failure_types"], s["policies"][n]["n"]
            rows.append({"Dataset": DATASET[ds], "Policy": LABEL[n], **{k: f.get(k, 0) / total for k in FAILURES}})
    return pd.DataFrame(rows)


def diagnostic_tables(diag):
    rows = []
    for variant, s in diag["pairs"].items():
        for mech in ("verifier", "judge"):
            r = s[mech]
            rows.append({"Pair (clean vs ...)": VARIANT[variant], "Family": s["family"], "Mechanism": mech,
                         "n": r["n"], "Prefers better": r["better"], "Tie": r["tie"], "Wrong preference": r["wrong"],
                         "Judge order-consistency": s["judge_order_consistency"] if mech == "judge" else "n/a"})
    pairs = pd.DataFrame(rows)
    sens = pd.DataFrame([{"Statistic": name, "Verifier": diag[key]["verifier"], "AI judge": diag[key]["judge"]}
                         for name, key in [("S_reason = Pr[R(clean) > R(reason-corrupt)]", "S_reason"),
                                           ("S_outcome = Pr[R(correct final) > R(wrong final)]", "S_outcome"),
                                           ("Filler susceptibility = Pr[R(filler) > R(clean)]", "filler_susceptibility"),
                                           ("Distractor robustness = Pr[R(clean) > R(distractor)]", "distractor_robustness")]])
    group = pd.DataFrame([{"Variant": VARIANT[v], "Exact reward (mean)": g["exact_mean"],
                           "RLAIF group reward K=5 (mean ± std)": f"{g['rlaif_group_reward_mean']:.3f} ± {g['rlaif_group_reward_std']:.3f}"}
                          for v, g in diag["group_rewards"].items()])
    return pairs, sens, group


def cost_table(cfg, summaries, diag):
    """Inference cost of each feedback source, measured on the same hardware run."""
    responses = [r["response"] for ds in summaries for n in POLICIES
                 for r in read_jsonl(repo_path(cfg["results_dir"]) / "task5_feedback" / ds / f"generations_{n}.jsonl")]
    t0 = time.perf_counter()
    for text in responses:
        exact_reward(text, "0")
    verifier_s = (time.perf_counter() - t0) / max(len(responses), 1)
    rows = [{"Feedback source": "Exact verifier (CPU regex)", "Calls": len(responses), "Seconds / call": f"{verifier_s:.2e}",
             "Calls per K-response group": "K", "Notes": "measured locally when building the report"}]
    for name, c in [*[(DATASET[ds], s["judge_cost"]) for ds, s in summaries.items()], ("diagnostic set", diag["judge_cost"])]:
        rows.append({"Feedback source": f"Pairwise AI judge ({name})", "Calls": c["new_judge_calls"],
                     "Seconds / call": fmt(c["seconds_per_call"], 2), "Calls per K-response group": "K(K-1)/2",
                     "Notes": f"{cfg['ai_judge_model']}, 4-bit, one T4; 0 calls = all cached"})
    return pd.DataFrame(rows)


def plot_diagnostics(diag, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    variants = list(PAIRS)
    for ax, mech in zip(axes, ("verifier", "judge")):
        left = np.zeros(len(variants))
        for k in ("better", "tie", "wrong"):
            vals = np.array([diag["pairs"][v][mech][k] for v in variants])
            ax.barh(range(len(variants)), vals, left=left, color=COLORS[k], label=k)
            left += vals
        ax.set_title("Exact verifier" if mech == "verifier" else "Pairwise AI judge")
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of controlled pairs")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(variants)), [f"clean vs\n{VARIANT[v]}" for v in variants])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["prefers clean (better)", "tie", "prefers perturbed (wrong)"], loc="upper center",
               ncol=3, bbox_to_anchor=(0.55, 1.08))
    save_fig(fig, fig_dir, "diagnostic_preferences")


def plot_in_vs_out(summaries, fig_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    x = np.arange(len(summaries))
    w = 0.26
    for i, n in enumerate(POLICIES):
        acc = [s["policies"][n]["accuracy"] for s in summaries.values()]
        win = [s["pairwise_vs_sft"][n]["win_rate"] if n != "sft" else 0.5 for s in summaries.values()]
        tok = [s["policies"][n]["tokens_mean"] for s in summaries.values()]
        for ax, vals in zip(axes, (acc, win, tok)):
            ax.bar(x + (i - 1) * w, vals, w, color=POLICY_COLORS[n], label=LABEL[n])
    for ax, title in zip(axes, ("Exact accuracy", "Judge win rate vs SFT", "Mean response tokens")):
        ax.set_xticks(x, [DATASET[d] for d in summaries])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[1].axhline(0.5, color="#52514e", lw=0.8, ls="--")
    axes[0].legend(loc="lower left")
    save_fig(fig, fig_dir, "in_vs_out_of_domain")


def plot_failures(fail, fig_dir):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    labels = [f"{r.Policy}\n{r.Dataset.split()[0]}" for r in fail.itertuples()]
    left = np.zeros(len(fail))
    palette = ["#2a7f62", "#c8553d", "#e0a458", "#6a4c93", "#b9b7af"]
    for k, c in zip(FAILURES, palette):
        ax.barh(range(len(fail)), fail[k], left=left, color=c, label=k)
        left += fail[k].to_numpy()
    ax.set_yticks(range(len(fail)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of prompts")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.25))
    save_fig(fig, fig_dir, "failure_types")


def qualitative(cfg, datasets, diag_dir, res_dir, out_path):
    """Short examples for the report: diagnostic disagreements and policy-level verifier/judge conflicts."""
    rows = {(str(r["problem_id"]), r["variant_type"]): r for r in read_jsonl(repo_path(cfg["paths"]["task5_diagnostics"]))}
    pairs = read_jsonl(diag_dir / "pairs.jsonl")
    md = ["# Task 5 qualitative evidence (selected automatically; verify before quoting)", ""]

    def diag_block(title, pick, n=2):
        md.extend([f"## {title}", ""])
        chosen = [p for p in pairs if pick(p)][:n]
        if not chosen:
            md.extend(["_No pair matched this condition._", ""])
        for p in chosen:
            clean, var = rows[(p["problem_id"], "clean_correct")], rows[(p["problem_id"], p["variant"])]
            md.extend([f"**Problem {p['problem_id']}** (gold {clean['gold_final']}) - verifier: {p['verifier_outcome']}, "
                       f"judge: {p['judge_outcome']} (reversed call: {p['judge_reversed']}, consistent={p['judge_consistent']})", "",
                       "Clean response:", "", quote(clean["response"], 700), "", f"{VARIANT[p['variant']]}:", "",
                       quote(var["response"], 700), ""])

    diag_block("Reasoning-only corruption: verifier ties (outcome-blind); what does the judge do?",
               lambda p: p["family"] == "reasoning" and p["judge_outcome"] != "better")
    diag_block("Reasoning-only corruption detected by the judge", lambda p: p["family"] == "reasoning" and p["judge_outcome"] == "better", 1)
    diag_block("Persuasive filler: judge prefers the filler (style/persuasion bias)",
               lambda p: p["family"] == "filler" and p["judge_outcome"] == "wrong")
    diag_block("Persuasive filler: judge ties or prefers clean", lambda p: p["family"] == "filler" and p["judge_outcome"] != "wrong", 1)
    diag_block("Gold distractor / wrong final: judge is fooled", lambda p: p["family"] in ("distractor", "outcome") and p["judge_outcome"] != "better")
    diag_block("Gold distractor: judge and verifier both correct", lambda p: p["family"] == "distractor" and p["judge_outcome"] == "better", 1)

    for ds in datasets:
        gens = {n: {r["prompt_id"]: r for r in read_jsonl(res_dir / ds / f"generations_{n}.jsonl")} for n in POLICIES}
        conflicts = [r for r in read_jsonl(res_dir / ds / "pairwise.jsonl")
                     if r["verifier"] != "TIE" and r["judge"] not in ("TIE", r["verifier"])]
        md.extend([f"## {DATASET[ds]}: judge prefers the response the verifier marks wrong ({len(conflicts)} pairs)", ""])
        for r in conflicts[:2]:
            a, b = gens[r["policy_a"]][r["prompt_id"]], gens[r["policy_b"]][r["prompt_id"]]
            md.extend([f"**{r['prompt_id']}** (gold {a['gold_final']}) - judge prefers "
                       f"{LABEL[r['policy_a'] if r['judge'] == 'A' else r['policy_b']]}", "", quote(a["question"], 400), "",
                       f"{LABEL[r['policy_a']]} (exact={a['exact']:.0f}, final={a['pred_final']}):", "", quote(a["response"], 600), "",
                       f"{LABEL[r['policy_b']]} (exact={b['exact']:.0f}, final={b['pred_final']}):", "", quote(b["response"], 600), ""])
        for n in ("rlvr", "rlaif"):
            fails = [r for r in gens[n].values() if r["failure_type"] in ("wrong_final_gold_in_text", "no_final", "no_final_truncated")]
            if fails:
                r = fails[0]
                md.extend([f"**{DATASET[ds]} {LABEL[n]} failure ({r['failure_type']})**, {r['prompt_id']} (gold {r['gold_final']})", "",
                           quote(r["response"], 600), ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    out = repo_path("report/task5")
    tab, fig_dir = out / "tables", out / "figures"
    summaries = {ds: load_json(res_dir / ds / "summary.json") for ds in ("gsm", "transfer")
                 if (res_dir / ds / "summary.json").exists()}
    diag = load_json(res_dir / "diagnostics" / "summary.json")
    if "gsm" not in summaries:
        raise FileNotFoundError("Run python -m task5_feedback.evaluate_math --dataset gsm first.")
    gsm = summaries["gsm"]
    n_gsm = gsm["n_prompts"]
    setup = f"greedy decoding, max {gsm['max_new_tokens']} new tokens; judge {gsm['judge_model']}"

    write_table(policy_table(gsm), tab, "in_domain",
                f"In-domain comparison on the fixed GSM8K subset (n={n_gsm}; {setup}). Win rate = (wins + 0.5 ties)/n against "
                "SFT on the same prompt; strict format = response ends with '#### <number>'.")
    if "transfer" in summaries:
        svamp = summaries["transfer"]
        write_table(policy_table(svamp), tab, "transfer",
                    f"Out-of-domain comparison on the fixed SVAMP subset (n={svamp['n_prompts']}; same protocol).")
        write_table(transfer_table(gsm, svamp), tab, "transfer_drop", "In-domain vs transfer: drop from the GSM8K metric.")
        if "accuracy_by_type" in svamp:
            by_type = pd.DataFrame([{"SVAMP type": t, "n": svamp["type_counts"][t],
                                     **{LABEL[n]: svamp["accuracy_by_type"][n][t] for n in POLICIES}}
                                    for t in svamp["type_counts"]])
            write_table(by_type, tab, "transfer_by_type", "SVAMP exact accuracy by problem type.")
    write_table(agreement_table(summaries), tab, "verifier_judge_agreement",
                "Verifier-judge agreement on policy responses to the same prompt. 'Verifier separates' = exactly one response "
                "is correct; then the judge agrees, ties, or prefers the wrong one. When both are correct/wrong the verifier "
                "ties, and the judge may still break the tie.")
    fail = failure_table(summaries)
    write_table(fail, tab, "failure_types",
                "Failure types (fractions). wrong_final_gold_in_text: the gold number appears in the response but the "
                "designated final is wrong (can be incidental for small gold numbers); no_final: no parsable '#### <number>'.")
    pairs, sens, group = diagnostic_tables(diag)
    write_table(pairs, tab, "diagnostic_pairs",
                f"Controlled diagnostic pairs ({diag['n_problems']} problems; each perturbed variant vs the clean correct "
                "response). For the filler pair a tie is also acceptable; preferring the filler is a wrong preference.")
    write_table(sens, tab, "diagnostic_sensitivity", "Reasoning/outcome sensitivity and robustness statistics.")
    write_table(group, tab, "diagnostic_group_rewards",
                "RLAIF group reward (wins + 0.5 ties)/(K-1) with the five variants of each problem as one K=5 group, "
                "next to the exact reward.")
    write_table(cost_table(cfg, summaries, diag), tab, "feedback_cost", "Inference cost of the two feedback sources.")

    plot_diagnostics(diag, fig_dir)
    plot_in_vs_out(summaries, fig_dir)
    plot_failures(fail, fig_dir)
    qualitative(cfg, list(summaries), res_dir / "diagnostics", res_dir, out / "qualitative" / "examples.md")

    save_json(res_dir / "feedback_comparison.json", {"in_domain": gsm, "transfer": summaries.get("transfer"), "diagnostics": diag})
    print("Wrote report material ->", out)


if __name__ == "__main__":
    main()
