"""Build the Task 1 report material from the saved results.

Reads results/task1_dpo/ (written by train/evaluate/ablate_beta/analyze_length)
and writes tables (CSV + Markdown + LaTeX), figures (PNG + PDF) and qualitative
example candidates to report/task1/. No model is loaded; runs on CPU in seconds.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.report_utils import clip_text, fmt, plt, quote, save_fig, write_table

CONDITIONS = [
    ("sft", "SFT (no DPO)"),
    ("standard", "Standard DPO"),
    ("beta_0.03", "Short fork beta=0.03"),
    ("beta_0.10", "Short fork beta=0.10"),
    ("beta_0.30", "Short fork beta=0.30"),
    ("length_balanced", "Length-balanced DPO"),
]
LABEL = dict(CONDITIONS)
# One fixed color per condition (reference categorical palette, fixed order; SFT is the gray reference).
COLOR = {
    "sft": "#8a8984",
    "standard": "#2a78d6",
    "length_balanced": "#eb6834",
    "beta_0.03": "#1baf7a",
    "beta_0.10": "#eda100",
    "beta_0.30": "#e87ba4",
}
STRATA = ("preferred_longer", "length_matched", "rejected_longer")
STRATUM_LABEL = {"preferred_longer": "Preferred longer", "length_matched": "Length matched", "rejected_longer": "Rejected longer"}

def summary_table(summaries, train):
    rows = []
    for name, s in summaries.items():
        p, g, wl = s["heldout_pairs"], s["heldout_generations"], s["word_limit"]
        t = train.get(name)
        is_sft = name == "sft"
        L = g["length_tokens"]
        rows.append({
            "Condition": LABEL[name],
            "Train data": "-" if is_sft else ("length-balanced" if name == "length_balanced" else "standard"),
            "Budget": "-" if is_sft else ("1 epoch" if name in ("standard", "length_balanced") else "short fork"),
            "Train pairs": t["n_train_pairs"] if t else "-",
            "Opt. steps": t["optimizer_steps"] if t else "-",
            "beta": "-" if is_sft else s["beta"],
            "Held-out DPO loss": None if is_sft else p["dpo_loss"],
            "Pref. acc.": None if is_sft else p["preference_accuracy"],
            "Margin m": None if is_sft else p["margin_mean"],
            "Chosen reward": None if is_sft else p["chosen_reward_mean"],
            "Rejected reward": None if is_sft else p["rejected_reward_mean"],
            "KL/token": g["kl_token_mean"],
            "KL/seq": g["kl_sequence_mean"],
            "RM score": f"{g['reward_mean']:.3f} ± {g['reward_std']:.3f}",
            "Len mean ± std": f"{L['mean']:.1f} ± {L['std']:.1f}",
            "Len median [IQR]": f"{L['median']:.0f} [{L['q25']:.0f}-{L['q75']:.0f}]",
            "Trunc. rate": g["truncation_rate"],
            "Word-limit compl.": wl["compliance_rate"],
            "Train min": round(t["wall_clock_s"] / 60, 1) if t else "-",
            "Peak VRAM GB": round(t["peak_vram_gb"], 2) if t else "-",
        })
    return pd.DataFrame(rows)


def strata_table(summaries, analysis):
    rows = []
    for name in ("standard", "length_balanced"):
        if name not in summaries:
            continue
        for s in STRATA:
            p = summaries[name]["stratified_pairs"][s]
            rows.append({"Model": LABEL[name], "Stratum": STRATUM_LABEL[s], "n": p["n"],
                         "Pref. acc.": p["preference_accuracy"], "Margin m": p["margin_mean"],
                         "DPO loss": p["dpo_loss"], "Chosen reward": p["chosen_reward_mean"],
                         "Rejected reward": p["rejected_reward_mean"]})
    if analysis:
        for s in STRATA:
            o = analysis["learned_overlap_standard_vs_balanced"][s]
            d = o["accuracy_diff_balanced_minus_standard"]
            rows.append({"Model": "Balanced - Standard (paired)", "Stratum": STRATUM_LABEL[s], "n": o["n"],
                         "Pref. acc.": f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]",
                         "Margin m": f"r={o['margin_corr_standard_vs_balanced']:.2f}",
                         "DPO loss": f"both {o['both_correct']:.2f}", "Chosen reward": f"only std {o['only_standard']:.2f}",
                         "Rejected reward": f"only bal {o['only_balanced']:.2f}"})
    return pd.DataFrame(rows)


def word_limit_tables(summaries, wl_rows):
    agg = []
    for name, s in summaries.items():
        w = s["word_limit"]
        agg.append({"Condition": LABEL[name], "Samples": w["n"], "Compliance": w["compliance_rate"],
                    "Words mean": w["length_words"]["mean"], "Words / limit": w["words_over_limit_mean"],
                    "Tokens mean": w["length_tokens"]["mean"], "Trunc. rate": w["truncation_rate"],
                    "RM score": w["reward_mean"]})
    prompts = {}
    for name, rows in wl_rows.items():
        for r in rows:
            d = prompts.setdefault(r["prompt_id"], {"Prompt": r["prompt_id"], "Limit": r["word_limit"]})
            d.setdefault(name, []).append(r["compliant"])
    per_prompt = pd.DataFrame([{**{k: v for k, v in d.items() if k in ("Prompt", "Limit")},
                                **{LABEL[n]: float(np.mean(d[n])) for n in wl_rows if n in d}}
                               for d in prompts.values()])
    return pd.DataFrame(agg), per_prompt


def dataset_table(stats):
    names = {"dpo_standard_train": "Standard train", "dpo_standard_eval": "Standard held-out",
             "dpo_length_train": "Length-balanced train", "dpo_length_eval": "Length-stratified eval"}
    rows = []
    for key, s in stats.items():
        rows.append({"File": names[key], "n": s["n"], "Frac. chosen longer": s["frac_chosen_longer"],
                     "Chosen tok. mean": s["chosen_tokens_mean"], "Rejected tok. mean": s["rejected_tokens_mean"],
                     "Len. diff mean": s["length_diff_mean"], "Len. diff median": s["length_diff_median"],
                     "corr(score gap, len diff)": s["corr_score_gap_vs_length_diff"],
                     **{f"% {STRATUM_LABEL[k]}": 100 * s["rule_strata"][k] for k in STRATA},
                     "% intermediate": 100 * s["rule_strata"]["intermediate"],
                     "Rule = supplied label": s.get("rule_agrees_with_supplied_label")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def fig_beta_sweep(summaries, out):
    betas = [b for b in ("beta_0.03", "beta_0.10", "beta_0.30") if b in summaries]
    if not betas:
        return
    x = [summaries[b]["beta"] for b in betas]
    panels = [
        ("Held-out preference accuracy", lambda s: s["heldout_pairs"]["preference_accuracy"], False),
        ("KL from reference (per token)", lambda s: s["heldout_generations"]["kl_token_mean"], True),
        ("Reward-model score (mean)", lambda s: s["heldout_generations"]["reward_mean"], True),
        ("Generated length (tokens, mean)", lambda s: s["heldout_generations"]["length_tokens"]["mean"], True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, (title, get, show_sft) in zip(axes, panels):
        ax.plot(x, [get(summaries[b]) for b in betas], "-o", color="#1baf7a", lw=2, ms=6, label="Short forks (600 pairs)")
        if "standard" in summaries:
            s = summaries["standard"]
            ax.plot([s["beta"]], [get(s)], "*", color=COLOR["standard"], ms=12, label="Standard DPO (1 epoch)")
        if show_sft and "sft" in summaries:
            ax.axhline(get(summaries["sft"]), color=COLOR["sft"], ls="--", lw=1.2, label="SFT reference")
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x])
        ax.minorticks_off()
        ax.set_xlabel("beta")
        ax.set_title(title, fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()  # panel with all three series
    axes[0].legend(handles, labels, fontsize=7, loc="lower right")
    fig.tight_layout()
    save_fig(fig, out, "beta_sweep")


def fig_training_curves(logs, out):
    if not logs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    keys = [("loss", "Train DPO loss"), ("preference_accuracy", "Train preference accuracy"), ("margin_mean", "Train margin m")]
    for name, rows in logs.items():
        x = [r["examples_seen"] for r in rows]
        for ax, (k, title) in zip(axes, keys):
            y = pd.Series([r[k] for r in rows]).rolling(5, min_periods=1).mean()
            ax.plot(x, y, color=COLOR[name], lw=2, label=LABEL[name])
            ax.set_title(title + " (5-step mean)", fontsize=9)
            ax.set_xlabel("training pairs seen")
    axes[0].axhline(np.log(2), color=COLOR["sft"], ls=":", lw=1)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, "training_curves")


def fig_length_strata(summaries, out):
    names = [n for n in ("standard", "length_balanced") if n in summaries]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5, 2.8))
    width = 0.38
    for i, name in enumerate(names):
        acc = [summaries[name]["stratified_pairs"][s]["preference_accuracy"] for s in STRATA]
        pos = np.arange(len(STRATA)) + (i - 0.5) * width
        bars = ax.bar(pos, acc, width - 0.03, color=COLOR[name], label=LABEL[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color="#52514e", padding=2)
    ax.axhline(0.5, color=COLOR["sft"], ls="--", lw=1)
    ax.set_xticks(np.arange(len(STRATA)))
    ax.set_xticklabels([STRATUM_LABEL[s] for s in STRATA])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Preference accuracy (m > 0)")
    ax.set_title("Length-stratified held-out pairs (82 per stratum)", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_fig(fig, out, "length_strata_accuracy")


def fig_margin_vs_length(strat_pairs, out):
    names = [n for n in ("standard", "length_balanced") if n in strat_pairs]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        rows = strat_pairs[name]
        d = np.array([r["chosen_tokens"] - r["rejected_tokens"] for r in rows])
        m = np.array([r["margin"] for r in rows])
        ax.scatter(d, m, s=14, color=COLOR[name], edgecolor="white", linewidth=0.6)
        if len(d) > 1:
            k, c = np.polyfit(d, m, 1)
            xs = np.linspace(d.min(), d.max(), 50)
            ax.plot(xs, k * xs + c, color="#0b0b0b", lw=1.2)
            r = np.corrcoef(d, m)[0, 1]
            ax.set_title(f"{LABEL[name]}  (Pearson r = {r:.2f})", fontsize=9)
        ax.axhline(0, color=COLOR["sft"], lw=0.8)
        ax.axvline(0, color=COLOR["sft"], lw=0.8)
        ax.set_xlabel("chosen - rejected length (tokens)")
    axes[0][0].set_ylabel("DPO margin m (held-out)")
    fig.tight_layout()
    save_fig(fig, out, "margin_vs_length_difference")


def fig_generated_length(gens, wl_rows, out):
    names = [n for n, _ in CONDITIONS if n in gens]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    data = [[r["response_tokens"] for r in gens[n]] for n in names]
    bp = axes[0].boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=False)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLOR[n])
        patch.set_edgecolor("#52514e")
    for med in bp["medians"]:
        med.set_color("#0b0b0b")
    axes[0].set_yticks(range(1, len(names) + 1))
    axes[0].set_yticklabels([LABEL[n] for n in names])
    axes[0].set_xlabel("generated response tokens (held-out prompts, cap 256)")
    axes[0].set_title("Generated length", fontsize=9)

    wnames = [n for n in names if n in wl_rows]
    ratio = [[r["words"] / r["word_limit"] for r in wl_rows[n]] for n in wnames]
    rng = np.random.default_rng(0)
    for i, (n, vals) in enumerate(zip(wnames, ratio)):
        axes[1].scatter(vals, i + 1 + rng.uniform(-0.15, 0.15, len(vals)), s=12, color=COLOR[n], edgecolor="white", linewidth=0.5)
        axes[1].plot([np.mean(vals)], [i + 1], "|", color="#0b0b0b", ms=14, mew=2)
    axes[1].axvline(1.0, color="#e34948", ls="--", lw=1.2)
    axes[1].set_yticks(range(1, len(wnames) + 1))
    axes[1].set_yticklabels([LABEL[n] for n in wnames])
    axes[1].set_xlabel("words / requested word limit (right of red line = violation)")
    axes[1].set_title("Word-limit prompts (10 prompts x samples)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out, "generated_length_and_word_limit")


# ---------------------------------------------------------------- qualitative candidates

def qual_reward_vs_quality(gens, wl_rows, out, top=8):
    lines = ["# Candidates: reward/preference signal vs response quality", "",
             "Automatically selected candidates; read them and keep only cases you judge to be real",
             "quality regressions (correctness, concision, instruction compliance). Quote minimally in the report.", ""]
    if "sft" in gens and "standard" in gens:
        sft = {r["prompt_id"]: r for r in gens["sft"]}
        cands = []
        for r in gens["standard"]:
            b = sft.get(r["prompt_id"])
            if b and r["reward"] > b["reward"] and (r["truncated"] or r["response_tokens"] >= 1.5 * b["response_tokens"]):
                cands.append((r["reward"] - b["reward"], r, b))
        cands.sort(key=lambda x: -x[0])
        lines += ["## A. Held-out prompts: standard DPO scores higher than SFT but is much longer or truncated", ""]
        for gain, r, b in cands[:top]:
            lines += [f"### prompt_id `{r['prompt_id']}` (reward gain {gain:+.2f})", "", "**Prompt**", "", quote(r["prompt"], 800), "",
                      f"**SFT** - reward {b['reward']:.2f}, {b['response_tokens']} tokens, truncated={b['truncated']}", "", quote(b["response"]), "",
                      f"**Standard DPO** - reward {r['reward']:.2f}, {r['response_tokens']} tokens, truncated={r['truncated']}", "", quote(r["response"]), ""]
    lines += ["## B. Word-limit prompts: non-compliant DPO responses that out-score a compliant SFT response", ""]
    if "sft" in wl_rows:
        best_sft = {}
        for r in wl_rows["sft"]:
            if r["compliant"]:
                cur = best_sft.get(r["prompt_id"])
                if cur is None or r["reward"] > cur["reward"]:
                    best_sft[r["prompt_id"]] = r
        for name in ("standard", "length_balanced", "beta_0.30", "beta_0.10", "beta_0.03"):
            for r in wl_rows.get(name, []):
                b = best_sft.get(r["prompt_id"])
                if b and not r["compliant"] and r["reward"] > b["reward"] and r["sample"] == 0:
                    lines += [f"### {LABEL[name]} on `{r['prompt_id']}` (limit {r['word_limit']} words)", "", quote(r["prompt"]), "",
                              f"**SFT (compliant)** - {b['words']} words, reward {b['reward']:.2f}", "", quote(b["response"]), "",
                              f"**{LABEL[name]} (violates)** - {r['words']} words, reward {r['reward']:.2f}", "", quote(r["response"]), ""]
    (out / "reward_vs_quality_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_word_limit(wl_rows, out):
    names = [n for n in ("sft", "standard", "length_balanced") if n in wl_rows]
    by_prompt = {}
    for n in names:
        for r in wl_rows[n]:
            if r["sample"] == 0:
                by_prompt.setdefault(r["prompt_id"], {})[n] = r
    lines = ["# Word-limit prompts: first sample per condition", ""]
    for pid, d in sorted(by_prompt.items()):
        first = next(iter(d.values()))
        lines += [f"## `{pid}` (limit {first['word_limit']} words)", "", quote(first["prompt"]), ""]
        for n in names:
            if n in d:
                r = d[n]
                lines += [f"**{LABEL[n]}** - {r['words']} words, compliant={bool(r['compliant'])}, reward {r['reward']:.2f}", "", quote(r["response"], 1200), ""]
    (out / "word_limit_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_length_pairs(cfg, strat_pairs, out, per_stratum=4):
    if not all(n in strat_pairs for n in ("standard", "length_balanced")):
        return
    rows = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["dpo_length_eval"])}
    bal = {r["prompt_id"]: r for r in strat_pairs["length_balanced"]}
    lines = ["# Length-stratified pairs where standard and length-balanced DPO disagree most", "",
             "m > 0 means the model prefers the dataset's chosen response relative to the reference.", ""]
    for s in STRATA:
        pairs = [(r, bal[r["prompt_id"]]) for r in strat_pairs["standard"] if r["length_stratum"] == s and r["prompt_id"] in bal]
        pairs.sort(key=lambda x: -abs(x[0]["margin"] - x[1]["margin"]))
        lines += [f"## {STRATUM_LABEL[s]}", ""]
        for a, b in pairs[:per_stratum]:
            row = rows[a["prompt_id"]]
            lines += [f"### `{a['prompt_id'][:12]}` - standard m={a['margin']:+.2f}, balanced m={b['margin']:+.2f}; "
                      f"chosen {a['chosen_tokens']} vs rejected {a['rejected_tokens']} tokens; "
                      f"scores {row['score_chosen']} vs {row['score_rejected']}", "",
                      "**Prompt**", "", quote(row["prompt"], 600), "",
                      "**Chosen**", "", quote(row["chosen"][-1]["content"], 700), "",
                      "**Rejected**", "", quote(row["rejected"][-1]["content"], 700), ""]
    (out / "length_pair_disagreements.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_beta(gens, out, top=5):
    names = [n for n in ("sft", "beta_0.03", "beta_0.10", "beta_0.30") if n in gens]
    if len(names) < 3:
        return
    by = {n: {r["prompt_id"]: r for r in gens[n]} for n in names}
    common = set.intersection(*(set(d) for d in by.values()))
    spread = sorted(common, key=lambda p: -np.ptp([by[n][p]["response_tokens"] for n in names]))
    lines = ["# Same held-out prompts across beta forks (largest length spread)", ""]
    for pid in spread[:top]:
        lines += [f"## `{pid[:12]}`", "", quote(by[names[0]][pid]["prompt"], 600), ""]
        for n in names:
            r = by[n][pid]
            lines += [f"**{LABEL[n]}** - reward {r['reward']:.2f}, {r['response_tokens']} tokens, "
                      f"seq-KL {r['kl_sum']:.2f}, truncated={r['truncated']}", "", quote(r["response"], 1000), ""]
    (out / "beta_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--out", default="report/task1")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    out = repo_path(args.out)
    tables, figs, qual = out / "tables", out / "figures", out / "qualitative"
    qual.mkdir(parents=True, exist_ok=True)

    names = [n for n, _ in CONDITIONS if (res / "eval" / n / "summary.json").exists()]
    print("Conditions found:", names)
    summaries = {n: load_json(res / "eval" / n / "summary.json") for n in names}
    train = {n: load_json(res / "train_logs" / f"{n}_summary.json") for n in names
             if (res / "train_logs" / f"{n}_summary.json").exists()}
    logs = {n: read_jsonl(res / "train_logs" / f"{n}.jsonl") for n, _ in CONDITIONS
            if (res / "train_logs" / f"{n}.jsonl").exists()}
    gens = {n: read_jsonl(res / "eval" / n / "generations_heldout.jsonl") for n in names}
    wl_rows = {n: read_jsonl(res / "eval" / n / "generations_word_limit.jsonl") for n in names}
    strat = {n: read_jsonl(res / "eval" / n / "pairs_stratified.jsonl") for n in names}
    analysis = load_json(res / "length_analysis.json") if (res / "length_analysis.json").exists() else None

    if summaries:
        write_table(summary_table(summaries, train), tables, "dpo_summary",
                    "Task 1 DPO summary. Standard and length-balanced runs: 1 epoch on all usable pairs; "
                    "beta forks: short runs on the first 600 usable standard pairs (different budget). "
                    "Pair metrics on the held-out pairs; RM/KL/length on sampled responses to the held-out prompts.")
        write_table(strata_table(summaries, analysis), tables, "length_strata",
                    "Standard vs length-balanced DPO on the length-stratified held-out set (paired bootstrap 95% CI).")
        agg, per_prompt = word_limit_tables(summaries, wl_rows)
        write_table(agg, tables, "word_limit", "Word-limit compliance on the 10 common prompts.")
        write_table(per_prompt, tables, "word_limit_per_prompt", "Per-prompt word-limit compliance (fraction of samples).")
    if (res / "dataset_length_stats.json").exists():
        write_table(dataset_table(load_json(res / "dataset_length_stats.json")), tables, "dataset_length_stats",
                    "Length structure of the preference DATA (Qwen tokenizer, untruncated responses).")
    if analysis:
        gen_rows = []
        for tag in ("standard_minus_sft", "length_balanced_minus_sft", "length_balanced_minus_standard"):
            g, w = analysis["generation"][tag], analysis["word_limit"][tag]
            ci = lambda d: f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
            gen_rows.append({"Comparison": tag.replace("_minus_", " - "), "Gen. tokens": ci(g["response_tokens"]),
                             "RM score": ci(g["reward"]), "WL compliance": ci(w["compliance"]), "WL words": ci(w["words"])})
        write_table(pd.DataFrame(gen_rows), tables, "paired_generation_differences",
                    "Paired differences on identical prompts (mean, bootstrap 95% CI).")
        rule_rows = [{"Model": LABEL[n], **{STRATUM_LABEL.get(s, s): f"{v['preference_accuracy']:.3f} (n={v['n']})"
                                            if v["preference_accuracy"] is not None else "n/a"
                                            for s, v in analysis["heldout_by_rule_stratum"][n].items()}}
                     for n in ("standard", "length_balanced")]
        write_table(pd.DataFrame(rule_rows), tables, "heldout_accuracy_by_length_rule",
                    "Preference accuracy on the ordinary held-out set split by the length rule.")

    fig_beta_sweep(summaries, figs)
    fig_training_curves(logs, figs)
    fig_length_strata(summaries, figs)
    fig_margin_vs_length({n: strat[n] for n in ("standard", "length_balanced") if n in strat}, figs)
    fig_generated_length(gens, wl_rows, figs)

    qual_reward_vs_quality(gens, wl_rows, qual)
    qual_word_limit(wl_rows, qual)
    qual_length_pairs(cfg, strat, qual)
    qual_beta(gens, qual)

    for f in ("dpo_loss_validation.json", "dataset_length_stats.json", "length_analysis.json"):
        if (res / f).exists():
            shutil.copy(res / f, out / f)
    print("Report material written to", out)


if __name__ == "__main__":
    main()
