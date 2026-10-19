"""Build the Task 3 report material from the saved results.

Reads results/task3_grpo/ (written by continue_train/evaluate/analyze_group_size/
compare_normalization) and writes tables (CSV + Markdown + LaTeX), figures (PNG + PDF)
and qualitative candidates to report/task3/. No model is loaded; runs on CPU in seconds.
"""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.metrics import safe_corr
from common.report_utils import paired_vs_reference, plt, quote, rl_heldout_table, save_fig, write_table
from task3_grpo.compare_normalization import LOSS_TYPES, fork_name

GRAY, INK, MUTED = "#8a8984", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order (reference palette)
BINS = ("hard", "medium", "easy")


def labels(cfg):
    return {"sft": "SFT reference", "midpoint": "GRPO midpoint (start)", "standard": f"Standard GRPO ({cfg['updates']} updates)",
            fork_name("grpo"): f"Fork canonical GRPO ({cfg['fork_updates']} upd.)",
            fork_name("dr_grpo"): f"Fork Dr. GRPO ({cfg['fork_updates']} upd.)"}


# ---------------------------------------------------------------- tables

def standard_table(logs):
    rows = []
    for r in logs:
        rows.append({"update": r["update"], "prompt_id": r["prompt_id"][:10], "reward mean": r["reward"],
                     "group std": r["group_reward_std"], "informative": r["informative"], "KL": r["kl"], "KL k3": r["kl_k3"],
                     "entropy": r["entropy"], "policy loss": r["policy_term"], "loss": r["loss"], "grad norm": r["grad_norm"],
                     "len mean": r["response_tokens_mean"], "masked": r["n_masked"],
                     "tokens in loss / generated": r["tokens_in_loss"] / max(1, r["tokens_generated"]),
                     "elapsed s": r["elapsed_s"], "peak VRAM GB": r["peak_vram_gb"]})
    return pd.DataFrame(rows)


def efficiency_table(logs, label):
    """Where do generated samples stop contributing signal? (RQ3)"""
    rows = []
    for n, lg in logs.items():
        stds = np.array([r["group_reward_std"] for r in lg])
        gn = np.array([r["grad_norm"] for r in lg])
        rewards = np.array([r["reward"] for r in lg])
        n_comp = sum(len(r["rewards"]) for r in lg)
        zero_adv = sum(abs(a) <= 1e-9 for r in lg for a in r["advantages"])
        rows.append({
            "Run": label[n], "Updates": len(lg), "Completions": n_comp,
            "Uninformative groups": int(sum(not r["informative"] for r in lg)),
            "Masked (truncated) completions": int(sum(r["n_masked"] for r in lg)),
            "Zero-advantage completions": int(zero_adv),
            "Tokens in loss / generated": sum(r["tokens_in_loss"] for r in lg) / max(1, sum(r["tokens_generated"] for r in lg)),
            "Group std mean": float(stds.mean()), "Group std min": float(stds.min()),
            "Reward mean (across updates) ± std": f"{rewards.mean():.3f} ± {rewards.std():.3f}",
            "Grad-norm CV": float(gn.std() / gn.mean()) if gn.mean() > 0 else float("nan"),
            "corr(group std, grad norm)": safe_corr(stds, gn),
        })
    return pd.DataFrame(rows)


def group_size_tables(gs):
    overall, by_bin = [], []
    for k, m in gs["per_k"].items():
        sub = gs["per_k_all_subsets"][k]
        overall.append({"K": int(k), "Groups": m["n_groups"], "Generations": m["n_generations"],
                        "Informative rate": m["informative_rate"], "Mean within-group std": m["mean_within_group_std"],
                        "Var(r - group mean)": m["var_relative_signal"], "Var(norm. adv.)": m["var_normalized_advantage"],
                        "Frac. completions w/ gradient": m["frac_completions_with_gradient"],
                        "Groups std<0.1": m["frac_groups_std_below_0.1"], "Groups std<0.25": m["frac_groups_std_below_0.25"],
                        "All-subsets informative": sub["informative_rate"], "All-subsets mean std": sub["mean_within_group_std"]})
        for b, v in gs["per_k_per_bin"][k].items():
            lo, hi = gs["bins"][b]["mean_reward_range"]
            by_bin.append({"K": int(k), "Difficulty bin": f"{b} [{lo:.2f}, {hi:.2f}]", "Groups": v["n_groups"],
                           "Informative rate": v["informative_rate"], "Mean within-group std": v["mean_within_group_std"],
                           "Var(r - group mean)": v["var_relative_signal"], "Groups std<0.25": v["frac_groups_std_below_0.25"],
                           "All-subsets informative": v["all_subsets"]["informative_rate"],
                           "All-subsets mean std": v["all_subsets"]["mean_within_group_std"]})
    return pd.DataFrame(overall), pd.DataFrame(by_bin).sort_values(["Difficulty bin", "K"])


def seq_grad_records(log):
    out = []
    for r in log:
        for T, a, gnorm, used in zip(r["response_tokens"], r["advantages"], r["seq_grad_norms"], r["in_loss"]):
            if gnorm > 0 and used:
                out.append({"update": r["update"], "length": T, "adv": a, "grad_norm": gnorm})
    return out


def normalization_table(logs, summaries, label):
    """Held-out metrics plus length-conditioned gradient statistics for the two normalizations."""
    recs = {n: seq_grad_records(logs[n]) for n in (fork_name(t) for t in LOSS_TYPES) if n in logs}
    pooled = [x["length"] for v in recs.values() for x in v]
    split = float(np.median(pooled)) if pooled else 0.0
    rows = []
    for n in (fork_name(t) for t in LOSS_TYPES):
        if n not in logs:
            continue
        r = recs[n]
        L = np.array([x["length"] for x in r], dtype=float)
        G = np.array([x["grad_norm"] for x in r])
        Gn = G / np.maximum(np.abs([x["adv"] for x in r]), 1e-6)  # per unit |advantage|
        short, long_ = L <= split, L > split
        s = summaries.get(n, {})
        rows.append({
            "Condition": label[n],
            "Held-out raw RM": s.get("raw_reward_mean"), "Held-out KL/token": s.get("kl_token_mean"),
            "Held-out len mean": s.get("length_tokens", {}).get("mean"), "Held-out len median": s.get("length_tokens", {}).get("median"),
            "Held-out trunc.": s.get("truncation_rate"), "Held-out entropy": s.get("entropy_token_mean"),
            "Train len mean": float(np.mean([x["response_tokens_mean"] for x in logs[n]])),
            "Gen. tokens": logs[n][-1]["generated_tokens_cum"],
            "Completions (unmasked)": len(r),
            f"Grad norm short (<= {split:.0f} tok)": float(G[short].mean()) if short.any() else None,
            f"Grad norm long (> {split:.0f} tok)": float(G[long_].mean()) if long_.any() else None,
            "Long/short grad ratio": float(G[long_].mean() / G[short].mean()) if short.any() and long_.any() else None,
            "Long share of grad norm": float(G[long_].sum() / G.sum()) if G.sum() > 0 else None,
            "corr(len, grad norm)": safe_corr(L, G),
            "corr(len, grad norm / |A|)": safe_corr(L, Gn),
        })
    return pd.DataFrame(rows), split


# ---------------------------------------------------------------- figures

def fig_standard(logs, out):
    x = [r["update"] for r in logs]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    ax = axes[0, 0]
    for r in logs:
        ax.scatter([r["update"]] * len(r["rewards"]), r["rewards"], s=10, color=GRAY, edgecolor="white", linewidth=0.4)
    ax.plot(x, [r["reward"] for r in logs], "-o", color=SERIES[0], ms=4, label="group mean")
    ax.set_title("Learned reward (dots = K completions)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    ax.plot(x, [r["group_reward_std"] for r in logs], "-o", color=SERIES[0], ms=4)
    bad = [r["update"] for r in logs if not r["informative"]]
    ax.scatter(bad, [0] * len(bad), marker="x", s=40, color="#e34948", label="uninformative group", zorder=3)
    ax.set_title("Within-group reward std", fontsize=9)
    if bad:
        ax.legend(fontsize=7)
    panels = [(axes[0, 2], "kl", "KL from reference (token)"), (axes[0, 3], "entropy", "Entropy (token)"),
              (axes[1, 0], "response_tokens_mean", "Response length (mean of K)"), (axes[1, 1], "policy_term", "Policy loss"),
              (axes[1, 2], "grad_norm", "Gradient norm")]
    for ax, k, title in panels:
        ax.plot(x, [r[k] for r in logs], "-o", color=SERIES[0], ms=4)
        ax.set_title(title, fontsize=9)
    ax = axes[1, 3]
    ax.bar(x, [r["n_masked"] for r in logs], color=SERIES[1], width=0.6)
    ax.set_title("Truncated completions masked (of K)", fontsize=9)
    for a in axes.flat:
        a.set_xlabel("update")
    fig.tight_layout()
    save_fig(fig, out, "standard_continuation_diagnostics")


def fig_group_size(gs, out):
    ks = list(gs["per_k"])
    metrics = [("informative_rate", "Informative-group rate"), ("mean_within_group_std", "Mean within-group reward std"),
               ("var_relative_signal", "Var(r - group mean)"), ("frac_groups_std_below_0.25", "Groups with std < 0.25")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.9))
    width = 0.25
    cats = ["all"] + [b for b in BINS if b in gs["bins"]]
    for ax, (m, title) in zip(axes, metrics):
        for i, k in enumerate(ks):
            vals = [gs["per_k"][k][m]] + [gs["per_k_per_bin"][k][b][m] for b in cats[1:]]
            ax.bar(np.arange(len(cats)) + (i - 1) * width, vals, width - 0.02, color=SERIES[i], label=f"K={k}")
        ax.set_xticks(np.arange(len(cats)))
        ax.set_xticklabels(cats)
        ax.set_title(title, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(ks), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, out, "group_size_by_difficulty")

    fig, ax = plt.subplots(figsize=(4.5, 3))
    pp = gs["per_prompt"]
    ax.scatter([p["mean_reward"] for p in pp], [p["std_k8"] for p in pp], s=24, color=SERIES[0], edgecolor="white", linewidth=0.6)
    for b in ("hard", "medium"):
        if b in gs["bins"]:
            ax.axvline(gs["bins"][b]["mean_reward_range"][1], color=GRAY, ls=":", lw=1)
    ax.set_xlabel("prompt mean reward over 8 cached completions (difficulty)")
    ax.set_ylabel("within-prompt reward std (K=8)")
    ax.set_title("Prompt difficulty vs reward spread (dotted = tertile edges)", fontsize=8)
    fig.tight_layout()
    save_fig(fig, out, "prompt_difficulty_vs_std")


def fig_normalization(logs, gens, label, out):
    names = [fork_name(t) for t in LOSS_TYPES if fork_name(t) in logs]
    if not names:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for i, n in enumerate(names):
        r = seq_grad_records(logs[n])
        axes[0].scatter([x["length"] for x in r], [x["grad_norm"] for x in r], s=14, color=SERIES[i], label=label[n],
                        edgecolor="white", linewidth=0.4)
        axes[1].plot([x["update"] for x in logs[n]], [x["response_tokens_mean"] for x in logs[n]], "-o", ms=4,
                     color=SERIES[i], label=label[n])
    axes[0].set_xlabel("completion length (tokens)")
    axes[0].set_ylabel("per-completion policy-gradient norm")
    if any(x["grad_norm"] > 0 for n in names for x in seq_grad_records(logs[n])):
        axes[0].set_yscale("log")
    else:
        axes[0].text(0.5, 0.5, "no unmasked completions", transform=axes[0].transAxes, ha="center", color=MUTED)
    axes[0].set_title("Gradient magnitude vs length", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("update")
    axes[1].set_title("Training completion length (mean of K)", fontsize=9)
    present = [n for n in ["midpoint"] + names if n in gens]
    data = [[r["response_tokens"] for r in gens[n]] for n in present]
    bp = axes[2].boxplot(data, vert=False, patch_artist=True, widths=0.6, showfliers=False)
    for patch, n in zip(bp["boxes"], present):
        patch.set_facecolor(GRAY if n == "midpoint" else SERIES[names.index(n)])
    for med in bp["medians"]:
        med.set_color(INK)
    axes[2].set_yticks(range(1, len(present) + 1))
    axes[2].set_yticklabels([label[n] for n in present], fontsize=7)
    axes[2].set_xlabel("held-out response tokens")
    axes[2].set_title("Held-out length", fontsize=9)
    fig.tight_layout()
    save_fig(fig, out, "normalization_comparison")


# ---------------------------------------------------------------- qualitative candidates

def qual_groups(cfg, gs, out):
    """Cached groups that illustrate informativeness: the lowest- and highest-spread prompts."""
    prompts = {r["prompt_id"]: r for r in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    cache = {}
    for r in read_jsonl(cfg["group_cache"]):
        cache.setdefault(str(r["source_index"]), []).append(r)
    pp = sorted(gs["per_prompt"], key=lambda p: p["std_k8"])
    lines = ["# Cached K=8 groups: lowest vs highest within-prompt reward spread", "",
             "Use these to illustrate why some groups give no relative signal. Quote minimally.", ""]
    for tag, sel in (("lowest spread", pp[:3]), ("highest spread", pp[-2:])):
        lines += [f"## {tag}", ""]
        for p in sel:
            rows = sorted(cache[p["source_index"]], key=lambda x: x["reward"])
            prompt = prompts.get(p["prompt_id"], {}).get("messages", [{"content": "(prompt not found)"}])[-1]["content"]
            lines += [f"### source_index {p['source_index']} ({p['bin']}): mean {p['mean_reward']:.2f}, std {p['std_k8']:.3f}, "
                      f"truncated {p['n_truncated']}/8", "", quote(prompt, 500), "",
                      "rewards: " + ", ".join(f"{x['reward']:.2f}" for x in rows), ""]
            for x in (rows[0], rows[-1]):
                lines += [f"**reward {x['reward']:.2f}, {x['completion_tokens']} tokens, truncated={x['clipped_at_max']}**", "",
                          quote(x["completion"], 700), ""]
    (out / "group_informativeness_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_normalization(gens, label, out, top=4):
    a, b = fork_name("grpo"), fork_name("dr_grpo")
    if a not in gens or b not in gens:
        return
    gb = {r["prompt_id"]: r for r in gens[b]}
    pairs = [(r, gb[r["prompt_id"]]) for r in gens[a] if r["prompt_id"] in gb]
    pairs.sort(key=lambda p: -abs(p[0]["response_tokens"] - p[1]["response_tokens"]))
    lines = ["# Held-out prompts where canonical GRPO and Dr. GRPO responses differ most in length", ""]
    for x, y in pairs[:top]:
        lines += [f"## `{x['prompt_id'][:12]}`", "", quote(x["prompt"], 500), ""]
        for n, r in ((a, x), (b, y)):
            lines += [f"**{label[n]}** - raw reward {r['raw_reward']:.2f}, {r['response_tokens']} tokens, "
                      f"truncated={r['truncated']}, repetition {r['repetition']:.2f}", "", quote(r["response"], 1000), ""]
    (out / "normalization_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_training_groups(logs, label, out):
    lines = ["# Training groups from the standard run: lowest and highest within-group reward std", ""]
    if "standard" not in logs:
        return
    ranked = sorted(logs["standard"], key=lambda r: r["group_reward_std"])
    for tag, sel in (("lowest std", ranked[:2]), ("highest std", ranked[-2:])):
        lines += [f"## {tag}", ""]
        for r in sel:
            lines += [f"### update {r['update']} `{r['prompt_id'][:12]}` std {r['group_reward_std']:.3f}", ""]
            for resp, rew, adv, T, tr in zip(r["responses"], r["rewards"], r["advantages"], r["response_tokens"], r["truncated"]):
                lines += [f"**reward {rew:.2f}, advantage {adv:+.2f}, {T} tokens, truncated={tr}**", "", quote(resp, 500), ""]
    (out / "training_group_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--out", default="report/task3")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res, out = repo_path(cfg["results_dir"]), repo_path(args.out)
    tables, figs, qual = out / "tables", out / "figures", out / "qualitative"
    qual.mkdir(parents=True, exist_ok=True)
    label = labels(cfg)
    order = ["sft", "midpoint", "standard"] + [fork_name(t) for t in LOSS_TYPES]

    names = [n for n in order if (res / "eval" / n / "summary.json").exists()]
    print("Conditions found:", names)
    summaries = {n: load_json(res / "eval" / n / "summary.json") for n in names}
    gens = {n: read_jsonl(res / "eval" / n / "generations.jsonl") for n in names}
    logs = {n: read_jsonl(res / "train_logs" / f"{n}.jsonl") for n in order if (res / "train_logs" / f"{n}.jsonl").exists()}

    if summaries:
        write_table(rl_heldout_table(summaries, label), tables, "heldout_all_conditions",
                    f"Held-out evaluation (rl_prompt_pool_eval, identical sampling/seed, cap {cfg['eval_max_response_length']} tokens).")
    if "midpoint" in gens:
        write_table(paired_vs_reference(gens, label, names), tables, "paired_vs_midpoint",
                    "Paired held-out differences versus the GRPO midpoint (same prompts; mean, bootstrap 95% CI).")
    if "standard" in logs:
        write_table(standard_table(logs["standard"]), tables, "standard_continuation_per_update",
                    "Standard GRPO continuation from the supplied midpoint: per-update diagnostics (K completions per update).")
        t = load_json(res / "train_logs" / "standard_summary.json")
        write_table(pd.DataFrame([{"Run": label["standard"], "Updates": t["updates"], "Generated tokens": t["generated_tokens"],
                                   "Wall-clock (min)": t["wall_clock_s"] / 60, "Peak VRAM (GB)": t["peak_vram_gb"], "GPU": t["gpu"]}]),
                    tables, "standard_continuation_compute", "Compute for the standard GRPO continuation.")
        fig_standard(logs["standard"], figs)
    if logs:
        write_table(efficiency_table(logs, label), tables, "signal_and_sample_efficiency",
                    "Where generated samples stop carrying learning signal (uninformative groups, masked truncations, zero advantages).")
    if (res / "group_size.json").exists():
        gs = load_json(res / "group_size.json")
        overall, by_bin = group_size_tables(gs)
        write_table(overall, tables, "group_size_equal_generations",
                    f"Equal-generation group-size study ({gs['total_generations']} cached completions, {gs['n_prompts']} prompts). "
                    "Informative = group reward std > 1e-6.")
        write_table(by_bin, tables, "group_size_by_difficulty", "Group-size study per prompt-difficulty bin (" + gs["difficulty_rule"] + ").")
        fig_group_size(gs, figs)
        qual_groups(cfg, gs, qual)
    norm, split = normalization_table(logs, summaries, label)
    if not norm.empty:
        write_table(norm, tables, "normalization_comparison",
                    f"Canonical GRPO vs Dr. GRPO normalization ({cfg['fork_updates']} matched updates). Gradient statistics use each "
                    f"unmasked completion's own policy-term gradient norm before the update; short/long split at the pooled median ({split:.0f} tokens).")
        fig_normalization(logs, gens, label, figs)
        qual_normalization(gens, label, qual)
    qual_training_groups(logs, label, qual)

    for f in ("grpo_validation.json", "group_size.json"):
        if (res / f).exists():
            shutil.copy(res / f, out / f)
    print("Report material written to", out)


if __name__ == "__main__":
    main()
