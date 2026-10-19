"""Build the Task 2 report material from the saved results.

Reads results/task2_ppo/ (written by continue_train/evaluate/analyze_clipping/ablate_kl)
and writes tables (CSV + Markdown + LaTeX), figures (PNG + PDF) and qualitative
candidates to report/task2/. No model is loaded; runs on CPU in seconds.
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
from task2_ppo.forks import fork_name

GRAY, INK, MUTED = "#8a8984", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order (reference palette)


def conditions(cfg):
    eps0, beta0 = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    eps_forks = [fork_name(float(e), beta0) for e in cfg["clip_values"]]
    kl_forks = [fork_name(eps0, float(b)) for b in cfg["kl_values"]]
    label = {"sft": "SFT reference", "midpoint": "PPO midpoint (start)", "standard": f"Standard ({cfg['updates']} updates)"}
    for e in cfg["clip_values"]:
        label[fork_name(float(e), beta0)] = f"Fork eps={float(e):.2f}, beta_KL={beta0:.2f}"
    for b in cfg["kl_values"]:
        label[fork_name(eps0, float(b))] = f"Fork eps={eps0:.2f}, beta_KL={float(b):.2f}"
    order = ["sft", "midpoint", "standard"] + list(dict.fromkeys(eps_forks + kl_forks))
    return order, label, eps_forks, kl_forks


# ---------------------------------------------------------------- tables


def standard_table(logs):
    keys = ["update", "prompt_id", "reward", "raw_reward", "kl", "entropy", "policy_loss", "value_loss", "explained_variance",
            "policy_grad_norm", "value_grad_norm", "clip_fraction_last_epoch", "affected_fraction_last_epoch",
            "update_kl_old_new", "response_tokens", "truncated", "elapsed_s", "peak_vram_gb"]
    df = pd.DataFrame([{k: r[k] for k in keys} for r in logs])
    df["prompt_id"] = df["prompt_id"].str[:10]
    return df


def stability_row(name, label, logs, s):
    gn = [r["policy_grad_norm"] for r in logs]
    shift = [r["update_kl_old_new"] for r in logs]
    return {
        "Condition": label[name],
        "Updates": len(logs), "Gen. tokens": logs[-1]["generated_tokens_cum"],
        "Train reward (mean)": float(np.mean([r["reward"] for r in logs])),
        "Clip frac. (last epoch, mean)": float(np.mean([r["clip_fraction_last_epoch"] for r in logs])),
        "Affected frac. (mean)": float(np.mean([r["affected_fraction_last_epoch"] for r in logs])),
        "Update KL old->new mean": float(np.mean(shift)), "Update KL old->new max": float(np.max(shift)),
        "Max |log ratio|": float(np.max([r["update_max_abs_log_ratio"] for r in logs])),
        "Policy grad-norm CV": float(np.std(gn) / np.mean(gn)) if np.mean(gn) > 0 else float("nan"),
        "Held-out reward": s["reward_mean"] if s else None, "Held-out KL/token": s["kl_token_mean"] if s else None,
        "Held-out len": s["length_tokens"]["mean"] if s else None, "Held-out entropy": s["entropy_token_mean"] if s else None,
    }


def kl_row(name, label, logs, s):
    first, last = logs[: len(logs) // 2], logs[len(logs) // 2:]
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    return {
        "Condition": label[name], "beta_KL": logs[0]["kl_beta"],
        "Updates": len(logs), "Gen. tokens": logs[-1]["generated_tokens_cum"],
        "Held-out reward": s["reward_mean"], "Held-out raw RM": s["raw_reward_mean"],
        "Held-out KL/token": s["kl_token_mean"], "Held-out KL/seq": s["kl_sequence_mean"],
        "Held-out entropy": s["entropy_token_mean"], "Held-out len": s["length_tokens"]["mean"],
        "Trunc. rate": s["truncation_rate"], "Repetition": s["repetition_mean"],
        "Train reward 1st/2nd half": f"{m(first, 'reward'):.3f} / {m(last, 'reward'):.3f}",
        "Train KL 1st/2nd half": f"{m(first, 'kl'):.4f} / {m(last, 'kl'):.4f}",
        "Train entropy 1st/2nd half": f"{m(first, 'entropy'):.3f} / {m(last, 'entropy'):.3f}",
        "Train len 1st/2nd half": f"{m(first, 'response_tokens'):.0f} / {m(last, 'response_tokens'):.0f}",
    }



def reward_by_kl_bins(gens, label, names, bins=4):
    """Within each condition: raw reward and length by quartile of per-response sequence KL."""
    rows = []
    for n in names:
        if n not in gens or n == "sft":
            continue
        g = gens[n]
        kl = np.array([r["kl_sum"] for r in g])
        edges = np.quantile(kl, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, kl, side="right") - 1, 0, bins - 1)
        for b in range(bins):
            sel = [r for r, i in zip(g, idx) if i == b]
            if sel:
                rows.append({"Condition": label[n], "KL/seq quartile": f"Q{b + 1} [{edges[b]:.2f}, {edges[b + 1]:.2f}]",
                             "n": len(sel), "Raw RM": float(np.mean([r["raw_reward"] for r in sel])),
                             "Len": float(np.mean([r["response_tokens"] for r in sel])),
                             "Trunc. rate": float(np.mean([r["truncated"] for r in sel])),
                             "Repetition": float(np.mean([r["repetition"] for r in sel]))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def series(logs, key):
    return [r["update"] for r in logs], [r[key] for r in logs]


def fig_standard(logs, out):
    panels = [("reward", "Learned reward (eff.)"), ("kl", "KL from reference (token)"), ("entropy", "Entropy (token)"),
              ("response_tokens", "Response length (tokens)"), ("policy_loss", "Policy loss"), ("value_loss", "Value loss"),
              ("policy_grad_norm", "Policy grad norm"), ("clip_fraction_last_epoch", "Clip fraction (last epoch)")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5))
    for ax, (k, title) in zip(axes.flat, panels):
        x, y = series(logs, k)
        ax.plot(x, y, "-o", color=SERIES[0], lw=1.5, ms=4)
        ax.plot(x, pd.Series(y).expanding().mean(), color=INK, lw=1, ls="--", label="running mean")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("update")
    x, y = series(logs, "raw_reward")
    axes[0, 0].plot(x, y, "x", color=GRAY, ms=5, label="raw (before EOS penalty)")
    axes[0, 0].legend(fontsize=7)
    ax2 = axes[1, 1]
    x, y = series(logs, "explained_variance")
    ax2.text(0.02, 0.95, f"critic explained variance: median {np.nanmedian(y):.2f}", transform=ax2.transAxes,
             fontsize=7, va="top", color=MUTED)
    x, y = series(logs, "affected_fraction_last_epoch")
    axes[1, 3].plot(x, y, "-s", color=SERIES[1], lw=1.2, ms=3, label="affected fraction")
    axes[1, 3].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, "standard_continuation_diagnostics")


def fig_clipping_cached(cached, out):
    grid = cached["step0_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    e = [g["eps"] for g in grid]
    axes[0].plot(e, [g["clip_fraction"] for g in grid], color=SERIES[0], lw=2, label="clip fraction (rho outside band)")
    axes[0].plot(e, [g["affected_fraction"] for g in grid], color=SERIES[1], lw=2, label="affected fraction (clip binds)")
    for k in cached["step0"]:
        axes[0].axvline(float(k), color=GRAY, ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("epsilon")
    axes[0].set_title("Cached batch, step 0 (midpoint vs old policy)", fontsize=9)
    axes[0].legend(fontsize=7)
    for i, (k, trace) in enumerate(cached["update_steps"].items()):
        s = [t["step"] for t in trace]
        axes[1].plot(s, [t["clip_fraction"] for t in trace], "-o", color=SERIES[i], ms=4, label=f"eps={k} clip")
        axes[1].plot(s, [t["affected_fraction"] for t in trace], "--", color=SERIES[i], lw=1, label=f"eps={k} affected")
        axes[2].plot(s, [t["kl_midpoint_new"] for t in trace], "-o", color=SERIES[i], ms=4, label=f"eps={k}")
    axes[1].set_title("Fractions during PPO steps on the cached batch", fontsize=9)
    axes[1].set_xlabel("full-batch PPO step")
    axes[1].legend(fontsize=6, ncol=2)
    axes[2].set_title("Drift from midpoint, KL(start || current)", fontsize=9)
    axes[2].set_xlabel("full-batch PPO step")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, "clipping_cached_batch")


def fig_forks(logs, names, label, keys, out, fname):
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.8))
    for i, n in enumerate(names):
        if n not in logs:
            continue
        for ax, (k, title) in zip(axes, keys):
            x, y = series(logs[n], k)
            ax.plot(x, y, "-o", color=SERIES[i % 3], lw=1.5, ms=4, label=label[n])
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("update")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, out, fname)


def fig_kl_differences(logs, kl_forks, label, out):
    """Paired per-update differences against the strongest-KL fork (same prompt and sampling seed per update)."""
    present = [n for n in kl_forks if n in logs]
    if len(present) < 2:
        return
    ref = present[-1]
    keys = [("reward", "Reward"), ("kl", "KL (token)"), ("entropy", "Entropy"), ("response_tokens", "Length")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for i, n in enumerate(present[:-1]):
        for ax, (k, title) in zip(axes, keys):
            d = [a[k] - b[k] for a, b in zip(logs[n], logs[ref])]
            ax.plot(range(1, len(d) + 1), d, "-o", color=SERIES[i], ms=4, label=f"{label[n]} - {label[ref]}")
            ax.axhline(0, color=GRAY, lw=0.8)
            ax.set_title(f"{title}: paired difference", fontsize=9)
            ax.set_xlabel("update")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    save_fig(fig, out, "kl_forks_paired_differences")


def fig_reward_vs_kl(summaries, gens, label, detail, out):
    """Left: one point per condition (gray = the two reference policies). Right: per-response detail panels."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for n, s in summaries.items():
        axes[0].scatter(s["kl_token_mean"], s["raw_reward_mean"], s=40, color=GRAY if n in ("sft", "midpoint") else SERIES[0],
                        edgecolor="white", linewidth=1, zorder=3)
        axes[0].annotate(label[n], (s["kl_token_mean"], s["raw_reward_mean"]), fontsize=6, color=MUTED,
                         xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("held-out KL from reference (token)")
    axes[0].set_ylabel("held-out raw learned reward")
    axes[0].set_title("Condition means", fontsize=9)
    for ax, n in zip(axes[1:], [x for x in detail if x in gens]):
        g = gens[n]
        ax.scatter([r["kl_sum"] for r in g], [r["raw_reward"] for r in g], s=10, color=SERIES[0],
                   edgecolor="white", linewidth=0.4)
        ax.set_title(f"{label[n]}: per response (r={safe_corr([r['kl_sum'] for r in g], [r['raw_reward'] for r in g]):.2f})", fontsize=8)
        ax.set_xlabel("sequence KL from reference")
        ax.set_ylabel("raw learned reward")
    fig.tight_layout()
    save_fig(fig, out, "reward_vs_kl")


# ---------------------------------------------------------------- qualitative candidates

def flags(a, b):
    out = []
    if b["truncated"] and not a["truncated"]:
        out.append("new truncation")
    if b["response_tokens"] > 1.5 * max(a["response_tokens"], 1):
        out.append("length x%.1f" % (b["response_tokens"] / max(a["response_tokens"], 1)))
    if b["repetition"] > a["repetition"] + 0.1:
        out.append("repetition +%.2f" % (b["repetition"] - a["repetition"]))
    return out


def qual_reward_quality(gens, label, compare, out, top=5):
    lines = ["# Candidates: learned reward vs response quality (held-out prompts)", "",
             "Auto-selected by reward change versus the PPO midpoint on the same prompt. 'Agree' candidates have a",
             "reward gain without length/truncation/repetition red flags; 'disagree' candidates gain reward together",
             "with such flags, or lose reward despite looking similar. Read each pair and keep only cases you judge",
             "yourself. Quote minimally in the report.", ""]
    base = {r["prompt_id"]: r for r in gens["midpoint"]}
    for n in compare:
        if n not in gens:
            continue
        pairs = [(base[r["prompt_id"]], r) for r in gens[n] if r["prompt_id"] in base]
        gain = lambda p: p[1]["raw_reward"] - p[0]["raw_reward"]
        agree = sorted([p for p in pairs if gain(p) > 0 and not flags(*p)], key=lambda p: -gain(p))[:top]
        disagree = sorted([p for p in pairs if gain(p) > 0 and flags(*p)], key=lambda p: -gain(p))[:top]
        for title, sel in (("agree", agree), ("disagree", disagree)):
            lines += [f"## {label[n]} vs midpoint - {title} candidates", ""]
            for a, b in sel:
                lines += [f"### `{a['prompt_id'][:12]}` raw reward {a['raw_reward']:.2f} -> {b['raw_reward']:.2f}"
                          f" ({', '.join(flags(a, b)) or 'no flags'})", "", "**Prompt**", "", quote(a["prompt"], 700), "",
                          f"**Midpoint** - {a['response_tokens']} tokens, truncated={a['truncated']}, repetition {a['repetition']:.2f}",
                          "", quote(a["response"], 1200), "",
                          f"**{label[n]}** - {b['response_tokens']} tokens, truncated={b['truncated']}, repetition {b['repetition']:.2f}",
                          "", quote(b["response"], 1200), ""]
    (out / "reward_vs_quality_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_kl_forks(gens, label, kl_forks, out, top=4):
    present = [n for n in kl_forks if n in gens]
    if len(present) < 2:
        return
    by = {n: {r["prompt_id"]: r for r in gens[n]} for n in present}
    common = set.intersection(*(set(d) for d in by.values()))
    spread = sorted(common, key=lambda p: -np.ptp([by[n][p]["raw_reward"] for n in present]))
    lines = ["# Same held-out prompts across beta_KL forks (largest reward spread)", ""]
    for pid in spread[:top]:
        lines += [f"## `{pid[:12]}`", "", quote(by[present[0]][pid]["prompt"], 600), ""]
        for n in present:
            r = by[n][pid]
            lines += [f"**{label[n]}** - raw reward {r['raw_reward']:.2f}, {r['response_tokens']} tokens, seq-KL {r['kl_sum']:.2f}, "
                      f"truncated={r['truncated']}, repetition {r['repetition']:.2f}", "", quote(r["response"], 1000), ""]
    (out / "kl_fork_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qual_training_rollouts(logs, label, names, out, top=3):
    lines = ["# Highest- and lowest-reward training rollouts per run (on-policy samples during the continuation)", ""]
    for n in names:
        if n not in logs:
            continue
        ranked = sorted(logs[n], key=lambda r: -r["raw_reward"])
        for tag, sel in (("highest", ranked[:top]), ("lowest", ranked[-top:])):
            lines += [f"## {label[n]} - {tag} reward", ""]
            for r in sel:
                lines += [f"**update {r['update']}** `{r['prompt_id'][:12]}` - raw {r['raw_reward']:.2f}, eff. {r['reward']:.2f}, "
                          f"{r['response_tokens']} tokens, KL {r['kl']:.4f}", "", quote(r["response"], 900), ""]
    (out / "training_rollouts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--out", default="report/task2")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res, out = repo_path(cfg["results_dir"]), repo_path(args.out)
    tables, figs, qual = out / "tables", out / "figures", out / "qualitative"
    qual.mkdir(parents=True, exist_ok=True)

    order, label, eps_forks, kl_forks = conditions(cfg)
    names = [n for n in order if (res / "eval" / n / "summary.json").exists()]
    print("Conditions found:", names)
    summaries = {n: load_json(res / "eval" / n / "summary.json") for n in names}
    gens = {n: read_jsonl(res / "eval" / n / "generations.jsonl") for n in names}
    logs = {n: read_jsonl(res / "train_logs" / f"{n}.jsonl") for n in order if (res / "train_logs" / f"{n}.jsonl").exists()}
    train = {n: load_json(res / "train_logs" / f"{n}_summary.json") for n in logs}

    if summaries:
        write_table(rl_heldout_table(summaries, label), tables, "heldout_all_conditions",
                    "Held-out evaluation (rl_prompt_pool_eval, identical sampling/seed, cap "
                    f"{cfg['eval_max_response_length']} tokens). Reward (eff.) includes the missing-EOS penalty.")
    if "standard" in logs:
        write_table(standard_table(logs["standard"]), tables, "standard_continuation_per_update",
                    "Standard PPO continuation from the supplied midpoint: per-update diagnostics.")
        t = train["standard"]
        compute = pd.DataFrame([{"Run": label["standard"], "Updates": t["updates"], "Generated tokens": t["generated_tokens"],
                                 "Wall-clock (min)": t["wall_clock_s"] / 60, "Peak VRAM (GB)": t["peak_vram_gb"], "GPU": t["gpu"]}])
        write_table(compute, tables, "standard_continuation_compute", "Compute for the standard continuation.")
        fig_standard(logs["standard"], figs)
    if (res / "clipping_cached.json").exists():
        cached = load_json(res / "clipping_cached.json")
        rows = []
        for k, s0 in cached["step0"].items():
            last = cached["update_steps"][k][-1]
            rows.append({"epsilon": float(k), "Clip frac. (step 0)": s0["clip_fraction"], "Affected frac. (step 0)": s0["affected_fraction"],
                         "Surrogate (step 0)": s0["surrogate"], "Unclipped surrogate (step 0)": s0["unclipped_surrogate"],
                         f"Clip frac. (after {cached['cache_update_steps']} steps)": last["clip_fraction"],
                         f"Affected frac. (after {cached['cache_update_steps']} steps)": last["affected_fraction"],
                         "Surrogate (after)": last["surrogate"], "KL midpoint->after": last["kl_midpoint_new"]})
        write_table(pd.DataFrame(rows), tables, "clipping_cached_batch",
                    f"Cached rollout batch ({cached['n_rebuilt']} rollouts, {cached['response_tokens_total']} response tokens): "
                    "clipped surrogate, clip fraction and affected-token fraction under the same response mask.")
        rec = cached["reconstruction"]
        write_table(pd.DataFrame([{"Check": k, "Mean abs. diff": v} for k, v in rec.items() if k != "per_row"]
                                 + [{"Check": "token-length mismatches", "Mean abs. diff": len(cached["token_mismatch_prompt_ids"])}]),
                    tables, "cached_batch_reconstruction", "Reconstruction of the cached batch (recomputed vs cached values).")
        fig_clipping_cached(cached, figs)
    stab = [stability_row(n, label, logs[n], summaries.get(n)) for n in eps_forks if n in logs]
    if stab:
        write_table(pd.DataFrame(stab), tables, "clipping_forks",
                    f"Matched short forks ({cfg['fork_updates']} updates, beta_KL={cfg['kl_beta']}) for each epsilon. "
                    "Stability statistic: per-update KL(old||new) on the update's own rollout (mean and max).")
        fig_forks(logs, eps_forks, label, [("reward", "Train reward"), ("update_kl_old_new", "Update KL(old||new)"),
                                           ("clip_fraction_last_epoch", "Clip fraction (last epoch)"),
                                           ("policy_grad_norm", "Policy grad norm")], figs, "clipping_forks_trajectories")
    klr = [kl_row(n, label, logs[n], summaries[n]) for n in kl_forks if n in logs and n in summaries]
    if klr:
        write_table(pd.DataFrame(klr), tables, "kl_forks",
                    f"Matched short forks ({cfg['fork_updates']} updates, eps={cfg['clip_epsilon']}) for each beta_KL.")
        fig_forks(logs, kl_forks, label, [("reward", "Train reward"), ("kl", "KL from reference"), ("entropy", "Entropy"),
                                          ("response_tokens", "Response length")], figs, "kl_forks_trajectories")
        fig_kl_differences(logs, kl_forks, label, figs)
        per_update = pd.DataFrame([{"Condition": label[n], "update": r["update"], "prompt_id": r["prompt_id"][:10],
                                    "reward": r["reward"], "kl": r["kl"], "entropy": r["entropy"],
                                    "response_tokens": r["response_tokens"]} for n in kl_forks if n in logs for r in logs[n]])
        per_update.to_csv(tables / "kl_forks_per_update.csv", index=False)
    weakest_kl = fork_name(float(cfg["clip_epsilon"]), min(map(float, cfg["kl_values"])))
    if "midpoint" in gens:
        write_table(paired_vs_reference(gens, label, names), tables, "paired_vs_midpoint",
                    "Paired held-out differences versus the PPO midpoint (same prompts; mean, bootstrap 95% CI).")
        write_table(reward_by_kl_bins(gens, label, names), tables, "reward_by_kl_quartile",
                    "Within each condition: held-out responses grouped by quartile of sequence KL from the reference.")
        qual_reward_quality(gens, label, ["standard", weakest_kl], qual)
    if summaries:
        fig_reward_vs_kl(summaries, gens, label, ["standard", weakest_kl], figs)
    qual_kl_forks(gens, label, kl_forks, qual)
    qual_training_rollouts(logs, label, ["standard"] + kl_forks, qual)

    for f in ("ppo_validation.json", "clipping_cached.json"):
        if (res / f).exists():
            shutil.copy(res / f, out / f)
    print("Report material written to", out)


if __name__ == "__main__":
    main()
