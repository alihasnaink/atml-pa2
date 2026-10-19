"""Small helpers shared by the per-task report builders (tables, figures, quotes)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.metrics import paired_bootstrap

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e4e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_table(df: pd.DataFrame, out_dir, name: str, caption: str):
    """CSV (machine readable) + Markdown (for reading) + LaTeX (for the paper)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    cells = [[fmt(v) for v in row] for row in df.itertuples(index=False)]
    md = [f"**{caption}**", "", "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(r) + " |" for r in cells]
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")
    tex = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}",
           r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
           " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    tex += [" & ".join(esc(c) for c in r) + r" \\" for r in cells]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def save_fig(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)


def clip_text(text: str, n: int = 1500) -> str:
    text = str(text).strip()
    return text if len(text) <= n else text[:n] + " [...]"


def quote(text: str, n: int = 1500) -> str:
    return "\n".join("> " + line for line in clip_text(text, n).splitlines()) or "> (empty)"


def rl_heldout_table(summaries, label):
    """One row per condition from common.rl_utils held-out summaries."""
    rows = []
    for n, s in summaries.items():
        L = s["length_tokens"]
        rows.append({"Condition": label[n], "Reward (eff.)": f"{s['reward_mean']:.3f} ± {s['reward_std']:.3f}",
                     "Raw RM": s["raw_reward_mean"], "KL/token": s["kl_token_mean"], "KL/seq": s["kl_sequence_mean"],
                     "Entropy": s["entropy_token_mean"], "Len mean ± std": f"{L['mean']:.1f} ± {L['std']:.1f}",
                     "Len median [IQR]": f"{L['median']:.0f} [{L['q25']:.0f}-{L['q75']:.0f}]",
                     "EOS rate": s["eos_rate"], "Trunc. rate": s["truncation_rate"], "Repetition": s["repetition_mean"],
                     "corr(RM, len)": s["corr_reward_vs_length"], "corr(RM, KL)": s["corr_reward_vs_kl_sequence"]})
    return pd.DataFrame(rows)


def paired_vs_reference(gens, label, names, ref="midpoint"):
    """Paired per-prompt differences of each condition against `ref` (mean, bootstrap 95% CI)."""
    rows = []
    base = {r["prompt_id"]: r for r in gens[ref]}
    ci = lambda d: f"{d['mean_diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
    for n in names:
        if n == ref or n not in gens:
            continue
        pairs = [(base[r["prompt_id"]], r) for r in gens[n] if r["prompt_id"] in base]
        get = lambda k: ([a[k] for a, _ in pairs], [b[k] for _, b in pairs])
        rows.append({f"Condition - {ref}": label[n],
                     "Raw RM": ci(paired_bootstrap(*get("raw_reward"))),
                     "Reward (eff.)": ci(paired_bootstrap(*get("reward"))),
                     "Len (tokens)": ci(paired_bootstrap(*get("response_tokens"))),
                     "Truncated": ci(paired_bootstrap(*[[float(x) for x in v] for v in get("truncated")])),
                     "Repetition": ci(paired_bootstrap(*get("repetition"))),
                     "KL/seq": ci(paired_bootstrap(*get("kl_sum")))})
    return pd.DataFrame(rows)
