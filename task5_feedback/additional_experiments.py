"""Task 5 additional experiments (beyond the required steps). They help interpret the main Task 5 results.

A. Lenient answer extraction (`--part lenient`, CPU). The official verifier only reads `#### <number>`.
   Many responses end with `\\boxed{n}`, `#### $n`, `#### Answer: n` or a final sentence ("... is **n**").
   The existing generations are re-scored with cumulative extraction tiers, so the official accuracy can be
   split into "wrong maths" vs "unparsable format". The official exact metric stays the reported one.
     tier 1 strict    : the supplied verifier (last `#### <number>`)
     tier 2 ####-loose: first number after the last `####` on that line (`#### $300`, `#### Answer: 50`)
     tier 3 boxed     : number in the last `\\boxed{...}`
     tier 4 last line : last bold number `**n**` in the last non-empty line, else the last number in that line

B. Judge raw-output probe (`--part judge_probe`, GPU). The supplied judge maps anything it cannot parse to
   TIE and caches only the parsed label. The probe reruns the same rubric prompt and chat template with
   greedy decoding, through the same 4-bit judge, and saves:
     - the raw text of the first 4 generated tokens (exactly what the supplied judge parses), plus 16 tokens for context;
     - whether the supplied regex finds A / B / TIE (otherwise it is a parse failure that became TIE);
     - next-token probabilities of "A", "B" and "T" (the first token of "TIE"), i.e. a soft preference.
   Every pair is asked in BOTH explicit orientations (the probe sets the order itself instead of using
   the hash-based swap), so position bias can be measured directly. The probed pairs are:
     - the 80 controlled diagnostic pairs (clean vs each perturbed variant);
     - every non-identical policy pair from the GSM8K and SVAMP evaluations;
     - a seeded sample of identical-response pairs, where any non-TIE verdict comes from position bias alone.

C. `--part report` (CPU) writes report/task5/additional/ from A and B.

Outputs: results/task5_feedback/additional/{lenient_<dataset>.jsonl, lenient_summary.json, judge_probe.jsonl, judge_probe_summary.json}
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import load_json, save_json
from common.report_utils import quote, write_table
from task5_feedback.evaluate_math import POLICIES
from task5_feedback.rlvr import extract_designated_final, numerically_equal
from task5_feedback.score_perturbations import PAIRS, REFERENCE, load_diagnostic_groups

LABEL = {"sft": "SFT", "rlvr": "RLVR", "rlaif": "RLAIF"}
DATASETS = {"gsm": "GSM8K", "transfer": "SVAMP"}
TIERS = ["strict", "hash_loose", "boxed", "last_line"]
TIER_NAME = {"strict": "1 strict #### (official)", "hash_loose": "2 + loose ####", "boxed": "3 + \\boxed{}",
             "last_line": "4 + last-line number"}
_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_BOLD_NUM = re.compile(r"\*\*[^*\d]*([-+]?\d[\d,]*(?:\.\d+)?)[^*]*\*\*")
_JUDGE_RE = re.compile(r"\b(A|B|TIE)\b")   # same regex as the supplied PairwiseAIJudge.compare


def first_number(text):
    m = _NUM.search(text or "")
    return m.group(0).replace(",", "") if m else None


def extract_tiered(response: str):
    """Return (tier, number) for the first tier that yields a number, or (None, None)."""
    strict = extract_designated_final(response)
    if strict is not None:
        return "strict", strict
    if "####" in response:
        after = response[response.rfind("####") + 4:].strip()
        n = first_number(after.splitlines()[0]) if after else None
        if n is not None:
            return "hash_loose", n
    boxed = _BOXED.findall(response)
    if boxed and first_number(boxed[-1]) is not None:
        return "boxed", first_number(boxed[-1])
    lines = [ln for ln in response.strip().splitlines() if ln.strip()]
    if lines:
        bold = _BOLD_NUM.findall(lines[-1])
        if bold:
            return "last_line", bold[-1].replace(",", "")
        nums = _NUM.findall(lines[-1])
        if nums:
            return "last_line", nums[-1].replace(",", "")
    return None, None


def run_lenient(cfg):
    src = repo_path(cfg["results_dir"]) / "task5_feedback"
    out = src / "additional"
    summary = {}
    for ds in DATASETS:
        rows, summary[ds] = [], {}
        for n in POLICIES:
            gens = read_jsonl(src / ds / f"generations_{n}.jsonl")
            for r in gens:
                tier, pred = extract_tiered(r["response"])
                rows.append({"prompt_id": r["prompt_id"], "policy": n, "gold_final": r["gold_final"], "official_exact": r["exact"],
                             "tier": tier, "pred": pred, "correct": bool(pred is not None and numerically_equal(pred, r["gold_final"]))})
            pol = [x for x in rows if x["policy"] == n]
            assert all(x["correct"] == bool(x["official_exact"]) for x in pol if x["tier"] == "strict"), "tier 1 must equal the verifier"
            summary[ds][n] = {
                "n": len(pol),
                # cumulative accuracy: answers extracted at tier <= t and correct
                "cumulative_accuracy": {t: float(np.mean([x["correct"] and x["tier"] in TIERS[:TIERS.index(t) + 1]
                                                          for x in pol])) for t in TIERS},
                "tier_counts": {str(t): c for t, c in Counter(x["tier"] for x in pol).items()},
                "tier_accuracy": {t: float(np.mean([x["correct"] for x in pol if x["tier"] == t])) if any(x["tier"] == t for x in pol) else None
                                  for t in TIERS},
                "no_number_found": sum(x["tier"] is None for x in pol),
            }
        write_jsonl(out / f"lenient_{ds}.jsonl", rows)
    save_json(out / "lenient_summary.json", summary)
    for ds, s in summary.items():
        for n, v in s.items():
            print(f"[{ds}/{n}] cumulative acc:", {t: round(a, 3) for t, a in v["cumulative_accuracy"].items()})
    return summary


# ----------------------------------------------------------------------------- judge probe (GPU)

@torch.no_grad()
def probe_one(judge, ids_ab, problem, a, b):
    """Same prompt/template/decoding as PairwiseAIJudge.compare, with A=a, B=b fixed (no hash swap)."""
    from task5_feedback.rlaif import PAIRWISE_RUBRIC
    tok, model = judge.tokenizer, judge.model
    ids = tok.apply_chat_template([{"role": "user", "content": PAIRWISE_RUBRIC.format(problem=problem, a=a, b=b)}],
                                  return_tensors="pt", add_generation_prompt=True).to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=16, do_sample=False, output_scores=True, return_dict_in_generate=True,
                         pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    gen = out.sequences[0, ids.shape[1]:]
    raw4 = tok.decode(gen[:4], skip_special_tokens=True).strip()       # what the supplied judge parses
    raw16 = tok.decode(gen, skip_special_tokens=True).strip()
    m = _JUDGE_RE.search(raw4.upper())
    probs = torch.softmax(out.scores[0][0].float(), dim=-1)
    pa, pb, pt = (float(probs[i]) for i in ids_ab)
    return {"raw4": raw4, "raw16": raw16, "parsed": m.group(1) if m else "TIE", "parse_failure": m is None,
            "p_A": pa, "p_B": pb, "p_T": pt, "prompt_tokens": int(ids.shape[1])}


def probe_items(cfg):
    """(set, id, problem, x, y) items; x is the 'reference' side (clean / first policy)."""
    items = []
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    for pid, g in sorted(groups.items(), key=lambda kv: int(kv[0])):
        for variant in PAIRS:
            items.append({"set": f"diag:{variant}", "id": pid, "problem": g[REFERENCE]["question"],
                          "x": g[REFERENCE]["response"], "y": g[variant]["response"], "x_label": REFERENCE, "y_label": variant})
    src = repo_path(cfg["results_dir"]) / "task5_feedback"
    rng = np.random.default_rng(int(cfg["seed"]))
    for ds in DATASETS:
        gens = {n: {r["prompt_id"]: r for r in read_jsonl(src / ds / f"generations_{n}.jsonl")} for n in POLICIES}
        pairs = read_jsonl(src / ds / "pairwise.jsonl")
        identical = []
        for p in pairs:
            a, b = gens[p["policy_a"]][p["prompt_id"]], gens[p["policy_b"]][p["prompt_id"]]
            item = {"id": p["prompt_id"], "problem": a["question"], "x": a["response"], "y": b["response"],
                    "x_label": p["policy_a"], "y_label": p["policy_b"], "exact_x": a["exact"], "exact_y": b["exact"],
                    "logged_judge": p["judge"]}
            if p["identical"]:
                if p["policy_a"] == "sft" and p["policy_b"] == "rlvr":   # one copy of each identical prompt
                    identical.append({**item, "set": f"{ds}:identical"})
            else:
                items.append({**item, "set": f"{ds}:policy_pair"})
        n_id = min(int(cfg.get("probe_identical_per_dataset", 50)), len(identical))
        if n_id:
            items += [identical[i] for i in sorted(rng.choice(len(identical), size=n_id, replace=False))]
    return items


def run_judge_probe(cfg):
    from task5_feedback.rlaif import PairwiseAIJudge
    out = repo_path(cfg["results_dir"]) / "task5_feedback" / "additional"
    path = out / "judge_probe.jsonl"
    items = probe_items(cfg)
    done = {(r["set"], r["id"], r.get("x_label"), r.get("y_label")) for r in read_jsonl(path)} if path.exists() else set()
    todo = [it for it in items if (it["set"], it["id"], it["x_label"], it["y_label"]) not in done]
    print(f"Judge probe: {len(items)} pairs x 2 orientations ({len(todo)} to do)", flush=True)
    judge = PairwiseAIJudge(cfg, out / "probe_cache_unused.json")   # loads the fixed judge; its cache is not used
    tok = judge.tokenizer
    ids_ab = [tok.encode("A")[0], tok.encode("B")[0], tok.encode("TIE")[0]]
    rows = read_jsonl(path) if path.exists() else []
    for k, it in enumerate(todo):
        xy = probe_one(judge, ids_ab, it["problem"], it["x"], it["y"])   # A = x
        yx = probe_one(judge, ids_ab, it["problem"], it["y"], it["x"])   # A = y
        rec = {k2: v for k2, v in it.items() if k2 not in ("problem", "x", "y")}
        rec.update({f"xy_{k2}": v for k2, v in xy.items()})
        rec.update({f"yx_{k2}": v for k2, v in yx.items()})
        rows.append(rec)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            write_jsonl(path, rows)
            print(f"  {k + 1}/{len(todo)}", flush=True)
    write_jsonl(path, rows)
    summary = summarize_probe(rows)
    save_json(out / "judge_probe_summary.json", summary)
    for s, v in summary.items():
        print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, dict)})


def preference_for_x(row, prefix):
    """Map the verdict of one orientation to 'x' / 'y' / 'TIE'."""
    p = row[f"{prefix}_parsed"]
    if p == "TIE":
        return "TIE"
    first_is_x = prefix == "xy"
    return ("x" if p == "A" else "y") if first_is_x else ("y" if p == "A" else "x")


def soft_x(row, prefix):
    """P(prefer x) - P(prefer y) from the next-token probabilities of one orientation."""
    pa, pb = row[f"{prefix}_p_A"], row[f"{prefix}_p_B"]
    return pa - pb if prefix == "xy" else pb - pa


def summarize_probe(rows):
    out = {}
    for s in sorted({r["set"] for r in rows}):
        sub = [r for r in rows if r["set"] == s]
        both = [(preference_for_x(r, "xy"), preference_for_x(r, "yx")) for r in sub]
        raw = Counter(r[f"{o}_raw4"] for r in sub for o in ("xy", "yx"))
        soft = np.array([(soft_x(r, "xy") + soft_x(r, "yx")) / 2 for r in sub])
        out[s] = {
            "n_pairs": len(sub),
            "parse_failure_rate": float(np.mean([r[f"{o}_parse_failure"] for r in sub for o in ("xy", "yx")])),
            "label_rate_first_position": {lab: float(np.mean([r[f"{o}_parsed"] == lab for r in sub for o in ("xy", "yx")]))
                                          for lab in ("A", "B", "TIE")},
            "x_preferred_both_orders": float(np.mean([a == b == "x" for a, b in both])),
            "y_preferred_both_orders": float(np.mean([a == b == "y" for a, b in both])),
            "tie_both_orders": float(np.mean([a == b == "TIE" for a, b in both])),
            "order_dependent": float(np.mean([a != b for a, b in both])),
            "mean_p_A": float(np.mean([r[f"{o}_p_A"] for r in sub for o in ("xy", "yx")])),
            "mean_p_B": float(np.mean([r[f"{o}_p_B"] for r in sub for o in ("xy", "yx")])),
            "mean_p_T": float(np.mean([r[f"{o}_p_T"] for r in sub for o in ("xy", "yx")])),
            "soft_pref_x_mean": float(soft.mean()),
            "soft_pref_x_positive": float(np.mean(soft > 0)),
            "top_raw_outputs": dict(raw.most_common(6)),
        }
    return out


# ----------------------------------------------------------------------------- report (CPU)

SET_NAME = {f"diag:{v}": f"diagnostic: clean vs {v}" for v in PAIRS} | {
    "gsm:policy_pair": "GSM8K: non-identical policy pairs", "transfer:policy_pair": "SVAMP: non-identical policy pairs",
    "gsm:identical": "GSM8K: identical responses (sample)", "transfer:identical": "SVAMP: identical responses (sample)"}


def run_report(cfg):
    res = repo_path(cfg["results_dir"]) / "task5_feedback" / "additional"
    out = repo_path("report/task5/additional")
    tab = out / "tables"
    if (res / "lenient_summary.json").exists():
        lenient = load_json(res / "lenient_summary.json")
        rows = []
        for ds, s in lenient.items():
            for n, v in s.items():
                rows.append({"Dataset": DATASETS[ds], "Policy": LABEL[n],
                             **{TIER_NAME[t]: v["cumulative_accuracy"][t] for t in TIERS},
                             "No number found": v["no_number_found"]})
        write_table(pd.DataFrame(rows), tab, "lenient_accuracy",
                    "Cumulative accuracy when the answer is also extracted from looser formats (tier 1 = the official "
                    "exact verifier; each later column adds one fallback). A gap between tier 1 and tier 4 means formatting "
                    "failures, not wrong maths. Supplementary to the official metric.")
        rows = []
        for ds, s in lenient.items():
            for n, v in s.items():
                rows.append({"Dataset": DATASETS[ds], "Policy": LABEL[n],
                             **{f"{TIER_NAME[t]} (n / acc)": f"{v['tier_counts'].get(t, 0)} / "
                                f"{'n/a' if v['tier_accuracy'][t] is None else format(v['tier_accuracy'][t], '.2f')}" for t in TIERS},
                             "None": v["tier_counts"].get("None", 0)})
        write_table(pd.DataFrame(rows), tab, "answer_formats",
                    "Where the final answer was found (first matching tier), with the accuracy within that tier.")
    if (res / "judge_probe_summary.json").exists():
        probe = load_json(res / "judge_probe_summary.json")
        rows = [{"Probe set": SET_NAME.get(s, s), "Pairs": v["n_pairs"], "Parse failure": v["parse_failure_rate"],
                 "First-token A / B / TIE": f"{v['label_rate_first_position']['A']:.2f} / {v['label_rate_first_position']['B']:.2f} / "
                                             f"{v['label_rate_first_position']['TIE']:.2f}",
                 "Mean p(A) / p(B) / p(T)": f"{v['mean_p_A']:.2f} / {v['mean_p_B']:.2f} / {v['mean_p_T']:.2f}",
                 "Top raw outputs": "; ".join(f"{k!r}x{c}" for k, c in list(v["top_raw_outputs"].items())[:3])}
                for s, v in probe.items()]
        write_table(pd.DataFrame(rows), tab, "judge_raw_outputs",
                    "Raw judge outputs (first 4 generated tokens, exactly what the supplied judge parses), in both "
                    "orientations. Parse failure = no A/B/TIE match; the supplied judge maps it to TIE.")
        rows = [{"Probe set": SET_NAME.get(s, s), "Pairs": v["n_pairs"],
                 "Prefers x (clean / first policy) in both orders": v["x_preferred_both_orders"],
                 "Prefers y in both orders": v["y_preferred_both_orders"], "TIE in both orders": v["tie_both_orders"],
                 "Order-dependent verdict": v["order_dependent"],
                 "Soft pref. for x: mean / frac > 0": f"{v['soft_pref_x_mean']:+.3f} / {v['soft_pref_x_positive']:.2f}"}
                for s, v in probe.items()]
        write_table(pd.DataFrame(rows), tab, "judge_position_bias",
                    "Each pair is judged in both orientations. Order-dependent = the verdict changes with the A/B order "
                    "(position bias). Soft preference = p(A) - p(B) of the first generated token, mapped to x and "
                    "averaged over both orders.")
        probe_rows = read_jsonl(res / "judge_probe.jsonl")
        md = ["# Judge raw-output examples (selected automatically)", ""]
        for s in probe:
            md.extend([f"## {SET_NAME.get(s, s)}", ""])
            for r in [r for r in probe_rows if r["set"] == s][:3]:
                md.append(f"- id {r['id']} ({r['x_label']} vs {r['y_label']}): x-first -> `{r['xy_raw16']!r}` "
                          f"(p A/B/T {r['xy_p_A']:.2f}/{r['xy_p_B']:.2f}/{r['xy_p_T']:.2f}); "
                          f"y-first -> `{r['yx_raw16']!r}` (p A/B/T {r['yx_p_A']:.2f}/{r['yx_p_B']:.2f}/{r['yx_p_T']:.2f})")
            md.append("")
        (out / "qualitative").mkdir(parents=True, exist_ok=True)
        (out / "qualitative" / "judge_raw_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if (res / "lenient_gsm.jsonl").exists():
        gens = {r["prompt_id"]: r for r in read_jsonl(repo_path(cfg["results_dir"]) / "task5_feedback" / "gsm" / "generations_sft.jsonl")}
        md = ["# Answers the official verifier misses (SFT, GSM8K; selected automatically)", ""]
        for t in TIERS[1:]:
            ex = [r for r in read_jsonl(res / "lenient_gsm.jsonl") if r["policy"] == "sft" and r["tier"] == t][:2]
            md.extend([f"## Found at tier {TIER_NAME[t]}", ""])
            for r in ex:
                md.extend([f"**{r['prompt_id']}** gold {r['gold_final']}, extracted {r['pred']} (correct={r['correct']})", "",
                           quote(gens[r["prompt_id"]]["response"][-400:], 400), ""])
        (out / "qualitative").mkdir(parents=True, exist_ok=True)
        (out / "qualitative" / "format_examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--part", choices=["lenient", "judge_probe", "report", "all"], default="all")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.part in ("lenient", "all"):
        run_lenient(cfg)
    if args.part in ("judge_probe", "all"):
        run_judge_probe(cfg)
    if args.part in ("report", "all"):
        run_report(cfg)


if __name__ == "__main__":
    main()
