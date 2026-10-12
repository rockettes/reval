"""F4c-1: Statistical analysis on REVAL multi-judge scores.

Reads from scores/reval_scores.json.

Computes:
- Aggregate REVAL per generator (mean across non-self judges) + bootstrap CI
- Per-dimension means per (judge, generator) — 6 dimensions
- Bias matrix B[j,g] = mean(j→g) - global_mean(g)
- Inter-judge agreement (Krippendorff's alpha per dimension + REVAL)
- Paired Wilcoxon Ours vs each baseline (REVAL)
- Bonferroni correction
- Cohen's d + Cliff's delta
- Disagreement classification (σ < 0.5 high vs ≥ 0.5 low agreement)

Output: results/stats_reval.json
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _stats_utils import bootstrap_ci, paired_test_full, bonferroni

DRY = Path(__file__).resolve().parents[1]
SCORES_FILE = DRY / "scores" / "reval_scores.json"
OUT = DRY / "results" / "stats_reval.json"

DIMENSIONS = [
    ("correcao_cientifica", "Correção", 0.25),
    ("verificab_citacoes",  "Citações", 0.25),
    ("especificidade",      "Especif.", 0.20),
    ("completude",          "Compl.",   0.15),
    ("assertividade",       "Assert.",  0.10),
    ("aplicacao_pratica",   "Aplic.",   0.05),
]
# Updated 2026-05-12: K=5 judges and 6 generators (previously stale K=4 / 4 gens).
# stats_reval.json output now includes Krippendorff α, bias matrix, per-dim means,
# and pairwise comparisons.
GENERATORS = ["Ours", "GPT54", "GLM", "Gemini", "Grok", "GemSearch"]
JUDGES = ["Atla", "GPT54", "GLM", "Gemini", "Grok"]


def load_scores():
    """Load scores. Excluded entries are dropped."""
    raw = json.load(open(SCORES_FILE))
    s = [x for x in raw if x.get("scores") and not x.get("excluded")]
    excluded_count = sum(1 for x in raw if x.get("excluded"))
    return s, excluded_count


def reval(scores):
    return sum(scores[k]*w for k,_,w in DIMENSIONS)


def krippendorff_alpha_interval(items_ratings):
    """Krippendorff's alpha for interval data.
    items_ratings: list of lists, inner = ratings for one item from different judges."""
    pairs = []
    for ratings in items_ratings:
        clean = [r for r in ratings if r is not None]
        if len(clean) < 2: continue
        for i in range(len(clean)):
            for j in range(len(clean)):
                if i != j:
                    pairs.append((clean[i], clean[j]))
    if not pairs: return 0
    Do = sum((a-b)**2 for a, b in pairs) / len(pairs)
    all_vals = [v for ratings in items_ratings for v in ratings if v is not None]
    if len(all_vals) < 2: return 0
    N = len(all_vals)
    De = sum((a-b)**2 for i,a in enumerate(all_vals) for b in all_vals[i+1:]) * 2 / (N*(N-1))
    return 1 - Do/De if De > 0 else 0


def main():
    scores, excluded = load_scores()
    print(f"Loaded {len(scores)} scores ({excluded} excluded GLM judge bugs)")

    # Group: scores[judge][generator][qid] = score_dict
    grouped = defaultdict(lambda: defaultdict(dict))
    for s in scores:
        if s.get("scores"):
            grouped[s["judge"]][s["generator"]][s["qid"]] = s["scores"]

    # ========================================================================
    # 1. Aggregate REVAL per generator (mean across non-self judges)
    # ========================================================================
    gen_aggregates = {}
    for g in GENERATORS:
        per_question_means = {}
        all_judges_revs = []
        for qid in range(1, 351):
            judge_revs = []
            for j in JUDGES:
                if j == g: continue
                if qid in grouped.get(j, {}).get(g, {}):
                    s = grouped[j][g][qid]
                    judge_revs.append(reval(s))
            if judge_revs:
                per_question_means[qid] = statistics.mean(judge_revs)
                all_judges_revs.extend(judge_revs)

        if not all_judges_revs:
            continue

        per_q_values = list(per_question_means.values())
        mean_v, lo, hi = bootstrap_ci(per_q_values, n_iter=10000)
        gen_aggregates[g] = {
            "n_evaluations": len(all_judges_revs),
            "n_questions": len(per_question_means),
            "mean_reval": round(mean_v, 4),
            "median_reval": round(statistics.median(per_q_values), 4),
            "std": round(statistics.stdev(per_q_values), 4) if len(per_q_values) > 1 else 0,
            "ci_95_lo": round(lo, 4),
            "ci_95_hi": round(hi, 4),
            "_per_q_means": per_question_means,
        }

    # ========================================================================
    # 2. Per-dimension means per (judge, generator)
    # ========================================================================
    per_dim_means = {}
    for j in JUDGES:
        per_dim_means[j] = {}
        for g in GENERATORS:
            if j == g: continue
            scores_list = list(grouped.get(j, {}).get(g, {}).values())
            if not scores_list: continue
            dim_means = {}
            for k, _, _ in DIMENSIONS:
                dim_means[k] = round(statistics.mean(s[k] for s in scores_list), 4)
            dim_means["reval"] = round(sum(dim_means[k]*w for k,_,w in DIMENSIONS), 4)
            dim_means["n"] = len(scores_list)
            per_dim_means[j][g] = dim_means

    # ========================================================================
    # 3. Bias matrix
    # ========================================================================
    bias_matrix = {}
    for g in GENERATORS:
        all_judge_means = []
        for j in JUDGES:
            if j == g: continue
            if g in per_dim_means.get(j, {}):
                all_judge_means.append(per_dim_means[j][g]["reval"])
        if not all_judge_means: continue
        global_mean = statistics.mean(all_judge_means)
        for j in JUDGES:
            if j == g: continue
            if g not in per_dim_means.get(j, {}): continue
            bias_matrix[f"{j}→{g}"] = round(per_dim_means[j][g]["reval"] - global_mean, 4)

    # ========================================================================
    # 4. Inter-judge agreement (Krippendorff's α)
    # ========================================================================
    alphas = {}
    for k, label, _ in DIMENSIONS:
        items = defaultdict(list)
        for s in scores:
            if not s.get("scores"): continue
            j = s["judge"]; g = s["generator"]
            if j == g: continue
            items[(g, s["qid"])].append(s["scores"][k])
        alphas[k] = round(krippendorff_alpha_interval(list(items.values())), 4)
    items = defaultdict(list)
    for s in scores:
        if not s.get("scores"): continue
        if s["judge"] == s["generator"]: continue
        items[(s["generator"], s["qid"])].append(reval(s["scores"]))
    alphas["REVAL"] = round(krippendorff_alpha_interval(list(items.values())), 4)

    # ========================================================================
    # 5. Paired comparisons (Ours vs each baseline)
    # ========================================================================
    comparisons = {}
    if "Ours" in gen_aggregates:
        ours_per_q = gen_aggregates["Ours"]["_per_q_means"]
        for g in ["GLM", "Gemini", "Grok"]:
            if g not in gen_aggregates: continue
            other_per_q = gen_aggregates[g]["_per_q_means"]
            common_qids = sorted(set(ours_per_q.keys()) & set(other_per_q.keys()))
            ours_vals = [ours_per_q[q] for q in common_qids]
            other_vals = [other_per_q[q] for q in common_qids]
            comparisons[g] = paired_test_full(ours_vals, other_vals)

        # Bonferroni
        p_values = [comparisons[g]["p_value"] for g in comparisons if comparisons[g]]
        bonf = bonferroni(p_values)
        for i, g in enumerate(comparisons):
            if comparisons[g]:
                comparisons[g]["p_bonferroni"] = round(bonf[i], 6)
                comparisons[g]["significant_05"] = bonf[i] < 0.05

    # ========================================================================
    # 6. Disagreement
    # ========================================================================
    sigmas = []
    high_agreement = low_agreement = 0
    for g in GENERATORS:
        for qid in range(1, 351):
            judge_revs = []
            for j in JUDGES:
                if j == g: continue
                if qid in grouped.get(j, {}).get(g, {}):
                    judge_revs.append(reval(grouped[j][g][qid]))
            if len(judge_revs) >= 2:
                sigma = statistics.stdev(judge_revs)
                sigmas.append(sigma)
                if sigma < 0.5: high_agreement += 1
                else: low_agreement += 1

    total = high_agreement + low_agreement
    disagreement = {
        "high_agreement": high_agreement,
        "low_agreement": low_agreement,
        "total": total,
        "high_pct": round(high_agreement/total*100, 2) if total else 0,
        "mean_sigma": round(statistics.mean(sigmas), 4) if sigmas else 0,
        "median_sigma": round(statistics.median(sigmas), 4) if sigmas else 0,
    }

    # Strip _per_q_means before saving
    out_aggregates = {g: {k: v for k, v in d.items() if not k.startswith("_")}
                       for g, d in gen_aggregates.items()}

    out = {
        "n_scores_total": len(scores),
        "excluded_count": excluded,
        "gen_aggregates": out_aggregates,
        "per_dim_means": per_dim_means,
        "bias_matrix": bias_matrix,
        "krippendorff_alpha": alphas,
        "paired_comparisons": comparisons,
        "disagreement": disagreement,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    # Print summary
    print("\n" + "="*100)
    print("REVAL STATS SUMMARY")
    print("="*100)
    print(f"\n📊 Aggregate REVAL (multi-judge mean):")
    for g in GENERATORS:
        if g not in out_aggregates: continue
        a = out_aggregates[g]
        print(f"  {g:<10}: {a['mean_reval']:.3f} [CI {a['ci_95_lo']:.3f}, {a['ci_95_hi']:.3f}] (n={a['n_evaluations']})")

    print(f"\n🎯 Bias Matrix B[j,g]:")
    print(f"  {'Judge\\Gen':<10}", end=""); [print(f"{g:>10}", end="") for g in GENERATORS]; print()
    for j in JUDGES:
        print(f"  {j:<10}", end="")
        for g in GENERATORS:
            if j == g: print(f"{'---':>10}", end="")
            else:
                v = bias_matrix.get(f"{j}→{g}")
                print(f"{v:>10.3f}" if v is not None else f"{'?':>10}", end="")
        print()

    print(f"\n📐 Krippendorff α (inter-judge agreement):")
    for k, label, _ in DIMENSIONS:
        print(f"  {label:<12}: α = {alphas[k]:+.4f}")
    print(f"  {'REVAL':<12}: α = {alphas['REVAL']:+.4f}")

    if comparisons:
        print(f"\n🔬 Paired tests (Ours vs each baseline):")
        for g, t in comparisons.items():
            if not t: continue
            sig = "✓" if t.get("significant_05") else "✗"
            print(f"  Ours vs {g:<6}: Δ={t['delta']:+.3f} [{t['delta_ci_95'][0]:+.3f}, {t['delta_ci_95'][1]:+.3f}] "
                  f"d={t['cohens_d']:+.3f} cliffδ={t['cliffs_delta']:+.3f} "
                  f"p={t['p_value']:.4g} p_bonf={t.get('p_bonferroni',0):.4g} {sig}")

    print(f"\n🤝 Inter-judge disagreement:")
    print(f"  High agreement (σ<0.5): {disagreement['high_agreement']}/{disagreement['total']} = {disagreement['high_pct']}%")
    print(f"  Low agreement  (σ≥0.5): {disagreement['low_agreement']}/{disagreement['total']}")
    print(f"  Mean σ: {disagreement['mean_sigma']:.3f}")

    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
