"""F4c-2: Statistical analysis on CrossRef DOI verification metrics.

Reads from results/stats_crossref.json input data.

Computes per generator:
- VCA existence (% DOIs validated by CrossRef)
- HR existence (% DOIs not in CrossRef)
- VCA semantic (% DOIs with author+year matching)
- HR real (% real hallucinations - C category)
- DOI density per response
- % responses with ≥1 verified DOI

Statistical tests:
- Bootstrap 95% CI per metric per generator
- Paired Wilcoxon: Ours vs each baseline
- Bonferroni correction
- Cohen's d + Cliff's delta

Output: results/stats_crossref_summary.json
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _stats_utils import bootstrap_ci, paired_test_full, bonferroni, summarize

DRY = Path(__file__).resolve().parents[1]

INPUTS = {
    "v2_with_doi": DRY / "results" / "stats_crossref.json",  # pre-computed CrossRef data
}
OUT = DRY / "results" / "stats_crossref_summary.json"  # output (kept separate from the input file)

GENERATORS = ["Ours", "GLM", "Gemini", "Grok"]


def per_question_metrics(crossref_data):
    """For each generator, compute per-question metrics arrays for stats."""
    out = {}
    for label, gen_data in crossref_data.items():
        results = gen_data.get("results", [])
        # Per-question: number of DOIs, number valid (existence + semantic), HR
        n_dois_per_q = []
        n_valid_existence_per_q = []
        n_valid_semantic_per_q = []  # OK or OK_year_drift
        n_real_halluc_per_q = []     # category C
        has_verified_doi = []         # binary 0/1
        for r in results:
            dois = r.get("dois", [])
            n_dois_per_q.append(len(dois))
            existence_valid = sum(1 for d in dois if d.get("category") != "existence_invalid")
            semantic_valid = sum(1 for d in dois if d.get("category") in ("OK", "OK_year_drift"))
            real_halluc = sum(1 for d in dois if d.get("category") == "C_real_hallucination")
            n_valid_existence_per_q.append(existence_valid)
            n_valid_semantic_per_q.append(semantic_valid)
            n_real_halluc_per_q.append(real_halluc)
            has_verified_doi.append(1 if semantic_valid > 0 else 0)
        out[label] = {
            "n_dois": n_dois_per_q,
            "n_valid_existence": n_valid_existence_per_q,
            "n_valid_semantic": n_valid_semantic_per_q,
            "n_real_halluc": n_real_halluc_per_q,
            "has_verified_doi": has_verified_doi,
        }
    return out


def aggregate_stats(per_q):
    """Aggregate per-question values to model-level stats."""
    out = {}
    for gen, m in per_q.items():
        # DOI density (per question)
        dd_mean, dd_lo, dd_hi = bootstrap_ci(m["n_dois"], n_iter=10000)

        # VCA existence rate (over total DOIs, not per question)
        total_dois = sum(m["n_dois"])
        total_valid_existence = sum(m["n_valid_existence"])
        total_valid_semantic = sum(m["n_valid_semantic"])
        total_halluc = sum(m["n_real_halluc"])

        vca_existence = (total_valid_existence / total_dois * 100) if total_dois else 0
        hr_existence = ((total_dois - total_valid_existence) / total_dois * 100) if total_dois else 0
        vca_semantic = (total_valid_semantic / total_dois * 100) if total_dois else 0
        hr_real = (total_halluc / total_dois * 100) if total_dois else 0

        # % responses with at least one verified DOI
        coverage = sum(m["has_verified_doi"]) / len(m["has_verified_doi"]) * 100 if m["has_verified_doi"] else 0

        # Bootstrap CI on coverage (binomial)
        cov_mean, cov_lo, cov_hi = bootstrap_ci(m["has_verified_doi"], n_iter=10000)

        out[gen] = {
            "n_questions": len(m["n_dois"]),
            "total_dois": total_dois,
            "doi_density": {
                "mean": round(dd_mean, 4),
                "median": statistics.median(m["n_dois"]) if m["n_dois"] else 0,
                "std": round(statistics.stdev(m["n_dois"]), 4) if len(m["n_dois"]) > 1 else 0,
                "ci_95_lo": round(dd_lo, 4),
                "ci_95_hi": round(dd_hi, 4),
            },
            "vca_existence_pct": round(vca_existence, 4),
            "hr_existence_pct": round(hr_existence, 4),
            "vca_semantic_pct": round(vca_semantic, 4),
            "hr_real_pct": round(hr_real, 4),
            "coverage_verified_pct": round(coverage, 4),
            "coverage_ci_95": [round(cov_lo*100, 4), round(cov_hi*100, 4)],
        }
    return out


def paired_comparisons(per_q):
    """Ours vs each baseline on coverage_verified (binary per question)."""
    if "Ours" not in per_q: return {}
    ours_cov = per_q["Ours"]["has_verified_doi"]
    ours_dois = per_q["Ours"]["n_dois"]
    out = {}
    for g in ["GLM", "Gemini", "Grok"]:
        if g not in per_q: continue
        baseline_cov = per_q[g]["has_verified_doi"]
        baseline_dois = per_q[g]["n_dois"]
        # Same length expected (350 each)
        n = min(len(ours_cov), len(baseline_cov))
        cov_test = paired_test_full(ours_cov[:n], baseline_cov[:n])
        dois_test = paired_test_full(ours_dois[:n], baseline_dois[:n])
        out[g] = {
            "coverage_verified": cov_test,
            "doi_density": dois_test,
        }
    # Bonferroni for the 3 coverage tests
    p_coverages = [out[g]["coverage_verified"]["p_value"] for g in out if out[g].get("coverage_verified")]
    if p_coverages:
        bonf = bonferroni(p_coverages)
        for i, g in enumerate([k for k in out if out[k].get("coverage_verified")]):
            out[g]["coverage_verified"]["p_bonferroni"] = round(bonf[i], 6)
            out[g]["coverage_verified"]["significant_05"] = bonf[i] < 0.05
    return out


def main():
    all_results = {}
    for variant_label, path in INPUTS.items():
        if not path.exists():
            print(f"SKIP {variant_label}: {path.name} not found")
            continue
        print(f"\n=== Processing {variant_label} ===")
        crossref_data = json.load(open(path))
        per_q = per_question_metrics(crossref_data)
        agg = aggregate_stats(per_q)
        comparisons = paired_comparisons(per_q)
        all_results[variant_label] = {
            "aggregate": agg,
            "paired_comparisons": comparisons,
        }

        print(f"\n{'Model':<10} {'DOIs':>6} {'/Q':>6} {'VCA_ex%':>9} {'HR_ex%':>8} {'HR_real%':>10} {'Coverage%':>11}")
        for g, s in agg.items():
            print(f"  {g:<10} {s['total_dois']:>6} {s['doi_density']['mean']:>6.2f} "
                  f"{s['vca_existence_pct']:>8.2f}% {s['hr_existence_pct']:>7.2f}% "
                  f"{s['hr_real_pct']:>9.2f}% {s['coverage_verified_pct']:>10.2f}%")

        if comparisons:
            print(f"\n  Paired tests (Ours vs each):")
            for g, t in comparisons.items():
                cv = t.get("coverage_verified")
                if cv:
                    sig = "✓" if cv.get("significant_05") else "✗"
                    print(f"    Ours vs {g:<6}: Δcov={cv['delta']:+.4f}, d={cv['cohens_d']:+.3f}, "
                          f"Cliff δ={cv['cliffs_delta']:+.3f}, p={cv['p_value']:.4g}, "
                          f"p_bonf={cv.get('p_bonferroni',0):.4g} {sig}")

    OUT.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
