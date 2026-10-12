"""Compute 95% bootstrap CIs (10k resamples, percentile method) for:
  - REVAL means (Table 4 in user nomenclature; tab:eval_graph in tex)
  - Citation metrics: Exist, Fabr, VCA, VCA±1yr (Table 5; tab:citations)
  - Ablation pair: Gemini base vs Gem+S (Table 7; tab:ablation_vca)

Outputs: results/bootstrap_cis.json
"""
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

random.seed(42)

DRY = Path(__file__).resolve().parents[1]
SCORES_FILE = DRY / "scores" / "reval_scores.json"
VCA_FILE = DRY / "results" / "vca_final_5way.json"
OUT = DRY / "results" / "bootstrap_cis.json"

N_ITER = 10000

DIMENSIONS = [
    ("correcao_cientifica", 0.25),
    ("verificab_citacoes",  0.25),
    ("especificidade",      0.20),
    ("completude",          0.15),
    ("assertividade",       0.10),
    ("aplicacao_pratica",   0.05),
]


def reval(scores):
    return sum(scores[k]*w for k, w in DIMENSIONS)


def bootstrap_pct(values, n_iter=N_ITER):
    """Returns (mean, p2.5, p97.5) via percentile bootstrap."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_iter):
        s = 0.0
        for _ in range(n):
            s += values[random.randint(0, n-1)]
        means.append(s / n)
    means.sort()
    lo = means[int(n_iter * 0.025)]
    hi = means[int(n_iter * 0.975)]
    return (statistics.mean(values), lo, hi)


def reval_cis():
    """Per-generator REVAL: bootstrap over 350 per-question means."""
    rows = json.load(open(SCORES_FILE))
    # build per (generator, qid) -> list of reval per judge
    bucket = defaultdict(list)
    for r in rows:
        if r["judge"] == r["generator"]:
            continue
        bucket[(r["generator"], r["qid"])].append(reval(r["scores"]))
    # collapse to per-question mean across judges
    per_q = defaultdict(list)
    for (g, qid), vs in bucket.items():
        per_q[g].append(statistics.mean(vs))
    cis = {}
    for g, vals in per_q.items():
        m, lo, hi = bootstrap_pct(vals)
        cis[g] = {"n_questions": len(vals), "mean": round(m, 3),
                  "ci_95_lo": round(lo, 3), "ci_95_hi": round(hi, 3)}
    return cis


def vca_cis():
    """Per-system citation metrics, bootstrap over the N DOIs.
    For each metric (exist, fabr, vca, vca_lenient): construct a binary array
    of length total_dois and resample.
    """
    data = json.load(open(VCA_FILE))["summary"]
    cis = {}
    for sys_name, d in data.items():
        N = d["total_dois"]
        # Build binary arrays from aggregate counts
        arrays = {
            "exist":   [1]*d["exist"]      + [0]*(N - d["exist"]),
            "fabr":    [1]*d["fabricated"] + [0]*(N - d["fabricated"]),
            "vca":     [1]*d["vca_strict"] + [0]*(N - d["vca_strict"]),
            "vca_pm1": [1]*d["vca_lenient"]+ [0]*(N - d["vca_lenient"]),
        }
        cis[sys_name] = {"n_dois": N}
        for k, arr in arrays.items():
            m, lo, hi = bootstrap_pct(arr)
            cis[sys_name][k] = {
                "pct": round(m*100, 2),
                "ci_95_lo": round(lo*100, 2),
                "ci_95_hi": round(hi*100, 2),
            }
    return cis


def main():
    print("Computing REVAL CIs (per generator, paired bootstrap on 350 questions, 10k iters)...")
    reval_out = reval_cis()
    for g, ci in reval_out.items():
        print(f"  {g:<10} {ci['mean']:.3f} [{ci['ci_95_lo']:.3f}, {ci['ci_95_hi']:.3f}] "
              f"(N_q={ci['n_questions']})")

    print("\nComputing VCA CIs (per system, bootstrap on N DOIs, 10k iters)...")
    vca_out = vca_cis()
    for s, ci in vca_out.items():
        print(f"  {s:<10} N={ci['n_dois']}")
        for k in ["exist", "fabr", "vca", "vca_pm1"]:
            d = ci[k]
            print(f"    {k:<8} {d['pct']:>5.2f} [{d['ci_95_lo']:.2f}, {d['ci_95_hi']:.2f}]")

    out = {
        "method": "Percentile bootstrap, 10,000 resamples",
        "seed": 42,
        "reval": reval_out,
        "citation": vca_out,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
