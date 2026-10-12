"""Reproduces paper Table 11 (15 pairwise comparisons among 6 systems).

Outputs results/pairwise_significance.json:
  rows = [{pair, n, delta, ci_lower, ci_upper, p, p_bonf, cohens_d, significant_05}, ...]

Methodology:
  - Per-question mean REVAL = mean across eligible judges (self excluded).
  - Paired bootstrap with B = 10,000 over the 350 questions.
  - 95% CI = percentile [2.5, 97.5] of bootstrap distribution.
  - p-value = 2 × P(bootstrap mean has opposite sign of observed Δ).
  - Bonferroni correction: k = 15.
  - Cohen's d (paired) = mean(diff) / std(diff).
  - Significant @ 0.05 after Bonferroni: p_bonf < 0.05.

Deterministic via random.seed(42).
"""
import json, random, statistics
from collections import defaultdict
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]

DIMS = [('correcao_cientifica',.25),('verificab_citacoes',.25),
        ('especificidade',.20),('completude',.15),
        ('assertividade',.10),('aplicacao_pratica',.05)]
def reval(s): return sum(s[k]*w for k,w in DIMS)

SCORES = json.load((ROOT/'scores/reval_scores.json').open())
gen_q_means = defaultdict(dict)
for r in SCORES:
    if r['judge']==r['generator']: continue
    g, qid = r['generator'], r['qid']
    gen_q_means[g].setdefault(qid, []).append(reval(r['scores']))
for g in gen_q_means:
    for qid in gen_q_means[g]:
        gen_q_means[g][qid] = statistics.mean(gen_q_means[g][qid])

def paired_test(a, b, n_iter=10000):
    diffs = [ai-bi for ai,bi in zip(a,b)]
    delta = statistics.mean(diffs)
    n = len(diffs)
    boot = []
    for _ in range(n_iter):
        s = 0
        for _ in range(n): s += diffs[random.randint(0,n-1)]
        boot.append(s/n)
    boot.sort()
    lo, hi = boot[int(n_iter*0.025)], boot[int(n_iter*0.975)]
    if delta > 0:
        p = 2*sum(1 for x in boot if x <= 0)/n_iter
    else:
        p = 2*sum(1 for x in boot if x >= 0)/n_iter
    p = max(p, 1/n_iter)
    sd = statistics.stdev(diffs) if len(diffs)>1 else 0
    d = delta/sd if sd > 0 else 0
    return {'n':n,'delta':round(delta,4),'ci_lower':round(lo,4),'ci_upper':round(hi,4),
            'p':round(p,6),'cohens_d':round(d,4)}

GENS = ['Ours','GPT54','GLM','Gemini','Grok','GemSearch']
results = []
for i in range(len(GENS)):
    for j in range(i+1, len(GENS)):
        a,b = GENS[i], GENS[j]
        common = sorted(set(gen_q_means[a].keys()) & set(gen_q_means[b].keys()))
        va = [gen_q_means[a][q] for q in common]
        vb = [gen_q_means[b][q] for q in common]
        t = paired_test(va, vb)
        t['pair'] = f'{a}_vs_{b}'
        results.append(t)

k = len(results)
BASE = {'Ours','GPT54','GLM','Gemini','Grok'}
k_base = sum(1 for r in results if set(r['pair'].split('_vs_')) <= BASE)  # 10 pairs (paper Table 10)
for r in results:
    r['p_bonf'] = round(min(r['p']*k, 1.0), 6)
    r['significant_05'] = r['p_bonf'] < 0.05
    if set(r['pair'].split('_vs_')) <= BASE:
        r['p_bonf_base10'] = round(min(r['p']*k_base, 1.0), 6)  # Bonferroni over the five base systems

out = {'method':'paired bootstrap B=10,000, Bonferroni k=15 (all six) and k=10 (five base systems, paper Table 10)','seed':42,'k_pairs':k,'pairs':results}
(ROOT/'results/pairwise_significance.json').write_text(json.dumps(out, indent=2))
print('Saved pairwise_significance.json with', len(results), 'pairs')
for r in results:
    print(f'  {r["pair"]:<25} Δ={r["delta"]:>+.3f} CI[{r["ci_lower"]:>+.3f},{r["ci_upper"]:>+.3f}] '
          f'd={r["cohens_d"]:>+.3f} p_raw={r["p"]:.4g} p_bonf(k=15)={r["p_bonf"]:.4g}'
          + (f' p_bonf(k=10, base)={r["p_bonf_base10"]:.4g}' if 'p_bonf_base10' in r else '')
          + f' {"sig" if r["significant_05"] else "ns"}')
