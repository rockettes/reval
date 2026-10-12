"""Reproduces paper Section 5.6 judge ablation: all C(5,k) subsets for k∈{2,3,4,5}.

Claim: SYSTEM-R ranks last in 100% of judge subsets.

Output: results/judge_ablation.json
"""
import json, statistics, itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGES = ['Atla','GPT54','GLM','Gemini','Grok']
GENS = ['Ours','GPT54','GLM','Gemini','Grok','GemSearch']
DIMS = [('correcao_cientifica',.25),('verificab_citacoes',.25),
        ('especificidade',.20),('completude',.15),
        ('assertividade',.10),('aplicacao_pratica',.05)]
def reval(s): return sum(s[k]*w for k,w in DIMS)

SCORES = json.load((ROOT/'scores/reval_scores.json').open())

# Precompute per (judge, generator) mean REVAL
pair_means = {}
for j in JUDGES:
    for g in GENS:
        if j == g: continue
        bucket = [reval(r['scores']) for r in SCORES if r['judge']==j and r['generator']==g]
        if bucket:
            pair_means[(j,g)] = statistics.mean(bucket)

out = {}
for k in [2,3,4,5]:
    subsets = list(itertools.combinations(JUDGES, k))
    ours_last = 0
    rank_results = []
    for sub in subsets:
        gen_reval = {}
        for g in GENS:
            eligible = [j for j in sub if j != g]
            if not eligible: continue
            vals = [pair_means.get((j,g)) for j in eligible if (j,g) in pair_means]
            if vals: gen_reval[g] = statistics.mean(vals)
        if gen_reval:
            order = sorted(gen_reval, key=gen_reval.get)
            last = order[0]
            rank_results.append({'judges': list(sub), 'ranking_low_to_high': order})
            if last == 'Ours': ours_last += 1
    out[f'k_{k}'] = {
        'n_subsets': len(subsets),
        'ours_last_count': ours_last,
        'pct_ours_last': round(ours_last/len(subsets)*100, 2) if subsets else 0,
        'all_subsets': rank_results if k>=4 else None,  # save details only for larger k
    }

(ROOT/'results/judge_ablation.json').write_text(json.dumps(out, indent=2))
print('Saved judge_ablation.json')
for k in [2,3,4,5]:
    d = out[f'k_{k}']
    print(f'  k={k}: {d["ours_last_count"]}/{d["n_subsets"]} = {d["pct_ours_last"]}% subsets have Ours last')
