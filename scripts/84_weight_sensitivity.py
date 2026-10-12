"""Reproduces paper Section 5.6 weight sensitivity.

Claim: in 10,000 random weight configurations sampled uniformly on the simplex,
SYSTEM-R ranks last in 100% of configurations. Also strict Pareto dominance:
SYSTEM-R has lower mean than every other system on every dimension.

Output: results/weight_sensitivity.json
"""
import json, random, math, statistics
from collections import defaultdict
from pathlib import Path

random.seed(42)
ROOT = Path(__file__).resolve().parents[1]

DIMS = [('correcao_cientifica','SciCorrect'),('verificab_citacoes','CiteVerif'),
        ('especificidade','Specificity'),('completude','Completeness'),
        ('assertividade','Assertiveness'),('aplicacao_pratica','PracticalApplic')]
GENS = ['Ours','GPT54','GLM','Gemini','Grok','GemSearch']

SCORES = json.load((ROOT/'scores/reval_scores.json').open())
per_dim = defaultdict(lambda: defaultdict(list))
for r in SCORES:
    if r['judge']==r['generator']: continue
    for d,_ in DIMS:
        per_dim[r['generator']][d].append(r['scores'][d])
mean_dim = {g: {d: statistics.mean(per_dim[g][d]) for d,_ in DIMS} for g in GENS}

# Strict Pareto dominance check
pareto = {}
for d, name in DIMS:
    o = mean_dim['Ours'][d]
    others = {g: mean_dim[g][d] for g in GENS if g != 'Ours'}
    pareto[name] = {'ours':round(o,4),'min_others':round(min(others.values()),4),
                    'min_other_gen': min(others, key=others.get),
                    'strict_dominance': o < min(others.values())}
all_pareto = all(p['strict_dominance'] for p in pareto.values())

# Dirichlet(α=1) sampling = uniform on simplex
def dirichlet():
    g = [-math.log(random.random()) for _ in range(6)]
    tot = sum(g)
    return [x/tot for x in g]

N = 10000
ours_last = 0
last_dist = {g:0 for g in GENS}
for _ in range(N):
    w = dirichlet()
    scores = {g: sum(mean_dim[g][d]*w[i] for i,(d,_) in enumerate(DIMS)) for g in GENS}
    last = min(scores, key=scores.get)
    last_dist[last] += 1
    if last == 'Ours': ours_last += 1

out = {
    'n_configs': N,
    'seed': 42,
    'pct_systemr_last': round(ours_last/N*100, 4),
    'pareto_dominance_strict': all_pareto,
    'per_dimension_dominance': pareto,
    'distribution_of_last_place': last_dist,
}
(ROOT/'results/weight_sensitivity.json').write_text(json.dumps(out, indent=2))
print(f'Saved weight_sensitivity.json')
print(f'Ours ranks last: {ours_last}/{N} = {ours_last/N*100:.2f}%')
print(f'Strict Pareto dominance: {all_pareto}')
for n,p in pareto.items():
    print(f'  {n}: Ours={p["ours"]}, min_others={p["min_others"]} ({p["min_other_gen"]}) → {p["strict_dominance"]}')
