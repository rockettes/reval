"""Reproduces paper Section 5.9 claim: |ρ| < 0.13 for every system in Spearman
correlation between RAGAS answer_relevancy and REVAL, per question.

Output: results/ragas_spearman.json
"""
import json, statistics
from collections import defaultdict
from pathlib import Path
import scipy.stats as ss

ROOT = Path(__file__).resolve().parents[1]
DIMS = [('correcao_cientifica',.25),('verificab_citacoes',.25),
        ('especificidade',.20),('completude',.15),
        ('assertividade',.10),('aplicacao_pratica',.05)]
def reval(s): return sum(s[k]*w for k,w in DIMS)

SCORES = json.load((ROOT/'scores/reval_scores.json').open())
rag = json.load((ROOT/'results/ragas_answer_relevancy.json').open())

gen_q_reval = defaultdict(dict)
tmp = defaultdict(lambda: defaultdict(list))
for r in SCORES:
    if r['judge']==r['generator']: continue
    tmp[r['generator']][r['qid']].append(reval(r['scores']))
for g, qs in tmp.items():
    for qid, vs in qs.items():
        gen_q_reval[g][qid] = statistics.mean(vs)

spearman = {}
for s, d in rag['results'].items():
    per_item = d.get('per_item', [])
    if not per_item: continue
    if isinstance(per_item[0], dict) and 'qid' in per_item[0]:
        rg = {p['qid']: p['score'] for p in per_item}
    elif isinstance(per_item[0], dict) and 'score' in per_item[0]:
        rg = {i+1: per_item[i]['score'] for i in range(len(per_item))}
    else:
        rg = {i+1: per_item[i] for i in range(len(per_item))}
    qids = sorted(set(rg) & set(gen_q_reval.get(s,{})))
    if not qids: continue
    x = [rg[q] for q in qids]
    y = [gen_q_reval[s][q] for q in qids]
    rho, p = ss.spearmanr(x, y)
    spearman[s] = {'rho': round(float(rho),4), 'p_value': float(p), 'n': len(qids),
                   'within_paper_bound_013': bool(abs(rho) < 0.13)}

out = {'method':'per-question Spearman correlation RAGAS vs REVAL','paper_claim':'|ρ|<0.13 for all',
       'systems': spearman, 'all_within_bound': bool(all(d['within_paper_bound_013'] for d in spearman.values()))}
(ROOT/'results/ragas_spearman.json').write_text(json.dumps(out, indent=2))
print('Saved ragas_spearman.json')
for s, d in spearman.items():
    flag = '✓' if d['within_paper_bound_013'] else '✗'
    print(f'  {s:<10} ρ={d["rho"]:+.4f}  n={d["n"]}  {flag}')
