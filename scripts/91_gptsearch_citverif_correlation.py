"""Reproduces the GPT+S claim in Section 5.7 / Appendix B: the judges' Citation
Verifiability score is uncorrelated with actual per-response verification
(Pearson r = -0.004 forced, r = -0.08 auto; n = 349 / 348 responses with DOIs).

Per response: x = mean Citation Verifiability across the four eligible judges,
y = fraction of the response's DOIs passing VCA+-1yr (definitions of 68_vca_final.py,
as implemented in 90_gptsearch_vca_by_type.py).

Inputs: scores/gpt54search_scores.json + the inputs of script 90.
Output: results/gptsearch_citverif_correlation.json
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_src = (ROOT / 'scripts/90_gptsearch_vca_by_type.py').read_text().split('out = {}')[0]
m90 = {'__file__': str(ROOT / 'scripts/90_gptsearch_vca_by_type.py')}
exec(_src, m90)  # reuse CR, CACHE, CONDS, year_near_doi without re-running script 90


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


scores = json.load((ROOT / 'scores/gpt54search_scores.json').open())
out = {}
for gen, path in m90['CONDS'].items():
    resp = {r['id']: r for r in json.load((ROOT / path).open())}
    n_doi, n_ok = defaultdict(int), defaultdict(int)
    for q in m90['CR'][gen]['results']:
        for d in q['dois']:
            n_doi[q['qid']] += 1
            v = m90['CACHE'].get(f"{gen}_{q['qid']}_{d['doi']}") or m90['CACHE'].get(f"{gen}_{q['qid']}_{d['doi'].lower()}")
            if not (d.get('cr_valid') and v == 'VALID'):
                continue
            cy = d.get('cited_year') or m90['year_near_doi'](resp[q['qid']].get('answer', ''), d['doi'])
            try:
                yd = abs(int(cy) - int(d.get('cr_year')))
            except (TypeError, ValueError):
                yd = None
            if yd is None or yd <= 1:
                n_ok[q['qid']] += 1
    cv = defaultdict(list)
    for r in scores:
        if r.get('generator') == gen and 'scores' in r:
            cv[r['qid']].append(r['scores']['verificab_citacoes'])
    qs = [q for q in n_doi if n_doi[q] and q in cv]
    x = [st.mean(cv[q]) for q in qs]
    y = [n_ok[q] / n_doi[q] for q in qs]
    out[gen] = {'n_responses': len(qs), 'pearson_r': round(pearson(x, y), 4),
                'mean_citverif': round(st.mean(x), 2), 'mean_vca_pm1_per_response': round(st.mean(y), 3)}

(ROOT / 'results/gptsearch_citverif_correlation.json').write_text(json.dumps(out, indent=2))
print('Saved gptsearch_citverif_correlation.json')
for g, v in out.items():
    print(f"  {g:<14} n={v['n_responses']} r={v['pearson_r']} mean CitVerif={v['mean_citverif']}")
