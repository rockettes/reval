"""Per-response correlation between the judges' Citation Verifiability score and actual
citation verification, for all seven conditions (extends script 91, which covers GPT+S).

Per response: x = mean Citation Verifiability across the eligible judges,
y = fraction of the response's DOIs passing VCA+-1yr (definitions of 68_vca_final.py).
Responses without DOIs are excluded. Pearson r and Spearman rho, per system and pooled.

Inputs: results/crossref_doi_verification_all_conditions.json,
        results/vca_semantic_judge_cache.json, responses/*.json,
        scores/reval_scores.json, scores/gpt54search_scores.json
Output: results/citverif_vs_vca_per_response.json
"""
import json, re, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CR = json.load((ROOT / 'results/crossref_doi_verification_all_conditions.json').open())
CACHE = json.load((ROOT / 'results/vca_semantic_judge_cache.json').open())
FILES = {'Ours': 'ours', 'GPT54': 'gpt54', 'GLM': 'glm', 'Gemini': 'gemini', 'Grok': 'grok',
         'GemSearch': 'gemsearch', 'GPT54S_forced': 'gpt54search'}
NAMES = {'Ours': 'SYSTEM-R', 'GPT54': 'GPT', 'GLM': 'GLM', 'Gemini': 'Gemini', 'Grok': 'Grok',
         'GemSearch': 'Gemini+S', 'GPT54S_forced': 'GPT+S'}
SCORES = [r for r in json.load((ROOT / 'scores/reval_scores.json').open()) if 'scores' in r]
SCORES += [r for r in json.load((ROOT / 'scores/gpt54search_scores.json').open()) if 'scores' in r]


def year_near_doi(text, doi):
    pos = text.lower().find(doi.lower())
    if pos < 0:
        return None
    years = re.findall(r'\b(19[5-9]\d|20[0-3]\d)\b', text[max(0, pos - 200):pos])
    return years[-1] if years else None


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


out, all_x, all_y = {}, [], []
for g, f in FILES.items():
    resp = {r['id']: r for r in json.load((ROOT / f'responses/{f}.json').open())}
    n_doi, n_ok = defaultdict(int), defaultdict(int)
    for q in CR[g]['results']:
        for d in q['dois']:
            n_doi[q['qid']] += 1
            v = CACHE.get(f"{g}_{q['qid']}_{d['doi']}") or CACHE.get(f"{g}_{q['qid']}_{d['doi'].lower()}")
            if not (d.get('cr_valid') and v == 'VALID'):
                continue
            cy = d.get('cited_year') or year_near_doi(resp[q['qid']].get('answer', ''), d['doi'])
            try:
                yd = abs(int(cy) - int(d.get('cr_year')))
            except (TypeError, ValueError):
                yd = None
            if yd is None or yd <= 1:
                n_ok[q['qid']] += 1
    cv = defaultdict(list)
    for r in SCORES:
        if r['generator'] == g:
            cv[r['qid']].append(r['scores']['verificab_citacoes'])
    qs = [q for q in n_doi if n_doi[q] and q in cv]
    x = [st.mean(cv[q]) for q in qs]
    y = [n_ok[q] / n_doi[q] for q in qs]
    all_x += x
    all_y += y
    out[NAMES[g]] = {'n_responses': len(qs), 'pearson_r': round(pearson(x, y), 3),
                     'spearman_rho': round(pearson(ranks(x), ranks(y)), 3),
                     'mean_citverif': round(st.mean(x), 2), 'mean_vca_pm1_per_response': round(st.mean(y), 2)}
out['pooled'] = {'n_responses': len(all_x), 'pearson_r': round(pearson(all_x, all_y), 3),
                 'spearman_rho': round(pearson(ranks(all_x), ranks(all_y)), 3)}
(ROOT / 'results/citverif_vs_vca_per_response.json').write_text(json.dumps(out, indent=2))
print('Saved citverif_vs_vca_per_response.json')
for k, v in out.items():
    print(f"  {k:<9} n={v['n_responses']:>4} r={v['pearson_r']:+.3f} rho={v['spearman_rho']:+.3f}"
          + (f"  mean CitVerif={v['mean_citverif']} mean VCA/resp={v['mean_vca_pm1_per_response']}" if 'mean_citverif' in v else ''))
