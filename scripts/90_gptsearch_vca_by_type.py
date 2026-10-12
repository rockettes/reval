"""Reproduces the GPT+S column of paper Table 17 (VCA by question type) and the
GPT+S rows of Tables 8 and 11 (VCA strict / +-1yr), forced and auto search.

Same definitions as 68_vca_final.py, reading only released files:
  results/crossref_doi_verification_all_conditions.json  (CrossRef existence, years)
  results/vca_semantic_judge_cache.json                  (Llama 3.3 70B verdicts)
  responses/gpt54search.json, responses/gpt54search_auto.json
Output: results/gptsearch_vca_by_type.json
"""
import json, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CR = json.load((ROOT / 'results/crossref_doi_verification_all_conditions.json').open())
CACHE = json.load((ROOT / 'results/vca_semantic_judge_cache.json').open())
CONDS = {'GPT54S_forced': 'responses/gpt54search.json',
         'GPT54S_auto': 'responses/gpt54search_auto.json'}
TYPES = ['DIVERGENT', 'POPULATION', 'MULTI_HOP', 'COMP', 'TEMPORAL', 'MECHANISM', 'NUM', 'ADV_OOS']


def year_near_doi(text, doi):
    pos = text.lower().find(doi.lower())
    if pos < 0:
        return None
    years = re.findall(r'\b(19[5-9]\d|20[0-3]\d)\b', text[max(0, pos - 200):pos])
    return years[-1] if years else None


def outcomes(gen, resp_path):
    """Per-DOI (qtype, strict, lenient) following 68_vca_final.py."""
    resp = {r['id']: r for r in json.load((ROOT / resp_path).open())}
    res = []
    for q in CR[gen]['results']:
        qt = resp[q['qid']]['question_type'].upper()
        text = resp[q['qid']].get('answer', '')
        for d in q['dois']:
            if not d.get('cr_valid'):
                res.append((qt, 0, 0)); continue
            v = CACHE.get(f"{gen}_{q['qid']}_{d['doi']}") or CACHE.get(f"{gen}_{q['qid']}_{d['doi'].lower()}")
            if v != 'VALID':
                res.append((qt, 0, 0)); continue
            cy = d.get('cited_year') or year_near_doi(text, d['doi'])
            try:
                yd = abs(int(cy) - int(d.get('cr_year')))
            except (TypeError, ValueError):
                yd = None
            if yd == 0:
                res.append((qt, 1, 1))
            elif yd == 1 or yd is None:
                res.append((qt, 0, 1))
            else:
                res.append((qt, 0, 0))
    return res


def ci(xs, seed=42, b=10000):
    random.seed(seed)
    n = len(xs)
    bs = sorted(sum(random.choices(xs, k=n)) / n * 100 for _ in range(b))
    return [round(bs[int(0.025 * b)], 1), round(bs[int(0.975 * b) - 1], 1)]


out = {}
for gen, path in CONDS.items():
    res = outcomes(gen, path)
    strict = [s for _, s, _ in res]
    len_ = [l for _, _, l in res]
    by_t = defaultdict(list)
    for t, _, l in res:
        by_t[t].append(l)
    out[gen] = {'n_dois': len(res),
                'vca_strict_pct': round(100 * sum(strict) / len(res), 1), 'vca_strict_ci95': ci(strict),
                'vca_pm1_pct': round(100 * sum(len_) / len(res), 1), 'vca_pm1_ci95': ci(len_),
                'vca_pm1_by_type': {t: round(100 * sum(by_t[t]) / len(by_t[t]), 1) for t in TYPES}}
(ROOT / 'results/gptsearch_vca_by_type.json').write_text(json.dumps(out, indent=2))
print('Saved gptsearch_vca_by_type.json')
for g, v in out.items():
    print(f"  {g:<14} n={v['n_dois']} strict={v['vca_strict_pct']} {v['vca_strict_ci95']} "
          f"pm1={v['vca_pm1_pct']} {v['vca_pm1_ci95']}")
    print('   by type (pm1):', ' '.join(f"{t[:6]}={v['vca_pm1_by_type'][t]}" for t in TYPES))
