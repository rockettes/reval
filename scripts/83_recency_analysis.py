"""Reproduces paper Table 18 (Citation Recency Analysis), all seven conditions.

Reads the released per-condition CrossRef log (the same snapshot used for the
VCA tables): results/crossref_doi_verification_all_conditions.json.
N = unique resolved DOIs with a CrossRef year, per condition.

Output: results/recency_analysis.json
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'results' / 'crossref_doi_verification_all_conditions.json'
CONDS = {'Ours': 'Ours', 'GemSearch': 'GemSearch', 'GPT54S': 'GPT54S_forced',
         'Gemini': 'Gemini', 'GPT54': 'GPT54', 'Grok': 'Grok', 'GLM': 'GLM'}

# unique DOIs per system (dedup, case-insensitive)
data = json.load(LOG.open())
sys_dois = defaultdict(dict)  # system -> {doi: year}
for s, key in CONDS.items():
    for q in data[key]['results']:
        for d in q['dois']:
            if not d.get('cr_valid'):
                continue
            try:
                sys_dois[s][d['doi'].lower()] = int(d['cr_year'])
            except (TypeError, ValueError, KeyError):
                continue

out = {}
for s in CONDS:
    years = list(sys_dois.get(s, {}).values())
    if not years:
        out[s] = {'n_unique_dois':0,'mean_year':None,'pct_2020plus':None,'pct_2023plus':None}
        continue
    mean_year = sum(years)/len(years)
    pct_2020 = sum(1 for y in years if y >= 2020)/len(years)*100
    pct_2023 = sum(1 for y in years if y >= 2023)/len(years)*100
    out[s] = {'n_unique_dois':len(years),'mean_year':round(mean_year,1),
              'pct_2020plus':round(pct_2020,1),'pct_2023plus':round(pct_2023,1)}

(ROOT/'results/recency_analysis.json').write_text(json.dumps(out, indent=2))
print('Saved recency_analysis.json')
print(f'{"System":<10} {"N_DOIs":>8} {"MeanYr":>8} {"≥2020":>7} {"≥2023":>7}')
for s in CONDS:
    d = out[s]
    print(f'  {s:<10} {d["n_unique_dois"]:>8} {d["mean_year"] or "--":>8} {(d["pct_2020plus"] or 0):>6.1f}% {(d["pct_2023plus"] or 0):>6.1f}%')
