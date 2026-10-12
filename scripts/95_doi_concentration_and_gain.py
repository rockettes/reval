"""(a) DOI concentration per system: DOI mentions, CrossRef-resolved mentions, unique
resolved DOIs, and mentions per unique DOI (Limitation 8).
(b) Decomposition of the REVAL gain Gemini -> Gemini+S into weighted per-dimension
contributions (same judge panel: Atla, GLM, GPT, Grok), Section 5.8.

Inputs: results/crossref_doi_verification_all_conditions.json, scores/reval_scores.json
Output: results/doi_concentration_and_gain.json
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CR = json.load((ROOT / 'results/crossref_doi_verification_all_conditions.json').open())
NAMES = {'Ours': 'SYSTEM-R', 'GemSearch': 'Gemini+S', 'Gemini': 'Gemini', 'Grok': 'Grok',
         'GPT54S_forced': 'GPT+S', 'GPT54': 'GPT', 'GLM': 'GLM'}
conc = {}
for g, n in NAMES.items():
    mentions = [d for q in CR[g]['results'] for d in q['dois']]
    resolved = [d['doi'].lower() for d in mentions if d.get('cr_valid')]
    u = len(set(resolved))
    conc[n] = {'doi_mentions': len(mentions), 'resolved_mentions': len(resolved),
               'unique_resolved': u, 'mentions_per_unique': round(len(resolved) / u, 2)}

W = {'correcao_cientifica': .25, 'verificab_citacoes': .25, 'especificidade': .20,
     'completude': .15, 'assertividade': .10, 'aplicacao_pratica': .05}
rows = [r for r in json.load((ROOT / 'scores/reval_scores.json').open()) if 'scores' in r]


def means(g):
    acc = defaultdict(list)
    for r in rows:
        if r['generator'] == g:
            for k in W:
                acc[k].append(r['scores'][k])
            acc['reval'].append(r['reval'])
    return {k: st.mean(v) for k, v in acc.items()}


a, b = means('Gemini'), means('GemSearch')
total = b['reval'] - a['reval']
gain = {'delta_reval': round(total, 3),
        'by_dimension': {k: {'delta': round(b[k] - a[k], 3), 'weight': w,
                             'contribution': round(w * (b[k] - a[k]), 3),
                             'share_pct': round(100 * w * (b[k] - a[k]) / total, 1)} for k, w in W.items()}}
(ROOT / 'results/doi_concentration_and_gain.json').write_text(
    json.dumps({'doi_concentration': conc, 'gemini_to_gemsearch_gain': gain}, indent=2))
print('Saved doi_concentration_and_gain.json')
for n, c in conc.items():
    print(f"  {n:<9} mentions={c['doi_mentions']} resolved={c['resolved_mentions']} unique={c['unique_resolved']} per-unique={c['mentions_per_unique']}")
print(f"  Gemini->Gemini+S delta REVAL={gain['delta_reval']}")
for k, v in gain['by_dimension'].items():
    print(f"    {k:<20} contribution={v['contribution']:+.3f} ({v['share_pct']}%)")
