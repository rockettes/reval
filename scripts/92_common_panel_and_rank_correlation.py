"""Common-judge-panel pairwise comparisons and system-level rank correlation.

Each generator is scored by a different judge set (self-evaluation excluded), and the
GPT judge is the most severe, so GPT and GPT+S are never scored by the harshest judge.
For every pair of the seven conditions we therefore recompute the paired comparison
using only the judges common to both generators (paired bootstrap over questions,
10,000 resamples, seed 42; two-sided p from the bootstrap distribution).

Also reports the Spearman rank correlation between system-level mean REVAL (full
eligible panel, Table 7) and VCA+-1yr (Table 8), with and without SYSTEM-R.

Inputs: scores/reval_scores.json, scores/gpt54search_scores.json,
        results/vca_final_5way.json, results/gptsearch_vca_by_type.json
Output: results/common_panel_comparisons.json
"""
import json, random, statistics as st, itertools
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'Ours': 'SYSTEM-R', 'GPT54': 'GPT', 'GLM': 'GLM', 'Gemini': 'Gemini',
         'Grok': 'Grok', 'GemSearch': 'Gemini+S', 'GPT54S_forced': 'GPT+S'}

rows = [r for r in json.load((ROOT / 'scores/reval_scores.json').open()) if 'reval' in r]
rows += [r for r in json.load((ROOT / 'scores/gpt54search_scores.json').open())
         if 'reval' in r and r['generator'] == 'GPT54S_forced']
S = defaultdict(lambda: defaultdict(dict))  # generator -> qid -> judge -> REVAL
for r in rows:
    S[r['generator']][r['qid']][r['judge']] = r['reval']
panel = {g: sorted({j for q in S[g].values() for j in q}) for g in S}


def full_mean(g):
    return st.mean(st.mean(v.values()) for v in S[g].values())


random.seed(42)
pairs = []
for a, b in itertools.combinations(NAMES, 2):
    common = sorted(set(panel[a]) & set(panel[b]))
    qs = [q for q in S[a] if q in S[b]
          and all(j in S[a][q] for j in common) and all(j in S[b][q] for j in common)]
    ma = [st.mean(S[a][q][j] for j in common) for q in qs]
    mb = [st.mean(S[b][q][j] for j in common) for q in qs]
    d = [x - y for x, y in zip(ma, mb)]
    bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(10000))
    p = min(1.0, 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / 10000)
    pairs.append({'a': NAMES[a], 'b': NAMES[b], 'common_judges': common, 'n_questions': len(qs),
                  'mean_a': round(st.mean(ma), 3), 'mean_b': round(st.mean(mb), 3),
                  'delta': round(st.mean(d), 3), 'ci95': [round(bs[250], 3), round(bs[9749], 3)],
                  'p': round(p, 4), 'delta_full_panel': round(full_mean(a) - full_mean(b), 3)})

# system-level Spearman (no ties in either ranking)
reval = {NAMES[g]: full_mean(g) for g in NAMES}
v5 = json.load((ROOT / 'results/vca_final_5way.json').open())
vca = {NAMES[g]: v5['summary'][g]['vca_lenient_pct'] for g in NAMES if g in v5['summary']}
vca['GPT+S'] = json.load((ROOT / 'results/gptsearch_vca_by_type.json').open())['GPT54S_forced']['vca_pm1_pct']


def spearman(x, y):
    ks = list(x)
    rx = {k: i for i, k in enumerate(sorted(ks, key=lambda k: -x[k]))}
    ry = {k: i for i, k in enumerate(sorted(ks, key=lambda k: -y[k]))}
    n = len(ks)
    return 1 - 6 * sum((rx[k] - ry[k]) ** 2 for k in ks) / (n * (n * n - 1))


no_r = lambda d: {k: v for k, v in d.items() if k != 'SYSTEM-R'}
out = {'pairs': pairs, 'system_mean_reval': {k: round(v, 3) for k, v in reval.items()}, 'vca_pm1': vca,
       'spearman_reval_vca_all7': round(spearman(reval, vca), 3),
       'spearman_reval_vca_without_system_r': round(spearman(no_r(reval), no_r(vca)), 3)}
(ROOT / 'results/common_panel_comparisons.json').write_text(json.dumps(out, indent=2))
print('Saved common_panel_comparisons.json')
for p in pairs:
    print(f"  {p['a']+' vs '+p['b']:<22} judges={','.join(p['common_judges'])} n={p['n_questions']} "
          f"delta={p['delta']:+.3f} {p['ci95']} p={p['p']} (full panel {p['delta_full_panel']:+.3f})")
print(f"  Spearman REVAL x VCA+-1: all 7 = {out['spearman_reval_vca_all7']}, "
      f"without SYSTEM-R = {out['spearman_reval_vca_without_system_r']}")
