"""Reproduces remaining paper numbers that have no dedicated script:
 (a) REVAL by question type for the five base systems and Delta_max (Table 9);
 (b) GPT -> GPT+S paired comparison on the full eligible panels (Section 5.8);
 (c) per-dimension means of GPT+S (Table 19, last column);
 (d) error-analysis union counts among the 50 worst SYSTEM-R responses (Section 5.11);
 (e) response-level TOST of the format-preserving perturbation (Appendix A).

Inputs: benchmark/, scores/, results/error_analysis.json
Output: results/paper_details.json
"""
import json, math, random, statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.load((ROOT / 'scores/reval_scores.json').open())
G = json.load((ROOT / 'scores/gpt54search_scores.json').open())
TYPES = {q['id']: q['question_type'] for q in map(json.loads, (ROOT / 'benchmark/bodybuilding_qa_350.jsonl').open())}
DIMS = ['correcao_cientifica', 'verificab_citacoes', 'especificidade', 'completude', 'assertividade', 'aplicacao_pratica']


def per_question(rows, g):
    acc = defaultdict(list)
    for x in rows:
        if x['generator'] == g and 'reval' in x:
            acc[x['qid']].append(x['reval'])
    return {q: st.mean(v) for q, v in acc.items()}


def boot_ci(d, seed=42, b=10000, lo=0.025, hi=0.975):
    random.seed(seed)
    bs = sorted(st.mean(random.choices(d, k=len(d))) for _ in range(b))
    return [round(bs[int(lo * b)], 3), round(bs[int(hi * b) - 1], 3)]


out = {}
# (a)
by_type = {}
for g in ['Ours', 'GPT54', 'GLM', 'Gemini', 'Grok']:
    acc = defaultdict(list)
    for q, v in per_question(R, g).items():
        acc[TYPES[q]].append(v)
    by_type[g] = {t: round(st.mean(v), 2) for t, v in acc.items()}
by_type['delta_max'] = {t: round(by_type['Ours'][t] - max(by_type[g][t] for g in ['GPT54', 'GLM', 'Gemini', 'Grok']), 2)
                        for t in by_type['Ours']}
out['reval_by_type'] = by_type
# (b)
a, b = per_question(R, 'GPT54'), per_question(G, 'GPT54S_forced')
d = [b[q] - a[q] for q in a if q in b]
out['gpt_to_gptsearch'] = {'n': len(d), 'delta': round(st.mean(d), 3), 'ci95': boot_ci(d),
                           'cohens_d': round(st.mean(d) / st.stdev(d), 2)}
# (c)
out['gptsearch_per_dimension'] = {k: round(st.mean(x['scores'][k] for x in G
                                                   if x['generator'] == 'GPT54S_forced' and 'scores' in x), 2) for k in DIMS}
# (d)
bot = json.load((ROOT / 'results/error_analysis.json').open())['bottom_50']
pats = [set(x.get('patterns', [])) for x in bot]
out['error_analysis_bottom50'] = {
    'incomplete_synthesis': sum('sintese_incompleta' in p for p in pats),
    'weak_or_missing_citation': sum(bool(p & {'citacao_fraca', 'citacao_ausente'}) for p in pats),
    'refusal': sum('recusa' in p for p in pats)}
# (e)
orig = {(x['judge'], x['qid']): x['reval'] for x in R if x['generator'] == 'GemSearch' and 'reval' in x}
per = defaultdict(list)
for x in json.load((ROOT / 'scores/perturbation_format_preserving_scores.json').open()):
    if 'reval' in x and (x['judge'], x['qid']) in orig:
        per[x['qid']].append(x['reval'] - orig[(x['judge'], x['qid'])])
dr = [st.mean(v) for v in per.values()]
m, se = st.mean(dr), st.stdev(dr) / math.sqrt(len(dr))
t_lo, t_hi = (m + 0.15) / se, (0.15 - m) / se
# one-sided p under the normal approximation (df = n-1 > 300); log10 for very small values
log10p = lambda t: math.log10(0.5 * math.erfc(t / math.sqrt(2))) if t < 37 else float('-inf')
out['perturbation_response_level'] = {'n': len(dr), 'delta': round(m, 4),
                                      'ci90_t': [round(m - 1.6493 * se, 4), round(m + 1.6493 * se, 4)],
                                      't_lower': round(t_lo, 2), 't_upper': round(t_hi, 2),
                                      'log10_p_lower': round(log10p(t_lo), 1), 'log10_p_upper': round(log10p(t_hi), 1)}
(ROOT / 'results/paper_details.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
