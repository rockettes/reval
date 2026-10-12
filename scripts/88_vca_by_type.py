"""Reproduces paper Table 17 (VCA±1yr by question type).

Reads from results/vca_final_5way.json (aggregate by_type produced by 68_vca_final.py).
Output: results/vca_by_type.json with cleaner shape (6 systems × 8 types).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
v = json.load((ROOT/'results/vca_final_5way.json').open())

out = {}
for s, types in v['by_type'].items():
    out[s] = {t: {'n_dois': d['total_dois'],
                  'vca_strict_pct': d['vca_strict_pct'],
                  'vca_lenient_pct': d['vca_lenient_pct']}
              for t, d in types.items()}

(ROOT/'results/vca_by_type.json').write_text(json.dumps(out, indent=2))
print('Saved vca_by_type.json')
print(f"{'System':<10} " + ' '.join(f'{t:>8}' for t in ['DIVERG','POPUL','M_HOP','COMP','TEMP','MECH','NUM','ADV']))
for s in ['Ours','GPT54','GemSearch','Gemini','Grok','GLM']:
    if s not in out: continue
    types = ['DIVERGENT','POPULATION','MULTI_HOP','COMP','TEMPORAL','MECHANISM','NUM','ADV_OOS']
    row = [out[s].get(t,{'vca_lenient_pct':None})['vca_lenient_pct'] for t in types]
    print(f"  {s:<10} " + ' '.join(f'{r:>8}' if r is not None else f'{"--":>8}' for r in row))
