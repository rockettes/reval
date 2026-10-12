"""Reproduces the retest noise floor in the perturbation table (Delta REVAL = -0.0045, n = 200).

50 Gemini+S responses (every 7th question id) were re-scored, unchanged, by the same four
judges; the noise floor is the mean paired difference between the re-score and the
original score. Failed judge calls (rate limit, timeout, unparseable output) carry no
score and are excluded, as in 71_perturbation_study.py.

Inputs: scores/retest_noise_floor_scores.json, scores/reval_scores.json
Output: results/retest_noise_floor.json
"""
import json, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
orig = {(r['judge'], r['qid']): r['reval'] for r in json.load((ROOT / 'scores/reval_scores.json').open())
        if r['generator'] == 'GemSearch' and 'reval' in r}
retest = [r for r in json.load((ROOT / 'scores/retest_noise_floor_scores.json').open())
          if r.get('generator') == 'GemSearchRetest' and 'reval' in r]
d = [r['reval'] - orig[(r['judge'], r['qid'])] for r in retest if (r['judge'], r['qid']) in orig]
out = {'n': len(d), 'mean_delta': round(st.mean(d), 4), 'sd': round(st.stdev(d), 4)}
(ROOT / 'results/retest_noise_floor.json').write_text(json.dumps(out, indent=2))
print('Saved retest_noise_floor.json:', out)
