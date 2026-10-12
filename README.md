# BodybuildingQA: A Scientific QA Benchmark for Exercise Science

This repository contains the dataset, model responses, evaluation scores, and analysis scripts accompanying the paper:

> **Grounding-Blind Judges: Multi-Judge Evaluation of Scientific RAG Reveals Systematic Divergence Between Subjective Scores and Citation Validity**
> Anonymous Authors, 2026.

## Overview

**BodybuildingQA** is a domain-specific scientific QA benchmark comprising **350 questions** in Brazilian Portuguese (pt-BR) across **8 taxonomic types** covering exercise science, sports nutrition, training periodization, and bodybuilding physiology.

**REVAL** is a multi-judge evaluation framework anchored in objective citation verification via CrossRef. It combines a cross-evaluation graph (5 LLM judges, self-evaluation excluded), Verified Citation Accuracy (VCA) separating DOI existence from semantic correspondence, a 6-dimension domain-weighted rubric, and paired statistical testing with Bonferroni correction.

**Key findings:** (1) The LLM-judge ranking is approximately inverted relative to the objective citation ranking across systems: models that fabricate 8–50% of DOIs score 0.88–1.24 REVAL points above the RAG system, which achieves 86% metadata-verified citation accuracy yet is ranked last by every judge. (2) A controlled perturbation study (identical answers, only the DOI strings replaced by surface-plausible non-resolving fakes) leaves composite judge scores TOST-equivalent — judges respond to the surface form of citations, not to whether they resolve. (3) The effect replicates with web search across two provider families: search halves DOI fabrication but leaves a substantial metadata-verification gap, and the judge panel assigns the search-augmented responses the highest scores of the study. RAGAS `answer_relevancy` reproduces the same blind spot.

## Question Types

| Type | Description | Count |
|------|-------------|-------|
| DIVERGENT | Conflicting evidence | 70 |
| POPULATION | Population-dependent | 70 |
| MULTI_HOP | Multi-variable synthesis | 70 |
| COMP | A vs. B comparison | 35 |
| TEMPORAL | Evidence evolution | 30 |
| MECHANISM | Physiological mechanism | 25 |
| NUM | Numeric answer | 25 |
| ADV_OOS | Adversarial / out-of-scope | 25 |

## Repository Structure

```
├── README.md
├── LICENSE                            # CC-BY-NC-4.0
├── benchmark/
│   └── bodybuilding_qa_350.jsonl      # 350 questions
├── responses/
│   ├── ours.json                      # Proprietary RAG (350 Qs)
│   ├── gpt54.json                     # GPT-5.4 (350 Qs)
│   ├── glm.json                       # GLM-5.1 (350 Qs)
│   ├── gemini.json                    # Gemini 3.1 Pro (350 Qs)
│   ├── gemsearch.json                 # Gemini 3.1 Pro + Google Search (350 Qs)
│   ├── grok.json                      # Grok 4.20 (350 Qs)
│   ├── gpt54search.json               # GPT-5.4 + web search, forced (350 Qs)
│   ├── gpt54search_auto.json          # GPT-5.4 + web search, model-elected (350 Qs)
│   ├── gemsearch_perturbed_format_preserving.json  # Perturbation arm 1 (350 Qs)
│   └── gemsearch_perturbed_format_breaking.json    # Perturbation arm 2 (350 Qs)
├── scores/
│   ├── reval_scores.json              # 8,750 judge scores (5 judges × 6 generators, self-excluded)
│   ├── gpt54search_scores.json        # 2,766 judge scores for both GPT+S conditions
│   ├── perturbation_format_preserving_scores.json  # 1,382 paired re-scores
│   ├── perturbation_format_breaking_scores.json    # 1,362 paired re-scores
│   └── retest_noise_floor_scores.json # 200 unchanged re-scores (noise floor)
├── results/
│   ├── stats_reval.json               # Aggregated REVAL statistics
│   ├── stats_crossref.json            # CrossRef DOI verification (input to scripts 51 and 64)
│   ├── vca_70b_results.json           # VCA 2.0 semantic validation (Llama 3.3 70B)
│   ├── vca_final_5way.json            # VCA strict + VCA±1yr (all systems)
│   ├── ragas_answer_relevancy.json    # RAGAS comparison (Llama 3.3 70B)
│   ├── human_vca_300_results.json     # Human VCA validation (N=300, 2 annotators; κ_hh=0.96, κ_hv=0.81)
│   ├── human_vca_annotator1_300.csv   # Annotator 1: 300 blind stratified DOIs
│   ├── human_vca_annotator2_300.csv   # Annotator 2: 300 blind stratified DOIs
│   ├── human_vca_300_annotation_guide.txt # Annotation instructions (kit)
│   ├── human_vca_300_validator_verdicts.csv # Automated matcher verdict per validation item (script 70)
│   ├── error_analysis.json            # Error pattern classification
│   ├── robustness.json                # Robustness by question type
│   ├── perturbation_analysis_format_preserving.json # Δ REVAL, per-judge, per-dimension, TOST
│   ├── perturbation_analysis_format_breaking.json   # Δ REVAL, per-judge, per-dimension, TOST
│   ├── fake_doi_global_resolution_check.json        # Fabricated DOIs verified non-resolving
│   ├── crossref_doi_verification_all_conditions.json # Per-DOI CrossRef record, all 8 conditions
│   ├── gptsearch_vca_by_type.json     # GPT+S VCA strict/±1yr, by type (script 90)
│   ├── gptsearch_citverif_correlation.json # GPT+S CitVerif vs per-response VCA (script 91)
│   ├── recency_analysis.json          # Citation recency, 7 conditions (script 83)
│   ├── common_panel_comparisons.json  # script 92
│   ├── citverif_vs_vca_per_response.json # script 93
│   ├── retest_noise_floor.json        # script 94
│   ├── doi_concentration_and_gain.json # script 95
│   ├── paper_details.json             # script 96
│   ├── stats_crossref_summary.json    # script 51 output
│   └── vca_semantic_judge_cache.json  # Cached VCA author/year verdicts
└── scripts/
    ├── _stats_utils.py                # Shared statistical utilities
    ├── 50_stats_REVAL.py              # REVAL aggregate scores
    ├── 51_stats_crossref.py           # CrossRef verification stats
    ├── 55_robustness_tests.py         # Robustness by question type
    ├── 56_error_analysis.py           # Error pattern analysis
    ├── 64_vca_llm_70b.py              # VCA semantic validation (Llama 3.3 70B)
    ├── 68_vca_final.py                # Combined VCA strict + VCA±1yr
    ├── 69_ragas_answer_relevancy.py   # RAGAS answer_relevancy comparison
    ├── 70_human_vca_score.py          # Human VCA validation (Cohen's κ)
    ├── 71_perturbation_study.py       # Controlled citation-perturbation study (both arms)
    ├── 72_gpt54_search_ablation.py    # GPT-5.4 + web search generation
    ├── 73_gpt54search_pipeline.py     # GPT+S CrossRef + VCA + judging pipeline
    ├── 80_bootstrap_cis.py            # 95% bootstrap CIs (REVAL, VCA)
    ├── 81_pairwise_significance.py    # Pairwise comparisons (Table 10)
    ├── 83_recency_analysis.py         # Citation recency (all seven conditions)
    ├── 84_weight_sensitivity.py       # 10,000 random rubric weightings
    ├── 85_judge_ablation.py           # All judge subsets
    ├── 86_behavioral_signatures.py    # Behavioral signatures + regex refusal (Table 4)
    ├── 87_ragas_spearman.py           # RAGAs vs REVAL Spearman
    ├── 88_vca_by_type.py              # VCA±1yr by question type
    ├── 90_gptsearch_vca_by_type.py    # GPT+S VCA strict/±1yr and by question type
    ├── 91_gptsearch_citverif_correlation.py # GPT+S CitVerif vs per-response VCA
    ├── 92_common_panel_and_rank_correlation.py # Common-judge-panel comparisons; REVAL×VCA rank correlation
    ├── 93_citverif_vs_vca_per_response.py # CitVerif vs per-response VCA, all seven conditions
    ├── 94_retest_noise_floor.py       # Retest noise floor of the perturbation table
    ├── 95_doi_concentration_and_gain.py # Unique-DOI concentration; Gemini→Gemini+S gain by dimension
    ├── 96_paper_details.py            # REVAL by type, GPT→GPT+S, GPT+S per-dimension, error analysis, response-level TOST
    ├── 44_judge_REVAL.py              # Multi-judge REVAL scoring
    ├── 46b_crossref_v2.py             # CrossRef DOI verification
    └── 48_baselines_v2_with_doi.py    # Generalist baseline generation
```

## Usage

### Reproduce statistics

```bash
cd scripts/
python 50_stats_REVAL.py       # REVAL score aggregation
python 51_stats_crossref.py    # DOI verification stats
python 68_vca_final.py         # VCA strict + VCA±1yr
python 55_robustness_tests.py  # Robustness by question type
python 70_human_vca_score.py   # Human VCA validation (κ)
```

### Verify the human citation-validation agreement

The two annotators' labels are released so the inter-annotator and human-vs-validator agreement can be recomputed independently:

```bash
# Cohen's kappa between the two annotators (expected kappa ~ 0.96, 300 items)
python - <<'PY'
import csv
a=[r['human_verdict'] for r in csv.DictReader(open('results/human_vca_annotator1_300.csv'))]
b=[r['human_verdict'] for r in csv.DictReader(open('results/human_vca_annotator2_300.csv'))]
agree=sum(x==y for x,y in zip(a,b)); n=len(a)
print(f"raw agreement: {agree}/{n} = {agree/n:.3f}")
PY
```

Summary statistics (κ human–human = 0.96, κ human–validator = 0.81, 0 false positives) are in `results/human_vca_300_results.json`.

### Perturbation study and GPT+S

The perturbed responses hold each answer's text byte-identical and replace only its DOI
strings; each item carries a `replacements` field listing every substitution, so the
manipulation can be audited directly against `responses/gemsearch.json`. The paired
re-scores are in `scores/`, and `results/perturbation_analysis_*.json` carry Δ REVAL with
its CI, the per-judge and per-dimension breakdowns, the TOST verdict, and the retest noise
floor. The two arms use 1,382 and 1,362 usable paired scores out of 1,400 possible; the
remainder are dropped where a response carries no resolvable DOI to perturb or a judge call
failed transiently (only successful scores are released).

The perturbation table can be recomputed from the released scores alone:

```bash
python - <<'EOF'
import json, statistics
orig = {(r['judge'], r['qid']): r for r in json.load(open('scores/reval_scores.json'))
        if r['generator'] == 'GemSearch'}
for arm, f in [('format-preserving', 'scores/perturbation_format_preserving_scores.json'),
               ('format-breaking',   'scores/perturbation_format_breaking_scores.json')]:
    pert = json.load(open(f))
    d  = [r['reval'] - orig[(r['judge'], r['qid'])]['reval'] for r in pert
          if (r['judge'], r['qid']) in orig]
    cv = [r['scores']['verificab_citacoes']
          - orig[(r['judge'], r['qid'])]['scores']['verificab_citacoes']
          for r in pert if (r['judge'], r['qid']) in orig]
    print(f"{arm}: n={len(d)}  dREVAL={statistics.mean(d):+.4f}  dCitVerif={statistics.mean(cv):+.3f}")
EOF
```

Expected output (paper values in parentheses):

```
format-preserving: n=1382  dREVAL=-0.0656 (-0.066)  dCitVerif=-0.211 (-0.21)
format-breaking:   n=1362  dREVAL=-0.5819 (-0.582)  dCitVerif=-1.907 (-1.91)
```

### File-name mapping

Scripts 44, 46b, 48 and 71-73 are published **as run**. They read the working-directory
layout of the original experiments rather than this repository's folders:

| Script expects | Released as |
|---|---|
| `responses_350_<system>.json` | `responses/<system>.json` |
| `responses_350_gpt54search_forced.json` | `responses/gpt54search.json` |
| `responses_350_gemsearch_perturbMIN.json` | `responses/gemsearch_perturbed_format_preserving.json` |
| `responses_350_gemsearch_perturbFAB.json` | `responses/gemsearch_perturbed_format_breaking.json` |
| `doi_4way_v2.json` | `results/crossref_doi_verification_all_conditions.json` |
| `vca_70b_cache.json` | `results/vca_semantic_judge_cache.json` |
| `scores_retest50.json` | `scores/retest_noise_floor_scores.json` |
| `scores_perturbMIN.json` / `scores_perturbFAB.json` | `scores/perturbation_format_*_scores.json` |
| `scores_gpt54search.json` | `scores/gpt54search_scores.json` |

One correction was applied to the published copy of `44_judge_REVAL.py`: its `GENERATORS`
dict had gone stale after the run and pointed the GLM, Gemini and Grok lanes at the earlier
response files. The judges scored the responses released here, which the dict now names.
No reported number is affected.

**Reproducibility notes:**
- All bootstrap scripts use `random.seed(42)` for deterministic results.
- CrossRef API uses the Polite Pool (User-Agent with mailto). No API key needed.
- VCA semantic validation (script 64) requires a Groq API key (see `.env.example`).
- RAGAS comparison (script 69) requires Groq + OpenAI API keys.
- Dependencies: `pip install requests ragas langchain-groq langchain-openai datasets`.

## License

This dataset and code are released under [CC-BY-NC-4.0](LICENSE).

## Citation

```bibtex
@inproceedings{anonymous2026reval,
  title     = {Grounding-Blind Judges: Multi-Judge Evaluation of Scientific {RAG} Reveals Systematic Divergence Between Subjective Scores and Citation Validity},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026},
  note      = {Under review}
}
```
