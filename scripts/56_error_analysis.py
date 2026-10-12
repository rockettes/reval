"""F4g: Error analysis — classify Ours' lowest-scoring responses into 5 patterns.

5 patterns from paper:
1. Contexto inadequado (specific protocol used as general recommendation)
2. Síntese incompleta (covers core but omits relevant condition)
3. Citação ausente (evidence available but not cited with DOI)
4. Recusa excessiva (refuses with sufficient support)
5. Conflito temporal (date/version not reconciled)

Heuristic-based classification + manual review of top-N worst.

Output: results/error_analysis.json
"""
import json
import re
import statistics
from collections import defaultdict, Counter
from pathlib import Path

DRY = Path(__file__).resolve().parents[1]

OURS = DRY / "responses" / "ours.json"
SCORES = DRY / "scores" / "reval_scores.json"
OUT = DRY / "results" / "error_analysis.json"

DIMS = [("correcao_cientifica", 0.25), ("verificab_citacoes", 0.25),
        ("especificidade", 0.20), ("completude", 0.15),
        ("assertividade", 0.10), ("aplicacao_pratica", 0.05)]

# Heuristic patterns

PATTERN_PROTOCOL_SPECIFIC = [
    r'peak ?week', r'pré[-\s]?compet', r'desidrata',
    r'protocolo específico', r'fase específica',
    r'imediatamente antes', r'24 horas antes',
]

PATTERN_INCOMPLETE = [
    r'(varia|depende)', r'em alguns casos', r'pode (ser|variar)',
    r'mais (estudos|pesquisa) (são|são necessári)',
]

PATTERN_NO_CITATION = [
    # response without [DOI: ...] or [ISBN: ...] anywhere
]

PATTERN_REFUSAL = [
    r'não tenho.*informação', r'consulte', r'redirecione',
    r'não recomendo', r'risco', r'profissional',
]

PATTERN_TEMPORAL_CONFLICT = [
    r'(antes|antigamente).*(?:atualmente|agora|hoje)',
    r'mudou', r'(novas?|recentes?) evidências?',
]


def reval(scores):
    return sum(scores[k]*w for k,w in DIMS)


def load_scores():
    raw = json.load(open(SCORES))
    return [x for x in raw if x.get("scores") and not x.get("excluded")]


def matches_any(text, patterns):
    if not patterns: return False
    return any(re.search(p, text.lower(), re.IGNORECASE) for p in patterns)


def classify(qid, response, scores):
    """Classify failure pattern. Returns set of pattern labels."""
    answer = response.get("answer", "")
    patterns = set()

    # 3. Citação ausente
    if scores.get("verificab_citacoes", 5) < 3:
        if "[DOI:" not in answer and "[ISBN:" not in answer:
            patterns.add("citacao_ausente")
        else:
            patterns.add("citacao_fraca")

    # 1. Contexto inadequado
    if scores.get("correcao_cientifica", 5) < 4:
        if matches_any(answer, PATTERN_PROTOCOL_SPECIFIC):
            patterns.add("contexto_inadequado")

    # 2. Síntese incompleta
    if scores.get("completude", 5) < 4:
        patterns.add("sintese_incompleta")

    # 4. Recusa excessiva (refusal text)
    if matches_any(answer, PATTERN_REFUSAL):
        patterns.add("recusa")

    # 5. Conflito temporal
    if matches_any(answer, PATTERN_TEMPORAL_CONFLICT) and scores.get("correcao_cientifica", 5) < 4:
        patterns.add("conflito_temporal")

    if not patterns:
        patterns.add("outros")
    return patterns


def main():
    ours = json.load(open(OURS))
    ours_by_id = {r["id"]: r for r in ours}
    scores = load_scores()

    # Per-question mean scores for Ours (across non-self judges)
    by_qid = defaultdict(list)
    for s in scores:
        if s.get("judge") == "Ours" or s.get("generator") != "Ours" or not s.get("scores"):
            continue
        by_qid[s["qid"]].append(s["scores"])

    # Compute mean per question + REVAL
    per_q_mean = {}
    for qid, score_list in by_qid.items():
        avg = {k: statistics.mean(s[k] for s in score_list) for k, _ in DIMS}
        avg["reval"] = reval(avg)
        per_q_mean[qid] = avg

    # Sort by REVAL ascending (worst first)
    sorted_qids = sorted(per_q_mean.keys(), key=lambda q: per_q_mean[q]["reval"])
    bottom_n = sorted_qids[:50]  # 50 worst

    # Classify each
    classifications = []
    pattern_counts = Counter()
    for qid in bottom_n:
        if qid not in ours_by_id: continue
        avg = per_q_mean[qid]
        patterns = classify(qid, ours_by_id[qid], avg)
        for p in patterns:
            pattern_counts[p] += 1
        classifications.append({
            "qid": qid,
            "reval": round(avg["reval"], 3),
            "question": ours_by_id[qid]["question"][:120],
            "question_type": ours_by_id[qid].get("question_type"),
            "patterns": sorted(patterns),
        })

    # Also classify ALL questions (not just bottom 50) for coverage
    pattern_counts_all = Counter()
    for qid in per_q_mean:
        if qid not in ours_by_id: continue
        patterns = classify(qid, ours_by_id[qid], per_q_mean[qid])
        for p in patterns:
            pattern_counts_all[p] += 1

    out = {
        "n_questions_evaluated": len(per_q_mean),
        "bottom_50": classifications,
        "pattern_counts_bottom_50": dict(pattern_counts),
        "pattern_counts_all_350": dict(pattern_counts_all),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("="*100)
    print("F4g ERROR ANALYSIS — Ours RAG")
    print("="*100)
    print(f"\nQuestions analyzed: {len(per_q_mean)}/350")
    print(f"\n📊 Pattern counts in BOTTOM 50 (worst REVAL scores):")
    for p, c in pattern_counts.most_common():
        print(f"  {p:<22}: {c:>3} ({c/len(bottom_n)*100:.0f}%)")
    print(f"\n📊 Pattern counts in ALL 350:")
    for p, c in pattern_counts_all.most_common():
        print(f"  {p:<22}: {c:>3} ({c/len(per_q_mean)*100:.0f}%)")

    print(f"\n🔍 Top 10 worst responses:")
    for c in classifications[:10]:
        print(f"  Q{c['qid']:3d} REVAL={c['reval']:.2f} ({c['question_type']:<10}) {','.join(c['patterns'])}")
        print(f"    {c['question'][:100]}")

    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
