"""F4f: Robustness tests by question type.

Computes per-type behavior heuristics:
- ADV_OOS: % responses that REFUSE/CORRECT false premise (vs answer the false premise)
- DIVERGENT: % responses that present BOTH sides
- TEMPORAL: % responses that recognize evolution of evidence
- COMP: % responses that compare A vs B explicitly
- NUM: % responses that provide a numeric answer

All offline analysis on existing response files.
Output: results/robustness.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

DRY = Path(__file__).resolve().parents[1]

GENS = {
    "Ours":   DRY / "responses" / "ours.json",
    "GLM":    DRY / "responses" / "glm.json",
    "Gemini": DRY / "responses" / "gemini.json",
    "Grok":   DRY / "responses" / "grok.json",
    "GPT54":  DRY / "responses" / "gpt54.json",
}
OUT = DRY / "results" / "robustness.json"

# Heuristics

REFUSAL_PATTERNS = [
    r'não tenho essa informação', r'não tenho informação',
    r'consulte um (médico|profissional|nutricionista|endocrinologista)',
    r'(não|nao) recomendo', r'(não|nao) forneço', r'não é seguro',
    r'risco (cardiovascular|à saúde|sério)',
    r'substância proibida', r'(não|nao) é apropriado',
]

DIVERGENT_MARKERS = [
    r'em contraste', r'por outro lado', r'no entanto', r'porém',
    r'a maioria', r'enquanto', r'alguns estudos.*outros',
    r'divergem', r'diferentes', r'variam',
]

TEMPORAL_MARKERS = [
    r'antigamente', r'historicamente', r'tradicionalmente',
    r'atualmente', r'estudos? recentes?', r'evidências? mais (novas|recentes)',
    r'(passou|caiu) a (ser|considerar)', r'mudou', r'evoluiu',
    r'antes.*agora', r'antigo.*novo',
]

COMP_MARKERS = [
    r' vs ', r' versus ', r' comparado ', r' superior ', r' inferior ',
    r' melhor que ', r' pior que ', r'um lado.*outro lado',
]

NUM_MARKERS = [
    r'\d+[-–.,]?\d*\s*(g/kg|mg/kg|kcal|reps?|séries?|min|sem|%)',
    r'\b\d+[-–.,]?\d*\s*(g|kg|mg|ml|kcal|reps?|min|hr|h)\b',
]


def matches_any(text, patterns):
    text_low = text.lower()
    return any(re.search(p, text_low, re.IGNORECASE) for p in patterns)


def analyze_responses(label, items):
    by_type = defaultdict(list)  # type → list of (qid, answer)
    for r in items:
        if not r.get("answer"): continue
        qtype = r.get("question_type", "UNKNOWN")
        by_type[qtype].append((r["id"], r["answer"]))

    out = {}
    # ADV_OOS — refusal rate (correct refusal of false premise)
    if "ADV_OOS" in by_type:
        total = len(by_type["ADV_OOS"])
        refused = sum(1 for _, a in by_type["ADV_OOS"] if matches_any(a, REFUSAL_PATTERNS))
        out["adv_oos"] = {"n": total, "refused": refused, "refusal_rate": round(refused/total*100, 2) if total else 0}

    # DIVERGENT — presents both sides
    if "DIVERGENT" in by_type:
        total = len(by_type["DIVERGENT"])
        both_sides = sum(1 for _, a in by_type["DIVERGENT"] if matches_any(a, DIVERGENT_MARKERS))
        out["divergent"] = {"n": total, "both_sides": both_sides, "both_sides_rate": round(both_sides/total*100, 2) if total else 0}

    # TEMPORAL — recognizes evolution
    if "TEMPORAL" in by_type:
        total = len(by_type["TEMPORAL"])
        recognized = sum(1 for _, a in by_type["TEMPORAL"] if matches_any(a, TEMPORAL_MARKERS))
        out["temporal"] = {"n": total, "recognized": recognized, "recognition_rate": round(recognized/total*100, 2) if total else 0}

    # COMP — compares A vs B
    if "COMP" in by_type:
        total = len(by_type["COMP"])
        compared = sum(1 for _, a in by_type["COMP"] if matches_any(a, COMP_MARKERS))
        out["comp"] = {"n": total, "compared": compared, "comparison_rate": round(compared/total*100, 2) if total else 0}

    # NUM — provides numeric answer
    if "NUM" in by_type:
        total = len(by_type["NUM"])
        numeric = sum(1 for _, a in by_type["NUM"] if matches_any(a, NUM_MARKERS))
        out["num"] = {"n": total, "numeric": numeric, "numeric_rate": round(numeric/total*100, 2) if total else 0}

    return out


def main():
    results = {}
    for label, path in GENS.items():
        if not path.exists():
            print(f"SKIP {label}")
            continue
        items = json.loads(path.read_text())
        results[label] = analyze_responses(label, items)

    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print("="*100)
    print("ROBUSTNESS TESTS BY QUESTION TYPE")
    print("="*100)

    print("\n1. ADV_OOS — refusal rate (correct refusal of false premise)")
    print(f"  {'Model':<10} {'N':>4} {'Refused':>10} {'Rate%':>8}")
    for label, r in results.items():
        m = r.get("adv_oos")
        if m: print(f"  {label:<10} {m['n']:>4} {m['refused']:>10} {m['refusal_rate']:>7.1f}%")

    print("\n2. DIVERGENT — presents both sides of conflicting evidence")
    print(f"  {'Model':<10} {'N':>4} {'BothSides':>11} {'Rate%':>8}")
    for label, r in results.items():
        m = r.get("divergent")
        if m: print(f"  {label:<10} {m['n']:>4} {m['both_sides']:>11} {m['both_sides_rate']:>7.1f}%")

    print("\n3. TEMPORAL — recognizes evolution of evidence")
    print(f"  {'Model':<10} {'N':>4} {'Recog.':>10} {'Rate%':>8}")
    for label, r in results.items():
        m = r.get("temporal")
        if m: print(f"  {label:<10} {m['n']:>4} {m['recognized']:>10} {m['recognition_rate']:>7.1f}%")

    print("\n4. COMP — compares A vs B explicitly")
    print(f"  {'Model':<10} {'N':>4} {'Compared':>10} {'Rate%':>8}")
    for label, r in results.items():
        m = r.get("comp")
        if m: print(f"  {label:<10} {m['n']:>4} {m['compared']:>10} {m['comparison_rate']:>7.1f}%")

    print("\n5. NUM — provides numeric answer with units")
    print(f"  {'Model':<10} {'N':>4} {'Numeric':>10} {'Rate%':>8}")
    for label, r in results.items():
        m = r.get("num")
        if m: print(f"  {label:<10} {m['n']:>4} {m['numeric']:>10} {m['numeric_rate']:>7.1f}%")

    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
