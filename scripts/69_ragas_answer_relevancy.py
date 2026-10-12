"""RAGAS answer_relevancy for all 6 generators.

Uses Llama 3.3 70B (Groq) for question generation + text-embedding-3-small for similarity.
No generator family serves as its own evaluator.

Usage: python 69_ragas_answer_relevancy.py [N]
  N = number of questions (default: 350, use 10 for testing)
"""
import json
import os
import sys
import time
from pathlib import Path

DRY = Path(__file__).resolve().parents[1]

# Set GROQ_API_KEY and OPENAI_API_KEY as environment variables before running.
# See .env.example for required keys.

RESPONSE_FILES = {
    "Ours": "responses_350_RAG.json",
    "GPT54": "responses_350_gpt54_v2.json",
    "GLM": "responses_350_glm_v2.json",
    "Gemini": "responses_350_gemini_v2.json",
    "GemSearch": "responses_350_gemsearch.json",
    "Grok": "responses_350_grok_v2.json",
}

OUT_FILE = DRY / "ragas_answer_relevancy.json"


def main():
    n_questions = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    target_gen = sys.argv[2] if len(sys.argv) > 2 else "all"
    print(f"RAGAS answer_relevancy — N={n_questions}, target={target_gen}", flush=True)

    # Import ragas (heavy, do it here)
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy
    from langchain_groq import ChatGroq
    from langchain_openai import OpenAIEmbeddings

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Groq doesn't support n>1, set strictness=1 (generate 1 question per response)
    answer_relevancy.strictness = 1

    # Load all responses
    all_responses = {}
    for gen, fname in RESPONSE_FILES.items():
        path = DRY / fname
        if not path.exists():
            print(f"SKIP {gen}: {fname} not found")
            continue
        data = json.load(open(path))
        all_responses[gen] = sorted(data, key=lambda x: x["id"])[:n_questions]
        print(f"Loaded {gen}: {len(all_responses[gen])} responses")

    results = {}
    gens_order = ["Ours", "GPT54", "GLM", "Gemini", "GemSearch", "Grok"]

    if target_gen != "all":
        gens_order = [target_gen]
        OUT_FILE = DRY / f"ragas_{target_gen}.json"

    for gen in gens_order:
        if gen not in all_responses:
            continue
        subset = all_responses[gen]
        print(f"\n=== {gen} ({len(subset)} Qs) ===", flush=True)

        ds = Dataset.from_dict({
            "question": [r["question"] for r in subset],
            "answer": [r.get("answer", "") for r in subset],
            "contexts": [[] for _ in subset],
        })

        t0 = time.time()
        try:
            score = evaluate(
                dataset=ds,
                metrics=[answer_relevancy],
                llm=llm,
                embeddings=embeddings,
            )
            elapsed = time.time() - t0
            ar = score["answer_relevancy"]
            # ar may be a list of per-item scores or a float
            if isinstance(ar, list):
                valid = [x for x in ar if x is not None and not (isinstance(x, float) and x != x)]
                ar_mean = sum(valid) / len(valid) if valid else 0.0
            else:
                ar_mean = float(ar)
            results[gen] = {
                "mean": round(ar_mean, 4),
                "per_item": [round(float(x), 4) if x is not None else None for x in ar] if isinstance(ar, list) else [round(ar_mean, 4)],
                "n": len(subset),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  {gen}: answer_relevancy = {ar_mean:.4f} ({elapsed:.1f}s)", flush=True)
            if isinstance(ar, list):
                print(f"    per-item sample: {[round(float(x),3) for x in ar[:5]]}...", flush=True)
        except Exception as e:
            print(f"  {gen}: ERROR — {e}", flush=True)
            results[gen] = None

        # Save incrementally
        with open(OUT_FILE, "w") as f:
            json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 50)
    print("RAGAS answer_relevancy RESULTS")
    print("=" * 50)
    for gen in gens_order:
        if gen in results and results[gen] is not None:
            print(f"  {gen:>12}: {results[gen]['mean']:.4f}")

    # Save with metadata
    output = {
        "methodology": {
            "metric": "answer_relevancy",
            "llm": "llama-3.3-70b-versatile (Groq, temp 0.0)",
            "embeddings": "text-embedding-3-small (OpenAI)",
            "strictness": 1,
            "ragas_version": "0.4.3",
            "n_questions": n_questions,
            "justification": "No generator family serves as evaluator. Llama 3.3 70B (Meta) is neutral. Consistent with VCA semantic validation.",
        },
        "results": results,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUT_FILE.name}")


if __name__ == "__main__":
    main()
