"""VCA 2.0 Semantic Validation via Llama 3.3 70B (Groq).

Improved over 63_vca_llm_judge.py:
- Model: llama-3.3-70b-versatile (was 8b)
- Prompt: few-shot with positive AND negative examples
- Conservative but not paranoid: if author+year+DOI all match, it's VALID

For each DOI that EXISTS on CrossRef, classifies as VALID or INVALID.
Temperature 0.0 for reproducibility.
"""
import json
import os
import time
import requests
from pathlib import Path
from collections import defaultdict, Counter

DRY = Path(__file__).resolve().parents[1]
# Set GROQ_API_KEY (or OPENAI_API_KEY) as environment variable before running.
# See .env.example for required keys.

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

CACHE_FILE = DRY / "results" / "vca_70b_cache.json"
OUT_FILE = DRY / "results" / "vca_70b_results.json"

SYSTEM_PROMPT = """You are a citation verification expert. You will receive:
1. A DOI and its real metadata from CrossRef (title, first author, year)
2. A snippet of text from an AI-generated response that cites this DOI

Your job: determine if the citation in the response text is VALID or INVALID.

VALID means: the response associates this DOI with the correct author (any author of the paper, not just first) AND approximately the correct year (±1 year tolerance). The author name may have minor spelling variations.

INVALID means: the response either (a) associates this DOI with a completely wrong author, or (b) there is no author context near the DOI at all, or (c) the year is off by more than 1 year.

EXAMPLES:

Example 1 - VALID:
DOI: 10.1186/1550-2783-11-20 | Real: Helms (2014)
Text: "...de acordo com Helms et al. (2014) [DOI: 10.1186/1550-2783-11-20], a ingestão proteica..."
→ VALID (author Helms matches, year 2014 matches)

Example 2 - VALID:
DOI: 10.1519/jsc.0000000000002776 | Real: Baz-Valle (2021)
Text: "...Baz-Valle et al. [DOI: 10.1519/jsc.0000000000002776] demonstraram que..."
→ VALID (author matches, year not stated but author is correct)

Example 3 - INVALID:
DOI: 10.3390/jfmk5010019 | Real: Campbell (2020)
Text: "...de acordo com Byrne et al. [DOI: 10.3390/jfmk5010019], os participantes..."
→ INVALID (response says Byrne but real first author is Campbell)

Example 4 - VALID:
DOI: 10.1139/apnm-2014-0530 | Real: Arentson-Lantz (2015)
Text: "...Arentson-Lantz et al. (2015) [DOI: 10.1139/apnm-2014-0530], uma dieta..."
→ VALID (author matches exactly, year matches)

Example 5 - INVALID:
DOI: 10.1080/17461391.2022.2100279 | Real: Maeo (2022)
Text: "...Eric Helms recomenda amplitudes completas [DOI: 10.1080/17461391.2022.2100279]..."
→ INVALID (response says Helms but real author is Maeo)

Respond with EXACTLY one word: VALID or INVALID"""

USER_TEMPLATE = """DOI: {doi}
Real metadata from CrossRef:
- Title: {cr_title}
- First author: {cr_author}
- Year: {cr_year}

Response text snippet:
{response_excerpt}

Is this citation VALID or INVALID?"""


def call_groq(system, user, max_retries=3):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
                timeout=30,
            )
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip().upper()
                return "VALID" if "VALID" in text and "INVALID" not in text else "INVALID"
            elif r.status_code == 429:
                wait = min(2 ** attempt * 2, 30)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  API error {r.status_code}: {r.text[:80]}")
                time.sleep(2)
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(2)
    return "INVALID"


def get_crossref_info(doi):
    """Fetch title + all authors from CrossRef."""
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "REVAL-VCA/2.0 (mailto:anonymous@review.org)"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()["message"]
            title = data.get("title", ["Unknown"])[0] if data.get("title") else "Unknown"
            authors = [a.get("family", "") for a in data.get("author", [])]
            year = str(data.get("published-print", data.get("published-online", data.get("created", {})))
                      .get("date-parts", [[0]])[0][0])
            return {"title": title, "authors": authors, "year": year}
    except:
        pass
    return None


def main():
    print(f"Model: {MODEL}")
    print(f"VCA 2.0 with improved prompt + 70B model")

    v2 = json.load(open(DRY / "results" / "stats_crossref.json"))

    responses = {}
    for gen, fname in [
        ("Ours", "ours.json"),
        ("GLM", "glm.json"),
        ("Gemini", "gemini.json"),
        ("Grok", "grok.json"),
    ]:
        data = json.load(open(DRY / "responses" / fname))
        responses[gen] = {r["id"]: r.get("answer", "") for r in data}

    cache = {}
    if CACHE_FILE.exists():
        cache = json.load(open(CACHE_FILE))

    cr_cache_path = DRY / "results" / "crossref_title_cache.json"
    cr_cache = {}
    if cr_cache_path.exists():
        cr_cache = json.load(open(cr_cache_path))

    results = {}
    api_calls = 0
    cr_fetches = 0

    for gen in ["Ours", "GLM", "Gemini", "Grok"]:
        gen_valid = 0
        gen_invalid = 0
        gen_total = 0

        for q in v2[gen]["results"]:
            qid = q["qid"]
            answer = responses[gen].get(qid, "")

            for doi_entry in q["dois"]:
                if not doi_entry.get("cr_valid"):
                    continue

                gen_total += 1
                doi = doi_entry["doi"]
                cache_key = f"{gen}_{qid}_{doi}"

                if cache_key in cache:
                    verdict = cache[cache_key]
                else:
                    # Get CrossRef title
                    cr_author = doi_entry.get("cr_first_author", "unknown")
                    cr_year = doi_entry.get("cr_year", "unknown")

                    if doi in cr_cache:
                        cr_title = cr_cache[doi].get("title", "Unknown")
                    else:
                        info = get_crossref_info(doi)
                        if info:
                            cr_cache[doi] = info
                            cr_title = info["title"]
                            cr_fetches += 1
                            if cr_fetches % 5 == 0:
                                time.sleep(0.5)
                        else:
                            cr_title = "Unknown"

                    # Extract context around DOI
                    doi_pos = answer.lower().find(doi.lower())
                    if doi_pos >= 0:
                        start = max(0, doi_pos - 400)
                        end = min(len(answer), doi_pos + 400)
                        excerpt = answer[start:end]
                    else:
                        excerpt = answer[:1200]

                    user_msg = USER_TEMPLATE.format(
                        doi=doi, cr_title=cr_title, cr_author=cr_author,
                        cr_year=cr_year, response_excerpt=excerpt,
                    )

                    verdict = call_groq(SYSTEM_PROMPT, user_msg)
                    cache[cache_key] = verdict
                    api_calls += 1

                    if api_calls % 20 == 0:
                        time.sleep(0.5)
                    if api_calls % 100 == 0:
                        with open(CACHE_FILE, "w") as f:
                            json.dump(cache, f)
                        with open(cr_cache_path, "w") as f:
                            json.dump(cr_cache, f)
                        print(f"  {gen}: {api_calls} API calls, {gen_total} DOIs processed...")

                if verdict == "VALID":
                    gen_valid += 1
                else:
                    gen_invalid += 1

        pct = gen_valid / gen_total * 100 if gen_total > 0 else 0
        results[gen] = {
            "total_cr_valid": gen_total,
            "valid": gen_valid,
            "invalid": gen_invalid,
            "vca_semantic_pct": round(pct, 2),
        }
        print(f"\n{gen}: {gen_valid}/{gen_total} = {pct:.1f}%")

    # Save
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    with open(cr_cache_path, "w") as f:
        json.dump(cr_cache, f, indent=2)
    with open(OUT_FILE, "w") as f:
        json.dump({"summary": results, "model": MODEL, "temperature": 0.0}, f, indent=2)

    # Comparison
    old_8b = {"Ours": 69.65, "GLM": 37.79, "Gemini": 50.41, "Grok": 61.48}
    old_heur = {"Ours": 82.16, "GLM": 9.98, "Gemini": 1.24, "Grok": 34.28}
    exi = {"Ours": 98.13, "GLM": 50.00, "Gemini": 84.83, "Grok": 80.10}

    print("\n" + "=" * 70)
    print("VCA COMPARISON: Heuristic vs 8B vs 70B")
    print("=" * 70)
    print(f"{'Gen':<10} {'Exist':>7} {'Heur':>7} {'8B':>7} {'70B':>7}")
    for gen in ["Ours", "GLM", "Gemini", "Grok"]:
        new = results[gen]["vca_semantic_pct"]
        print(f"{gen:<10} {exi[gen]:>6.1f}% {old_heur[gen]:>6.1f}% {old_8b[gen]:>6.1f}% {new:>6.1f}%")

    print(f"\nAPI calls: {api_calls}, CrossRef fetches: {cr_fetches}")
    print(f"Saved: {OUT_FILE.name}")


if __name__ == "__main__":
    main()
