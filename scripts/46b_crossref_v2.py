"""F4a: CrossRef DOI verification for all 4 generators × 350 questions.

Computes:
- Existence rate (DOI exists in CrossRef)
- Semantic match (cited author + year vs CrossRef metadata)
- Classification: OK / OK_year_drift / D_secondary / C_real_hallucination
- VCA per model
- HR per model
"""
# NOTE (release): this script is published as run. It reads and writes the working
# directory layout used during the experiments ("responses_350_<system>.json",
# "scores_*.json"); see the file-name mapping in README.md to match these against
# the released responses/, scores/ and results/ folders. API keys are read from a
# local .env; no credentials are included here.
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DRY = Path(__file__).resolve().parents[1]
HDRS = {"User-Agent": "REVAL/1.0 (mailto:anonymous@review.org)"}

GENERATORS = {
    "Ours":   DRY / "responses_350_RAG.json",
    "GLM":    DRY / "responses_350_glm_v2.json",
    "Gemini": DRY / "responses_350_gemini_v2.json",
    "Grok":   DRY / "responses_350_grok_v2.json",
    "GemSearch": DRY / "responses_350_gemsearch.json",
    "GPT54":     DRY / "responses_350_gpt54_v2.json",
}
OUT = DRY / "doi_4way_v2.json"

DOI_PLAIN = re.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', re.IGNORECASE)
PT_STOP = {"além","ainda","apesar","assim","como","contudo","entretanto","então","já","logo","mas","mesmo","não","onde","ora","outra","outro","para","pela","pelo","por","pois","porém","porque","portanto","quando","quanto","que","sem","somente","sobre","talvez","todavia","também","todos","tudo","entre","embora","depois","enquanto","durante","conforme","segundo","desde"}


def extract_author_year(text, doi_idx, window=200):
    pre = text[max(0, doi_idx-window):doi_idx]
    pre = re.sub(r'\s+', ' ', pre)
    year_matches = list(re.finditer(r'\b(19[5-9]\d|20[0-3]\d)\b', pre))
    if not year_matches: return None, None
    yr_m = year_matches[-1]
    year = yr_m.group(1)
    pre_year = pre[:yr_m.start()].rstrip(' ,([')
    m = re.search(r'((?:[A-Z][a-zA-ZÀ-ÿ\-]+|[A-Z]\.|et\s+al\.?|et)(?:\s+(?:[A-Z][a-zA-ZÀ-ÿ\-]+|[A-Z]\.|et\s+al\.?|et|e\s+colegas))*)\s*$', pre_year)
    if not m: return None, year
    words = re.findall(r'\b[A-Z][a-zA-ZÀ-ÿ\-]+\b', m.group(1))
    words = [w for w in words if w.lower() not in PT_STOP and w.lower() not in ('et','al')]
    if not words: return None, year
    return words[0].lower(), year


CR_CACHE = {}
cache_lock = threading.Lock()

def crossref_lookup(doi):
    with cache_lock:
        if doi in CR_CACHE: return CR_CACHE[doi]
    for attempt in range(3):
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20, headers=HDRS)
            if r.status_code == 200:
                m = r.json().get("message", {})
                authors = m.get("author") or []
                all_families = [a.get("family","").lower() for a in authors if a.get("family")]
                year = ""
                dp = m.get("issued", {}).get("date-parts") or m.get("published-print", {}).get("date-parts") or m.get("created", {}).get("date-parts")
                if dp and dp[0]: year = str(dp[0][0])
                res = {"valid": True, "all_authors": all_families, "year": year, "title": (m.get("title") or [""])[0][:100]}
                break
            elif r.status_code == 404:
                res = {"valid": False, "status": 404}
                break
            else:
                time.sleep(2 ** attempt)
                res = {"valid": False, "status": r.status_code}
        except Exception as e:
            time.sleep(2 ** attempt)
            res = {"valid": False, "error": str(e)[:80]}
    with cache_lock:
        CR_CACHE[doi] = res
    time.sleep(0.1)
    return res


def classify(cited_a, cited_y, real_authors, real_year):
    if not real_authors: return "no_data"
    if not cited_a: return "no_context"
    is_first = real_authors and (cited_a == real_authors[0] or cited_a in real_authors[0] or real_authors[0] in cited_a)
    is_in_list = any(cited_a == a or cited_a in a or a in cited_a for a in real_authors)
    try: yd = abs(int(cited_y) - int(real_year)) if cited_y and real_year else 99
    except: yd = 99
    if is_first and yd == 0: return "OK"
    if is_first and yd <= 1: return "OK_year_drift"
    if is_in_list and yd <= 1: return "D_secondary"
    if is_in_list: return "D_secondary_drift"
    return "C_real_hallucination"


def process_response(q, label):
    ans = q.get("answer","") or ""
    dois_data = []
    for m in DOI_PLAIN.finditer(ans):
        doi = m.group(1).rstrip('.,;:)]}>').lower()
        cr = crossref_lookup(doi)
        cited_a, cited_y = extract_author_year(ans, m.start())
        if cr.get("valid"):
            cat = classify(cited_a, cited_y, cr["all_authors"], cr["year"])
        else:
            cat = "existence_invalid"
        dois_data.append({
            "doi": doi,
            "cited_author": cited_a, "cited_year": cited_y,
            "cr_valid": cr.get("valid", False),
            "cr_first_author": (cr.get("all_authors") or [None])[0],
            "cr_year": cr.get("year"),
            "category": cat,
        })
    return {"qid": q["id"], "n_dois": len(dois_data), "dois": dois_data}


def run_model(label, path):
    items = json.loads(path.read_text())
    print(f"\n=== {label} (n={len(items)}) ===")
    results = []
    done = [0]
    lock = threading.Lock()

    def proc(q):
        r = process_response(q, label)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f"  [{label}] {done[0]}/{len(items)}", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(proc, q) for q in items]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x["qid"])

    # Aggregate
    total_dois = sum(r["n_dois"] for r in results)
    cats = Counter()
    for r in results:
        for d in r["dois"]:
            cats[d["category"]] += 1

    qs_with_dois = sum(1 for r in results if r["n_dois"] > 0)
    existence_valid = total_dois - cats.get("existence_invalid", 0)
    halluc_real = cats.get("C_real_hallucination", 0)
    ok = cats.get("OK", 0)
    ok_drift = cats.get("OK", 0) + cats.get("OK_year_drift", 0)
    secondary = cats.get("D_secondary", 0) + cats.get("D_secondary_drift", 0)

    summary = {
        "label": label,
        "n_questions": len(items),
        "questions_with_dois": qs_with_dois,
        "total_dois": total_dois,
        "dois_per_q": round(total_dois / len(items), 2),
        "existence_valid": existence_valid,
        "vca_existence": round(existence_valid / total_dois * 100, 2) if total_dois else 0,
        "hr_existence": round((cats.get("existence_invalid", 0)) / total_dois * 100, 2) if total_dois else 0,
        "ok_strict": ok,
        "ok_pct": round(ok / total_dois * 100, 2) if total_dois else 0,
        "ok_drift_total": ok_drift,
        "ok_drift_pct": round(ok_drift / total_dois * 100, 2) if total_dois else 0,
        "secondary_pct": round(secondary / total_dois * 100, 2) if total_dois else 0,
        "halluc_real": halluc_real,
        "halluc_real_pct": round(halluc_real / total_dois * 100, 2) if total_dois else 0,
        "no_context": cats.get("no_context", 0),
        "categories": dict(cats),
    }
    return {"summary": summary, "results": results}


def main():
    # Load existing results to skip already-processed generators
    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text())
    for label, path in GENERATORS.items():
        if label in out:
            print(f"SKIP {label}: already in {OUT.name}")
            continue
        if not path.exists():
            print(f"SKIP {label}: file not found ({path.name})")
            continue
        out[label] = run_model(label, path)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n" + "="*100)
    print("F4a CROSSREF SUMMARY (4 generators × 350 questions)")
    print("="*100)
    print(f"\n{'Model':<10} {'nQ':>4} {'DOIs':>6} {'/Q':>5} {'Exist%':>8} {'OK%':>7} {'OK+drift%':>11} {'Sec%':>6} {'Halluc%':>8}")
    print("-"*100)
    for label, data in out.items():
        s = data["summary"]
        print(f"{label:<10} {s['n_questions']:>4} {s['total_dois']:>6} {s['dois_per_q']:>5.1f} {s['vca_existence']:>7.2f}% {s['ok_pct']:>6.2f}% {s['ok_drift_pct']:>10.2f}% {s['secondary_pct']:>5.2f}% {s['halluc_real_pct']:>7.2f}%")
    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
